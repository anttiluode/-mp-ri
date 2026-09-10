"""
run_clean_associative.py — Clean Differential Routing Test.

Background: Two separated vortex cores create a divergent saddle.
Pulse A routes to Detector A (left). Detector B (right) stays dark at baseline.
We test whether co-injecting (A + B) writes a transverse bridge in Omega_slow
that reroutes Pulse A into Detector B.
"""

import numpy as np

class CleanFluidMemory:
    def __init__(self, N=128, L=2*np.pi, nu=5e-4, dt=2e-3, k_split=5):
        self.N, self.L, self.nu, self.dt = N, L, nu, dt
        self.k_split = k_split

        k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(k, k, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2

        self.K2_inv = np.zeros_like(self.K2)
        self.K2_inv[self.K2 > 0] = 1.0 / self.K2[self.K2 > 0]

        kmax = np.max(np.abs(k)) * (2 / 3)
        self.dealias = (np.abs(self.KX) <= kmax) & (np.abs(self.KY) <= kmax)

        self.slow_mask = np.sqrt(self.K2) <= k_split
        self.fast_mask = (np.sqrt(self.K2) > k_split) & self.dealias

        x = np.linspace(0, L, N, endpoint=False)
        self.x, self.y = np.meshgrid(x, x, indexing='ij')
        self.omega = np.zeros((N, N))

        Lop = -nu * self.K2
        self._E = np.exp(Lop * dt)
        self._phi1 = np.ones_like(Lop)
        mask = np.abs(Lop * dt) > 1e-10
        self._phi1[mask] = np.expm1(Lop[mask] * dt) / (Lop[mask] * dt)

    def add_vortex(self, x0, y0, sign, amp=14.0, sigma=0.45):
        dx = (self.x - x0 + self.L / 2) % self.L - self.L / 2
        dy = (self.y - y0 + self.L / 2) % self.L - self.L / 2
        self.omega += sign * amp * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    def inject_pulse(self, x0, y0, k_carrier=14.0, amp=20.0, sigma=0.25, direction='x'):
        dx = (self.x - x0 + self.L / 2) % self.L - self.L / 2
        dy = (self.y - y0 + self.L / 2) % self.L - self.L / 2
        env = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        carrier = np.cos(k_carrier * self.x) if direction == 'x' else np.cos(k_carrier * self.y)
        self.omega += amp * env * carrier

    def step(self):
        w_hat = np.fft.fft2(self.omega)
        psi_hat = w_hat * self.K2_inv
        u = np.real(np.fft.ifft2(1j * self.KY * psi_hat))
        v = np.real(np.fft.ifft2(-1j * self.KX * psi_hat))
        omega_x = np.real(np.fft.ifft2(1j * self.KX * w_hat))
        omega_y = np.real(np.fft.ifft2(1j * self.KY * w_hat))
        N1 = -np.fft.fft2(u * omega_x + v * omega_y) * self.dealias
        a_hat = self._E * w_hat + self.dt * self._phi1 * N1
        
        psi_a = a_hat * self.K2_inv
        ua = np.real(np.fft.ifft2(1j * self.KY * psi_a))
        va = np.real(np.fft.ifft2(-1j * self.KX * psi_a))
        omega_ax = np.real(np.fft.ifft2(1j * self.KX * a_hat))
        omega_ay = np.real(np.fft.ifft2(1j * self.KY * a_hat))
        N2 = -np.fft.fft2(ua * omega_ax + va * omega_ay) * self.dealias
        
        w_hat = a_hat + self.dt * self._phi1 * (N2 - N1)
        self.omega = np.real(np.fft.ifft2(w_hat))

    def get_scale_norms(self):
        w_hat = np.fft.fft2(self.omega)
        w_slow = np.real(np.fft.ifft2(w_hat * self.slow_mask))
        w_fast = np.real(np.fft.ifft2(w_hat * self.fast_mask))
        return np.linalg.norm(w_slow), np.linalg.norm(w_fast)

    def probe_patch(self, x0, y0, half_px=5):
        """Extracts flat vorticity patch."""
        ix = int(round(x0 / self.L * self.N)) % self.N
        iy = int(round(y0 / self.L * self.N)) % self.N
        idx_x = [(ix + d) % self.N for d in range(-half_px, half_px + 1)]
        idx_y = [(iy + d) % self.N for d in range(-half_px, half_px + 1)]
        return self.omega[np.ix_(idx_x, idx_y)].flatten()


def run_experiment():
    L = 2 * np.pi
    CENTER = L / 2

    # Two co-rotating vortices setting up an asymmetric dividing streamline
    VORTEX_LEFT = (CENTER - 1.0, CENTER)
    VORTEX_RIGHT = (CENTER + 1.0, CENTER)

    # Inputs at the base
    SITE_A = (CENTER - 0.7, CENTER - 1.2)
    SITE_B = (CENTER + 0.7, CENTER - 1.2)

    # Downstream detectors
    DET_A = (CENTER - 0.8, CENTER + 1.1)
    DET_B = (CENTER + 0.8, CENTER + 1.1)

    print("=== Step 1: Settling Background Operator ===")
    sim = CleanFluidMemory(N=128, nu=5e-4, dt=2e-3, k_split=5)
    sim.add_vortex(*VORTEX_LEFT, sign=+1, amp=14.0)
    sim.add_vortex(*VORTEX_RIGHT, sign=+1, amp=14.0)
    for _ in range(400):
        sim.step()
    base_omega = np.copy(sim.omega)

    # Function to get differential patch: delta = patch(cue) - patch(base_evolved)
    def probe_differential(cue_sites, steps=500):
        test_sim = CleanFluidMemory(N=128, nu=5e-4, dt=2e-3, k_split=5)
        test_sim.omega = np.copy(base_omega)
        for (x, y, d) in cue_sites:
            test_sim.inject_pulse(x, y, amp=22.0, direction=d)
        
        # Also evolve clean background to subtract it out exactly
        ref_sim = CleanFluidMemory(N=128, nu=5e-4, dt=2e-3, k_split=5)
        ref_sim.omega = np.copy(base_omega)

        diff_b_history = []
        for _ in range(steps):
            test_sim.step()
            ref_sim.step()
            patch_b = test_sim.probe_patch(*DET_B) - ref_sim.probe_patch(*DET_B)
            diff_b_history.append(np.linalg.norm(patch_b))
        return max(diff_b_history)

    print("\n=== Step 2: Baseline Differential Signal (Cue A Alone) ===")
    baseline_leak = probe_differential([(SITE_A[0], SITE_A[1], 'x')])
    print(f"Differential perturbation at Detector B from Cue A: {baseline_leak:.6f}")

    print("\n=== Step 3: Training via Co-occurrence (A + B) ===")
    STEPS_PER_CYCLE = 1000  # Give fast modes enough time to dissipate
    for cycle in range(1, 5):
        sim.inject_pulse(*SITE_A, amp=24.0, direction='x')
        sim.inject_pulse(*SITE_B, amp=24.0, direction='y')
        for _ in range(STEPS_PER_CYCLE):
            sim.step()
        s_norm, f_norm = sim.get_scale_norms()
        print(f"Cycle {cycle}/4 complete | ||w_fast|| / ||Omega_slow|| = {f_norm / s_norm:.6e}")

    # The new trained background
    base_omega = np.copy(sim.omega)

    print("\n=== Step 4: Recall (Cue A Alone on Trained Operator) ===")
    recalled_signal = probe_differential([(SITE_A[0], SITE_A[1], 'x')])
    print(f"Differential perturbation at Detector B from Cue A: {recalled_signal:.6f}")

    gain = recalled_signal / (baseline_leak + 1e-12)
    print("\n=== Results ===")
    print(f"Baseline |dA -> B| : {baseline_leak:.6f}")
    print(f"Trained  |dA -> B| : {recalled_signal:.6f}")
    print(f"Routing Gain       : {gain:.2f}x")

if __name__ == "__main__":
    run_experiment()