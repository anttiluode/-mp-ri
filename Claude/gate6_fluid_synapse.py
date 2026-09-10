"""
gate6_fluid_synapse.py — Five-World Fluid Synapse Benchmark.

Evaluates whether the quadratic Navier-Stokes convective cross-term 
acts as an autonomous fluid synapse.

The 5 Worlds (all run for the exact same physical duration T_train):
  W_0   : Sham (background flow only, no pulses)
  W_A   : Pulse A alone (tests unilateral A write & aging)
  W_B   : Pulse B alone (tests unilateral B write & aging)
  W_AB  : Pulses A + B (tests true collision & interaction)
  W_sep : Pulses A + B spatially separated (controls for total energy without collision)

Recall Test:
  Inject Cue A into all 5 frozen/relaxed operators.
  Measures isolated synaptic routing:
      M = R_AB(A) - R_A(A) - R_B(A) + R_0(A)
"""

import numpy as np

class FluidSynapseSubstrate:
    def __init__(self, N=128, L=2*np.pi, nu=6e-4, dt=2e-3, k_split=5):
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

    def add_vortex(self, x0, y0, sign, amp=12.0, sigma=0.5):
        dx = (self.x - x0 + self.L / 2) % self.L - self.L / 2
        dy = (self.y - y0 + self.L / 2) % self.L - self.L / 2
        self.omega += sign * amp * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    def inject_pulse(self, x0, y0, k_carrier=16.0, amp=20.0, sigma=0.22, direction='x'):
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

    def get_slow(self):
        w_hat = np.fft.fft2(self.omega)
        return np.real(np.fft.ifft2(w_hat * self.slow_mask))

    def get_fast_norm(self):
        w_hat = np.fft.fft2(self.omega)
        w_fast = np.real(np.fft.ifft2(w_hat * self.fast_mask))
        return np.linalg.norm(w_fast)

    def probe_patch(self, x0, y0, half_px=5):
        ix = int(round(x0 / self.L * self.N)) % self.N
        iy = int(round(y0 / self.L * self.N)) % self.N
        idx_x = [(ix + d) % self.N for d in range(-half_px, half_px + 1)]
        idx_y = [(iy + d) % self.N for d in range(-half_px, half_px + 1)]
        return self.omega[np.ix_(idx_x, idx_y)].flatten()


def run_benchmark():
    L = 2 * np.pi
    CENTER = L / 2

    # Baseline Jet Architecture: Counter-rotating dipole
    V_LEFT = (CENTER - 0.75, CENTER)
    V_RIGHT = (CENTER + 0.75, CENTER)

    # Input Ports: A enters jet throat; B enters cross-stream
    SITE_A = (CENTER - 0.20, CENTER - 1.15)
    SITE_B = (CENTER + 0.35, CENTER - 0.85)

    # Non-interacting control for B (placed in quiescent boundary zone)
    SITE_B_SEP = (CENTER + 2.0, CENTER - 1.15)

    # Downstream Detectors
    DET_A = (CENTER - 0.20, CENTER + 1.20)
    DET_B = (CENTER + 0.90, CENTER + 0.70)

    print("=== Step 1: Generating Initial Settled State ===")
    init_sim = FluidSynapseSubstrate(N=128, nu=6e-4, dt=2e-3, k_split=5)
    init_sim.add_vortex(*V_LEFT, sign=+1, amp=12.0)
    init_sim.add_vortex(*V_RIGHT, sign=-1, amp=12.0)
    for _ in range(300):
        init_sim.step()
    omega_0 = np.copy(init_sim.omega)

    # Helper to instantiate an identical branch world
    def make_world():
        w = FluidSynapseSubstrate(N=128, nu=6e-4, dt=2e-3, k_split=5)
        w.omega = np.copy(omega_0)
        return w

    worlds = {
        'W0':   make_world(),
        'WA':   make_world(),
        'WB':   make_world(),
        'WAB':  make_world(),
        'Wsep': make_world()
    }

    print("\n=== Step 2: Training 5 Parallel Worlds (Exact Step Parity) ===")
    # Inject respective pulses
    worlds['WA'].inject_pulse(*SITE_A, amp=22.0, direction='x')
    worlds['WB'].inject_pulse(*SITE_B, amp=22.0, direction='y')
    
    worlds['WAB'].inject_pulse(*SITE_A, amp=22.0, direction='x')
    worlds['WAB'].inject_pulse(*SITE_B, amp=22.0, direction='y')

    worlds['Wsep'].inject_pulse(*SITE_A, amp=22.0, direction='x')
    worlds['Wsep'].inject_pulse(*SITE_B_SEP, amp=22.0, direction='y')

    TRAIN_STEPS = 1000
    for step in range(TRAIN_STEPS):
        for w in worlds.values():
            w.step()

    print(f"Elapsed training steps: {TRAIN_STEPS}")
    for name, w in worlds.items():
        f_norm = w.get_fast_norm()
        s_norm = np.linalg.norm(w.get_slow())
        print(f"  World {name:5s} | Fast/Slow Ratio: {f_norm / (s_norm + 1e-12):.6e}")

    print("\n=== Step 3: Measuring Physical Collision-Specific Operator Write ===")
    omega_slow = {k: v.get_slow() for k, v in worlds.items()}
    # Delta Omega_collision = P_slow[omega_AB - omega_A - omega_B + omega_0]
    delta_omega_collision = omega_slow['WAB'] - omega_slow['WA'] - omega_slow['WB'] + omega_slow['W0']
    collision_write_norm = np.linalg.norm(delta_omega_collision)
    print(f"Collision-specific write ||Delta Omega_collision||: {collision_write_norm:.6f}")

    print("\n=== Step 4: Recall Probing (Cue A Injected into All 5 Worlds) ===")
    RECALL_STEPS = 450

    def test_recall_at_detector(target_world_omega):
        # Evolve cue branch
        sim_cue = FluidSynapseSubstrate(N=128, nu=6e-4, dt=2e-3, k_split=5)
        sim_cue.omega = np.copy(target_world_omega)
        sim_cue.inject_pulse(*SITE_A, amp=20.0, direction='x')

        # Evolve background reference
        sim_ref = FluidSynapseSubstrate(N=128, nu=6e-4, dt=2e-3, k_split=5)
        sim_ref.omega = np.copy(target_world_omega)

        peak_dA = 0.0
        peak_dB = 0.0
        for _ in range(RECALL_STEPS):
            sim_cue.step()
            sim_ref.step()
            patch_dA = np.linalg.norm(sim_cue.probe_patch(*DET_A) - sim_ref.probe_patch(*DET_A))
            patch_dB = np.linalg.norm(sim_cue.probe_patch(*DET_B) - sim_ref.probe_patch(*DET_B))
            if patch_dA > peak_dA: peak_dA = patch_dA
            if patch_dB > peak_dB: peak_dB = patch_dB
        return peak_dA, peak_dB

    readouts_B = {}
    readouts_A = {}
    for name, w in worlds.items():
        rA, rB = test_recall_at_detector(w.omega)
        readouts_A[name] = rA
        readouts_B[name] = rB
        print(f"  World {name:5s} Recall -> Det A (Straight): {rA:.6f} | Det B (Cross): {rB:.6f}")

    # Isolated interaction routing metric: M = R_AB(A) - R_A(A) - R_B(A) + R_0(A)
    M_isolated = readouts_B['WAB'] - readouts_B['WA'] - readouts_B['WB'] + readouts_B['W0']
    M_sep_control = readouts_B['Wsep'] - readouts_B['WA'] - readouts_B['WB'] + readouts_B['W0']

    print("\n=== Gate 6 Verdict: Fluid Synapse Isolation ===")
    print(f"Collision-Specific Routing Metric M_isolated   : {M_isolated:+.6f}")
    print(f"Separated Control Metric M_sep_control         : {M_sep_control:+.6f}")

    if abs(M_isolated) > 0.05 and abs(M_isolated) > 3 * abs(M_sep_control):
        print("\n[CONFIRMED]: Fluid Synapse detected. The quadratic collision wrote an autonomous routing bias distinct from drift, energy dumping, and unilateral writes.")
    else:
        print("\n[UNESTABLISHED]: Collision effect not isolated from background maturation or baseline leakage.")

if __name__ == "__main__":
    run_benchmark()