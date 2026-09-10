"""
ns_operator.py — Self-Writing Fluid Operator
Implements scale separation: Slow Field (Weights) + Fast Packets (Activations).
Demonstrates permanent backreaction of fast collision stress onto the slow operator.
"""
import numpy as np
from ns_core import FlowField

class ScaleSeparatedField(FlowField):
    def __init__(self, N=128, L=2*np.pi, nu=4e-4, dt=2e-3, k_split=6):
        super().__init__(N=N, L=L, nu=nu, dt=dt)
        self.k_split = k_split
        self.slow_mask = np.sqrt(self.K2) <= k_split
        self.fast_mask = (np.sqrt(self.K2) > k_split) & self.dealias

    def get_slow_fast(self):
        w_hat = np.fft.fft2(self.omega)
        w_slow = np.real(np.fft.ifft2(w_hat * self.slow_mask))
        w_fast = np.real(np.fft.ifft2(w_hat * self.fast_mask))
        return w_slow, w_fast

    def measure_reynolds_stress_transfer(self):
        """
        Computes the instantaneous momentum flux transfer:
        T_reynolds = P_slow( curl( div( u_fast (x) u_fast ) ) )
        """
        _, w_fast = self.get_slow_fast()
        w_fast_hat = np.fft.fft2(w_fast)
        psi_fast_hat = w_fast_hat * self.K2_inv
        
        u_f = np.real(np.fft.ifft2(1j * self.KY * psi_fast_hat))
        v_f = np.real(np.fft.ifft2(-1j * self.KX * psi_fast_hat))
        
        # Convective advection of fast field
        dx_w = np.real(np.fft.ifft2(1j * self.KX * w_fast_hat))
        dy_w = np.real(np.fft.ifft2(1j * self.KY * w_fast_hat))
        adv_fast = u_f * dx_w + v_f * dy_w
        
        # Project into slow scale
        transfer_hat = np.fft.fft2(adv_fast) * self.slow_mask
        return np.real(np.fft.ifft2(transfer_hat))


def run_plasticity_test():
    L = 2 * np.pi
    CENTER = L / 2
    sim = ScaleSeparatedField(N=128, dt=2e-3, nu=4e-4, k_split=5)

    # 1. Initialize Slow Operator: Dipole background (the weights)
    sim.add_gaussian_vortex(CENTER - 0.8, CENTER, sign=+1, amp=12.0, sigma=0.6)
    sim.add_gaussian_vortex(CENTER + 0.8, CENTER, sign=-1, amp=12.0, sigma=0.6)
    w_slow_init, _ = sim.get_slow_fast()

    # 2. Inject Fast Packet Collision (the activations)
    k_carrier = 14.0
    Y_INJECT = CENTER - 1.2
    
    # Fast oscillatory pulse A (sinusoidally modulated core)
    rA2 = (sim.x - (CENTER - 0.4))**2 + (sim.y - Y_INJECT)**2
    envA = np.exp(-rA2 / (2 * 0.25**2))
    pulseA = 18.0 * envA * np.cos(k_carrier * sim.x)
    
    # Fast oscillatory pulse B
    rB2 = (sim.x - (CENTER + 0.4))**2 + (sim.y - Y_INJECT)**2
    envB = np.exp(-rB2 / (2 * 0.25**2))
    pulseB = 18.0 * envB * np.cos(k_carrier * sim.y)
    
    sim.omega += (pulseA + pulseB)
    
    print("Simulating pulse transit and viscous relaxation...")
    # Evolve forward until fast pulses transfer stress and dissipate
    for step in range(1200):
        sim.step()

    # 3. Analyze Operator Residuals
    w_slow_final, w_fast_final = sim.get_slow_fast()
    
    delta_slow = np.linalg.norm(w_slow_final - w_slow_init) / np.linalg.norm(w_slow_init)
    fast_residual = np.linalg.norm(w_fast_final) / np.linalg.norm(w_slow_final)
    
    print(f"Permanent operator shift (||Delta Omega_slow|| / ||Omega_slow||): {delta_slow:.6f}")
    print(f"Fast activation dissipation (||w_fast|| / ||Omega_slow||):      {fast_residual:.6f}")

if __name__ == "__main__":
    run_plasticity_test()