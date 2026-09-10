"""
ns_core.py — the substrate.

2D incompressible Navier-Stokes, vorticity-streamfunction form, pseudo-spectral
(FFT), periodic domain [0, 2pi) x [0, 2pi). Standard method, nothing novel here:

    omega_t + u.grad(omega) = nu * lap(omega) - hypernu * (-lap)^p omega
    lap(psi) = -omega
    u = psi_y ,  v = -psi_x

The "weights" of a FlowMatrix system are NOT a matrix. They are this field:
omega(x,y,t). A background coherent structure (a vortex pair) plays the role
CausalHorizon/Operaattori called "the operator" -- except here it is not fixed,
it is advected and reshaped by whatever gets injected into it. That backreaction
is the whole bet. This file only provides the physics; it makes no claim about
whether that bet pays off as a computer. run_additivity.py and run_xor.py test
that honestly.
"""
import numpy as np

class FlowField:
    def __init__(self, N=64, L=2*np.pi, nu=5e-4, hyper_nu=1e-8, hyper_p=8, dt=5e-3):
        self.N, self.L, self.nu, self.dt = N, L, nu, dt
        self.hyper_nu, self.hyper_p = hyper_nu, hyper_p

        k = np.fft.fftfreq(N, d=L/N) * 2*np.pi
        KX, KY = np.meshgrid(k, k, indexing='ij')
        self.KX, self.KY = KX, KY
        K2 = KX**2 + KY**2
        self.K2 = K2
        K2_inv = np.zeros_like(K2)
        K2_inv[K2 > 0] = 1.0 / K2[K2 > 0]
        self.K2_inv = K2_inv

        # 2/3 dealiasing mask
        kmax = np.max(np.abs(k)) * (2/3)
        self.dealias = (np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)

        x = np.linspace(0, L, N, endpoint=False)
        self.x, self.y = np.meshgrid(x, x, indexing='ij')

        self.omega = np.zeros((N, N))

        # linear operator (diffusion + hyperviscosity), diagonal in Fourier space
        Lop = -nu*K2 - hyper_nu*(K2**hyper_p)
        self._Lop = Lop
        self._E = np.exp(Lop*dt)
        self._E2 = np.exp(Lop*dt/2)
        self._phi1_full = self._phi1(Lop*dt)
        self._phi1_half = self._phi1(Lop*dt/2)

    @staticmethod
    def _phi1(z):
        """phi1(z) = (e^z - 1)/z, elementwise, safe at z=0."""
        out = np.ones_like(z)
        mask = np.abs(z) > 1e-10
        out[mask] = (np.expm1(z[mask])) / z[mask]
        return out

    def add_gaussian_vortex(self, x0, y0, sign, amp=8.0, sigma=0.35):
        """Inject a localized Gaussian vortex blob -- a 'wave packet' / bit."""
        L = self.L
        dx = (self.x - x0 + L/2) % L - L/2
        dy = (self.y - y0 + L/2) % L - L/2
        r2 = dx**2 + dy**2
        self.omega = self.omega + sign * amp * np.exp(-r2 / (2*sigma**2))

    def velocity(self):
        omega_hat = np.fft.fft2(self.omega)
        psi_hat = omega_hat * self.K2_inv
        u = np.real(np.fft.ifft2(1j*self.KY*psi_hat))
        v = np.real(np.fft.ifft2(-1j*self.KX*psi_hat))
        return u, v

    def _nonlinear(self, omega_hat):
        """N(w) = -(u.grad)w, dealiased. Diffusion/hyperviscosity are handled
        separately (exactly, via the integrating factor), so this excludes them."""
        psi_hat = omega_hat * self.K2_inv
        u = np.real(np.fft.ifft2(1j*self.KY*psi_hat))
        v = np.real(np.fft.ifft2(-1j*self.KX*psi_hat))
        omega_x = np.real(np.fft.ifft2(1j*self.KX*omega_hat))
        omega_y = np.real(np.fft.ifft2(1j*self.KY*omega_hat))
        nonlinear = u*omega_x + v*omega_y
        return -np.fft.fft2(nonlinear) * self.dealias

    def step(self):
        """One step of ETD-RK2 (Cox & Matthews style): the linear part
        (diffusion + hyperviscosity) is solved exactly via the integrating
        factor, the nonlinear advection term via a 2nd-order correction.
        Unconditionally stable w.r.t. the (stiff) linear part."""
        w_hat = np.fft.fft2(self.omega)
        N1 = self._nonlinear(w_hat)
        a_hat = self._E*w_hat + self.dt*self._phi1_full*N1
        N2 = self._nonlinear(a_hat)
        w_hat = a_hat + self.dt*self._phi1_full*(N2 - N1)
        self.omega = np.real(np.fft.ifft2(w_hat))

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()

    def probe_patch(self, x0, y0, half_px=6):
        """Downstream readout as a flattened patch of the vorticity field
        (a richer, multi-dimensional analogue of a single-point probe --
        this is what run_additivity.py compares across trials)."""
        N, L = self.N, self.L
        ix = int(round(x0 / L * N)) % N
        iy = int(round(y0 / L * N)) % N
        idx_x = [(ix + d) % N for d in range(-half_px, half_px+1)]
        idx_y = [(iy + d) % N for d in range(-half_px, half_px+1)]
        patch = self.omega[np.ix_(idx_x, idx_y)]
        return patch.flatten()

    def probe(self, x0, y0, radius=0.4):
        """Local readout: mean vorticity and mean speed in a disk around (x0,y0)."""
        L = self.L
        dx = (self.x - x0 + L/2) % L - L/2
        dy = (self.y - y0 + L/2) % L - L/2
        mask = (dx**2 + dy**2) <= radius**2
        u, v = self.velocity()
        return {
            'omega_mean': float(self.omega[mask].mean()),
            'omega_absmean': float(np.abs(self.omega[mask]).mean()),
            'speed_mean': float(np.sqrt(u[mask]**2 + v[mask]**2).mean()),
        }