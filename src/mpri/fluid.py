from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class VortexObject:
    """A localized coherent object used as a physicalized model weight.

    Parameters are dimensionless in a 2π-periodic box.
    circulation controls signed vorticity amplitude, while sigma controls size.
    """

    x: float
    y: float
    circulation: float
    sigma: float = 0.28


@dataclass(frozen=True)
class Probe:
    x: float
    y: float
    sigma: float = 0.35


class VorticityFluid2D:
    """Small pseudo-spectral 2-D incompressible Navier–Stokes solver.

    Evolves scalar vorticity ω on a 2π-periodic square:

        ∂t ω + u·∇ω = ν Δω - μ ω + f,
        u = (∂y ψ, -∂x ψ),  -Δψ = ω.

    The implementation is deliberately compact and deterministic. It is a
    research toy, not a CFD production solver.
    """

    def __init__(
        self,
        n: int = 48,
        viscosity: float = 3e-3,
        drag: float = 2e-2,
        dt: float = 0.012,
        dealias: bool = True,
    ) -> None:
        if n < 16 or n % 2:
            raise ValueError("n must be an even integer >= 16")
        self.n = int(n)
        self.viscosity = float(viscosity)
        self.drag = float(drag)
        self.dt = float(dt)
        self.length = 2.0 * np.pi
        x = np.linspace(0.0, self.length, self.n, endpoint=False)
        self.x, self.y = np.meshgrid(x, x, indexing="ij")

        # With a 2π box, FFT integer frequencies are exactly the wave numbers.
        k = np.fft.fftfreq(self.n, d=1.0 / self.n)
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.k2_safe = self.k2.copy()
        self.k2_safe[0, 0] = 1.0

        if dealias:
            cutoff = self.n / 3.0
            self.dealias_mask = (np.abs(self.kx) <= cutoff) & (np.abs(self.ky) <= cutoff)
        else:
            self.dealias_mask = np.ones((self.n, self.n), dtype=bool)

    def _periodic_delta(self, a: Array, b: float) -> Array:
        d = a - b
        return (d + np.pi) % (2.0 * np.pi) - np.pi

    def gaussian_vortex(self, x: float, y: float, circulation: float, sigma: float) -> Array:
        dx = self._periodic_delta(self.x, x)
        dy = self._periodic_delta(self.y, y)
        r2 = dx * dx + dy * dy
        blob = np.exp(-0.5 * r2 / (sigma * sigma))
        # Remove the spatial mean so the periodic Poisson solve is well-defined.
        blob -= blob.mean()
        norm = np.max(np.abs(blob))
        if norm > 0:
            blob /= norm
        return circulation * blob

    def field_from_objects(self, objects: Iterable[VortexObject]) -> Array:
        omega = np.zeros((self.n, self.n), dtype=np.float64)
        for obj in objects:
            omega += self.gaussian_vortex(obj.x, obj.y, obj.circulation, obj.sigma)
        return omega

    def velocity(self, omega: Array) -> tuple[Array, Array]:
        wh = np.fft.fft2(omega)
        psi_h = wh / self.k2_safe
        psi_h[0, 0] = 0.0
        u = np.fft.ifft2(1j * self.ky * psi_h).real
        v = np.fft.ifft2(-1j * self.kx * psi_h).real
        return u, v

    def divergence_rms(self, omega: Array) -> float:
        u, v = self.velocity(omega)
        uh = np.fft.fft2(u)
        vh = np.fft.fft2(v)
        div = np.fft.ifft2(1j * self.kx * uh + 1j * self.ky * vh).real
        return float(np.sqrt(np.mean(div * div)))

    def rhs(self, omega: Array, forcing: Array | None = None) -> Array:
        wh = np.fft.fft2(omega)
        psi_h = wh / self.k2_safe
        psi_h[0, 0] = 0.0

        u = np.fft.ifft2(1j * self.ky * psi_h).real
        v = np.fft.ifft2(-1j * self.kx * psi_h).real
        wx = np.fft.ifft2(1j * self.kx * wh).real
        wy = np.fft.ifft2(1j * self.ky * wh).real

        nonlinear = -(u * wx + v * wy)
        nh = np.fft.fft2(nonlinear)
        nh *= self.dealias_mask
        nonlinear = np.fft.ifft2(nh).real

        lap = np.fft.ifft2(-self.k2 * wh).real
        out = nonlinear + self.viscosity * lap - self.drag * omega
        if forcing is not None:
            out = out + forcing
        return out

    def step(self, omega: Array, forcing: Array | None = None) -> Array:
        dt = self.dt
        k1 = self.rhs(omega, forcing)
        k2 = self.rhs(omega + 0.5 * dt * k1, forcing)
        k3 = self.rhs(omega + 0.5 * dt * k2, forcing)
        k4 = self.rhs(omega + dt * k3, forcing)
        nxt = omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Numerical roundoff can accumulate a tiny mean mode.
        return nxt - nxt.mean()

    def evolve(self, omega: Array, steps: int, forcing: Array | None = None) -> Array:
        state = np.array(omega, dtype=np.float64, copy=True)
        for _ in range(int(steps)):
            state = self.step(state, forcing)
        return state

    def kinetic_energy(self, omega: Array) -> float:
        u, v = self.velocity(omega)
        return float(0.5 * np.mean(u * u + v * v))

    def enstrophy(self, omega: Array) -> float:
        return float(0.5 * np.mean(omega * omega))

    def probe_masks(self, probes: Sequence[Probe]) -> Array:
        masks = []
        for p in probes:
            dx = self._periodic_delta(self.x, p.x)
            dy = self._periodic_delta(self.y, p.y)
            m = np.exp(-0.5 * (dx * dx + dy * dy) / (p.sigma * p.sigma))
            m /= m.sum()
            masks.append(m)
        return np.stack(masks, axis=0)

    def read(self, omega: Array, probes: Sequence[Probe], mode: str = "vorticity") -> Array:
        masks = self.probe_masks(probes)
        if mode == "vorticity":
            field = omega
        elif mode == "speed":
            u, v = self.velocity(omega)
            field = np.sqrt(u * u + v * v)
        elif mode == "energy":
            u, v = self.velocity(omega)
            field = 0.5 * (u * u + v * v)
        else:
            raise ValueError(f"unknown read mode: {mode}")
        return np.einsum("pij,ij->p", masks, field)
