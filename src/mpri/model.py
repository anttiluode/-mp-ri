from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .fluid import Probe, VortexObject, VorticityFluid2D

Array = np.ndarray


@dataclass
class FluidWeightLayer:
    """A tiny fluid layer where trainable parameters are vortex objects.

    Inputs are localized vorticity pulses, the vortex objects form a persistent
    background field, depth is physical time evolution, and outputs are local
    probes. This makes the phrase "weights as objects in the field" literal.
    """

    fluid: VorticityFluid2D
    weights: list[VortexObject]
    input_ports: list[VortexObject]
    output_probes: list[Probe]
    steps: int = 28
    input_scale: float = 0.35
    read_mode: str = "vorticity"

    def background(self) -> Array:
        return self.fluid.field_from_objects(self.weights)

    def encode(self, x: Sequence[float]) -> Array:
        if len(x) != len(self.input_ports):
            raise ValueError("input length does not match input ports")
        state = self.background()
        for amplitude, port in zip(x, self.input_ports):
            if amplitude == 0:
                continue
            state += self.fluid.gaussian_vortex(
                port.x,
                port.y,
                self.input_scale * float(amplitude) * port.circulation,
                port.sigma,
            )
        return state

    def forward(self, x: Sequence[float]) -> Array:
        state = self.fluid.evolve(self.encode(x), self.steps)
        return self.fluid.read(state, self.output_probes, self.read_mode)

    def transfer_matrix(self, pulse: float = 0.12) -> Array:
        """Finite-difference input→output response around the background."""
        baseline = self.forward([0.0] * len(self.input_ports))
        columns = []
        for j in range(len(self.input_ports)):
            x = np.zeros(len(self.input_ports), dtype=float)
            x[j] = pulse
            columns.append((self.forward(x) - baseline) / pulse)
        return np.stack(columns, axis=1)

    def with_circulations(self, gamma: Sequence[float]) -> "FluidWeightLayer":
        if len(gamma) != len(self.weights):
            raise ValueError("gamma length does not match weights")
        new_weights = [replace(w, circulation=float(g)) for w, g in zip(self.weights, gamma)]
        return FluidWeightLayer(
            fluid=self.fluid,
            weights=new_weights,
            input_ports=self.input_ports,
            output_probes=self.output_probes,
            steps=self.steps,
            input_scale=self.input_scale,
            read_mode=self.read_mode,
        )


def effective_rank(matrix: Array, eps: float = 1e-12) -> float:
    """Entropy effective rank of a matrix's singular spectrum."""
    s = np.linalg.svd(matrix, compute_uv=False)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def interaction_residual(
    fluid: VorticityFluid2D,
    background: Array,
    pulse_a: Array,
    pulse_b: Array,
    steps: int,
) -> float:
    """Measure nonlinear failure of superposition after finite evolution.

    R = F(B+a+b) - F(B+a) - F(B+b) + F(B).
    """
    f0 = fluid.evolve(background, steps)
    fa = fluid.evolve(background + pulse_a, steps)
    fb = fluid.evolve(background + pulse_b, steps)
    fab = fluid.evolve(background + pulse_a + pulse_b, steps)
    residual = fab - fa - fb + f0
    denom = np.linalg.norm(fab - f0) + 1e-12
    return float(np.linalg.norm(residual) / denom)
