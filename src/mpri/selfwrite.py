from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import FluidWeightLayer

Array = np.ndarray


@dataclass
class WriteStep:
    epoch: int
    response: Array
    error: float
    eligibility: Array
    delta_circulation: Array
    circulations: Array


def _weight_masks(layer: FluidWeightLayer) -> Array:
    """Positive local masks centred on the persistent vortex objects."""
    f = layer.fluid
    masks = []
    for w in layer.weights:
        dx = f._periodic_delta(f.x, w.x)
        dy = f._periodic_delta(f.y, w.y)
        mask = np.exp(-0.5 * (dx * dx + dy * dy) / (w.sigma * w.sigma))
        mask /= mask.sum()
        masks.append(mask)
    return np.stack(masks, axis=0)


def episode_response(
    layer: FluidWeightLayer,
    x: Sequence[float],
    steps: int = 30,
) -> Array:
    """Input-induced output response with the drifting background subtracted."""
    f = layer.fluid
    background_end = f.evolve(layer.background(), steps)
    active_end = f.evolve(layer.encode(x), steps)
    return (
        f.read(active_end, layer.output_probes, layer.read_mode)
        - f.read(background_end, layer.output_probes, layer.read_mode)
    )


def local_cross_write(
    layer: FluidWeightLayer,
    x: Sequence[float],
    target_probe: int,
    target_response: float,
    *,
    epochs: int = 8,
    write_steps: int = 30,
    write_rate: float = 5000.0,
    circulation_clip: float = 4.0,
) -> tuple[FluidWeightLayer, list[WriteStep]]:
    """Rewrite persistent vortex circulations from a local flow interaction.

    This is an *added plasticity rule*, not part of the Navier--Stokes
    equations. Each persistent object sees only a local eligibility signal:
    the masked cross-energy between the current background velocity and the
    input-induced fast velocity. A single scalar readout error is broadcast
    globally, giving a three-factor update

        delta_gamma_i = eta * error * integral mask_i (u_slow dot u_fast) dx.

    There is no finite-difference gradient and no direct access to the desired
    transfer matrix. The gate asks only whether a passing activation can leave
    a persistent change in the same coherent objects that route later inputs.
    """
    if not 0 <= target_probe < len(layer.output_probes):
        raise ValueError("target_probe is out of range")
    if len(x) != len(layer.input_ports):
        raise ValueError("input length does not match input ports")

    f = layer.fluid
    masks = _weight_masks(layer)
    gamma = np.array([w.circulation for w in layer.weights], dtype=float)
    history: list[WriteStep] = []

    for epoch in range(int(epochs)):
        current = layer.with_circulations(gamma)

        background_end = f.evolve(current.background(), write_steps)
        active_end = f.evolve(current.encode(x), write_steps)
        fast = active_end - background_end

        response = (
            f.read(active_end, current.output_probes, current.read_mode)
            - f.read(background_end, current.output_probes, current.read_mode)
        )
        error = float(target_response - response[target_probe])

        u_slow, v_slow = f.velocity(background_end)
        u_fast, v_fast = f.velocity(fast)
        local_cross_energy = u_slow * u_fast + v_slow * v_fast
        eligibility = np.einsum("kij,ij->k", masks, local_cross_energy)

        delta = write_rate * error * eligibility
        gamma = np.clip(gamma + delta, -circulation_clip, circulation_clip)

        history.append(
            WriteStep(
                epoch=epoch,
                response=np.array(response, copy=True),
                error=error,
                eligibility=np.array(eligibility, copy=True),
                delta_circulation=np.array(delta, copy=True),
                circulations=np.array(gamma, copy=True),
            )
        )

    return layer.with_circulations(gamma), history
