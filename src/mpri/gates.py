from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .fluid import Probe, VortexObject, VorticityFluid2D
from .model import FluidWeightLayer, effective_rank, interaction_residual
from .selfwrite import episode_response, local_cross_write


def default_layer(n: int = 28) -> FluidWeightLayer:
    f = VorticityFluid2D(n=n, viscosity=3e-3, drag=1e-2, dt=0.012)
    weights = [
        VortexObject(2.80, 2.40, +2.00, 0.30),
        VortexObject(3.20, 3.10, -1.60, 0.30),
        VortexObject(3.60, 3.80, +1.50, 0.30),
        VortexObject(4.10, 2.70, -1.20, 0.30),
    ]
    inputs = [
        VortexObject(2.00, 2.40, +1.0, 0.20),
        VortexObject(2.00, 3.80, -1.0, 0.20),
        VortexObject(2.20, 3.10, +1.0, 0.20),
    ]
    outputs = [
        Probe(4.50, 2.20, 0.30),
        Probe(4.50, 3.10, 0.30),
        Probe(4.50, 4.00, 0.30),
    ]
    return FluidWeightLayer(
        f, weights, inputs, outputs, steps=60, input_scale=0.80, read_mode="speed"
    )


def gate0_sanity(layer: FluidWeightLayer) -> dict:
    f = layer.fluid
    state = layer.encode([0.7, -0.4, 0.5])
    div0 = f.divergence_rms(state)
    e0 = f.enstrophy(state)
    state2 = f.evolve(state, 20)
    div1 = f.divergence_rms(state2)
    e1 = f.enstrophy(state2)
    return {
        "divergence_rms_initial": div0,
        "divergence_rms_final": div1,
        "enstrophy_initial": e0,
        "enstrophy_final": e1,
        "finite": bool(np.isfinite(state2).all()),
    }


def gate1_weights_are_objects(layer: FluidWeightLayer) -> dict:
    base = layer.transfer_matrix()
    gamma = np.array([w.circulation for w in layer.weights])
    perturbed = gamma.copy()
    perturbed[1] += 0.22
    changed = layer.with_circulations(perturbed).transfer_matrix()
    delta = changed - base
    return {
        "base_transfer": base.tolist(),
        "perturbed_transfer": changed.tolist(),
        "delta_frobenius": float(np.linalg.norm(delta)),
        "delta_effective_rank": effective_rank(delta),
        "full_effective_rank": effective_rank(base),
    }


def gate2_overlap_interference(layer: FluidWeightLayer) -> dict:
    f = layer.fluid
    bg = layer.background()
    a = f.gaussian_vortex(2.75, 3.10, 0.55, 0.23)
    b_near = f.gaussian_vortex(2.98, 3.16, -0.50, 0.23)
    b_far = f.gaussian_vortex(5.55, 5.20, -0.50, 0.23)
    near = interaction_residual(f, bg, a, b_near, steps=22)
    far = interaction_residual(f, bg, a, b_far, steps=22)
    return {
        "near_interaction": near,
        "far_interaction": far,
        "near_over_far": float(near / (far + 1e-12)),
    }


def _routing_loss(layer: FluidWeightLayer, target: np.ndarray) -> float:
    transfer = layer.transfer_matrix(pulse=0.10)
    scale = float(np.sum(transfer * target) / (np.sum(transfer * transfer) + 1e-12))
    return float(np.mean((scale * transfer - target) ** 2))


def gate3_train_vortex_weights(layer: FluidWeightLayer, epochs: int = 8) -> dict:
    """Coordinate finite-difference learning on circulation values."""
    target = np.eye(len(layer.output_probes), len(layer.input_ports))
    gamma = np.array([w.circulation for w in layer.weights], dtype=float)
    start = _routing_loss(layer.with_circulations(gamma), target)
    history = [start]
    eps = 0.10
    lr = 2.0

    for _ in range(epochs):
        grad = np.zeros_like(gamma)
        for i in range(gamma.size):
            gp = gamma.copy(); gp[i] += eps
            gm = gamma.copy(); gm[i] -= eps
            lp = _routing_loss(layer.with_circulations(gp), target)
            lm = _routing_loss(layer.with_circulations(gm), target)
            grad[i] = (lp - lm) / (2.0 * eps)
        gamma -= lr * grad
        gamma = np.clip(gamma, -3.5, 3.5)
        history.append(_routing_loss(layer.with_circulations(gamma), target))

    return {
        "target": target.tolist(),
        "initial_circulations": [w.circulation for w in layer.weights],
        "trained_circulations": gamma.tolist(),
        "loss_history": history,
        "loss_improvement": float(history[0] - history[-1]),
    }


def gate4_scale_gain(layer: FluidWeightLayer) -> dict:
    """Probe the shear-vs-viscosity tradeoff with sinusoidal wave packets."""
    f = layer.fluid
    bg = layer.background()
    cx, cy, sigma = 3.35, 3.05, 0.62
    dx = f._periodic_delta(f.x, cx)
    dy = f._periodic_delta(f.y, cy)
    envelope = np.exp(-0.5 * (dx * dx + dy * dy) / sigma**2)
    records = []
    for k in (1, 2, 3, 4, 5, 6):
        pulse = 0.22 * envelope * np.sin(k * dx)
        pulse -= pulse.mean()
        e0 = f.enstrophy(pulse)
        with_bg = f.evolve(bg + pulse, 16) - f.evolve(bg, 16)
        alone = f.evolve(pulse, 16)
        e_bg = f.enstrophy(with_bg)
        e_alone = f.enstrophy(alone)
        records.append({
            "k": k,
            "gain_with_background": float(e_bg / (e0 + 1e-12)),
            "gain_alone": float(e_alone / (e0 + 1e-12)),
            "background_over_alone": float(e_bg / (e_alone + 1e-12)),
        })
    return {"scales": records}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-15:
        return 0.0
    return float(np.dot(a, b) / denom)


def gate5_self_write(layer: FluidWeightLayer, epochs: int = 8) -> dict:
    """Let a passing activation rewrite the persistent vortex objects.

    Unlike Gate 3, this has no coordinate finite-difference gradient and no
    target transfer matrix. A single probe supplies a scalar error while each
    weight receives only a local background/fast-flow cross-energy eligibility.
    """
    x = np.zeros(len(layer.input_ports), dtype=float)
    x[0] = 1.0
    target_probe = 0
    positive_target = 0.010
    negative_target = -0.010
    write_steps = 30
    write_rate = 5000.0

    before = episode_response(layer, x, steps=write_steps)

    positive, history_pos = local_cross_write(
        layer,
        x,
        target_probe,
        positive_target,
        epochs=epochs,
        write_steps=write_steps,
        write_rate=write_rate,
    )
    negative, history_neg = local_cross_write(
        layer,
        x,
        target_probe,
        negative_target,
        epochs=epochs,
        write_steps=write_steps,
        write_rate=write_rate,
    )

    after_pos = episode_response(positive, x, steps=write_steps)
    after_neg = episode_response(negative, x, steps=write_steps)

    initial_gamma = np.array([w.circulation for w in layer.weights], dtype=float)
    gamma_pos = np.array([w.circulation for w in positive.weights], dtype=float)
    gamma_neg = np.array([w.circulation for w in negative.weights], dtype=float)
    write_pos = gamma_pos - initial_gamma
    write_neg = gamma_neg - initial_gamma

    j0 = layer.transfer_matrix(pulse=0.12)
    j1 = positive.transfer_matrix(pulse=0.12)
    delta_j = j1 - j0

    return {
        "rule": {
            "write_steps": write_steps,
            "write_rate": write_rate,
            "target_probe": target_probe,
            "input": x.tolist(),
            "description": "delta_gamma_i = eta * scalar_error * local_velocity_cross_energy",
        },
        "initial_response": before.tolist(),
        "positive_target": {
            "target": positive_target,
            "final_response": after_pos.tolist(),
            "initial_abs_error": float(abs(positive_target - before[target_probe])),
            "final_abs_error": float(abs(positive_target - after_pos[target_probe])),
            "trained_circulations": gamma_pos.tolist(),
            "target_response_history": [
                float(step.response[target_probe]) for step in history_pos
            ],
        },
        "negative_target": {
            "target": negative_target,
            "final_response": after_neg.tolist(),
            "initial_abs_error": float(abs(negative_target - before[target_probe])),
            "final_abs_error": float(abs(negative_target - after_neg[target_probe])),
            "trained_circulations": gamma_neg.tolist(),
            "target_response_history": [
                float(step.response[target_probe]) for step in history_neg
            ],
        },
        "opposite_write_cosine": _cosine(write_pos, write_neg),
        "delta_transfer_frobenius": float(np.linalg.norm(delta_j)),
        "delta_transfer_effective_rank": effective_rank(delta_j),
        "full_transfer_effective_rank": effective_rank(j0),
    }


def run_all(n: int = 28, train_epochs: int = 8) -> dict:
    layer = default_layer(n=n)
    return {
        "config": {
            "grid": n,
            "fluid": {
                "viscosity": layer.fluid.viscosity,
                "drag": layer.fluid.drag,
                "dt": layer.fluid.dt,
            },
            "weights": [asdict(w) for w in layer.weights],
        },
        "gate0_sanity": gate0_sanity(layer),
        "gate1_weights_are_objects": gate1_weights_are_objects(layer),
        "gate2_overlap_interference": gate2_overlap_interference(layer),
        "gate3_train_vortex_weights": gate3_train_vortex_weights(layer, epochs=train_epochs),
        "gate4_scale_gain": gate4_scale_gain(layer),
        "gate5_self_write": gate5_self_write(layer, epochs=train_epochs),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run the -mp-ri fluid-weight gates")
    p.add_argument("--grid", type=int, default=28)
    p.add_argument("--train-epochs", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("results/latest.json"))
    args = p.parse_args()

    result = run_all(args.grid, args.train_epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
