from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .fluid import Probe, VortexObject, VorticityFluid2D
from .machine import DecisionReceipt, FieldBodyMachine, MachineConfig


def default_machine(n: int = 20) -> FieldBodyMachine:
    fluid = VorticityFluid2D(n=n, viscosity=3e-3, drag=1e-2, dt=0.012)
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
    outputs = [Probe(4.50, 2.20, 0.30)]
    config = MachineConfig(
        trace_decay=0.94,
        input_scale=0.20,
        fluid_substeps=2,
        fast_leak=0.975,
        output_gain=60.0,
        write_rate=1200.0,
        assurance=0.60,
    )
    return FieldBodyMachine(
        fluid, weights, inputs, outputs, config=config, read_mode="vorticity"
    )


def continuous_world(t: int) -> tuple[np.ndarray, int]:
    """A continuous three-address stream with no episode markers.

    Address 0 is a slowly changing binary cue. Addresses 1 and 2 are distractor
    processes. The cue is not observed every step because the active observer
    must keep all addresses from becoming stale, so private trace is useful
    between visits. The target is the latent cue and arrives only later.
    """
    block = t // 11
    cue = 1 if ((37 * block + 11) % 7) < 3 else -1
    observations = np.array(
        [cue, np.sin(0.23 * t), np.cos(0.17 * t + 0.4)], dtype=float
    )
    return observations, cue


def _run_stream(
    machine: FieldBodyMachine,
    *,
    start_t: int,
    steps: int,
    delay: int,
    learn: bool,
) -> dict:
    pending: deque[tuple[int, DecisionReceipt, int]] = deque()
    correct: list[bool] = []
    predictions: list[float] = []
    addresses: list[int] = []
    writes_before = machine.write_count

    for local_t in range(steps):
        world_t = start_t + local_t
        observations, target = continuous_world(world_t)
        decision = machine.tick_once(observations)
        correct.append(decision.action == target)
        predictions.append(decision.prediction)
        addresses.append(decision.address)
        pending.append((local_t + delay, decision, target))

        while pending and pending[0][0] <= local_t:
            _, old, old_target = pending.popleft()
            machine.apply_consequence(old, old_target, learn=learn)

    # Consequence can arrive after the visible stream stops. This advances the
    # delayed ledger/plasticity path, not the fast field itself.
    while pending:
        _, old, old_target = pending.popleft()
        machine.apply_consequence(old, old_target, learn=learn)

    burn = min(max(steps // 4, 1), max(steps - 1, 1))
    return {
        "accuracy": float(np.mean(correct)),
        "late_accuracy": float(np.mean(correct[burn:])),
        "mean_abs_prediction": float(np.mean(np.abs(predictions))),
        "writes": int(machine.write_count - writes_before),
        "address_counts": dict(sorted(Counter(addresses).items())),
    }


def run_demo(
    n: int = 20,
    train_steps: int = 240,
    eval_steps: int = 120,
    delay: int = 5,
) -> dict:
    machine = default_machine(n=n)
    initial_slow = machine.slow_signature()
    train = _run_stream(
        machine, start_t=0, steps=train_steps, delay=delay, learn=True
    )
    learned_slow = machine.slow_signature()
    slow_delta = learned_slow - initial_slow

    machine.reset_fast(reset_observer=True)
    fast_zero_after_reset = bool(np.allclose(machine.fast, 0.0))
    slow_survived_reset = bool(np.allclose(machine.slow_signature(), learned_slow))

    frozen = _run_stream(
        machine,
        start_t=10_000,
        steps=eval_steps,
        delay=delay,
        learn=False,
    )

    return {
        "status": "assembled_continuous_machine_smoke",
        "claim_boundary": (
            "This receipt verifies one composed continuously running object and its "
            "resource/accounting paths. It is not a superiority claim, and the slow "
            "write remains an explicit three-factor rule rather than an endogenous "
            "Navier-Stokes backreaction law."
        ),
        "config": {
            "grid": n,
            "train_steps": train_steps,
            "eval_steps": eval_steps,
            "delayed_consequence": delay,
            "machine": asdict(machine.config),
        },
        "train": train,
        "fast_reset": {
            "fast_zero": fast_zero_after_reset,
            "slow_body_survived": slow_survived_reset,
        },
        "slow_body": {
            "initial_circulations": initial_slow.tolist(),
            "learned_circulations": learned_slow.tolist(),
            "delta_l2": float(np.linalg.norm(slow_delta)),
            "l1_after": float(np.sum(np.abs(learned_slow))),
            "l1_budget": float(machine.circulation_budget),
        },
        "frozen_after_fast_wipe": frozen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the assembled continuous field organism"
    )
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--eval-steps", type=int, default=120)
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument(
        "--out", type=Path, default=Path("results/organism_latest.json")
    )
    args = parser.parse_args()
    result = run_demo(args.grid, args.train_steps, args.eval_steps, args.delay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
