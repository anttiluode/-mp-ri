from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .fluid import Probe, VortexObject, VorticityFluid2D

Array = np.ndarray


def _project_l1_ball(x: Array, radius: float) -> Array:
    """Euclidean projection onto ||x||_1 <= radius."""
    x = np.asarray(x, dtype=float)
    if radius <= 0:
        return np.zeros_like(x)
    if np.sum(np.abs(x)) <= radius:
        return x.copy()
    u = np.sort(np.abs(x))[::-1]
    cssv = np.cumsum(u)
    idx = np.arange(1, u.size + 1)
    keep = u - (cssv - radius) / idx > 0
    rho = np.nonzero(keep)[0][-1]
    theta = (cssv[rho] - radius) / float(rho + 1)
    return np.sign(x) * np.maximum(np.abs(x) - theta, 0.0)


@dataclass(frozen=True)
class MachineConfig:
    """Resource and timescale knobs for the assembled field machine."""

    trace_decay: float = 0.94
    input_scale: float = 0.18
    fluid_substeps: int = 2
    fast_leak: float = 0.985
    output_gain: float = 45.0
    write_rate: float = 2500.0
    circulation_clip: float = 4.0
    circulation_budget: float | None = None
    assurance: float = 0.55
    target_response: float = 0.010
    relevance_decay: float = 0.92
    age_weight: float = 1.0
    relevance_weight: float = 0.8
    max_probe_age: int = 12


@dataclass(frozen=True)
class DecisionReceipt:
    """Causally addressed record retained until delayed consequence arrives."""

    tick: int
    address: int
    observed_value: float
    score: float
    prediction: float
    action: int
    response: Array
    eligibility: Array
    trace: Array


@dataclass(frozen=True)
class ConsequenceReceipt:
    tick: int
    source_tick: int
    target: int
    error: float
    failed_or_uncertain: bool
    wrote: bool
    delta_circulation: Array
    circulations: Array


class FieldBodyMachine:
    """One continuously running fast-field x slow-body machine.

    This assembles the surviving primitives from the repo line into one object:

    * slow, stably addressed vortex objects are the persistent body/operator;
    * a fast vorticity residual moves through that body without episode resets;
    * private receiver traces retain recently sampled evidence;
    * an active observer chooses which input address to refresh;
    * local slow/fast flow overlap is retained as delayed eligibility;
    * a separate output boundary emits one binary action;
    * delayed consequence updates the addressed body only when the old action
      failed or remained below an assurance margin;
    * total absolute circulation can be bounded, turning growth into allocation.

    The slow write is an explicit three-factor plasticity rule. It is not claimed
    to follow from Navier--Stokes by itself.
    """

    def __init__(
        self,
        fluid: VorticityFluid2D,
        weights: Sequence[VortexObject],
        input_ports: Sequence[VortexObject],
        output_probes: Sequence[Probe],
        *,
        config: MachineConfig | None = None,
        read_mode: str = "vorticity",
    ) -> None:
        if len(weights) == 0:
            raise ValueError("machine needs at least one persistent body object")
        if len(input_ports) == 0:
            raise ValueError("machine needs at least one input address")
        if len(output_probes) not in (1, 2):
            raise ValueError(
                "current output boundary expects one scalar probe or two competing probes"
            )
        self.fluid = fluid
        self.weight_template = list(weights)
        self.input_ports = list(input_ports)
        self.output_probes = list(output_probes)
        self.config = config or MachineConfig()
        self.read_mode = read_mode

        self.gamma = np.array([w.circulation for w in weights], dtype=float)
        if self.config.circulation_budget is None:
            self.circulation_budget = 1.10 * float(np.sum(np.abs(self.gamma)))
        else:
            self.circulation_budget = float(self.config.circulation_budget)

        self.fast = np.zeros((fluid.n, fluid.n), dtype=float)
        self.trace = np.zeros(len(input_ports), dtype=float)
        self.relevance = np.zeros(len(input_ports), dtype=float)
        self.last_seen = np.full(len(input_ports), -10_000, dtype=int)
        self.tick = 0
        self.write_count = 0
        self._weight_masks = self._make_weight_masks()

    @property
    def weights(self) -> list[VortexObject]:
        return [
            replace(template, circulation=float(g))
            for template, g in zip(self.weight_template, self.gamma)
        ]

    def background(self) -> Array:
        return self.fluid.field_from_objects(self.weights)

    def _make_weight_masks(self) -> Array:
        masks = []
        f = self.fluid
        for w in self.weight_template:
            dx = f._periodic_delta(f.x, w.x)
            dy = f._periodic_delta(f.y, w.y)
            m = np.exp(-0.5 * (dx * dx + dy * dy) / (w.sigma * w.sigma))
            m /= m.sum()
            masks.append(m)
        return np.stack(masks, axis=0)

    def choose_address(self) -> int:
        """Choose a sensor without inspecting its current unseen value.

        Old addresses become increasingly urgent; addresses whose past samples
        were followed by consequential error also become more valuable. This is
        deliberately a tiny bounded policy, not a trained attention network.
        """
        age = np.maximum(0, self.tick - self.last_seen)
        age_score = np.minimum(age / max(1, self.config.max_probe_age), 1.0)
        priority = (
            self.config.age_weight * age_score
            + self.config.relevance_weight * self.relevance
        )
        best = np.flatnonzero(priority == np.max(priority))
        if best.size > 1:
            oldest = age[best]
            best = best[oldest == np.max(oldest)]
        return int(best[0])

    def _trace_pulse(self) -> Array:
        pulse = np.zeros_like(self.fast)
        for q, port in zip(self.trace, self.input_ports):
            if abs(q) < 1e-15:
                continue
            pulse += self.fluid.gaussian_vortex(
                port.x,
                port.y,
                self.config.input_scale * float(q) * port.circulation,
                port.sigma,
            )
        return pulse

    def _advance_fast(self) -> None:
        # The slow body is re-instantiated each substep: a field skeleton. The
        # fast residual still feels the nonlinear total flow, but viscosity does
        # not silently erase the persistent operator itself.
        for _ in range(int(self.config.fluid_substeps)):
            bg = self.background()
            total_next = self.fluid.step(bg + self.fast)
            background_next = self.fluid.step(bg)
            self.fast = self.config.fast_leak * (total_next - background_next)

    def _response_and_eligibility(self) -> tuple[Array, Array]:
        bg = self.background()
        response = (
            self.fluid.read(bg + self.fast, self.output_probes, self.read_mode)
            - self.fluid.read(bg, self.output_probes, self.read_mode)
        )
        u_slow, v_slow = self.fluid.velocity(bg)
        u_fast, v_fast = self.fluid.velocity(self.fast)
        cross = u_slow * u_fast + v_slow * v_fast
        eligibility = np.einsum("kij,ij->k", self._weight_masks, cross)
        return response, eligibility

    def observe_and_act(self, address: int, value: float) -> DecisionReceipt:
        """Refresh one addressed observation and advance the living field once."""
        if not 0 <= address < len(self.input_ports):
            raise ValueError("input address out of range")
        self.trace *= self.config.trace_decay
        self.trace[address] += (1.0 - self.config.trace_decay) * float(value)
        self.last_seen[address] = self.tick

        self.fast += self._trace_pulse()
        self._advance_fast()
        response, eligibility = self._response_and_eligibility()
        score = float(response[0] if response.size == 1 else response[0] - response[1])
        prediction = float(np.tanh(self.config.output_gain * score))
        action = 1 if prediction >= 0.0 else -1

        receipt = DecisionReceipt(
            tick=self.tick,
            address=int(address),
            observed_value=float(value),
            score=score,
            prediction=prediction,
            action=action,
            response=np.array(response, copy=True),
            eligibility=np.array(eligibility, copy=True),
            trace=np.array(self.trace, copy=True),
        )
        self.tick += 1
        return receipt

    def tick_once(self, observations: Sequence[float]) -> DecisionReceipt:
        """Convenience wrapper; only the chosen element is consumed."""
        if len(observations) != len(self.input_ports):
            raise ValueError("observation length does not match input addresses")
        address = self.choose_address()
        return self.observe_and_act(address, float(observations[address]))

    def apply_consequence(
        self,
        receipt: DecisionReceipt,
        target: int,
        *,
        learn: bool = True,
    ) -> ConsequenceReceipt:
        """Apply delayed scalar consequence to the old causally addressed event."""
        if target not in (-1, 1):
            raise ValueError("binary target must be -1 or +1")
        desired_response = float(target) * self.config.target_response
        error = float(desired_response - receipt.score)
        failed_or_uncertain = bool(
            receipt.action != target
            or abs(receipt.score) < self.config.assurance * self.config.target_response
        )

        surprise = min(1.0, abs(error) / (2.0 * self.config.target_response + 1e-12))
        a = receipt.address
        self.relevance *= self.config.relevance_decay
        self.relevance[a] += (1.0 - self.config.relevance_decay) * surprise

        delta = np.zeros_like(self.gamma)
        wrote = bool(learn and failed_or_uncertain)
        if wrote:
            delta = self.config.write_rate * error * receipt.eligibility
            proposed = np.clip(
                self.gamma + delta,
                -self.config.circulation_clip,
                self.config.circulation_clip,
            )
            proposed = _project_l1_ball(proposed, self.circulation_budget)
            delta = proposed - self.gamma
            self.gamma = proposed
            self.write_count += 1

        return ConsequenceReceipt(
            tick=self.tick,
            source_tick=receipt.tick,
            target=int(target),
            error=error,
            failed_or_uncertain=failed_or_uncertain,
            wrote=wrote,
            delta_circulation=np.array(delta, copy=True),
            circulations=np.array(self.gamma, copy=True),
        )

    def reset_fast(self, *, reset_observer: bool = True) -> None:
        """Erase transient activity while leaving the learned slow body intact."""
        self.fast.fill(0.0)
        if reset_observer:
            self.trace.fill(0.0)
            self.relevance.fill(0.0)
            self.last_seen.fill(-10_000)

    def slow_signature(self) -> Array:
        return np.array(self.gamma, copy=True)
