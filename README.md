# -mp-ri — it's all buckets

A tiny falsifiable prototype of a **fluid computer whose persistent weights are objects in the flow**.

The idea is deliberately narrower than “Navier–Stokes is an AI.” The repo asks one concrete question:

> Can a computation be routed by coherent structures in a nonlinear incompressible field, with input pulses as activations, vortex circulations as mutable weights, and physical time as network depth?

This implementation uses a 2-D periodic incompressible vorticity equation,

\[
\partial_t\omega + u\cdot\nabla\omega = \nu\Delta\omega - \mu\omega,
\qquad
u = (\partial_y\psi,-\partial_x\psi),
\qquad
-\Delta\psi=\omega.
\]

The solver is pseudo-spectral (FFT Poisson solve, 2/3 de-aliasing, RK4). It is intentionally small enough to inspect rather than a production CFD package.

## The mapping

| Neural language | Fluid object in this repo |
| --- | --- |
| activation | localized vorticity pulse |
| weight | persistent Gaussian vortex object `(x, y, circulation, sigma)` |
| layer depth | forward integration time |
| routing | advection + nonlinear vorticity interaction |
| nonlinearity | `u · ∇ω` |
| readout | local speed/vorticity probes |
| weight update | changing vortex circulation |
| interference | failure of finite-time superposition |

The important difference from ordinary reservoir computing is the experiment around **weights as field objects**. The background is not just a fixed black-box reservoir: its coherent vortices are explicit parameters and can be changed or trained.

## What already happens

`results/receipt.json` is a deterministic receipt from the default 28×28 run.

**Gate 0 — solver sanity.** The recovered velocity stays divergence-free to numerical precision (~`1.6e-16` RMS in the receipt), the state stays finite, and viscosity/drag reduce enstrophy.

**Gate 1 — a weight really is an object.** Perturb one vortex circulation and recompute the finite-difference input→output transfer matrix `J`. The full transfer has effective rank `2.16`; the single-object deformation `ΔJ` has effective rank `1.37`. That is only a toy measurement, but it is exactly the Kompressori-shaped question: can a local structural edit deform a larger response operator through a low-dimensional channel?

**Gate 2 — overlap controls interference.** For equal pulse strengths, the finite-time nonlinear superposition error is about **9.44× larger** for an overlapping pulse pair than for a well-separated pair (`0.01816` vs `0.001924`). This is the first bridge to the CausalHorizon/Kompressori line: disjoint influence is approximately additive; overlapping influence is not.

**Gate 3 — vortex circulations can be trained.** Four persistent vortex circulations are optimized by finite-difference gradients to make a 3×3 physical transfer matrix more diagonal. After eight tiny epochs, routing loss falls from **0.33047 → 0.29339** (~11.2%). This is not a useful learner yet. It is a proof-of-mechanism that the “weights” can literally be mutable objects in the same field that carries activations.

**Gate 4 — gain has a scale.** A localized sinusoidal packet is tested at increasing wavenumber. The background flow gives low-frequency packets net gain (`k=1`: `1.070×` initial enstrophy) but viscosity wins by `k=5` (`0.991×`) and above. The toy therefore has an explicit **shear/gain versus viscous cutoff** instead of an unconstrained activation magnitude.

Run everything:

```bash
python -m pip install -e .[dev]
python -m mpri.gates --out results/latest.json
pytest -q
```

A faster smoke run:

```bash
python -m mpri.gates --grid 24 --train-epochs 1 --out /tmp/mpri.json
```

## Why the recent Navier–Stokes construction mattered to the idea

The attached OpenAI construction is not an AI recipe, and this repo does **not** reproduce its 3-D blowup. What is useful here is the mechanism vocabulary it makes unusually explicit: localized oscillatory pulses are amplified by background shear, their wavevectors are sheared toward shorter scales where viscosity eventually dominates, and carefully separated pulse families can contribute cleanly to a target quadratic stress.

That suggests a computational architecture with three physical operations rather than transformer-style all-to-all lookup:

1. **route** a localized packet through a background flow;
2. **amplify or suppress** it according to local shear and scale;
3. let **overlap create nonlinear cross-terms**, while separated packets remain approximately additive.

This repository tests those statements in a tiny 2-D system before making any stronger claim.

## The actual open question

The interesting next step is not “make it bigger.” It is to test whether the field can acquire **useful, reusable internal objects** rather than merely letting us hand-place vortices.

A proper Gate 5 should therefore let the circulation *and position* of vortex weights change under a local learning rule derived from the passing pulse and readout error, then ask whether the learned structures survive new inputs and whether their `ΔJ` remains low-rank. If that works, the project stops being “a fluid reservoir with tunable knobs” and becomes something closer to **a medium that stores its own routing law as coherent structures**.

## Status

This is a research toy and a mechanism probe. Current hits are reproducible on this implementation; none of them establish a general advantage over neural networks, reservoir computers, or standard numerical operators.
