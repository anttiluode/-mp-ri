# -mp-ri — it's all buckets

A tiny falsifiable prototype of a **fluid computer whose persistent weights are objects in the flow**.

The idea is deliberately narrower than “Navier–Stokes is an AI.” The repo asks a sequence of concrete questions:

> Can computation be routed by coherent structures in a nonlinear incompressible field, can passing activations interfere only when they physically overlap, and can those activations rewrite the same persistent structures that route later inputs?

The core implementation uses a 2-D periodic incompressible vorticity equation,

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
| weight update | changing persistent vortex circulation |
| interference | failure of finite-time superposition |

The important difference from ordinary reservoir computing is the experiment around **weights as field objects**. The background is not merely a fixed black-box reservoir: its coherent vortices are explicit persistent parameters and, beginning with Gate 5, the passing flow itself supplies the local eligibility signal that rewrites them.

## Core gates

`results/receipt.json` is the deterministic receipt from the default 28×28 run.

**Gate 0 — solver sanity.** The recovered velocity stays divergence-free to numerical precision (~`1.6e-16` RMS in the receipt), the state stays finite, and viscosity/drag reduce enstrophy.

**Gate 1 — a weight really is an object.** Perturb one vortex circulation and recompute the finite-difference input→output transfer matrix `J`. The full transfer has effective rank `2.16`; the single-object deformation `ΔJ` has effective rank `1.37`. That is only a toy measurement, but it is exactly the Kompressori-shaped question: can a local structural edit deform a larger response operator through a low-dimensional channel?

**Gate 2 — overlap controls interference.** For equal pulse strengths, the finite-time nonlinear superposition error is about **9.44× larger** for an overlapping pulse pair than for a well-separated pair (`0.01816` vs `0.001924`). This is the first bridge to the CausalHorizon/Kompressori line: disjoint influence is approximately additive; overlapping influence is not.

**Gate 3 — vortex circulations can be trained.** Four persistent vortex circulations are optimized by finite-difference gradients to make a 3×3 physical transfer matrix more diagonal. After eight tiny epochs, routing loss falls from **0.33047 → 0.29339** (~11.2%). This proves only that the persistent objects are tunable; the optimizer is conventional and external.

**Gate 4 — gain has a scale.** A localized sinusoidal packet is tested at increasing wavenumber. The background flow gives low-frequency packets net gain (`k=1`: `1.070×` initial enstrophy) but viscosity wins by `k=5` (`0.991×`) and above. The toy therefore has an explicit **shear/gain versus viscous cutoff** instead of an unconstrained activation magnitude.

**Gate 5 — the fast flow writes the slow operator.** This is the first step beyond “a fluid reservoir with knobs.” A fixed input pulse is presented repeatedly. The target probe broadcasts only one scalar error. Each persistent vortex receives a local eligibility signal equal to the masked velocity cross-energy between the slowly evolving background and the input-induced fast flow,

\[
\chi_i=\int m_i(x)\,u_{\rm slow}(x)\cdot u_{\rm fast}(x)\,dx,
\qquad
\Delta\gamma_i=\eta\,e\,\chi_i .
\]

There is **no finite-difference gradient and no target transfer matrix** in this gate. After eight writes toward a target response of `+0.01`, the measured response at the target probe moves from `-0.001356` to `+0.005627`; absolute error falls from `0.011356` to `0.004373`. The persistent circulations change from

`[2.0, -1.6, 1.5, -1.2]`

to

`[2.929, -0.311, 1.486, -0.797]`.

A sign-reversed target writes the objects in a substantially opposite direction (write-vector cosine `-0.801`). The resulting transfer-operator deformation has effective rank `1.270` versus `2.158` for the original full transfer.

Important limitation: **the plasticity rule is an added learning law, not a consequence of Navier–Stokes itself.** Gate 5 shows a plausible two-timescale mechanism — fast activation, slow persistent routing field — not a new theorem about fluids. Positions are still fixed; allowing the coherent objects themselves to move is a later gate.

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

## Claude's independent branch

The [`Claude/`](Claude/) directory is a **separate implementation and experiment line written by Claude**, added after the first core gates. It is intentionally kept separate from `src/mpri` so provenance is visible.

Claude's code does two useful things:

1. `run_additivity.py` tests the CausalHorizon/Kompressori prediction directly on an ordinary 2-D Navier–Stokes shear flow. It runs `base`, `A`, `B`, and `AB`, then compares the raw downstream field response `dAB` against the linear prediction `dA + dB` while sweeping packet separation. Unlike the OpenAI construction, additivity is not designed in; it is measured.
2. `Run_xor.py` + `eval_xor.py` ask the classic reservoir-computing question: does the nonlinear flow create a feature space in which XOR becomes linearly readable? The stored `xor_data.npz` contains **88 jittered trials** (22 for each bit pair) and a 13×13 downstream vorticity patch per trial.

The recorded evaluation was:

| Readout | XOR | OR | AND |
| --- | ---: | ---: | ---: |
| linear classifier on raw bits | `0.408 ± 0.035` | `1.000` | `1.000` |
| linear classifier on the full downstream flow patch | **`1.000 ± 0.000`** | `1.000` | `1.000` |
| only mean `ω` and mean `|ω|` | `0.707 ± 0.077` | `1.000` | `0.966 ± 0.045` |

That is a real demonstration that the medium's nonlinear evolution makes XOR linearly accessible. It is **not the novelty claim** of this repository: fluid/wave reservoir computing and “bucket of water” XOR predate this project by decades. Its role here is a sanity check that this particular substrate actually computes before we ask the harder self-writing question.

See [`Claude/README.md`](Claude/README.md) for the exact files and commands.

## Why the recent OpenAI Navier–Stokes construction mattered

The attached OpenAI construction is not an AI recipe, and this repo does **not** reproduce its 3-D forced blowup. What was useful to this project is the mechanism vocabulary it makes unusually explicit: localized oscillatory pulses are amplified by background shear, their wavevectors are sheared toward shorter scales where viscosity eventually dominates, and carefully separated pulse families can contribute cleanly to a target quadratic stress.

That suggested a computational architecture with three physical operations rather than transformer-style all-to-all lookup:

1. **route** a localized packet through a background flow;
2. **amplify or suppress** it according to local shear and scale;
3. let **overlap create nonlinear cross-terms**, while separated packets remain approximately additive.

Gate 5 adds the next operation:

4. let a fast packet's local interaction with the persistent flow leave a **slow structural write** that changes how later packets propagate.

The OpenAI result did **not** make fluid computing possible — that existed long before it. What it contributed to this line of thought is a rigorous example showing that strong quadratic fluid interactions can be deliberately organized: amplification can be separated from dissipation, and unwanted cross-terms can be eliminated by support engineering instead of accepted as turbulent noise.

## The actual open question

The repo has now crossed two different thresholds:

- “Can a nonlinear flow compute?” — yes; Claude's XOR branch demonstrates the old reservoir-computing result on this substrate.
- “Can persistent field objects be rewritten by the passing computation?” — Gate 5 gives a first, deliberately minimal yes.

The next question is harder:

> Can the coherent structures **move, form, split, and persist under a local rule**, so that useful routing objects emerge rather than being hand-placed?

That means Gate 6 should allow vortex position as well as circulation to change from local flow moments, freeze the readout, remove direct optimizer access to the target transfer matrix, and test whether a learned routing structure survives new inputs. If that succeeds, this stops looking like a reservoir with tunable parameters and starts looking like **a medium that stores its own routing law as coherent structures**.

## Status

This is a research toy and a mechanism probe. Current hits are reproducible on this implementation; none establish a general advantage over neural networks, reservoir computers, neural operators, or standard numerical methods.
