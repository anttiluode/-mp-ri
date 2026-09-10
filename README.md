# -mp-ri — it's all buckets (Ämpäri - Finnish for bucket, but Ä does not work as repo name) 

A tiny falsifiable prototype of a **fluid computer whose persistent weights are objects in the flow**.

The idea is deliberately narrower than “Navier–Stokes is an AI.” The repo asks a sequence of concrete questions:

> Can computation be routed by coherent structures in a nonlinear incompressible field, can passing activations interfere only when they physically overlap, and can those activations rewrite the same persistent structures that route later inputs?

The core implementation uses a 2-D periodic incompressible vorticity equation,

\[
\partial_t\omega + u\cdot\nabla\omega = \nu\Delta\omega - \mu\omega,
\qquad
u=(\partial_y\psi,-\partial_x\psi),
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
| weight update | changing persistent vortex circulation / slow field |
| interference | failure of finite-time superposition |

The important difference from ordinary reservoir computing is the experiment around **weights as field objects**. The background is not merely a fixed black-box reservoir: its coherent structures are meant to become a persistent routing law that later activations can alter.

## Core gates

`results/receipt.json` is the deterministic receipt from the default 28×28 run.

**Gate 0 — solver sanity.** Divergence remains at numerical precision (~`1.6e-16` RMS), the state stays finite, and viscosity/drag reduce enstrophy.

**Gate 1 — a weight really is an object.** Perturb one vortex circulation and recompute the finite-difference input→output transfer matrix `J`. The full transfer has effective rank `2.16`; the single-object deformation `ΔJ` has effective rank `1.37`.

**Gate 2 — overlap controls interference.** Equal-strength overlapping pulses produce about **9.44×** the finite-time nonlinear superposition residual of a separated pair (`0.01816` vs `0.001924`).

**Gate 3 — vortex circulations can be trained.** Four persistent vortex circulations are optimized by finite-difference gradients to make a 3×3 physical transfer matrix more diagonal. Routing loss falls from `0.33047 → 0.29339` after eight tiny epochs. This proves tunability, not a new learning principle.

**Gate 4 — gain has a scale.** A localized sinusoidal packet is tested at increasing wavenumber. The background gives low-frequency packets net gain (`k=1`: `1.070×` initial enstrophy), while viscosity wins by `k=5` (`0.991×`) and above.

**Gate 5 — the fast flow writes the slow operator.** A fixed input pulse is presented repeatedly. The target probe broadcasts one scalar error. Each persistent vortex receives a local eligibility signal from the masked velocity cross-energy between slow background flow and input-induced fast flow,

\[
\chi_i=\int m_i(x)\,u_{\rm slow}(x)\cdot u_{\rm fast}(x)\,dx,
\qquad
\Delta\gamma_i=\eta e\chi_i.
\]

There is no finite-difference gradient and no target transfer matrix in this gate. Eight writes toward target response `+0.01` move the measured response from `-0.001356` to `+0.005627`; absolute error falls from `0.011356` to `0.004373`. Reversing the target gives a substantially opposite structural write (write-vector cosine `-0.801`). The resulting `ΔJ` has effective rank `1.270` versus `2.158` for the original full transfer.

Important limitation: **the Gate 5 plasticity law is added; it is not a consequence of Navier–Stokes itself.**

Run the core:

```bash
python -m pip install -e .[dev]
python -m mpri.gates --out results/latest.json
pytest -q
```

## Claude branch, plus Gemini follow-ons

The [`Claude/`](Claude/) directory began as a **separate implementation and experiment line written by Claude**. It stays separate from `src/mpri` so provenance is visible. Two later Gemini proposals are also stored in that folder; they are identified below rather than being attributed to Claude.

Claude's original branch does two useful things:

1. `run_additivity.py` runs `base`, `A`, `B`, and `AB` and compares the raw downstream response `dAB` with `dA + dB` while sweeping packet separation. This is the clean state-space test of non-additivity; it avoids the nonlinear-detector artifact of the early HTML demo.
2. `Run_xor.py` + `eval_xor.py` ask the classic reservoir-computing question: does nonlinear fluid evolution make XOR linearly readable? The stored `xor_data.npz` has 88 jittered trials (22 per bit pair) and a 13×13 downstream vorticity patch per trial.

Recorded XOR evaluation:

| Readout | XOR | OR | AND |
| --- | ---: | ---: | ---: |
| linear classifier on raw bits | `0.408 ± 0.035` | `1.000` | `1.000` |
| linear classifier on full downstream patch | **`1.000 ± 0.000`** | `1.000` | `1.000` |
| mean `ω` + mean `|ω|` only | `0.707 ± 0.077` | `1.000` | `0.966 ± 0.045` |

That demonstrates nonlinear feature construction in this substrate. It is **not the novelty claim**; fluid/wave reservoir computing and bucket-of-water XOR predate this project.

### Gemini: `ns_operator.py`

Gemini's `ns_operator.py` is the first explicit **spectral slow/fast decomposition** in this branch:

\[
\omega=\Omega_{\rm slow}+w_{\rm fast},
\]

with `|k| <= k_split` treated as the persistent operator and higher resolved modes treated as transient activations. It also computes the low-pass projection of the fast-fast vorticity advection term, the 2-D vorticity-form analogue of a Reynolds-stress backreaction channel.

The proposed test initializes a slow vortex dipole, injects two `k≈14` localized oscillatory packets, evolves the full Navier–Stokes field, then reports how much low-frequency structure changed and how much high-frequency activity remains.

**Scientific status: mechanism probe, not yet a receipt.** As currently written, `||Omega_slow(final)-Omega_slow(initial)||` is confounded by ordinary evolution and viscous drift of the background itself. The localized carriers also have spectral sidebands, so some low-`k` content can be injected directly rather than transferred nonlinearly. A decisive version needs a matched no-pulse control and preferably the collision-specific residual

\[
P_{\rm slow}[\omega_{AB}-\omega_A-\omega_B+\omega_0],
\]

plus a post-write washout before calling the change persistent. The function `measure_reynolds_stress_transfer()` is conceptually useful, but the current `run_plasticity_test()` does not yet use it to isolate the write.

### Gemini: `run_clean_assosciative_memory.py`

Gemini's second file asks a stronger and more interesting question: **can co-occurrence rewrite routing so that a later partial cue reaches a previously weak detector?**

It settles a two-vortex background, measures the differential response at detector B from cue A alone, repeatedly co-injects A+B, lets fast modes decay between training cycles, then presents A alone again on the trained field. That is much closer to an associative-memory gate than XOR because the desired effect is not merely a richer readout; it is a changed propagation law.

The code already does one important thing correctly: during recall it evolves a matched clean background and subtracts that drifting reference from the cue response.

But a routing gain from the current script would still **not yet prove associative writing**. Training advances the background by thousands of steps, so natural field maturation can change cue sensitivity. The decisive controls are: sham training for the same elapsed time, A-only training, B-only training, and a non-overlapping or phase-scrambled A+B control. Recall should also begin only after a measured fast-mode washout threshold is met. The docstring says detector A is the baseline route and detector B is dark, but the current metric only measures B; a proper receipt should record both detectors and the timing of their peaks.

So these two Gemini files are important for architecture, even before they produce a trustworthy positive number. `ns_operator.py` asks whether the Navier–Stokes nonlinearity itself transfers fast activity into a slow operator. `run_clean_assosciative_memory.py` asks whether such a write has **functional consequence on a later cue**. Together they point at the correct next gate.

See [`Claude/README.md`](Claude/README.md) for provenance, files, commands, and caveats.

## Why the recent OpenAI Navier–Stokes construction mattered

The attached OpenAI construction is not an AI recipe, and this repo does **not** reproduce its 3-D forced blowup. Fluid computing existed long before it.

What mattered here is the mechanism vocabulary: localized oscillatory pulses can be organized around background shear, scale-dependent amplification/dissipation, and controlled quadratic interactions. In the construction, unwanted interactions are not merely tolerated as turbulence; pulse families are engineered so selected quadratic contributions survive while unwanted cross-terms are eliminated.

That suggested four computational operations:

1. **route** a localized packet through a background flow;
2. **amplify or suppress** it according to local shear and scale;
3. let **overlap create selected nonlinear cross-terms**, while separated packets remain approximately additive;
4. let those interactions leave a **slow structural write** that changes how later packets propagate.

The OpenAI result did **not** make fluid-field computing possible. The new possibility we are testing is narrower: whether strong fluid nonlinearity can be **organized into a self-writing operator instead of used only as a black-box reservoir**.

## The actual open question

The repo now separates three thresholds:

- “Can nonlinear flow compute?” — yes; the XOR branch demonstrates the old reservoir-computing result on this substrate.
- “Can an added local learning law rewrite persistent flow objects?” — Gate 5 gives a first yes.
- “Can the Navier–Stokes interaction itself produce a selective, persistent, functionally useful slow write?” — **still open.** The two Gemini probes target exactly this question.

The next credible success criterion is therefore not another classifier score. It is:

> After A+B training and complete fast-mode washout, does A alone produce a reproducible new route that is absent after sham, A-only, B-only, and non-overlap training, with the change traceable to a collision-specific low-frequency transfer term?

If that survives, this stops looking like a reservoir with tunable parameters and starts looking like **a medium that stores its own routing law as coherent structures**.

## Status

This is a research toy and mechanism probe. Current core receipts are reproducible on this implementation; the new Gemini files are experimental proposals and should not be presented as positive results until their controls are run. Nothing here establishes a general advantage over neural networks, reservoir computers, neural operators, or standard numerical methods.
