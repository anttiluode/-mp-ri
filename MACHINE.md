# The assembled machine

This is the first attempt in this repo to stop adding isolated gates and put the surviving pieces into **one continuously running object**.

It is intentionally small. The goal is not to claim a new AI architecture from a short toy. The goal is to make the entire causal loop executable in one place so later work can replace individual mechanisms without rebuilding the organism.

## One object, four state classes

```text
external world
      |
      v
ACTIVE ADDRESSED OBSERVER
choose one sensor without reading the others
      |
      v
PRIVATE RECEIVER TRACE q(t)
recent evidence survives between observations
      |
      v
FAST FLUID RESIDUAL w(t)
travels / mixes / interferes through the body
      |
      v
SLOW VORTEX BODY Omega(theta)
stable addressed coherent objects route later activity
      |
      v
OUTPUT BOUNDARY
one scalar or two competing probes -> binary action
      |
      v
world consequence arrives later
      |
      v
CAUSALLY ADDRESSED RECEIPT
old sensor address + response + local eligibility
      |
      v
FAILURE / ASSURANCE GATE
successful stable routes stop teaching themselves
      |
      v
SLOW STRUCTURAL WRITE
bounded circulation allocation
```

The concrete state is approximately

```text
fast fluid       w_t
receiver history q_t
slow body        gamma_t
observer state   relevance_t, last_seen_t
```

The machine never receives a global flattened state vector from the world. The convenience `tick_once()` accepts a list only so a toy environment can be written compactly; the address is chosen first and only that element is consumed.

## Where the pieces came from

This file is assembly, not a novelty claim. The components already existed elsewhere in the repo family.

- **-mp-ri / Kompressori:** a high-dimensional response operator can be changed by local low-dimensional edits; nonlinear overlap matters.
- **yrotisopeRweN / Twensday:** slow persistent structure plus fast state; stable internal addressability; finite structural budget as allocation.
- **T-800NNP / Child:** continuously running receiver history; no episode reset; delayed consequence must retain the old causal address.
- **GeometricNeuronV24 / LentoOrava:** bounded active sensing; choose where to look rather than pretend every variable is globally visible.
- **JelloBrain:** surprise is not relevance; stable success should not keep reinforcing itself; failure can reopen plasticity.
- **IttnasNoruen:** an update should be judged against retained behavior rather than trusted because a local tangent looked harmless.
- **FunctionalArbors:** delayed consequence needs a causally useful local eligibility mark and a finite structural substrate.

The current code takes the smallest executable version of each idea rather than copying the old implementations verbatim.

## Fast field x slow body

The slow operator is a set of persistent vortex objects with circulations `gamma_i`.

```text
Omega(gamma) = sum_i GaussianVortex_i(gamma_i)
```

The fast state is represented as a residual `w` relative to that body. For one internal substep we compute

```text
F(Omega + w) - F(Omega)
```

with the existing pseudo-spectral Navier--Stokes stepper and retain that difference as the new fast residual. Re-instantiating `Omega` each substep is deliberate: the current machine is a **field with a skeleton**. Viscosity may erase transient activity without erasing the operator object itself.

This is not yet the stronger Gemini proposal in which fast-fast nonlinear transfer autonomously creates the slow field. That remains the most important substrate replacement once the assembled loop is stable.

## Private receiver history

Each input address owns one scalar trace.

```text
q_i <- decay * q_i
q_address <- q_address + (1-decay) * observed_value
```

The current trace is re-injected through the corresponding physical input ports. Therefore the same new observation can enter a different fluid state because earlier observations are still represented in `q` and in the moving fast residual.

`reset_fast()` erases both fast fluid activity and, optionally, the observer traces. It does **not** erase the slow body. That gives us the clean test we kept wanting across Sigh/Jello/Child:

> after transient state is destroyed, did learning leave anything in the operator itself?

## Active observation

The observer does not inspect unseen current values to choose a sensor.

Each address receives priority from two bounded quantities:

```text
age since last observation
+
medium-timescale relevance learned from delayed consequence
```

Age prevents starvation. Relevance means an address that repeatedly participates in consequential errors becomes worth revisiting sooner.

This is deliberately much smaller than a learned attention network. It is an executable policy seam so a stronger GeometricNeuron/LentoOrava information-gain policy can be swapped in later.

## Delayed causal receipts

Every action produces a `DecisionReceipt` containing

```text
source tick
sensor address
observed scalar
output response / score
receiver trace
local slow-fast eligibility
```

The receipt is retained externally until consequence arrives. No reverse traversal through the intervening fluid trajectory is performed.

The current local eligibility is the same family used by Gate 5:

```text
chi_i = integral mask_i * (u_slow dot u_fast) dx
```

This says only that object `i` was locally involved in the passing flow. It does not prove that `i` caused the eventual behavioral error.

## Failure-gated structural plasticity

Delayed consequence supplies a desired signed output response. The machine writes only when the old event either

```text
made the wrong binary action
or
remained inside the configured assurance margin
```

An already-correct, sufficiently strong response does not keep strengthening itself.

For a writable event:

```text
delta_gamma_i = eta * output_error * chi_i
```

Then circulation is clipped and projected onto a finite L1 budget.

```text
sum_i |gamma_i| <= budget
```

That means acquiring more structure in one place can force reallocation elsewhere. The machine cannot solve adaptation by increasing every persistent object forever.

## What is assembled now

`src/mpri/machine.py` contains the reusable runtime.

`src/mpri/organism_demo.py` provides one uninterrupted toy stream:

- three possible observation addresses;
- one slowly changing binary cue plus two distractor processes;
- no episode/reset markers;
- one selected observation per tick;
- private traces;
- persistent fast fluid state;
- delayed scalar consequence;
- relevance update;
- failure-gated structural write;
- explicit fast-state wipe;
- frozen post-wipe evaluation.

Run it with:

```bash
python -m pip install -e .[dev]
mpri-organism --out results/organism_latest.json
pytest -q
```

The demo is a **smoke instrument**, not a benchmark victory. Its JSON receipt deliberately reports raw task accuracy, write count, address use, slow-body change, the finite circulation budget, and whether the slow body survives a complete fast wipe. A poor classifier score is allowed to stay poor; the purpose of this first assembly is to make failures localizable instead of hiding them behind another isolated positive gate.

## What is still missing

The current full loop contains one explicit engineered step that we already know is provisional:

```text
slow structural write = hand supplied three-factor rule
```

The strongest next replacement is therefore not another gate around the outside of the machine. It is **inside the machine**:

```text
current explicit local write
        |
        v
collision-specific fast-fast transfer
        |
        v
measured low-k slow change
        |
        v
same later routing effect
```

The controlled version should use the matched residual already identified in the README:

```text
P_slow[omega_AB - omega_A - omega_B + omega_0]
```

and require fast-mode washout before calling anything memory.

If that endogenous write fails, the assembled machine still remains useful: the substrate and the learning rule are now separable modules instead of one story.

## First serious task after the smoke

Do not return to an endless ladder of tiny gates. Give this one object a continuous visual world.

A useful first world should have:

- moving objects and distractors;
- a task-relevant relation that cannot be inferred from one frame;
- costly addressed observation;
- action that changes later evidence;
- delayed consequence;
- an occasional unannounced contingency reversal;
- a final fast-state wipe followed by frozen reuse.

The important measurements are then not just accuracy. Record:

```text
observation cost
route / probe usage
fast-state dependence
slow-body dependence
writes per useful adaptation
old-route damage
reversal recovery
post-wipe retained behavior
```

A small RNN/SSM/attention controller is allowed to beat it. The point of the assembled organism is that we finally have one place where every old idea can be attacked at once.
