# Mathematical target

The repo is built around one decomposition:

\[
\text{persistent structure} \rightarrow \text{flow operator} \rightarrow \text{pulse trajectory} \rightarrow \text{local readout}.
\]

## 1. Fluid state

We evolve scalar vorticity on a periodic 2-D domain:

\[
\dot\omega = \mathcal N(\omega)
= -u[\omega]\cdot\nabla\omega + \nu\Delta\omega - \mu\omega,
\]

with incompressible velocity recovered from a streamfunction,

\[
-\Delta\psi=\omega,
\qquad
u=(\partial_y\psi,-\partial_x\psi).
\]

The finite-time propagator is

\[
F_T(\omega_0)=\omega(T).
\]

Unlike a fixed matrix, `F_T` is state-dependent because the state generates the velocity field that advects the state.

## 2. Weight objects

A persistent model weight is a localized vortex

\[
W_j(x)=\Gamma_j\exp\!\left(-\frac{|x-x_j|^2}{2\sigma_j^2}\right),
\]

with its spatial mean removed for the periodic Poisson solve. The background is

\[
B(x;\theta)=\sum_j W_j(x),
\qquad
\theta=\{x_j, y_j, \Gamma_j, \sigma_j\}_j.
\]

In the first implementation only the circulations `Γ_j` are trained. Positions and core sizes are explicit future degrees of freedom.

## 3. Activation pulses and observation

Input channel `q` injects a localized pulse `P_q`. For a small amplitude `ε`, a probe vector `M` defines a finite-difference transfer matrix

\[
J_{pq}(\theta)
= \frac{M_p F_T(B+\varepsilon P_q)-M_pF_T(B)}{\varepsilon}.
\]

This is the closest analogue of a layer's input→output Jacobian.

A local weight edit changes the whole response operator:

\[
\Delta J_j
=J(\theta+\delta\Gamma_j e_j)-J(\theta).
\]

Gate 1 measures the singular spectrum of `ΔJ_j` and compares it with the spectrum of `J`.

## 4. Interference

For two pulses `a` and `b`, define the finite-time nonlinear interaction residual

\[
R_T(a,b)=F_T(B+a+b)-F_T(B+a)-F_T(B+b)+F_T(B).
\]

`R_T=0` means exact additivity at horizon `T`. A large norm means the two pulses have coupled through the nonlinear flow.

This is the operational bridge to the older CausalHorizon/Kompressori question. We do **not** assume a nilpotent event operator here. We directly measure where superposition breaks.

## 5. Training

Gate 3 currently uses the simplest possible objective. Let `J(Γ)` be the measured transfer matrix and `T` a desired routing map. A scalar readout gain `a` is eliminated analytically,

\[
a^*=\frac{\langle J,T\rangle_F}{\langle J,J\rangle_F},
\qquad
L(\Gamma)=\|a^*J(\Gamma)-T\|_F^2.
\]

The circulation gradient is estimated by centered finite differences and applied directly to the vortex objects.

That algorithm is intentionally primitive. Its purpose is to make the ontological claim testable: **the same field objects that shape propagation can be the parameters that learning changes.**

## 6. Scale selection

For a wave packet with local wavenumber `k`, viscosity contributes a decay rate proportional to

\[
-\nu k^2.
\]

Background shear can transiently amplify/tilt the packet while viscosity becomes increasingly punitive at fine scales. Gate 4 scans `k` and measures the crossover in the actual numerical field.

## What would count as a stronger result

A stronger version should pass all of these attacks:

- learn routing with local structural updates, not global finite differences;
- preserve learned routing when new pulses are added;
- show that structural-overlap predicts interference better than Euclidean distance;
- compare against a matched ordinary reservoir and a learned linear operator;
- demonstrate multiple persistent “weight objects” sharing the same medium without crosstalk collapse;
- measure compute/energy cost rather than only accuracy.
