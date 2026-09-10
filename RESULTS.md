# Gate receipt

Default run: `grid=28`, `dt=0.012`, viscosity `0.003`, drag `0.01`.

| Gate | Measurement | Receipt |
| --- | --- | ---: |
| G0 | divergence RMS, initial | `1.605e-16` |
| G0 | divergence RMS, final | `1.637e-16` |
| G1 | effective rank of full `J` | `2.158` |
| G1 | effective rank of one-vortex `ΔJ` | `1.375` |
| G2 | overlapping interaction residual | `0.018160` |
| G2 | separated interaction residual | `0.001924` |
| G2 | overlap / separated | **`9.440×`** |
| G3 | routing loss, before | `0.330466` |
| G3 | routing loss, after 8 epochs | **`0.293389`** |
| G4 | enstrophy gain with background, `k=1` | `1.06983×` |
| G4 | enstrophy gain with background, `k=5` | `0.99053×` |
| G4 | enstrophy gain with background, `k=6` | `0.97184×` |
| G5 | target response, before | `-0.001356` |
| G5 | target response, after +target self-write | **`+0.005627`** |
| G5 | absolute target error, before | `0.011356` |
| G5 | absolute target error, after | **`0.004373`** |
| G5 | cosine(+target write, -target write) | **`-0.801`** |
| G5 | effective rank of self-written `ΔJ` | `1.270` |
| G5 | Frobenius norm of self-written `ΔJ` | `0.018739` |

Gate 5 uses an added local plasticity law

```text
delta_gamma_i = eta * scalar_error * local_velocity_cross_energy_i
```

rather than a finite-difference gradient. The scalar error is global, but the eligibility term is measured locally around each persistent vortex object. This is a mechanism probe, not a claim that Navier–Stokes itself performs learning.

These are receipts for this toy configuration, not general fluid-computing claims. Re-run `python -m mpri.gates` to regenerate them.
