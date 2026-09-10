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

These are receipts for this toy configuration, not general fluid-computing claims. Re-run `python -m mpri.gates` to regenerate them.
