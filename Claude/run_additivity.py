"""
Experiment 1 -- does this fluid do what CausalHorizon/Kompressori predicted?

Prediction being tested (the thing "give it a chance" was about):
  two localized perturbations inserted into the flow should compose almost
  additively when their propagation cones don't overlap, and interfere when
  they do -- same shape as Kompressori's measured Delta-J cosine test.

Method: inject two Gaussian vortex packets at sites A and B, separated by a
controllable distance along the injection line. Run four sims per separation:
  base  = background flow alone
  A     = background + packet A
  B     = background + packet B
  AB    = background + packet A + packet B
Evolve all four for T steps, take a windowed vorticity patch at a fixed
downstream probe. Define:
  dA  = patch(A)  - patch(base)
  dB  = patch(B)  - patch(base)
  dAB = patch(AB) - patch(base)
Compare dAB against the linear prediction (dA + dB) via cosine similarity
and relative L2 error. Sweep separation and see whether cosine -> 1 as
separation grows, and drops as the packets get close together.

Honest note: unlike CausalHorizon (proved exactly zero) or the OpenAI NS
paper (engineered exactly zero via disjoint dyadic supports), this fluid was
NOT built to guarantee additivity -- it's an ordinary shear flow. Whatever
comes out is a measurement, not a designed-in guarantee.
"""
import numpy as np
from setup_geom import make_field, PROBE, CENTER

T_STEPS = 1200
Y_INJECT = CENTER - 1.1
AMP = 10.0   # comparable in strength to the background vortices themselves
SEPARATIONS = [0.0, 0.15, 0.3, 0.5, 0.7, 1.0, 1.4, 1.9, 2.5, 3.2]


def run_case(sites):
    """sites: list of (x, y, sign) packets to inject beyond the background."""
    f = make_field(dt=1e-3)
    for (x0, y0, sign) in sites:
        f.add_gaussian_vortex(x0, y0, sign=sign, amp=AMP, sigma=0.3)
    for _ in range(T_STEPS):
        f.step()
    return f.probe_patch(*PROBE, half_px=6)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float('nan')
    return float(np.dot(a, b) / (na*nb))


def main():
    print("Running baseline (background only)...")
    base = run_case([])

    results = []
    for sep in SEPARATIONS:
        xa = CENTER - sep/2
        xb = CENTER + sep/2
        print(f"separation={sep:.2f} ...", flush=True)

        pA = run_case([(xa, Y_INJECT, +1)])
        pB = run_case([(xb, Y_INJECT, +1)])
        pAB = run_case([(xa, Y_INJECT, +1), (xb, Y_INJECT, +1)])

        dA = pA - base
        dB = pB - base
        dAB = pAB - base
        predicted_linear = dA + dB

        cos = cosine(dAB, predicted_linear)
        rel_err = float(np.linalg.norm(dAB - predicted_linear) / (np.linalg.norm(dAB) + 1e-12))

        results.append(dict(sep=sep, cosine=cos, rel_err=rel_err,
                             norm_dAB=float(np.linalg.norm(dAB)),
                             norm_dA=float(np.linalg.norm(dA)),
                             norm_dB=float(np.linalg.norm(dB))))
        print(f"  cosine(dAB, dA+dB) = {cos:.4f}   rel_err = {rel_err:.4f}")

    np.save('additivity_results.npy', results, allow_pickle=True)
    print("\nsaved additivity_results.npy")
    print("\n%8s %10s %10s" % ("sep", "cosine", "rel_err"))
    for r in results:
        print("%8.2f %10.4f %10.4f" % (r['sep'], r['cosine'], r['rel_err']))


if __name__ == '__main__':
    main()
