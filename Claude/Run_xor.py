"""
Experiment 2 -- the actual build. Does the flow compute anything?

This is the Fernando & Sojakka "bucket of water" benchmark, run on this
substrate instead of a real bucket: inject two bits as signed vortex packets
at sites A and B, let the background flow nonlinearly mix their influence,
read out a downstream patch, and train a LINEAR classifier on that readout
to predict XOR(bit_A, bit_B).

Why this is the right test: XOR is not linearly separable in the *inputs*.
If the flow only superposed the two packets' effects linearly (the additivity
result above, in the high-cosine regime), a linear readout could NEVER solve
XOR -- superposition keeps you in the span of {response(A alone), response(B
alone)}, which is exactly the situation a linear classifier already handles
without any reservoir. XOR only becomes linearly readable if the medium's
OWN nonlinearity mixes the two inputs into a new, jointly-informative
direction. So XOR accuracy is a direct, checkable test of whether the
"backreaction" bet (flow reshapes itself around what's injected) is doing
real computational work, not just passing packets through.

Each of the 4 bit combinations is run with amplitude/position jitter so
there's an actual train/test split instead of 4 fixed points.
"""
import numpy as np
from setup_geom import make_field, PROBE, CENTER

rng = np.random.default_rng(0)

T_STEPS = 1200
Y_INJECT = CENTER - 1.1
BASE_X = {0: -0.9, 1: 0.9}   # bit -> nominal x-offset from center at site A / B... see below
SIGN = {0: -1.0, 1: +1.0}    # bit -> vortex circulation sign
AMP = 8.0
JITTER_POS = 0.06
JITTER_AMP = 0.6

# fixed site x-positions (site identity matters, not just the bit value)
SITE_A_X = CENTER - 0.5
SITE_B_X = CENTER + 0.5


def run_trial(bit_a, bit_b):
    f = make_field(dt=1e-3)
    xa = SITE_A_X + rng.normal(0, JITTER_POS)
    xb = SITE_B_X + rng.normal(0, JITTER_POS)
    amp_a = AMP + rng.normal(0, JITTER_AMP)
    amp_b = AMP + rng.normal(0, JITTER_AMP)
    f.add_gaussian_vortex(xa, Y_INJECT, sign=SIGN[bit_a], amp=amp_a, sigma=0.3)
    f.add_gaussian_vortex(xb, Y_INJECT, sign=SIGN[bit_b], amp=amp_b, sigma=0.3)
    for _ in range(T_STEPS):
        f.step()
    return f.probe_patch(*PROBE, half_px=6)


def main():
    N_PER_CLASS = 22
    X, y_xor, y_or, y_and, bits_log = [], [], [], [], []

    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (a, b) in combos:
        print(f"bits=({a},{b}) : {N_PER_CLASS} jittered trials...", flush=True)
        for _ in range(N_PER_CLASS):
            feat = run_trial(a, b)
            X.append(feat)
            y_xor.append(a ^ b)
            y_or.append(a | b)
            y_and.append(a & b)
            bits_log.append((a, b))

    X = np.array(X)
    y_xor = np.array(y_xor)
    y_or = np.array(y_or)
    y_and = np.array(y_and)
    np.savez('xor_data.npz', X=X, y_xor=y_xor, y_or=y_or, y_and=y_and, bits=np.array(bits_log))
    print("saved xor_data.npz", X.shape)


if __name__ == '__main__':
    main()