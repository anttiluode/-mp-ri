"""
The fixed geometry every experiment below shares.

Background flow = two counter-rotating Gaussian vortices ("the weights").
They set up a shear channel between them. Two input sites sit on either
side of the channel; a probe sits downstream on the far side.

This geometry is arbitrary -- picked once, then held fixed across all
experiments so results are comparable to each other.
"""
import numpy as np
from ns_core import FlowField

L = 2*np.pi
CENTER = L/2

# background vortex pair (counter-rotating -> shear region between them)
BG = [
    dict(x0=CENTER - 0.9, y0=CENTER, sign=+1, amp=10.0, sigma=0.5),
    dict(x0=CENTER + 0.9, y0=CENTER, sign=-1, amp=10.0, sigma=0.5),
]

# two input injection sites -- deliberately NOT mirror-symmetric about the
# background's own symmetry axis, so their individual responses don't
# cancel by construction
SITE_A = (CENTER - 0.55, CENTER - 1.05)
SITE_B = (CENTER + 0.20, CENTER - 0.85)

# downstream probe, off-axis too
PROBE = (CENTER + 0.4, CENTER + 1.3)


def make_field(N=64, dt=1e-3, nu=6e-4):
    f = FlowField(N=N, L=L, nu=nu, dt=dt)
    for v in BG:
        f.add_gaussian_vortex(**v)
    return f