import numpy as np

from mpri.fluid import Probe, VortexObject, VorticityFluid2D
from mpri.model import FluidWeightLayer, effective_rank, interaction_residual


def tiny_layer():
    f = VorticityFluid2D(n=24, dt=0.008)
    return FluidWeightLayer(
        fluid=f,
        weights=[VortexObject(3.0, 3.0, 0.8)],
        input_ports=[VortexObject(1.0, 2.0, 1.0), VortexObject(1.0, 4.0, -1.0)],
        output_probes=[Probe(5.0, 2.0), Probe(5.0, 4.0)],
        steps=6,
    )


def test_transfer_matrix_shape_and_weight_sensitivity():
    layer = tiny_layer()
    j0 = layer.transfer_matrix()
    j1 = layer.with_circulations([1.2]).transfer_matrix()
    assert j0.shape == (2, 2)
    assert np.linalg.norm(j1 - j0) > 1e-9


def test_effective_rank_bounds():
    x = np.diag([4.0, 1.0, 0.0])
    r = effective_rank(x)
    assert 1.0 < r < 2.1


def test_nonlinear_interaction_is_nonzero():
    layer = tiny_layer()
    f = layer.fluid
    bg = layer.background()
    a = f.gaussian_vortex(2.7, 3.0, 0.5, 0.24)
    b = f.gaussian_vortex(2.9, 3.1, -0.45, 0.24)
    r = interaction_residual(f, bg, a, b, steps=5)
    assert r > 1e-8
