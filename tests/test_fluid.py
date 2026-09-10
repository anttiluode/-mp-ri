import numpy as np

from mpri.fluid import VortexObject, VorticityFluid2D


def test_periodic_fluid_stays_finite_and_incompressible():
    f = VorticityFluid2D(n=24, dt=0.01)
    state = f.field_from_objects([
        VortexObject(2.0, 2.0, 1.0),
        VortexObject(4.0, 4.1, -0.8),
    ])
    end = f.evolve(state, 8)
    assert np.isfinite(end).all()
    assert abs(float(end.mean())) < 1e-12
    assert f.divergence_rms(end) < 1e-10


def test_viscosity_damps_single_mode_enstrophy():
    f = VorticityFluid2D(n=24, viscosity=0.02, drag=0.0, dt=0.008)
    state = np.sin(3 * f.x)
    before = f.enstrophy(state)
    after = f.enstrophy(f.evolve(state, 15))
    assert after < before
