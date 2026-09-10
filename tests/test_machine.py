import numpy as np

from mpri.fluid import Probe, VortexObject, VorticityFluid2D
from mpri.machine import (
    DecisionReceipt,
    FieldBodyMachine,
    MachineConfig,
    _project_l1_ball,
)


def make_machine():
    f = VorticityFluid2D(n=16, viscosity=3e-3, drag=1e-2, dt=0.01)
    weights = [
        VortexObject(2.7, 2.4, 1.3, 0.32),
        VortexObject(3.4, 3.2, -1.1, 0.32),
        VortexObject(4.0, 3.8, 0.9, 0.32),
    ]
    inputs = [
        VortexObject(1.7, 2.3, 1.0, 0.23),
        VortexObject(1.7, 3.9, -1.0, 0.23),
        VortexObject(2.0, 3.1, 1.0, 0.23),
    ]
    outputs = [Probe(4.8, 2.4, 0.32), Probe(4.8, 3.8, 0.32)]
    cfg = MachineConfig(fluid_substeps=1, write_rate=400.0, output_gain=20.0)
    return FieldBodyMachine(f, weights, inputs, outputs, config=cfg)


def test_l1_projection_obeys_budget():
    x = np.array([3.0, -2.0, 1.0])
    y = _project_l1_ball(x, 2.5)
    assert np.sum(np.abs(y)) <= 2.5 + 1e-12
    assert np.sign(y[0]) == 1
    assert np.sign(y[1]) == -1


def test_active_observer_does_not_starve_addresses():
    m = make_machine()
    picked = []
    for _ in range(6):
        a = m.choose_address()
        picked.append(a)
        m.observe_and_act(a, 0.0)
    assert set(picked[:3]) == {0, 1, 2}


def test_fast_reset_preserves_slow_body():
    m = make_machine()
    m.tick_once([1.0, -1.0, 0.5])
    before = m.slow_signature()
    assert np.linalg.norm(m.fast) > 0
    m.reset_fast()
    assert np.allclose(m.fast, 0.0)
    assert np.allclose(m.slow_signature(), before)


def test_assured_success_does_not_reinforce_itself():
    m = make_machine()
    receipt = DecisionReceipt(
        tick=0,
        address=0,
        observed_value=1.0,
        score=1.0,
        prediction=0.95,
        action=1,
        response=np.array([1.0, 0.0]),
        eligibility=np.array([0.2, -0.1, 0.05]),
        trace=np.zeros(3),
    )
    before = m.slow_signature()
    out = m.apply_consequence(receipt, target=1, learn=True)
    assert not out.wrote
    assert np.allclose(m.slow_signature(), before)


def test_failed_delayed_event_writes_its_retained_eligibility():
    m = make_machine()
    receipt = DecisionReceipt(
        tick=0,
        address=1,
        observed_value=-1.0,
        score=0.1,
        prediction=0.2,
        action=1,
        response=np.array([0.1, 0.0]),
        eligibility=np.array([2e-3, -1e-3, 0.5e-3]),
        trace=np.zeros(3),
    )
    before = m.slow_signature()
    out = m.apply_consequence(receipt, target=-1, learn=True)
    assert out.wrote
    assert not np.allclose(m.slow_signature(), before)
    assert np.sum(np.abs(m.slow_signature())) <= m.circulation_budget + 1e-12
