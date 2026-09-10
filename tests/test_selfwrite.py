import numpy as np

from mpri.gates import default_layer
from mpri.selfwrite import episode_response, local_cross_write


def test_local_cross_write_changes_persistent_objects():
    layer = default_layer(n=24)
    x = np.array([1.0, 0.0, 0.0])
    before = np.array([w.circulation for w in layer.weights])

    written, history = local_cross_write(
        layer,
        x,
        target_probe=0,
        target_response=0.02,
        epochs=1,
        write_steps=6,
        write_rate=5000.0,
    )
    after = np.array([w.circulation for w in written.weights])

    assert len(history) == 1
    assert np.linalg.norm(after - before) > 1e-10
    assert episode_response(written, x, steps=6).shape == (3,)
