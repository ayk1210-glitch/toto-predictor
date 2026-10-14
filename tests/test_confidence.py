import pytest

from toto.confidence import classify


@pytest.mark.parametrize("prob, label", [
    (0.90, "확실"), (0.65, "확실"), (0.6499, "유력"), (0.55, "유력"),
    (0.50, "애매"), (0.45, "애매"), (0.44, "예측 어려움"), (0.34, "예측 어려움"),
])
def test_tier_boundaries(prob, label):
    assert classify(prob).label == label
