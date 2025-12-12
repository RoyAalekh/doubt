import pytest

np = pytest.importorskip("numpy")

from doubt import doubt


def test_numpy_array_handling():
    @doubt()
    def f(arr):
        return arr.sum()

    result = f.check(np.array([1.0, 2.0, 3.0]))
    assert result.scenarios
