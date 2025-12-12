import pytest

from doubt import assert_missing_robust, doubt


@doubt()
def safe_sum(xs):
    return sum(x for x in xs if x is not None)


@doubt()
def unsafe_sum(xs):
    return sum(xs)


def test_assert_missing_robust_passes():
    assert_missing_robust(
        safe_sum,
        [1, 2, 3],
        max_relative_change=0.5,
    )


def test_assert_missing_robust_fails_on_crash():
    with pytest.raises(AssertionError):
        assert_missing_robust(unsafe_sum, [1, 2, 3])
