from doubt import doubt, DoubtResult


def test_decorator_preserves_function_behavior():
    @doubt()
    def f(x):
        return x + 1

    assert f(1) == 2


def test_check_returns_doubt_result():
    @doubt()
    def f(x):
        return x * 2

    result = f.check(10)
    assert isinstance(result, DoubtResult)
    assert result.baseline_output == 20
    assert result.baseline_ok


def test_baseline_crash_is_reported():
    @doubt()
    def f(x):
        return 1 / x

    result = f.check(0)
    assert not result.baseline_ok
    assert result.baseline_error is not None
