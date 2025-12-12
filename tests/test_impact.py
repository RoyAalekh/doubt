
from doubt import ImpactType, doubt


def test_crash_detection():
    @doubt()
    def f(xs):
        return sum(xs)

    result = f.check([1, 2, 3])

    assert any(s.impact_type == ImpactType.CRASH for s in result.scenarios)


def test_silent_change_detection():
    @doubt()
    def f(xs):
        return sum(x for x in xs if x is not None)

    result = f.check([1, 2, 3])

    assert any(s.impact_type == ImpactType.SILENT_CHANGE for s in result.scenarios)


def test_no_change_detection():
    @doubt()
    def f(x):
        return 42

    result = f.check(10)

    assert all(s.impact_type == ImpactType.NO_CHANGE for s in result.scenarios)
