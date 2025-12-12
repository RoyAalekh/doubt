from doubt import doubt


def test_list_perturbations_generated():
    @doubt(max_scenarios_per_arg=3)
    def f(xs):
        return len(xs)

    result = f.check([1, 2, 3, 4, 5])

    assert len(result.scenarios) == 3


def test_dict_perturbations_generated():
    @doubt()
    def f(d):
        return sum(d.values())

    result = f.check({"a": 1, "b": 2})

    assert len(result.scenarios) >= 1
