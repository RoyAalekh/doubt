from doubt import doubt

def test_check_does_not_modify_original_behavior():
    @doubt()
    def f(x):
        return x + 1

    assert f(1) == 2
