import random

import pytest

IS_FLAKY = False


@pytest.mark.parametrize("param", range(0, 90_000))
def test_flaky(param):
    assert param < 100_000
    if IS_FLAKY:
        assert random.randint(0, 1_000)
