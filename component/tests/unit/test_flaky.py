import random

import pytest


@pytest.mark.parametrize("input", range(0, 90000))
def test_flaky(input):
    assert input < 1000000
    assert random.randint(0, 1)
