import unittest

from wheel.pep425tags import get_supported
from wheel.install import WheelFile

WHEELPAT = "%(name)s-%(ver)s-%(pyver)s-%(abi)s-%(arch)s.whl"
def make_wheel(name, ver, pyver, abi, arch):
    name = WHEELPAT % dict(name=name, ver=ver, pyver=pyver, abi=abi,
            arch=arch)
    return WheelFile(name)

# This relies on the fact that generate_supported will always return the
# exact pyver, abi, and architecture for its first (best) match.
sup = get_supported()
pyver, abi, arch = sup[0]
genver = 'py' + pyver[2:]
majver = genver[:3]

COMBINATIONS = (
    ('bar', '0.9', 'py2.py3', 'none', 'any'),
    ('bar', '0.9', majver, 'none', 'any'),
    ('bar', '0.9', genver, 'none', 'any'),
    ('bar', '0.9', pyver, abi, arch),
    ('bar', '1.3.2', majver, 'none', 'any'),
    ('bar', '3.1', genver, 'none', 'any'),
    ('bar', '3.1', pyver, abi, arch),
    ('foo', '1.0', majver, 'none', 'any'),
    ('foo', '1.1', pyver, abi, arch),
    ('foo', '2.1', majver + '0', 'none', 'any'),
    # This will not be compatible for Python x.0. Beware when we hit Python
    # 4.0, and don't test with 3.0!!!
    ('foo', '2.1', majver + '1', 'none', 'any'),
    ('foo', '2.1', pyver , 'none', 'any'),
    ('foo', '2.1', pyver , abi, arch),
)

WHEELS = [ make_wheel(*args) for args in COMBINATIONS ]

class TestRanking(unittest.TestCase):
    def test_comparison(self):
        for i in range(len(WHEELS)-1):
            for j in range(i):
                self.assertTrue(WHEELS[j]<WHEELS[i])
