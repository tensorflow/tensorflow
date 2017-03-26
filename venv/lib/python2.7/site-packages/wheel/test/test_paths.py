import wheel.paths
from distutils.command.install import SCHEME_KEYS

def test_path():
    d = wheel.paths.get_install_paths('wheel')
    assert len(d) == len(SCHEME_KEYS)
