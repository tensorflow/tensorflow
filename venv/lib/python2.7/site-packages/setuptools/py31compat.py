import sys
import unittest

__all__ = ['get_config_vars', 'get_path']

try:
    # Python 2.7 or >=3.2
    from sysconfig import get_config_vars, get_path
except ImportError:
    from distutils.sysconfig import get_config_vars, get_python_lib

    def get_path(name):
        if name not in ('platlib', 'purelib'):
            raise ValueError("Name must be purelib or platlib")
        return get_python_lib(name == 'platlib')


try:
    # Python >=3.2
    from tempfile import TemporaryDirectory
except ImportError:
    import shutil
    import tempfile

    class TemporaryDirectory(object):
        """
        Very simple temporary directory context manager.
        Will try to delete afterward, but will also ignore OS and similar
        errors on deletion.
        """

        def __init__(self):
            self.name = None  # Handle mkdtemp raising an exception
            self.name = tempfile.mkdtemp()

        def __enter__(self):
            return self.name

        def __exit__(self, exctype, excvalue, exctrace):
            try:
                shutil.rmtree(self.name, True)
            except OSError:  # removal errors are not the only possible
                pass
            self.name = None


unittest_main = unittest.main

_PY31 = (3, 1) <= sys.version_info[:2] < (3, 2)
if _PY31:
    # on Python 3.1, translate testRunner==None to TextTestRunner
    # for compatibility with Python 2.6, 2.7, and 3.2+
    def unittest_main(*args, **kwargs):
        if 'testRunner' in kwargs and kwargs['testRunner'] is None:
            kwargs['testRunner'] = unittest.TextTestRunner
        return unittest.main(*args, **kwargs)
