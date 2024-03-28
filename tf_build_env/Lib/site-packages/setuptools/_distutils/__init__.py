"""distutils

The main package for the Python Module Distribution Utilities.  Normally
used from a setup script as

   from distutils.core import setup

   setup (...)
"""

import sys
import importlib

__version__ = sys.version[: sys.version.index(' ')]


try:
    # Allow Debian and pkgsrc (only) to customize system
    # behavior. Ref pypa/distutils#2 and pypa/distutils#16.
    # This hook is deprecated and no other environments
    # should use it.
    importlib.import_module('_distutils_system_mod')
except ImportError:
    pass
