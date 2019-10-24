import os

# This module should be kept compatible with Python 2.1.

__revision__ = "$Id: debug.py 37828 2004-11-10 22:23:15Z loewis $"

# If DISTUTILS_DEBUG is anything other than the empty string, we run in
# debug mode.
DEBUG = os.environ.get('DISTUTILS_DEBUG')
