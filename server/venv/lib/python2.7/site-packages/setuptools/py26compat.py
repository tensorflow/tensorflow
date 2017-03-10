"""
Compatibility Support for Python 2.6 and earlier
"""

import sys

try:
    from urllib.parse import splittag
except ImportError:
    from urllib import splittag


def strip_fragment(url):
    """
    In `Python 8280 <http://bugs.python.org/issue8280>`_, Python 2.7 and
    later was patched to disregard the fragment when making URL requests.
    Do the same for Python 2.6 and earlier.
    """
    url, fragment = splittag(url)
    return url


if sys.version_info >= (2, 7):
    strip_fragment = lambda x: x

try:
    from importlib import import_module
except ImportError:

    def import_module(module_name):
        return __import__(module_name, fromlist=['__name__'])
