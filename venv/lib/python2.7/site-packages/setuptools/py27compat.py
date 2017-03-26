"""
Compatibility Support for Python 2.7 and earlier
"""

import sys
import platform


def get_all_headers(message, key):
    """
    Given an HTTPMessage, return all headers matching a given key.
    """
    return message.get_all(key)


if sys.version_info < (3,):

    def get_all_headers(message, key):
        return message.getheaders(key)


linux_py2_ascii = (
	platform.system() == 'Linux' and
	sys.version_info < (3,)
)

rmtree_safe = str if linux_py2_ascii else lambda x: x
"""Workaround for http://bugs.python.org/issue24672"""
