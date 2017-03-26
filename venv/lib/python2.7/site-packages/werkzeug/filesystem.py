# -*- coding: utf-8 -*-
"""
    werkzeug.filesystem
    ~~~~~~~~~~~~~~~~~~~

    Various utilities for the local filesystem.

    :copyright: (c) 2015 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import codecs
import sys
import warnings

# We do not trust traditional unixes.
has_likely_buggy_unicode_filesystem = \
    sys.platform.startswith('linux') or 'bsd' in sys.platform


def _is_ascii_encoding(encoding):
    """
    Given an encoding this figures out if the encoding is actually ASCII (which
    is something we don't actually want in most cases). This is necessary
    because ASCII comes under many names such as ANSI_X3.4-1968.
    """
    if encoding is None:
        return False
    try:
        return codecs.lookup(encoding).name == 'ascii'
    except LookupError:
        return False


class BrokenFilesystemWarning(RuntimeWarning, UnicodeWarning):
    '''The warning used by Werkzeug to signal a broken filesystem. Will only be
    used once per runtime.'''


_warned_about_filesystem_encoding = False


def get_filesystem_encoding():
    """
    Returns the filesystem encoding that should be used. Note that this is
    different from the Python understanding of the filesystem encoding which
    might be deeply flawed. Do not use this value against Python's unicode APIs
    because it might be different. See :ref:`filesystem-encoding` for the exact
    behavior.

    The concept of a filesystem encoding in generally is not something you
    should rely on. As such if you ever need to use this function except for
    writing wrapper code reconsider.
    """
    global _warned_about_filesystem_encoding
    rv = sys.getfilesystemencoding()
    if has_likely_buggy_unicode_filesystem and not rv \
       or _is_ascii_encoding(rv):
        if not _warned_about_filesystem_encoding:
            warnings.warn(
                'Detected a misconfigured UNIX filesystem: Will use UTF-8 as '
                'filesystem encoding instead of {0!r}'.format(rv),
                BrokenFilesystemWarning)
            _warned_about_filesystem_encoding = True
        return 'utf-8'
    return rv
