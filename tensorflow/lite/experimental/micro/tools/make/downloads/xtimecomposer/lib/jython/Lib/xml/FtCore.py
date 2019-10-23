"""
Contains various definitions common to modules acquired from 4Suite
"""

__all__ = ["FtException", "get_translator"]


class FtException(Exception):
    def __init__(self, errorCode, messages, args):
        # By defining __str__, args will be available.  Otherwise
        # the __init__ of Exception sets it to the passed in arguments.
        self.params = args
        self.errorCode = errorCode
        self.message = messages[errorCode] % args
        Exception.__init__(self, self.message, args)

    def __str__(self):
        return self.message


# What follows is used to provide support for I18N in the rest of the
# 4Suite-derived packages in PyXML.
#
# Each sub-package of the top-level "xml" package that contains 4Suite
# code is really a separate text domain, but they're all called
# '4Suite'.  For each domain, a translation object is provided using
# message catalogs stored inside the package.  The code below defines
# a get_translator() function that returns an appropriate gettext
# function to be used as _() in the sub-package named by the
# parameter.  This handles all the compatibility issues related to
# Python versions (whether the gettext module can be found) and
# whether the message catalogs can actually be found.

def _(msg):
    return msg

try:
    import gettext

except (ImportError, IOError):
    def get_translator(pkg):
        return _

else:
    import os

    _cache = {}
    _top = os.path.dirname(os.path.abspath(__file__))

    def get_translator(pkg):
        if not _cache.has_key(pkg):
            locale_dir = os.path.join(_top, pkg.replace(".", os.sep))
            try:
                f = gettext.translation('4Suite', locale_dir).gettext
            except IOError:
                f = _
            _cache[pkg] = f
        return _cache[pkg]
