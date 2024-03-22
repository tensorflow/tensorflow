"""
    pygments.formatters
    ~~~~~~~~~~~~~~~~~~~

    Pygments formatters.

    :copyright: Copyright 2006-2022 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re
import sys
import types
from fnmatch import fnmatch
from os.path import basename

from pip._vendor.pygments.formatters._mapping import FORMATTERS
from pip._vendor.pygments.plugin import find_plugin_formatters
from pip._vendor.pygments.util import ClassNotFound

__all__ = ['get_formatter_by_name', 'get_formatter_for_filename',
           'get_all_formatters', 'load_formatter_from_file'] + list(FORMATTERS)

_formatter_cache = {}  # classes by name

def _load_formatters(module_name):
    """Load a formatter (and all others in the module too)."""
    mod = __import__(module_name, None, None, ['__all__'])
    for formatter_name in mod.__all__:
        cls = getattr(mod, formatter_name)
        _formatter_cache[cls.name] = cls


def get_all_formatters():
    """Return a generator for all formatter classes."""
    # NB: this returns formatter classes, not info like get_all_lexers().
    for info in FORMATTERS.values():
        if info[1] not in _formatter_cache:
            _load_formatters(info[0])
        yield _formatter_cache[info[1]]
    for _, formatter in find_plugin_formatters():
        yield formatter


def find_formatter_class(alias):
    """Lookup a formatter by alias.

    Returns None if not found.
    """
    for module_name, name, aliases, _, _ in FORMATTERS.values():
        if alias in aliases:
            if name not in _formatter_cache:
                _load_formatters(module_name)
            return _formatter_cache[name]
    for _, cls in find_plugin_formatters():
        if alias in cls.aliases:
            return cls


def get_formatter_by_name(_alias, **options):
    """Lookup and instantiate a formatter by alias.

    Raises ClassNotFound if not found.
    """
    cls = find_formatter_class(_alias)
    if cls is None:
        raise ClassNotFound("no formatter found for name %r" % _alias)
    return cls(**options)


def load_formatter_from_file(filename, formattername="CustomFormatter",
                             **options):
    """Load a formatter from a file.

    This method expects a file located relative to the current working
    directory, which contains a class named CustomFormatter. By default,
    it expects the Formatter to be named CustomFormatter; you can specify
    your own class name as the second argument to this function.

    Users should be very careful with the input, because this method
    is equivalent to running eval on the input file.

    Raises ClassNotFound if there are any problems importing the Formatter.

    .. versionadded:: 2.2
    """
    try:
        # This empty dict will contain the namespace for the exec'd file
        custom_namespace = {}
        with open(filename, 'rb') as f:
            exec(f.read(), custom_namespace)
        # Retrieve the class `formattername` from that namespace
        if formattername not in custom_namespace:
            raise ClassNotFound('no valid %s class found in %s' %
                                (formattername, filename))
        formatter_class = custom_namespace[formattername]
        # And finally instantiate it with the options
        return formatter_class(**options)
    except OSError as err:
        raise ClassNotFound('cannot read %s: %s' % (filename, err))
    except ClassNotFound:
        raise
    except Exception as err:
        raise ClassNotFound('error when loading custom formatter: %s' % err)


def get_formatter_for_filename(fn, **options):
    """Lookup and instantiate a formatter by filename pattern.

    Raises ClassNotFound if not found.
    """
    fn = basename(fn)
    for modname, name, _, filenames, _ in FORMATTERS.values():
        for filename in filenames:
            if fnmatch(fn, filename):
                if name not in _formatter_cache:
                    _load_formatters(modname)
                return _formatter_cache[name](**options)
    for cls in find_plugin_formatters():
        for filename in cls.filenames:
            if fnmatch(fn, filename):
                return cls(**options)
    raise ClassNotFound("no formatter found for file name %r" % fn)


class _automodule(types.ModuleType):
    """Automatically import formatters."""

    def __getattr__(self, name):
        info = FORMATTERS.get(name)
        if info:
            _load_formatters(info[0])
            cls = _formatter_cache[info[1]]
            setattr(self, name, cls)
            return cls
        raise AttributeError(name)


oldmod = sys.modules[__name__]
newmod = _automodule(__name__)
newmod.__dict__.update(oldmod.__dict__)
sys.modules[__name__] = newmod
del newmod.newmod, newmod.oldmod, newmod.sys, newmod.types
