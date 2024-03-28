"""
    pygments.lexers
    ~~~~~~~~~~~~~~~

    Pygments lexers.

    :copyright: Copyright 2006-2022 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re
import sys
import types
from fnmatch import fnmatch
from os.path import basename

from pip._vendor.pygments.lexers._mapping import LEXERS
from pip._vendor.pygments.modeline import get_filetype_from_buffer
from pip._vendor.pygments.plugin import find_plugin_lexers
from pip._vendor.pygments.util import ClassNotFound, guess_decode

COMPAT = {
    'Python3Lexer': 'PythonLexer',
    'Python3TracebackLexer': 'PythonTracebackLexer',
}

__all__ = ['get_lexer_by_name', 'get_lexer_for_filename', 'find_lexer_class',
           'guess_lexer', 'load_lexer_from_file'] + list(LEXERS) + list(COMPAT)

_lexer_cache = {}

def _load_lexers(module_name):
    """Load a lexer (and all others in the module too)."""
    mod = __import__(module_name, None, None, ['__all__'])
    for lexer_name in mod.__all__:
        cls = getattr(mod, lexer_name)
        _lexer_cache[cls.name] = cls


def get_all_lexers(plugins=True):
    """Return a generator of tuples in the form ``(name, aliases,
    filenames, mimetypes)`` of all know lexers.

    If *plugins* is true (the default), plugin lexers supplied by entrypoints
    are also returned.  Otherwise, only builtin ones are considered.
    """
    for item in LEXERS.values():
        yield item[1:]
    if plugins:
        for lexer in find_plugin_lexers():
            yield lexer.name, lexer.aliases, lexer.filenames, lexer.mimetypes


def find_lexer_class(name):
    """Lookup a lexer class by name.

    Return None if not found.
    """
    if name in _lexer_cache:
        return _lexer_cache[name]
    # lookup builtin lexers
    for module_name, lname, aliases, _, _ in LEXERS.values():
        if name == lname:
            _load_lexers(module_name)
            return _lexer_cache[name]
    # continue with lexers from setuptools entrypoints
    for cls in find_plugin_lexers():
        if cls.name == name:
            return cls


def find_lexer_class_by_name(_alias):
    """Lookup a lexer class by alias.

    Like `get_lexer_by_name`, but does not instantiate the class.

    .. versionadded:: 2.2
    """
    if not _alias:
        raise ClassNotFound('no lexer for alias %r found' % _alias)
    # lookup builtin lexers
    for module_name, name, aliases, _, _ in LEXERS.values():
        if _alias.lower() in aliases:
            if name not in _lexer_cache:
                _load_lexers(module_name)
            return _lexer_cache[name]
    # continue with lexers from setuptools entrypoints
    for cls in find_plugin_lexers():
        if _alias.lower() in cls.aliases:
            return cls
    raise ClassNotFound('no lexer for alias %r found' % _alias)


def get_lexer_by_name(_alias, **options):
    """Get a lexer by an alias.

    Raises ClassNotFound if not found.
    """
    if not _alias:
        raise ClassNotFound('no lexer for alias %r found' % _alias)

    # lookup builtin lexers
    for module_name, name, aliases, _, _ in LEXERS.values():
        if _alias.lower() in aliases:
            if name not in _lexer_cache:
                _load_lexers(module_name)
            return _lexer_cache[name](**options)
    # continue with lexers from setuptools entrypoints
    for cls in find_plugin_lexers():
        if _alias.lower() in cls.aliases:
            return cls(**options)
    raise ClassNotFound('no lexer for alias %r found' % _alias)


def load_lexer_from_file(filename, lexername="CustomLexer", **options):
    """Load a lexer from a file.

    This method expects a file located relative to the current working
    directory, which contains a Lexer class. By default, it expects the
    Lexer to be name CustomLexer; you can specify your own class name
    as the second argument to this function.

    Users should be very careful with the input, because this method
    is equivalent to running eval on the input file.

    Raises ClassNotFound if there are any problems importing the Lexer.

    .. versionadded:: 2.2
    """
    try:
        # This empty dict will contain the namespace for the exec'd file
        custom_namespace = {}
        with open(filename, 'rb') as f:
            exec(f.read(), custom_namespace)
        # Retrieve the class `lexername` from that namespace
        if lexername not in custom_namespace:
            raise ClassNotFound('no valid %s class found in %s' %
                                (lexername, filename))
        lexer_class = custom_namespace[lexername]
        # And finally instantiate it with the options
        return lexer_class(**options)
    except OSError as err:
        raise ClassNotFound('cannot read %s: %s' % (filename, err))
    except ClassNotFound:
        raise
    except Exception as err:
        raise ClassNotFound('error when loading custom lexer: %s' % err)


def find_lexer_class_for_filename(_fn, code=None):
    """Get a lexer for a filename.

    If multiple lexers match the filename pattern, use ``analyse_text()`` to
    figure out which one is more appropriate.

    Returns None if not found.
    """
    matches = []
    fn = basename(_fn)
    for modname, name, _, filenames, _ in LEXERS.values():
        for filename in filenames:
            if fnmatch(fn, filename):
                if name not in _lexer_cache:
                    _load_lexers(modname)
                matches.append((_lexer_cache[name], filename))
    for cls in find_plugin_lexers():
        for filename in cls.filenames:
            if fnmatch(fn, filename):
                matches.append((cls, filename))

    if isinstance(code, bytes):
        # decode it, since all analyse_text functions expect unicode
        code = guess_decode(code)

    def get_rating(info):
        cls, filename = info
        # explicit patterns get a bonus
        bonus = '*' not in filename and 0.5 or 0
        # The class _always_ defines analyse_text because it's included in
        # the Lexer class.  The default implementation returns None which
        # gets turned into 0.0.  Run scripts/detect_missing_analyse_text.py
        # to find lexers which need it overridden.
        if code:
            return cls.analyse_text(code) + bonus, cls.__name__
        return cls.priority + bonus, cls.__name__

    if matches:
        matches.sort(key=get_rating)
        # print "Possible lexers, after sort:", matches
        return matches[-1][0]


def get_lexer_for_filename(_fn, code=None, **options):
    """Get a lexer for a filename.

    If multiple lexers match the filename pattern, use ``analyse_text()`` to
    figure out which one is more appropriate.

    Raises ClassNotFound if not found.
    """
    res = find_lexer_class_for_filename(_fn, code)
    if not res:
        raise ClassNotFound('no lexer for filename %r found' % _fn)
    return res(**options)


def get_lexer_for_mimetype(_mime, **options):
    """Get a lexer for a mimetype.

    Raises ClassNotFound if not found.
    """
    for modname, name, _, _, mimetypes in LEXERS.values():
        if _mime in mimetypes:
            if name not in _lexer_cache:
                _load_lexers(modname)
            return _lexer_cache[name](**options)
    for cls in find_plugin_lexers():
        if _mime in cls.mimetypes:
            return cls(**options)
    raise ClassNotFound('no lexer for mimetype %r found' % _mime)


def _iter_lexerclasses(plugins=True):
    """Return an iterator over all lexer classes."""
    for key in sorted(LEXERS):
        module_name, name = LEXERS[key][:2]
        if name not in _lexer_cache:
            _load_lexers(module_name)
        yield _lexer_cache[name]
    if plugins:
        yield from find_plugin_lexers()


def guess_lexer_for_filename(_fn, _text, **options):
    """
    Lookup all lexers that handle those filenames primary (``filenames``)
    or secondary (``alias_filenames``). Then run a text analysis for those
    lexers and choose the best result.

    usage::

        >>> from pygments.lexers import guess_lexer_for_filename
        >>> guess_lexer_for_filename('hello.html', '<%= @foo %>')
        <pygments.lexers.templates.RhtmlLexer object at 0xb7d2f32c>
        >>> guess_lexer_for_filename('hello.html', '<h1>{{ title|e }}</h1>')
        <pygments.lexers.templates.HtmlDjangoLexer object at 0xb7d2f2ac>
        >>> guess_lexer_for_filename('style.css', 'a { color: <?= $link ?> }')
        <pygments.lexers.templates.CssPhpLexer object at 0xb7ba518c>
    """
    fn = basename(_fn)
    primary = {}
    matching_lexers = set()
    for lexer in _iter_lexerclasses():
        for filename in lexer.filenames:
            if fnmatch(fn, filename):
                matching_lexers.add(lexer)
                primary[lexer] = True
        for filename in lexer.alias_filenames:
            if fnmatch(fn, filename):
                matching_lexers.add(lexer)
                primary[lexer] = False
    if not matching_lexers:
        raise ClassNotFound('no lexer for filename %r found' % fn)
    if len(matching_lexers) == 1:
        return matching_lexers.pop()(**options)
    result = []
    for lexer in matching_lexers:
        rv = lexer.analyse_text(_text)
        if rv == 1.0:
            return lexer(**options)
        result.append((rv, lexer))

    def type_sort(t):
        # sort by:
        # - analyse score
        # - is primary filename pattern?
        # - priority
        # - last resort: class name
        return (t[0], primary[t[1]], t[1].priority, t[1].__name__)
    result.sort(key=type_sort)

    return result[-1][1](**options)


def guess_lexer(_text, **options):
    """Guess a lexer by strong distinctions in the text (eg, shebang)."""

    if not isinstance(_text, str):
        inencoding = options.get('inencoding', options.get('encoding'))
        if inencoding:
            _text = _text.decode(inencoding or 'utf8')
        else:
            _text, _ = guess_decode(_text)

    # try to get a vim modeline first
    ft = get_filetype_from_buffer(_text)

    if ft is not None:
        try:
            return get_lexer_by_name(ft, **options)
        except ClassNotFound:
            pass

    best_lexer = [0.0, None]
    for lexer in _iter_lexerclasses():
        rv = lexer.analyse_text(_text)
        if rv == 1.0:
            return lexer(**options)
        if rv > best_lexer[0]:
            best_lexer[:] = (rv, lexer)
    if not best_lexer[0] or best_lexer[1] is None:
        raise ClassNotFound('no lexer matching the text found')
    return best_lexer[1](**options)


class _automodule(types.ModuleType):
    """Automatically import lexers."""

    def __getattr__(self, name):
        info = LEXERS.get(name)
        if info:
            _load_lexers(info[0])
            cls = _lexer_cache[info[1]]
            setattr(self, name, cls)
            return cls
        if name in COMPAT:
            return getattr(self, COMPAT[name])
        raise AttributeError(name)


oldmod = sys.modules[__name__]
newmod = _automodule(__name__)
newmod.__dict__.update(oldmod.__dict__)
sys.modules[__name__] = newmod
del newmod.newmod, newmod.oldmod, newmod.sys, newmod.types
