# -*- coding: utf-8 -*-
"""
    markupsafe
    ~~~~~~~~~~

    Implements a Markup string.

    :copyright: (c) 2010 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""
import re
import string
from collections import Mapping
from markupsafe._compat import text_type, string_types, int_types, \
     unichr, iteritems, PY2

__version__ = "1.0"

__all__ = ['Markup', 'soft_unicode', 'escape', 'escape_silent']


_striptags_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
_entity_re = re.compile(r'&([^& ;]+);')


class Markup(text_type):
    r"""Marks a string as being safe for inclusion in HTML/XML output without
    needing to be escaped.  This implements the `__html__` interface a couple
    of frameworks and web applications use.  :class:`Markup` is a direct
    subclass of `unicode` and provides all the methods of `unicode` just that
    it escapes arguments passed and always returns `Markup`.

    The `escape` function returns markup objects so that double escaping can't
    happen.

    The constructor of the :class:`Markup` class can be used for three
    different things:  When passed an unicode object it's assumed to be safe,
    when passed an object with an HTML representation (has an `__html__`
    method) that representation is used, otherwise the object passed is
    converted into a unicode string and then assumed to be safe:

    >>> Markup("Hello <em>World</em>!")
    Markup(u'Hello <em>World</em>!')
    >>> class Foo(object):
    ...  def __html__(self):
    ...   return '<a href="#">foo</a>'
    ...
    >>> Markup(Foo())
    Markup(u'<a href="#">foo</a>')

    If you want object passed being always treated as unsafe you can use the
    :meth:`escape` classmethod to create a :class:`Markup` object:

    >>> Markup.escape("Hello <em>World</em>!")
    Markup(u'Hello &lt;em&gt;World&lt;/em&gt;!')

    Operations on a markup string are markup aware which means that all
    arguments are passed through the :func:`escape` function:

    >>> em = Markup("<em>%s</em>")
    >>> em % "foo & bar"
    Markup(u'<em>foo &amp; bar</em>')
    >>> strong = Markup("<strong>%(text)s</strong>")
    >>> strong % {'text': '<blink>hacker here</blink>'}
    Markup(u'<strong>&lt;blink&gt;hacker here&lt;/blink&gt;</strong>')
    >>> Markup("<em>Hello</em> ") + "<foo>"
    Markup(u'<em>Hello</em> &lt;foo&gt;')
    """
    __slots__ = ()

    def __new__(cls, base=u'', encoding=None, errors='strict'):
        if hasattr(base, '__html__'):
            base = base.__html__()
        if encoding is None:
            return text_type.__new__(cls, base)
        return text_type.__new__(cls, base, encoding, errors)

    def __html__(self):
        return self

    def __add__(self, other):
        if isinstance(other, string_types) or hasattr(other, '__html__'):
            return self.__class__(super(Markup, self).__add__(self.escape(other)))
        return NotImplemented

    def __radd__(self, other):
        if hasattr(other, '__html__') or isinstance(other, string_types):
            return self.escape(other).__add__(self)
        return NotImplemented

    def __mul__(self, num):
        if isinstance(num, int_types):
            return self.__class__(text_type.__mul__(self, num))
        return NotImplemented
    __rmul__ = __mul__

    def __mod__(self, arg):
        if isinstance(arg, tuple):
            arg = tuple(_MarkupEscapeHelper(x, self.escape) for x in arg)
        else:
            arg = _MarkupEscapeHelper(arg, self.escape)
        return self.__class__(text_type.__mod__(self, arg))

    def __repr__(self):
        return '%s(%s)' % (
            self.__class__.__name__,
            text_type.__repr__(self)
        )

    def join(self, seq):
        return self.__class__(text_type.join(self, map(self.escape, seq)))
    join.__doc__ = text_type.join.__doc__

    def split(self, *args, **kwargs):
        return list(map(self.__class__, text_type.split(self, *args, **kwargs)))
    split.__doc__ = text_type.split.__doc__

    def rsplit(self, *args, **kwargs):
        return list(map(self.__class__, text_type.rsplit(self, *args, **kwargs)))
    rsplit.__doc__ = text_type.rsplit.__doc__

    def splitlines(self, *args, **kwargs):
        return list(map(self.__class__, text_type.splitlines(
            self, *args, **kwargs)))
    splitlines.__doc__ = text_type.splitlines.__doc__

    def unescape(self):
        r"""Unescape markup again into an text_type string.  This also resolves
        known HTML4 and XHTML entities:

        >>> Markup("Main &raquo; <em>About</em>").unescape()
        u'Main \xbb <em>About</em>'
        """
        from markupsafe._constants import HTML_ENTITIES
        def handle_match(m):
            name = m.group(1)
            if name in HTML_ENTITIES:
                return unichr(HTML_ENTITIES[name])
            try:
                if name[:2] in ('#x', '#X'):
                    return unichr(int(name[2:], 16))
                elif name.startswith('#'):
                    return unichr(int(name[1:]))
            except ValueError:
                pass
            # Don't modify unexpected input.
            return m.group()
        return _entity_re.sub(handle_match, text_type(self))

    def striptags(self):
        r"""Unescape markup into an text_type string and strip all tags.  This
        also resolves known HTML4 and XHTML entities.  Whitespace is
        normalized to one:

        >>> Markup("Main &raquo;  <em>About</em>").striptags()
        u'Main \xbb About'
        """
        stripped = u' '.join(_striptags_re.sub('', self).split())
        return Markup(stripped).unescape()

    @classmethod
    def escape(cls, s):
        """Escape the string.  Works like :func:`escape` with the difference
        that for subclasses of :class:`Markup` this function would return the
        correct subclass.
        """
        rv = escape(s)
        if rv.__class__ is not cls:
            return cls(rv)
        return rv

    def make_simple_escaping_wrapper(name):
        orig = getattr(text_type, name)
        def func(self, *args, **kwargs):
            args = _escape_argspec(list(args), enumerate(args), self.escape)
            _escape_argspec(kwargs, iteritems(kwargs), self.escape)
            return self.__class__(orig(self, *args, **kwargs))
        func.__name__ = orig.__name__
        func.__doc__ = orig.__doc__
        return func

    for method in '__getitem__', 'capitalize', \
                  'title', 'lower', 'upper', 'replace', 'ljust', \
                  'rjust', 'lstrip', 'rstrip', 'center', 'strip', \
                  'translate', 'expandtabs', 'swapcase', 'zfill':
        locals()[method] = make_simple_escaping_wrapper(method)

    # new in python 2.5
    if hasattr(text_type, 'partition'):
        def partition(self, sep):
            return tuple(map(self.__class__,
                             text_type.partition(self, self.escape(sep))))
        def rpartition(self, sep):
            return tuple(map(self.__class__,
                             text_type.rpartition(self, self.escape(sep))))

    # new in python 2.6
    if hasattr(text_type, 'format'):
        def format(*args, **kwargs):
            self, args = args[0], args[1:]
            formatter = EscapeFormatter(self.escape)
            kwargs = _MagicFormatMapping(args, kwargs)
            return self.__class__(formatter.vformat(self, args, kwargs))

        def __html_format__(self, format_spec):
            if format_spec:
                raise ValueError('Unsupported format specification '
                                 'for Markup.')
            return self

    # not in python 3
    if hasattr(text_type, '__getslice__'):
        __getslice__ = make_simple_escaping_wrapper('__getslice__')

    del method, make_simple_escaping_wrapper


class _MagicFormatMapping(Mapping):
    """This class implements a dummy wrapper to fix a bug in the Python
    standard library for string formatting.

    See http://bugs.python.org/issue13598 for information about why
    this is necessary.
    """

    def __init__(self, args, kwargs):
        self._args = args
        self._kwargs = kwargs
        self._last_index = 0

    def __getitem__(self, key):
        if key == '':
            idx = self._last_index
            self._last_index += 1
            try:
                return self._args[idx]
            except LookupError:
                pass
            key = str(idx)
        return self._kwargs[key]

    def __iter__(self):
        return iter(self._kwargs)

    def __len__(self):
        return len(self._kwargs)


if hasattr(text_type, 'format'):
    class EscapeFormatter(string.Formatter):

        def __init__(self, escape):
            self.escape = escape

        def format_field(self, value, format_spec):
            if hasattr(value, '__html_format__'):
                rv = value.__html_format__(format_spec)
            elif hasattr(value, '__html__'):
                if format_spec:
                    raise ValueError('No format specification allowed '
                                     'when formatting an object with '
                                     'its __html__ method.')
                rv = value.__html__()
            else:
                # We need to make sure the format spec is unicode here as
                # otherwise the wrong callback methods are invoked.  For
                # instance a byte string there would invoke __str__ and
                # not __unicode__.
                rv = string.Formatter.format_field(
                    self, value, text_type(format_spec))
            return text_type(self.escape(rv))


def _escape_argspec(obj, iterable, escape):
    """Helper for various string-wrapped functions."""
    for key, value in iterable:
        if hasattr(value, '__html__') or isinstance(value, string_types):
            obj[key] = escape(value)
    return obj


class _MarkupEscapeHelper(object):
    """Helper for Markup.__mod__"""

    def __init__(self, obj, escape):
        self.obj = obj
        self.escape = escape

    __getitem__ = lambda s, x: _MarkupEscapeHelper(s.obj[x], s.escape)
    __unicode__ = __str__ = lambda s: text_type(s.escape(s.obj))
    __repr__ = lambda s: str(s.escape(repr(s.obj)))
    __int__ = lambda s: int(s.obj)
    __float__ = lambda s: float(s.obj)


# we have to import it down here as the speedups and native
# modules imports the markup type which is define above.
try:
    from markupsafe._speedups import escape, escape_silent, soft_unicode
except ImportError:
    from markupsafe._native import escape, escape_silent, soft_unicode

if not PY2:
    soft_str = soft_unicode
    __all__.append('soft_str')
