# -*- coding: utf-8 -*-
"""
    werkzeug.utils
    ~~~~~~~~~~~~~~

    This module implements various utilities for WSGI applications.  Most of
    them are used by the request and response wrappers but especially for
    middleware development it makes sense to use them without the wrappers.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import re
import os
import sys
import pkgutil
try:
    from html.entities import name2codepoint
except ImportError:
    from htmlentitydefs import name2codepoint

from werkzeug._compat import unichr, text_type, string_types, iteritems, \
    reraise, PY2
from werkzeug._internal import _DictAccessorProperty, \
    _parse_signature, _missing


_format_re = re.compile(r'\$(?:(%s)|\{(%s)\})' % (('[a-zA-Z_][a-zA-Z0-9_]*',) * 2))
_entity_re = re.compile(r'&([^;]+);')
_filename_ascii_strip_re = re.compile(r'[^A-Za-z0-9_.-]')
_windows_device_files = ('CON', 'AUX', 'COM1', 'COM2', 'COM3', 'COM4', 'LPT1',
                         'LPT2', 'LPT3', 'PRN', 'NUL')


class cached_property(property):

    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::

        class Foo(object):

            @cached_property
            def foo(self):
                # calculate something important here
                return 42

    The class has to have a `__dict__` in order for this property to
    work.
    """

    # implementation detail: A subclass of python's builtin property
    # decorator, we override __get__ to check for a cached value. If one
    # choses to invoke __get__ by hand the property will still work as
    # expected because the lookup logic is replicated in __get__ for
    # manual invocation.

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


class environ_property(_DictAccessorProperty):

    """Maps request attributes to environment variables. This works not only
    for the Werzeug request object, but also any other class with an
    environ attribute:

    >>> class Test(object):
    ...     environ = {'key': 'value'}
    ...     test = environ_property('key')
    >>> var = Test()
    >>> var.test
    'value'

    If you pass it a second value it's used as default if the key does not
    exist, the third one can be a converter that takes a value and converts
    it.  If it raises :exc:`ValueError` or :exc:`TypeError` the default value
    is used. If no default value is provided `None` is used.

    Per default the property is read only.  You have to explicitly enable it
    by passing ``read_only=False`` to the constructor.
    """

    read_only = True

    def lookup(self, obj):
        return obj.environ


class header_property(_DictAccessorProperty):

    """Like `environ_property` but for headers."""

    def lookup(self, obj):
        return obj.headers


class HTMLBuilder(object):

    """Helper object for HTML generation.

    Per default there are two instances of that class.  The `html` one, and
    the `xhtml` one for those two dialects.  The class uses keyword parameters
    and positional parameters to generate small snippets of HTML.

    Keyword parameters are converted to XML/SGML attributes, positional
    arguments are used as children.  Because Python accepts positional
    arguments before keyword arguments it's a good idea to use a list with the
    star-syntax for some children:

    >>> html.p(class_='foo', *[html.a('foo', href='foo.html'), ' ',
    ...                        html.a('bar', href='bar.html')])
    u'<p class="foo"><a href="foo.html">foo</a> <a href="bar.html">bar</a></p>'

    This class works around some browser limitations and can not be used for
    arbitrary SGML/XML generation.  For that purpose lxml and similar
    libraries exist.

    Calling the builder escapes the string passed:

    >>> html.p(html("<foo>"))
    u'<p>&lt;foo&gt;</p>'
    """

    _entity_re = re.compile(r'&([^;]+);')
    _entities = name2codepoint.copy()
    _entities['apos'] = 39
    _empty_elements = set([
        'area', 'base', 'basefont', 'br', 'col', 'command', 'embed', 'frame',
        'hr', 'img', 'input', 'keygen', 'isindex', 'link', 'meta', 'param',
        'source', 'wbr'
    ])
    _boolean_attributes = set([
        'selected', 'checked', 'compact', 'declare', 'defer', 'disabled',
        'ismap', 'multiple', 'nohref', 'noresize', 'noshade', 'nowrap'
    ])
    _plaintext_elements = set(['textarea'])
    _c_like_cdata = set(['script', 'style'])

    def __init__(self, dialect):
        self._dialect = dialect

    def __call__(self, s):
        return escape(s)

    def __getattr__(self, tag):
        if tag[:2] == '__':
            raise AttributeError(tag)

        def proxy(*children, **arguments):
            buffer = '<' + tag
            for key, value in iteritems(arguments):
                if value is None:
                    continue
                if key[-1] == '_':
                    key = key[:-1]
                if key in self._boolean_attributes:
                    if not value:
                        continue
                    if self._dialect == 'xhtml':
                        value = '="' + key + '"'
                    else:
                        value = ''
                else:
                    value = '="' + escape(value) + '"'
                buffer += ' ' + key + value
            if not children and tag in self._empty_elements:
                if self._dialect == 'xhtml':
                    buffer += ' />'
                else:
                    buffer += '>'
                return buffer
            buffer += '>'

            children_as_string = ''.join([text_type(x) for x in children
                                          if x is not None])

            if children_as_string:
                if tag in self._plaintext_elements:
                    children_as_string = escape(children_as_string)
                elif tag in self._c_like_cdata and self._dialect == 'xhtml':
                    children_as_string = '/*<![CDATA[*/' + \
                                         children_as_string + '/*]]>*/'
            buffer += children_as_string + '</' + tag + '>'
            return buffer
        return proxy

    def __repr__(self):
        return '<%s for %r>' % (
            self.__class__.__name__,
            self._dialect
        )


html = HTMLBuilder('html')
xhtml = HTMLBuilder('xhtml')


def get_content_type(mimetype, charset):
    """Returns the full content type string with charset for a mimetype.

    If the mimetype represents text the charset will be appended as charset
    parameter, otherwise the mimetype is returned unchanged.

    :param mimetype: the mimetype to be used as content type.
    :param charset: the charset to be appended in case it was a text mimetype.
    :return: the content type.
    """
    if mimetype.startswith('text/') or \
       mimetype == 'application/xml' or \
       (mimetype.startswith('application/') and
            mimetype.endswith('+xml')):
        mimetype += '; charset=' + charset
    return mimetype


def format_string(string, context):
    """String-template format a string:

    >>> format_string('$foo and ${foo}s', dict(foo=42))
    '42 and 42s'

    This does not do any attribute lookup etc.  For more advanced string
    formattings have a look at the `werkzeug.template` module.

    :param string: the format string.
    :param context: a dict with the variables to insert.
    """
    def lookup_arg(match):
        x = context[match.group(1) or match.group(2)]
        if not isinstance(x, string_types):
            x = type(string)(x)
        return x
    return _format_re.sub(lookup_arg, string)


def secure_filename(filename):
    r"""Pass it a filename and it will return a secure version of it.  This
    filename can then safely be stored on a regular file system and passed
    to :func:`os.path.join`.  The filename returned is an ASCII only string
    for maximum portability.

    On windows systems the function also makes sure that the file is not
    named after one of the special device files.

    >>> secure_filename("My cool movie.mov")
    'My_cool_movie.mov'
    >>> secure_filename("../../../etc/passwd")
    'etc_passwd'
    >>> secure_filename(u'i contain cool \xfcml\xe4uts.txt')
    'i_contain_cool_umlauts.txt'

    The function might return an empty filename.  It's your responsibility
    to ensure that the filename is unique and that you generate random
    filename if the function returned an empty one.

    .. versionadded:: 0.5

    :param filename: the filename to secure
    """
    if isinstance(filename, text_type):
        from unicodedata import normalize
        filename = normalize('NFKD', filename).encode('ascii', 'ignore')
        if not PY2:
            filename = filename.decode('ascii')
    for sep in os.path.sep, os.path.altsep:
        if sep:
            filename = filename.replace(sep, ' ')
    filename = str(_filename_ascii_strip_re.sub('', '_'.join(
                   filename.split()))).strip('._')

    # on nt a couple of special files are present in each folder.  We
    # have to ensure that the target file is not such a filename.  In
    # this case we prepend an underline
    if os.name == 'nt' and filename and \
       filename.split('.')[0].upper() in _windows_device_files:
        filename = '_' + filename

    return filename


def escape(s, quote=None):
    """Replace special characters "&", "<", ">" and (") to HTML-safe sequences.

    There is a special handling for `None` which escapes to an empty string.

    .. versionchanged:: 0.9
       `quote` is now implicitly on.

    :param s: the string to escape.
    :param quote: ignored.
    """
    if s is None:
        return ''
    elif hasattr(s, '__html__'):
        return text_type(s.__html__())
    elif not isinstance(s, string_types):
        s = text_type(s)
    if quote is not None:
        from warnings import warn
        warn(DeprecationWarning('quote parameter is implicit now'), stacklevel=2)
    s = s.replace('&', '&amp;').replace('<', '&lt;') \
        .replace('>', '&gt;').replace('"', "&quot;")
    return s


def unescape(s):
    """The reverse function of `escape`.  This unescapes all the HTML
    entities, not only the XML entities inserted by `escape`.

    :param s: the string to unescape.
    """
    def handle_match(m):
        name = m.group(1)
        if name in HTMLBuilder._entities:
            return unichr(HTMLBuilder._entities[name])
        try:
            if name[:2] in ('#x', '#X'):
                return unichr(int(name[2:], 16))
            elif name.startswith('#'):
                return unichr(int(name[1:]))
        except ValueError:
            pass
        return u''
    return _entity_re.sub(handle_match, s)


def redirect(location, code=302, Response=None):
    """Returns a response object (a WSGI application) that, if called,
    redirects the client to the target location.  Supported codes are 301,
    302, 303, 305, and 307.  300 is not supported because it's not a real
    redirect and 304 because it's the answer for a request with a request
    with defined If-Modified-Since headers.

    .. versionadded:: 0.6
       The location can now be a unicode string that is encoded using
       the :func:`iri_to_uri` function.

    .. versionadded:: 0.10
        The class used for the Response object can now be passed in.

    :param location: the location the response should redirect to.
    :param code: the redirect status code. defaults to 302.
    :param class Response: a Response class to use when instantiating a
        response. The default is :class:`werkzeug.wrappers.Response` if
        unspecified.
    """
    if Response is None:
        from werkzeug.wrappers import Response

    display_location = escape(location)
    if isinstance(location, text_type):
        # Safe conversion is necessary here as we might redirect
        # to a broken URI scheme (for instance itms-services).
        from werkzeug.urls import iri_to_uri
        location = iri_to_uri(location, safe_conversion=True)
    response = Response(
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">\n'
        '<title>Redirecting...</title>\n'
        '<h1>Redirecting...</h1>\n'
        '<p>You should be redirected automatically to target URL: '
        '<a href="%s">%s</a>.  If not click the link.' %
        (escape(location), display_location), code, mimetype='text/html')
    response.headers['Location'] = location
    return response


def append_slash_redirect(environ, code=301):
    """Redirects to the same URL but with a slash appended.  The behavior
    of this function is undefined if the path ends with a slash already.

    :param environ: the WSGI environment for the request that triggers
                    the redirect.
    :param code: the status code for the redirect.
    """
    new_path = environ['PATH_INFO'].strip('/') + '/'
    query_string = environ.get('QUERY_STRING')
    if query_string:
        new_path += '?' + query_string
    return redirect(new_path, code)


def import_string(import_name, silent=False):
    """Imports an object based on a string.  This is useful if you want to
    use import paths as endpoints or something similar.  An import path can
    be specified either in dotted notation (``xml.sax.saxutils.escape``)
    or with a colon as object delimiter (``xml.sax.saxutils:escape``).

    If `silent` is True the return value will be `None` if the import fails.

    :param import_name: the dotted name for the object to import.
    :param silent: if set to `True` import errors are ignored and
                   `None` is returned instead.
    :return: imported object
    """
    # force the import name to automatically convert to strings
    # __import__ is not able to handle unicode strings in the fromlist
    # if the module is a package
    import_name = str(import_name).replace(':', '.')
    try:
        try:
            __import__(import_name)
        except ImportError:
            if '.' not in import_name:
                raise
        else:
            return sys.modules[import_name]

        module_name, obj_name = import_name.rsplit('.', 1)
        try:
            module = __import__(module_name, None, None, [obj_name])
        except ImportError:
            # support importing modules not yet set up by the parent module
            # (or package for that matter)
            module = import_string(module_name)

        try:
            return getattr(module, obj_name)
        except AttributeError as e:
            raise ImportError(e)

    except ImportError as e:
        if not silent:
            reraise(
                ImportStringError,
                ImportStringError(import_name, e),
                sys.exc_info()[2])


def find_modules(import_path, include_packages=False, recursive=False):
    """Finds all the modules below a package.  This can be useful to
    automatically import all views / controllers so that their metaclasses /
    function decorators have a chance to register themselves on the
    application.

    Packages are not returned unless `include_packages` is `True`.  This can
    also recursively list modules but in that case it will import all the
    packages to get the correct load path of that module.

    :param import_name: the dotted name for the package to find child modules.
    :param include_packages: set to `True` if packages should be returned, too.
    :param recursive: set to `True` if recursion should happen.
    :return: generator
    """
    module = import_string(import_path)
    path = getattr(module, '__path__', None)
    if path is None:
        raise ValueError('%r is not a package' % import_path)
    basename = module.__name__ + '.'
    for importer, modname, ispkg in pkgutil.iter_modules(path):
        modname = basename + modname
        if ispkg:
            if include_packages:
                yield modname
            if recursive:
                for item in find_modules(modname, include_packages, True):
                    yield item
        else:
            yield modname


def validate_arguments(func, args, kwargs, drop_extra=True):
    """Checks if the function accepts the arguments and keyword arguments.
    Returns a new ``(args, kwargs)`` tuple that can safely be passed to
    the function without causing a `TypeError` because the function signature
    is incompatible.  If `drop_extra` is set to `True` (which is the default)
    any extra positional or keyword arguments are dropped automatically.

    The exception raised provides three attributes:

    `missing`
        A set of argument names that the function expected but where
        missing.

    `extra`
        A dict of keyword arguments that the function can not handle but
        where provided.

    `extra_positional`
        A list of values that where given by positional argument but the
        function cannot accept.

    This can be useful for decorators that forward user submitted data to
    a view function::

        from werkzeug.utils import ArgumentValidationError, validate_arguments

        def sanitize(f):
            def proxy(request):
                data = request.values.to_dict()
                try:
                    args, kwargs = validate_arguments(f, (request,), data)
                except ArgumentValidationError:
                    raise BadRequest('The browser failed to transmit all '
                                     'the data expected.')
                return f(*args, **kwargs)
            return proxy

    :param func: the function the validation is performed against.
    :param args: a tuple of positional arguments.
    :param kwargs: a dict of keyword arguments.
    :param drop_extra: set to `False` if you don't want extra arguments
                       to be silently dropped.
    :return: tuple in the form ``(args, kwargs)``.
    """
    parser = _parse_signature(func)
    args, kwargs, missing, extra, extra_positional = parser(args, kwargs)[:5]
    if missing:
        raise ArgumentValidationError(tuple(missing))
    elif (extra or extra_positional) and not drop_extra:
        raise ArgumentValidationError(None, extra, extra_positional)
    return tuple(args), kwargs


def bind_arguments(func, args, kwargs):
    """Bind the arguments provided into a dict.  When passed a function,
    a tuple of arguments and a dict of keyword arguments `bind_arguments`
    returns a dict of names as the function would see it.  This can be useful
    to implement a cache decorator that uses the function arguments to build
    the cache key based on the values of the arguments.

    :param func: the function the arguments should be bound for.
    :param args: tuple of positional arguments.
    :param kwargs: a dict of keyword arguments.
    :return: a :class:`dict` of bound keyword arguments.
    """
    args, kwargs, missing, extra, extra_positional, \
        arg_spec, vararg_var, kwarg_var = _parse_signature(func)(args, kwargs)
    values = {}
    for (name, has_default, default), value in zip(arg_spec, args):
        values[name] = value
    if vararg_var is not None:
        values[vararg_var] = tuple(extra_positional)
    elif extra_positional:
        raise TypeError('too many positional arguments')
    if kwarg_var is not None:
        multikw = set(extra) & set([x[0] for x in arg_spec])
        if multikw:
            raise TypeError('got multiple values for keyword argument ' +
                            repr(next(iter(multikw))))
        values[kwarg_var] = extra
    elif extra:
        raise TypeError('got unexpected keyword argument ' +
                        repr(next(iter(extra))))
    return values


class ArgumentValidationError(ValueError):

    """Raised if :func:`validate_arguments` fails to validate"""

    def __init__(self, missing=None, extra=None, extra_positional=None):
        self.missing = set(missing or ())
        self.extra = extra or {}
        self.extra_positional = extra_positional or []
        ValueError.__init__(self, 'function arguments invalid.  ('
                            '%d missing, %d additional)' % (
                                len(self.missing),
                                len(self.extra) + len(self.extra_positional)
                            ))


class ImportStringError(ImportError):

    """Provides information about a failed :func:`import_string` attempt."""

    #: String in dotted notation that failed to be imported.
    import_name = None
    #: Wrapped exception.
    exception = None

    def __init__(self, import_name, exception):
        self.import_name = import_name
        self.exception = exception

        msg = (
            'import_string() failed for %r. Possible reasons are:\n\n'
            '- missing __init__.py in a package;\n'
            '- package or module path not included in sys.path;\n'
            '- duplicated package or module name taking precedence in '
            'sys.path;\n'
            '- missing module, class, function or variable;\n\n'
            'Debugged import:\n\n%s\n\n'
            'Original exception:\n\n%s: %s')

        name = ''
        tracked = []
        for part in import_name.replace(':', '.').split('.'):
            name += (name and '.') + part
            imported = import_string(name, silent=True)
            if imported:
                tracked.append((name, getattr(imported, '__file__', None)))
            else:
                track = ['- %r found in %r.' % (n, i) for n, i in tracked]
                track.append('- %r not found.' % name)
                msg = msg % (import_name, '\n'.join(track),
                             exception.__class__.__name__, str(exception))
                break

        ImportError.__init__(self, msg)

    def __repr__(self):
        return '<%s(%r, %r)>' % (self.__class__.__name__, self.import_name,
                                 self.exception)


# DEPRECATED
# these objects were previously in this module as well.  we import
# them here for backwards compatibility with old pickles.
from werkzeug.datastructures import (  # noqa
    MultiDict, CombinedMultiDict, Headers, EnvironHeaders)
from werkzeug.http import parse_cookie, dump_cookie  # noqa
