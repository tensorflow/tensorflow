# -*- coding: utf-8 -*-
"""
    werkzeug.wsgi
    ~~~~~~~~~~~~~

    This module implements WSGI related helpers.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import re
import os
import posixpath
import mimetypes
from itertools import chain
from zlib import adler32
from time import time, mktime
from datetime import datetime
from functools import partial, update_wrapper

from werkzeug._compat import iteritems, text_type, string_types, \
    implements_iterator, make_literal_wrapper, to_unicode, to_bytes, \
    wsgi_get_bytes, try_coerce_native, PY2
from werkzeug._internal import _empty_stream, _encode_idna
from werkzeug.http import is_resource_modified, http_date
from werkzeug.urls import uri_to_iri, url_quote, url_parse, url_join
from werkzeug.filesystem import get_filesystem_encoding


def responder(f):
    """Marks a function as responder.  Decorate a function with it and it
    will automatically call the return value as WSGI application.

    Example::

        @responder
        def application(environ, start_response):
            return Response('Hello World!')
    """
    return update_wrapper(lambda *a: f(*a)(*a[-2:]), f)


def get_current_url(environ, root_only=False, strip_querystring=False,
                    host_only=False, trusted_hosts=None):
    """A handy helper function that recreates the full URL as IRI for the
    current request or parts of it.  Here an example:

    >>> from werkzeug.test import create_environ
    >>> env = create_environ("/?param=foo", "http://localhost/script")
    >>> get_current_url(env)
    'http://localhost/script/?param=foo'
    >>> get_current_url(env, root_only=True)
    'http://localhost/script/'
    >>> get_current_url(env, host_only=True)
    'http://localhost/'
    >>> get_current_url(env, strip_querystring=True)
    'http://localhost/script/'

    This optionally it verifies that the host is in a list of trusted hosts.
    If the host is not in there it will raise a
    :exc:`~werkzeug.exceptions.SecurityError`.

    Note that the string returned might contain unicode characters as the
    representation is an IRI not an URI.  If you need an ASCII only
    representation you can use the :func:`~werkzeug.urls.iri_to_uri`
    function:

    >>> from werkzeug.urls import iri_to_uri
    >>> iri_to_uri(get_current_url(env))
    'http://localhost/script/?param=foo'

    :param environ: the WSGI environment to get the current URL from.
    :param root_only: set `True` if you only want the root URL.
    :param strip_querystring: set to `True` if you don't want the querystring.
    :param host_only: set to `True` if the host URL should be returned.
    :param trusted_hosts: a list of trusted hosts, see :func:`host_is_trusted`
                          for more information.
    """
    tmp = [environ['wsgi.url_scheme'], '://', get_host(environ, trusted_hosts)]
    cat = tmp.append
    if host_only:
        return uri_to_iri(''.join(tmp) + '/')
    cat(url_quote(wsgi_get_bytes(environ.get('SCRIPT_NAME', ''))).rstrip('/'))
    cat('/')
    if not root_only:
        cat(url_quote(wsgi_get_bytes(environ.get('PATH_INFO', '')).lstrip(b'/')))
        if not strip_querystring:
            qs = get_query_string(environ)
            if qs:
                cat('?' + qs)
    return uri_to_iri(''.join(tmp))


def host_is_trusted(hostname, trusted_list):
    """Checks if a host is trusted against a list.  This also takes care
    of port normalization.

    .. versionadded:: 0.9

    :param hostname: the hostname to check
    :param trusted_list: a list of hostnames to check against.  If a
                         hostname starts with a dot it will match against
                         all subdomains as well.
    """
    if not hostname:
        return False

    if isinstance(trusted_list, string_types):
        trusted_list = [trusted_list]

    def _normalize(hostname):
        if ':' in hostname:
            hostname = hostname.rsplit(':', 1)[0]
        return _encode_idna(hostname)

    try:
        hostname = _normalize(hostname)
    except UnicodeError:
        return False
    for ref in trusted_list:
        if ref.startswith('.'):
            ref = ref[1:]
            suffix_match = True
        else:
            suffix_match = False
        try:
            ref = _normalize(ref)
        except UnicodeError:
            return False
        if ref == hostname:
            return True
        if suffix_match and hostname.endswith('.' + ref):
            return True
    return False


def get_host(environ, trusted_hosts=None):
    """Return the real host for the given WSGI environment.  This first checks
    the `X-Forwarded-Host` header, then the normal `Host` header, and finally
    the `SERVER_NAME` environment variable (using the first one it finds).

    Optionally it verifies that the host is in a list of trusted hosts.
    If the host is not in there it will raise a
    :exc:`~werkzeug.exceptions.SecurityError`.

    :param environ: the WSGI environment to get the host of.
    :param trusted_hosts: a list of trusted hosts, see :func:`host_is_trusted`
                          for more information.
    """
    if 'HTTP_X_FORWARDED_HOST' in environ:
        rv = environ['HTTP_X_FORWARDED_HOST'].split(',', 1)[0].strip()
    elif 'HTTP_HOST' in environ:
        rv = environ['HTTP_HOST']
    else:
        rv = environ['SERVER_NAME']
        if (environ['wsgi.url_scheme'], environ['SERVER_PORT']) not \
           in (('https', '443'), ('http', '80')):
            rv += ':' + environ['SERVER_PORT']
    if trusted_hosts is not None:
        if not host_is_trusted(rv, trusted_hosts):
            from werkzeug.exceptions import SecurityError
            raise SecurityError('Host "%s" is not trusted' % rv)
    return rv


def get_content_length(environ):
    """Returns the content length from the WSGI environment as
    integer.  If it's not available `None` is returned.

    .. versionadded:: 0.9

    :param environ: the WSGI environ to fetch the content length from.
    """
    content_length = environ.get('CONTENT_LENGTH')
    if content_length is not None:
        try:
            return max(0, int(content_length))
        except (ValueError, TypeError):
            pass


def get_input_stream(environ, safe_fallback=True):
    """Returns the input stream from the WSGI environment and wraps it
    in the most sensible way possible.  The stream returned is not the
    raw WSGI stream in most cases but one that is safe to read from
    without taking into account the content length.

    .. versionadded:: 0.9

    :param environ: the WSGI environ to fetch the stream from.
    :param safe: indicates whether the function should use an empty
                 stream as safe fallback or just return the original
                 WSGI input stream if it can't wrap it safely.  The
                 default is to return an empty string in those cases.
    """
    stream = environ['wsgi.input']
    content_length = get_content_length(environ)

    # A wsgi extension that tells us if the input is terminated.  In
    # that case we return the stream unchanged as we know we can safely
    # read it until the end.
    if environ.get('wsgi.input_terminated'):
        return stream

    # If we don't have a content length we fall back to an empty stream
    # in case of a safe fallback, otherwise we return the stream unchanged.
    # The non-safe fallback is not recommended but might be useful in
    # some situations.
    if content_length is None:
        return safe_fallback and _empty_stream or stream

    # Otherwise limit the stream to the content length
    return LimitedStream(stream, content_length)


def get_query_string(environ):
    """Returns the `QUERY_STRING` from the WSGI environment.  This also takes
    care about the WSGI decoding dance on Python 3 environments as a
    native string.  The string returned will be restricted to ASCII
    characters.

    .. versionadded:: 0.9

    :param environ: the WSGI environment object to get the query string from.
    """
    qs = wsgi_get_bytes(environ.get('QUERY_STRING', ''))
    # QUERY_STRING really should be ascii safe but some browsers
    # will send us some unicode stuff (I am looking at you IE).
    # In that case we want to urllib quote it badly.
    return try_coerce_native(url_quote(qs, safe=':&%=+$!*\'(),'))


def get_path_info(environ, charset='utf-8', errors='replace'):
    """Returns the `PATH_INFO` from the WSGI environment and properly
    decodes it.  This also takes care about the WSGI decoding dance
    on Python 3 environments.  if the `charset` is set to `None` a
    bytestring is returned.

    .. versionadded:: 0.9

    :param environ: the WSGI environment object to get the path from.
    :param charset: the charset for the path info, or `None` if no
                    decoding should be performed.
    :param errors: the decoding error handling.
    """
    path = wsgi_get_bytes(environ.get('PATH_INFO', ''))
    return to_unicode(path, charset, errors, allow_none_charset=True)


def get_script_name(environ, charset='utf-8', errors='replace'):
    """Returns the `SCRIPT_NAME` from the WSGI environment and properly
    decodes it.  This also takes care about the WSGI decoding dance
    on Python 3 environments.  if the `charset` is set to `None` a
    bytestring is returned.

    .. versionadded:: 0.9

    :param environ: the WSGI environment object to get the path from.
    :param charset: the charset for the path, or `None` if no
                    decoding should be performed.
    :param errors: the decoding error handling.
    """
    path = wsgi_get_bytes(environ.get('SCRIPT_NAME', ''))
    return to_unicode(path, charset, errors, allow_none_charset=True)


def pop_path_info(environ, charset='utf-8', errors='replace'):
    """Removes and returns the next segment of `PATH_INFO`, pushing it onto
    `SCRIPT_NAME`.  Returns `None` if there is nothing left on `PATH_INFO`.

    If the `charset` is set to `None` a bytestring is returned.

    If there are empty segments (``'/foo//bar``) these are ignored but
    properly pushed to the `SCRIPT_NAME`:

    >>> env = {'SCRIPT_NAME': '/foo', 'PATH_INFO': '/a/b'}
    >>> pop_path_info(env)
    'a'
    >>> env['SCRIPT_NAME']
    '/foo/a'
    >>> pop_path_info(env)
    'b'
    >>> env['SCRIPT_NAME']
    '/foo/a/b'

    .. versionadded:: 0.5

    .. versionchanged:: 0.9
       The path is now decoded and a charset and encoding
       parameter can be provided.

    :param environ: the WSGI environment that is modified.
    """
    path = environ.get('PATH_INFO')
    if not path:
        return None

    script_name = environ.get('SCRIPT_NAME', '')

    # shift multiple leading slashes over
    old_path = path
    path = path.lstrip('/')
    if path != old_path:
        script_name += '/' * (len(old_path) - len(path))

    if '/' not in path:
        environ['PATH_INFO'] = ''
        environ['SCRIPT_NAME'] = script_name + path
        rv = wsgi_get_bytes(path)
    else:
        segment, path = path.split('/', 1)
        environ['PATH_INFO'] = '/' + path
        environ['SCRIPT_NAME'] = script_name + segment
        rv = wsgi_get_bytes(segment)

    return to_unicode(rv, charset, errors, allow_none_charset=True)


def peek_path_info(environ, charset='utf-8', errors='replace'):
    """Returns the next segment on the `PATH_INFO` or `None` if there
    is none.  Works like :func:`pop_path_info` without modifying the
    environment:

    >>> env = {'SCRIPT_NAME': '/foo', 'PATH_INFO': '/a/b'}
    >>> peek_path_info(env)
    'a'
    >>> peek_path_info(env)
    'a'

    If the `charset` is set to `None` a bytestring is returned.

    .. versionadded:: 0.5

    .. versionchanged:: 0.9
       The path is now decoded and a charset and encoding
       parameter can be provided.

    :param environ: the WSGI environment that is checked.
    """
    segments = environ.get('PATH_INFO', '').lstrip('/').split('/', 1)
    if segments:
        return to_unicode(wsgi_get_bytes(segments[0]),
                          charset, errors, allow_none_charset=True)


def extract_path_info(environ_or_baseurl, path_or_url, charset='utf-8',
                      errors='replace', collapse_http_schemes=True):
    """Extracts the path info from the given URL (or WSGI environment) and
    path.  The path info returned is a unicode string, not a bytestring
    suitable for a WSGI environment.  The URLs might also be IRIs.

    If the path info could not be determined, `None` is returned.

    Some examples:

    >>> extract_path_info('http://example.com/app', '/app/hello')
    u'/hello'
    >>> extract_path_info('http://example.com/app',
    ...                   'https://example.com/app/hello')
    u'/hello'
    >>> extract_path_info('http://example.com/app',
    ...                   'https://example.com/app/hello',
    ...                   collapse_http_schemes=False) is None
    True

    Instead of providing a base URL you can also pass a WSGI environment.

    .. versionadded:: 0.6

    :param environ_or_baseurl: a WSGI environment dict, a base URL or
                               base IRI.  This is the root of the
                               application.
    :param path_or_url: an absolute path from the server root, a
                        relative path (in which case it's the path info)
                        or a full URL.  Also accepts IRIs and unicode
                        parameters.
    :param charset: the charset for byte data in URLs
    :param errors: the error handling on decode
    :param collapse_http_schemes: if set to `False` the algorithm does
                                  not assume that http and https on the
                                  same server point to the same
                                  resource.
    """
    def _normalize_netloc(scheme, netloc):
        parts = netloc.split(u'@', 1)[-1].split(u':', 1)
        if len(parts) == 2:
            netloc, port = parts
            if (scheme == u'http' and port == u'80') or \
               (scheme == u'https' and port == u'443'):
                port = None
        else:
            netloc = parts[0]
            port = None
        if port is not None:
            netloc += u':' + port
        return netloc

    # make sure whatever we are working on is a IRI and parse it
    path = uri_to_iri(path_or_url, charset, errors)
    if isinstance(environ_or_baseurl, dict):
        environ_or_baseurl = get_current_url(environ_or_baseurl,
                                             root_only=True)
    base_iri = uri_to_iri(environ_or_baseurl, charset, errors)
    base_scheme, base_netloc, base_path = url_parse(base_iri)[:3]
    cur_scheme, cur_netloc, cur_path, = \
        url_parse(url_join(base_iri, path))[:3]

    # normalize the network location
    base_netloc = _normalize_netloc(base_scheme, base_netloc)
    cur_netloc = _normalize_netloc(cur_scheme, cur_netloc)

    # is that IRI even on a known HTTP scheme?
    if collapse_http_schemes:
        for scheme in base_scheme, cur_scheme:
            if scheme not in (u'http', u'https'):
                return None
    else:
        if not (base_scheme in (u'http', u'https') and
                base_scheme == cur_scheme):
            return None

    # are the netlocs compatible?
    if base_netloc != cur_netloc:
        return None

    # are we below the application path?
    base_path = base_path.rstrip(u'/')
    if not cur_path.startswith(base_path):
        return None

    return u'/' + cur_path[len(base_path):].lstrip(u'/')


class SharedDataMiddleware(object):

    """A WSGI middleware that provides static content for development
    environments or simple server setups. Usage is quite simple::

        import os
        from werkzeug.wsgi import SharedDataMiddleware

        app = SharedDataMiddleware(app, {
            '/shared': os.path.join(os.path.dirname(__file__), 'shared')
        })

    The contents of the folder ``./shared`` will now be available on
    ``http://example.com/shared/``.  This is pretty useful during development
    because a standalone media server is not required.  One can also mount
    files on the root folder and still continue to use the application because
    the shared data middleware forwards all unhandled requests to the
    application, even if the requests are below one of the shared folders.

    If `pkg_resources` is available you can also tell the middleware to serve
    files from package data::

        app = SharedDataMiddleware(app, {
            '/shared': ('myapplication', 'shared_files')
        })

    This will then serve the ``shared_files`` folder in the `myapplication`
    Python package.

    The optional `disallow` parameter can be a list of :func:`~fnmatch.fnmatch`
    rules for files that are not accessible from the web.  If `cache` is set to
    `False` no caching headers are sent.

    Currently the middleware does not support non ASCII filenames.  If the
    encoding on the file system happens to be the encoding of the URI it may
    work but this could also be by accident.  We strongly suggest using ASCII
    only file names for static files.

    The middleware will guess the mimetype using the Python `mimetype`
    module.  If it's unable to figure out the charset it will fall back
    to `fallback_mimetype`.

    .. versionchanged:: 0.5
       The cache timeout is configurable now.

    .. versionadded:: 0.6
       The `fallback_mimetype` parameter was added.

    :param app: the application to wrap.  If you don't want to wrap an
                application you can pass it :exc:`NotFound`.
    :param exports: a dict of exported files and folders.
    :param disallow: a list of :func:`~fnmatch.fnmatch` rules.
    :param fallback_mimetype: the fallback mimetype for unknown files.
    :param cache: enable or disable caching headers.
    :param cache_timeout: the cache timeout in seconds for the headers.
    """

    def __init__(self, app, exports, disallow=None, cache=True,
                 cache_timeout=60 * 60 * 12, fallback_mimetype='text/plain'):
        self.app = app
        self.exports = {}
        self.cache = cache
        self.cache_timeout = cache_timeout
        for key, value in iteritems(exports):
            if isinstance(value, tuple):
                loader = self.get_package_loader(*value)
            elif isinstance(value, string_types):
                if os.path.isfile(value):
                    loader = self.get_file_loader(value)
                else:
                    loader = self.get_directory_loader(value)
            else:
                raise TypeError('unknown def %r' % value)
            self.exports[key] = loader
        if disallow is not None:
            from fnmatch import fnmatch
            self.is_allowed = lambda x: not fnmatch(x, disallow)
        self.fallback_mimetype = fallback_mimetype

    def is_allowed(self, filename):
        """Subclasses can override this method to disallow the access to
        certain files.  However by providing `disallow` in the constructor
        this method is overwritten.
        """
        return True

    def _opener(self, filename):
        return lambda: (
            open(filename, 'rb'),
            datetime.utcfromtimestamp(os.path.getmtime(filename)),
            int(os.path.getsize(filename))
        )

    def get_file_loader(self, filename):
        return lambda x: (os.path.basename(filename), self._opener(filename))

    def get_package_loader(self, package, package_path):
        from pkg_resources import DefaultProvider, ResourceManager, \
            get_provider
        loadtime = datetime.utcnow()
        provider = get_provider(package)
        manager = ResourceManager()
        filesystem_bound = isinstance(provider, DefaultProvider)

        def loader(path):
            if path is None:
                return None, None
            path = posixpath.join(package_path, path)
            if not provider.has_resource(path):
                return None, None
            basename = posixpath.basename(path)
            if filesystem_bound:
                return basename, self._opener(
                    provider.get_resource_filename(manager, path))
            return basename, lambda: (
                provider.get_resource_stream(manager, path),
                loadtime,
                0
            )
        return loader

    def get_directory_loader(self, directory):
        def loader(path):
            if path is not None:
                path = os.path.join(directory, path)
            else:
                path = directory
            if os.path.isfile(path):
                return os.path.basename(path), self._opener(path)
            return None, None
        return loader

    def generate_etag(self, mtime, file_size, real_filename):
        if not isinstance(real_filename, bytes):
            real_filename = real_filename.encode(get_filesystem_encoding())
        return 'wzsdm-%d-%s-%s' % (
            mktime(mtime.timetuple()),
            file_size,
            adler32(real_filename) & 0xffffffff
        )

    def __call__(self, environ, start_response):
        cleaned_path = get_path_info(environ)
        if PY2:
            cleaned_path = cleaned_path.encode(get_filesystem_encoding())
        # sanitize the path for non unix systems
        cleaned_path = cleaned_path.strip('/')
        for sep in os.sep, os.altsep:
            if sep and sep != '/':
                cleaned_path = cleaned_path.replace(sep, '/')
        path = '/' + '/'.join(x for x in cleaned_path.split('/')
                              if x and x != '..')
        file_loader = None
        for search_path, loader in iteritems(self.exports):
            if search_path == path:
                real_filename, file_loader = loader(None)
                if file_loader is not None:
                    break
            if not search_path.endswith('/'):
                search_path += '/'
            if path.startswith(search_path):
                real_filename, file_loader = loader(path[len(search_path):])
                if file_loader is not None:
                    break
        if file_loader is None or not self.is_allowed(real_filename):
            return self.app(environ, start_response)

        guessed_type = mimetypes.guess_type(real_filename)
        mime_type = guessed_type[0] or self.fallback_mimetype
        f, mtime, file_size = file_loader()

        headers = [('Date', http_date())]
        if self.cache:
            timeout = self.cache_timeout
            etag = self.generate_etag(mtime, file_size, real_filename)
            headers += [
                ('Etag', '"%s"' % etag),
                ('Cache-Control', 'max-age=%d, public' % timeout)
            ]
            if not is_resource_modified(environ, etag, last_modified=mtime):
                f.close()
                start_response('304 Not Modified', headers)
                return []
            headers.append(('Expires', http_date(time() + timeout)))
        else:
            headers.append(('Cache-Control', 'public'))

        headers.extend((
            ('Content-Type', mime_type),
            ('Content-Length', str(file_size)),
            ('Last-Modified', http_date(mtime))
        ))
        start_response('200 OK', headers)
        return wrap_file(environ, f)


class DispatcherMiddleware(object):

    """Allows one to mount middlewares or applications in a WSGI application.
    This is useful if you want to combine multiple WSGI applications::

        app = DispatcherMiddleware(app, {
            '/app2':        app2,
            '/app3':        app3
        })
    """

    def __init__(self, app, mounts=None):
        self.app = app
        self.mounts = mounts or {}

    def __call__(self, environ, start_response):
        script = environ.get('PATH_INFO', '')
        path_info = ''
        while '/' in script:
            if script in self.mounts:
                app = self.mounts[script]
                break
            script, last_item = script.rsplit('/', 1)
            path_info = '/%s%s' % (last_item, path_info)
        else:
            app = self.mounts.get(script, self.app)
        original_script_name = environ.get('SCRIPT_NAME', '')
        environ['SCRIPT_NAME'] = original_script_name + script
        environ['PATH_INFO'] = path_info
        return app(environ, start_response)


@implements_iterator
class ClosingIterator(object):

    """The WSGI specification requires that all middlewares and gateways
    respect the `close` callback of an iterator.  Because it is useful to add
    another close action to a returned iterator and adding a custom iterator
    is a boring task this class can be used for that::

        return ClosingIterator(app(environ, start_response), [cleanup_session,
                                                              cleanup_locals])

    If there is just one close function it can be passed instead of the list.

    A closing iterator is not needed if the application uses response objects
    and finishes the processing if the response is started::

        try:
            return response(environ, start_response)
        finally:
            cleanup_session()
            cleanup_locals()
    """

    def __init__(self, iterable, callbacks=None):
        iterator = iter(iterable)
        self._next = partial(next, iterator)
        if callbacks is None:
            callbacks = []
        elif callable(callbacks):
            callbacks = [callbacks]
        else:
            callbacks = list(callbacks)
        iterable_close = getattr(iterator, 'close', None)
        if iterable_close:
            callbacks.insert(0, iterable_close)
        self._callbacks = callbacks

    def __iter__(self):
        return self

    def __next__(self):
        return self._next()

    def close(self):
        for callback in self._callbacks:
            callback()


def wrap_file(environ, file, buffer_size=8192):
    """Wraps a file.  This uses the WSGI server's file wrapper if available
    or otherwise the generic :class:`FileWrapper`.

    .. versionadded:: 0.5

    If the file wrapper from the WSGI server is used it's important to not
    iterate over it from inside the application but to pass it through
    unchanged.  If you want to pass out a file wrapper inside a response
    object you have to set :attr:`~BaseResponse.direct_passthrough` to `True`.

    More information about file wrappers are available in :pep:`333`.

    :param file: a :class:`file`-like object with a :meth:`~file.read` method.
    :param buffer_size: number of bytes for one iteration.
    """
    return environ.get('wsgi.file_wrapper', FileWrapper)(file, buffer_size)


@implements_iterator
class FileWrapper(object):

    """This class can be used to convert a :class:`file`-like object into
    an iterable.  It yields `buffer_size` blocks until the file is fully
    read.

    You should not use this class directly but rather use the
    :func:`wrap_file` function that uses the WSGI server's file wrapper
    support if it's available.

    .. versionadded:: 0.5

    If you're using this object together with a :class:`BaseResponse` you have
    to use the `direct_passthrough` mode.

    :param file: a :class:`file`-like object with a :meth:`~file.read` method.
    :param buffer_size: number of bytes for one iteration.
    """

    def __init__(self, file, buffer_size=8192):
        self.file = file
        self.buffer_size = buffer_size

    def close(self):
        if hasattr(self.file, 'close'):
            self.file.close()

    def __iter__(self):
        return self

    def __next__(self):
        data = self.file.read(self.buffer_size)
        if data:
            return data
        raise StopIteration()


def _make_chunk_iter(stream, limit, buffer_size):
    """Helper for the line and chunk iter functions."""
    if isinstance(stream, (bytes, bytearray, text_type)):
        raise TypeError('Passed a string or byte object instead of '
                        'true iterator or stream.')
    if not hasattr(stream, 'read'):
        for item in stream:
            if item:
                yield item
        return
    if not isinstance(stream, LimitedStream) and limit is not None:
        stream = LimitedStream(stream, limit)
    _read = stream.read
    while 1:
        item = _read(buffer_size)
        if not item:
            break
        yield item


def make_line_iter(stream, limit=None, buffer_size=10 * 1024,
                   cap_at_buffer=False):
    """Safely iterates line-based over an input stream.  If the input stream
    is not a :class:`LimitedStream` the `limit` parameter is mandatory.

    This uses the stream's :meth:`~file.read` method internally as opposite
    to the :meth:`~file.readline` method that is unsafe and can only be used
    in violation of the WSGI specification.  The same problem applies to the
    `__iter__` function of the input stream which calls :meth:`~file.readline`
    without arguments.

    If you need line-by-line processing it's strongly recommended to iterate
    over the input stream using this helper function.

    .. versionchanged:: 0.8
       This function now ensures that the limit was reached.

    .. versionadded:: 0.9
       added support for iterators as input stream.

    :param stream: the stream or iterate to iterate over.
    :param limit: the limit in bytes for the stream.  (Usually
                  content length.  Not necessary if the `stream`
                  is a :class:`LimitedStream`.
    :param buffer_size: The optional buffer size.
    :param cap_at_buffer: if this is set chunks are split if they are longer
                          than the buffer size.  Internally this is implemented
                          that the buffer size might be exhausted by a factor
                          of two however.
    .. versionadded:: 0.11.10
       added support for the `cap_at_buffer` parameter.
    """
    _iter = _make_chunk_iter(stream, limit, buffer_size)

    first_item = next(_iter, '')
    if not first_item:
        return

    s = make_literal_wrapper(first_item)
    empty = s('')
    cr = s('\r')
    lf = s('\n')
    crlf = s('\r\n')

    _iter = chain((first_item,), _iter)

    def _iter_basic_lines():
        _join = empty.join
        buffer = []
        while 1:
            new_data = next(_iter, '')
            if not new_data:
                break
            new_buf = []
            buf_size = 0
            for item in chain(buffer, new_data.splitlines(True)):
                new_buf.append(item)
                buf_size += len(item)
                if item and item[-1:] in crlf:
                    yield _join(new_buf)
                    new_buf = []
                elif cap_at_buffer and buf_size >= buffer_size:
                    rv = _join(new_buf)
                    while len(rv) >= buffer_size:
                        yield rv[:buffer_size]
                        rv = rv[buffer_size:]
                    new_buf = [rv]
            buffer = new_buf
        if buffer:
            yield _join(buffer)

    # This hackery is necessary to merge 'foo\r' and '\n' into one item
    # of 'foo\r\n' if we were unlucky and we hit a chunk boundary.
    previous = empty
    for item in _iter_basic_lines():
        if item == lf and previous[-1:] == cr:
            previous += item
            item = empty
        if previous:
            yield previous
        previous = item
    if previous:
        yield previous


def make_chunk_iter(stream, separator, limit=None, buffer_size=10 * 1024,
                    cap_at_buffer=False):
    """Works like :func:`make_line_iter` but accepts a separator
    which divides chunks.  If you want newline based processing
    you should use :func:`make_line_iter` instead as it
    supports arbitrary newline markers.

    .. versionadded:: 0.8

    .. versionadded:: 0.9
       added support for iterators as input stream.

    .. versionadded:: 0.11.10
       added support for the `cap_at_buffer` parameter.

    :param stream: the stream or iterate to iterate over.
    :param separator: the separator that divides chunks.
    :param limit: the limit in bytes for the stream.  (Usually
                  content length.  Not necessary if the `stream`
                  is otherwise already limited).
    :param buffer_size: The optional buffer size.
    :param cap_at_buffer: if this is set chunks are split if they are longer
                          than the buffer size.  Internally this is implemented
                          that the buffer size might be exhausted by a factor
                          of two however.
    """
    _iter = _make_chunk_iter(stream, limit, buffer_size)

    first_item = next(_iter, '')
    if not first_item:
        return

    _iter = chain((first_item,), _iter)
    if isinstance(first_item, text_type):
        separator = to_unicode(separator)
        _split = re.compile(r'(%s)' % re.escape(separator)).split
        _join = u''.join
    else:
        separator = to_bytes(separator)
        _split = re.compile(b'(' + re.escape(separator) + b')').split
        _join = b''.join

    buffer = []
    while 1:
        new_data = next(_iter, '')
        if not new_data:
            break
        chunks = _split(new_data)
        new_buf = []
        buf_size = 0
        for item in chain(buffer, chunks):
            if item == separator:
                yield _join(new_buf)
                new_buf = []
                buf_size = 0
            else:
                buf_size += len(item)
                new_buf.append(item)

                if cap_at_buffer and buf_size >= buffer_size:
                    rv = _join(new_buf)
                    while len(rv) >= buffer_size:
                        yield rv[:buffer_size]
                        rv = rv[buffer_size:]
                    new_buf = [rv]
                    buf_size = len(rv)

        buffer = new_buf
    if buffer:
        yield _join(buffer)


@implements_iterator
class LimitedStream(object):

    """Wraps a stream so that it doesn't read more than n bytes.  If the
    stream is exhausted and the caller tries to get more bytes from it
    :func:`on_exhausted` is called which by default returns an empty
    string.  The return value of that function is forwarded
    to the reader function.  So if it returns an empty string
    :meth:`read` will return an empty string as well.

    The limit however must never be higher than what the stream can
    output.  Otherwise :meth:`readlines` will try to read past the
    limit.

    .. admonition:: Note on WSGI compliance

       calls to :meth:`readline` and :meth:`readlines` are not
       WSGI compliant because it passes a size argument to the
       readline methods.  Unfortunately the WSGI PEP is not safely
       implementable without a size argument to :meth:`readline`
       because there is no EOF marker in the stream.  As a result
       of that the use of :meth:`readline` is discouraged.

       For the same reason iterating over the :class:`LimitedStream`
       is not portable.  It internally calls :meth:`readline`.

       We strongly suggest using :meth:`read` only or using the
       :func:`make_line_iter` which safely iterates line-based
       over a WSGI input stream.

    :param stream: the stream to wrap.
    :param limit: the limit for the stream, must not be longer than
                  what the string can provide if the stream does not
                  end with `EOF` (like `wsgi.input`)
    """

    def __init__(self, stream, limit):
        self._read = stream.read
        self._readline = stream.readline
        self._pos = 0
        self.limit = limit

    def __iter__(self):
        return self

    @property
    def is_exhausted(self):
        """If the stream is exhausted this attribute is `True`."""
        return self._pos >= self.limit

    def on_exhausted(self):
        """This is called when the stream tries to read past the limit.
        The return value of this function is returned from the reading
        function.
        """
        # Read null bytes from the stream so that we get the
        # correct end of stream marker.
        return self._read(0)

    def on_disconnect(self):
        """What should happen if a disconnect is detected?  The return
        value of this function is returned from read functions in case
        the client went away.  By default a
        :exc:`~werkzeug.exceptions.ClientDisconnected` exception is raised.
        """
        from werkzeug.exceptions import ClientDisconnected
        raise ClientDisconnected()

    def exhaust(self, chunk_size=1024 * 64):
        """Exhaust the stream.  This consumes all the data left until the
        limit is reached.

        :param chunk_size: the size for a chunk.  It will read the chunk
                           until the stream is exhausted and throw away
                           the results.
        """
        to_read = self.limit - self._pos
        chunk = chunk_size
        while to_read > 0:
            chunk = min(to_read, chunk)
            self.read(chunk)
            to_read -= chunk

    def read(self, size=None):
        """Read `size` bytes or if size is not provided everything is read.

        :param size: the number of bytes read.
        """
        if self._pos >= self.limit:
            return self.on_exhausted()
        if size is None or size == -1:  # -1 is for consistence with file
            size = self.limit
        to_read = min(self.limit - self._pos, size)
        try:
            read = self._read(to_read)
        except (IOError, ValueError):
            return self.on_disconnect()
        if to_read and len(read) != to_read:
            return self.on_disconnect()
        self._pos += len(read)
        return read

    def readline(self, size=None):
        """Reads one line from the stream."""
        if self._pos >= self.limit:
            return self.on_exhausted()
        if size is None:
            size = self.limit - self._pos
        else:
            size = min(size, self.limit - self._pos)
        try:
            line = self._readline(size)
        except (ValueError, IOError):
            return self.on_disconnect()
        if size and not line:
            return self.on_disconnect()
        self._pos += len(line)
        return line

    def readlines(self, size=None):
        """Reads a file into a list of strings.  It calls :meth:`readline`
        until the file is read to the end.  It does support the optional
        `size` argument if the underlaying stream supports it for
        `readline`.
        """
        last_pos = self._pos
        result = []
        if size is not None:
            end = min(self.limit, last_pos + size)
        else:
            end = self.limit
        while 1:
            if size is not None:
                size -= last_pos - self._pos
            if self._pos >= end:
                break
            result.append(self.readline(size))
            if size is not None:
                last_pos = self._pos
        return result

    def tell(self):
        """Returns the position of the stream.

        .. versionadded:: 0.9
        """
        return self._pos

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration()
        return line
