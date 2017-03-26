# -*- coding: utf-8 -*-
"""
    werkzeug.test
    ~~~~~~~~~~~~~

    This module implements a client to WSGI applications for testing.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import sys
import mimetypes
from time import time
from random import random
from itertools import chain
from tempfile import TemporaryFile
from io import BytesIO

try:
    from urllib2 import Request as U2Request
except ImportError:
    from urllib.request import Request as U2Request
try:
    from http.cookiejar import CookieJar
except ImportError:  # Py2
    from cookielib import CookieJar

from werkzeug._compat import iterlists, iteritems, itervalues, to_bytes, \
    string_types, text_type, reraise, wsgi_encoding_dance, \
    make_literal_wrapper
from werkzeug._internal import _empty_stream, _get_environ
from werkzeug.wrappers import BaseRequest
from werkzeug.urls import url_encode, url_fix, iri_to_uri, url_unquote, \
    url_unparse, url_parse
from werkzeug.wsgi import get_host, get_current_url, ClosingIterator
from werkzeug.utils import dump_cookie
from werkzeug.datastructures import FileMultiDict, MultiDict, \
    CombinedMultiDict, Headers, FileStorage


def stream_encode_multipart(values, use_tempfile=True, threshold=1024 * 500,
                            boundary=None, charset='utf-8'):
    """Encode a dict of values (either strings or file descriptors or
    :class:`FileStorage` objects.) into a multipart encoded string stored
    in a file descriptor.
    """
    if boundary is None:
        boundary = '---------------WerkzeugFormPart_%s%s' % (time(), random())
    _closure = [BytesIO(), 0, False]

    if use_tempfile:
        def write_binary(string):
            stream, total_length, on_disk = _closure
            if on_disk:
                stream.write(string)
            else:
                length = len(string)
                if length + _closure[1] <= threshold:
                    stream.write(string)
                else:
                    new_stream = TemporaryFile('wb+')
                    new_stream.write(stream.getvalue())
                    new_stream.write(string)
                    _closure[0] = new_stream
                    _closure[2] = True
                _closure[1] = total_length + length
    else:
        write_binary = _closure[0].write

    def write(string):
        write_binary(string.encode(charset))

    if not isinstance(values, MultiDict):
        values = MultiDict(values)

    for key, values in iterlists(values):
        for value in values:
            write('--%s\r\nContent-Disposition: form-data; name="%s"' %
                  (boundary, key))
            reader = getattr(value, 'read', None)
            if reader is not None:
                filename = getattr(value, 'filename',
                                   getattr(value, 'name', None))
                content_type = getattr(value, 'content_type', None)
                if content_type is None:
                    content_type = filename and \
                        mimetypes.guess_type(filename)[0] or \
                        'application/octet-stream'
                if filename is not None:
                    write('; filename="%s"\r\n' % filename)
                else:
                    write('\r\n')
                write('Content-Type: %s\r\n\r\n' % content_type)
                while 1:
                    chunk = reader(16384)
                    if not chunk:
                        break
                    write_binary(chunk)
            else:
                if not isinstance(value, string_types):
                    value = str(value)

                value = to_bytes(value, charset)
                write('\r\n\r\n')
                write_binary(value)
            write('\r\n')
    write('--%s--\r\n' % boundary)

    length = int(_closure[0].tell())
    _closure[0].seek(0)
    return _closure[0], length, boundary


def encode_multipart(values, boundary=None, charset='utf-8'):
    """Like `stream_encode_multipart` but returns a tuple in the form
    (``boundary``, ``data``) where data is a bytestring.
    """
    stream, length, boundary = stream_encode_multipart(
        values, use_tempfile=False, boundary=boundary, charset=charset)
    return boundary, stream.read()


def File(fd, filename=None, mimetype=None):
    """Backwards compat."""
    from warnings import warn
    warn(DeprecationWarning('werkzeug.test.File is deprecated, use the '
                            'EnvironBuilder or FileStorage instead'))
    return FileStorage(fd, filename=filename, content_type=mimetype)


class _TestCookieHeaders(object):

    """A headers adapter for cookielib
    """

    def __init__(self, headers):
        self.headers = headers

    def getheaders(self, name):
        headers = []
        name = name.lower()
        for k, v in self.headers:
            if k.lower() == name:
                headers.append(v)
        return headers

    def get_all(self, name, default=None):
        rv = []
        for k, v in self.headers:
            if k.lower() == name.lower():
                rv.append(v)
        return rv or default or []


class _TestCookieResponse(object):

    """Something that looks like a httplib.HTTPResponse, but is actually just an
    adapter for our test responses to make them available for cookielib.
    """

    def __init__(self, headers):
        self.headers = _TestCookieHeaders(headers)

    def info(self):
        return self.headers


class _TestCookieJar(CookieJar):

    """A cookielib.CookieJar modified to inject and read cookie headers from
    and to wsgi environments, and wsgi application responses.
    """

    def inject_wsgi(self, environ):
        """Inject the cookies as client headers into the server's wsgi
        environment.
        """
        cvals = []
        for cookie in self:
            cvals.append('%s=%s' % (cookie.name, cookie.value))
        if cvals:
            environ['HTTP_COOKIE'] = '; '.join(cvals)

    def extract_wsgi(self, environ, headers):
        """Extract the server's set-cookie headers as cookies into the
        cookie jar.
        """
        self.extract_cookies(
            _TestCookieResponse(headers),
            U2Request(get_current_url(environ)),
        )


def _iter_data(data):
    """Iterates over a dict or multidict yielding all keys and values.
    This is used to iterate over the data passed to the
    :class:`EnvironBuilder`.
    """
    if isinstance(data, MultiDict):
        for key, values in iterlists(data):
            for value in values:
                yield key, value
    else:
        for key, values in iteritems(data):
            if isinstance(values, list):
                for value in values:
                    yield key, value
            else:
                yield key, values


class EnvironBuilder(object):

    """This class can be used to conveniently create a WSGI environment
    for testing purposes.  It can be used to quickly create WSGI environments
    or request objects from arbitrary data.

    The signature of this class is also used in some other places as of
    Werkzeug 0.5 (:func:`create_environ`, :meth:`BaseResponse.from_values`,
    :meth:`Client.open`).  Because of this most of the functionality is
    available through the constructor alone.

    Files and regular form data can be manipulated independently of each
    other with the :attr:`form` and :attr:`files` attributes, but are
    passed with the same argument to the constructor: `data`.

    `data` can be any of these values:

    -   a `str`: If it's a string it is converted into a :attr:`input_stream`,
        the :attr:`content_length` is set and you have to provide a
        :attr:`content_type`.
    -   a `dict`: If it's a dict the keys have to be strings and the values
        any of the following objects:

        -   a :class:`file`-like object.  These are converted into
            :class:`FileStorage` objects automatically.
        -   a tuple.  The :meth:`~FileMultiDict.add_file` method is called
            with the tuple items as positional arguments.

    .. versionadded:: 0.6
       `path` and `base_url` can now be unicode strings that are encoded using
       the :func:`iri_to_uri` function.

    :param path: the path of the request.  In the WSGI environment this will
                 end up as `PATH_INFO`.  If the `query_string` is not defined
                 and there is a question mark in the `path` everything after
                 it is used as query string.
    :param base_url: the base URL is a URL that is used to extract the WSGI
                     URL scheme, host (server name + server port) and the
                     script root (`SCRIPT_NAME`).
    :param query_string: an optional string or dict with URL parameters.
    :param method: the HTTP method to use, defaults to `GET`.
    :param input_stream: an optional input stream.  Do not specify this and
                         `data`.  As soon as an input stream is set you can't
                         modify :attr:`args` and :attr:`files` unless you
                         set the :attr:`input_stream` to `None` again.
    :param content_type: The content type for the request.  As of 0.5 you
                         don't have to provide this when specifying files
                         and form data via `data`.
    :param content_length: The content length for the request.  You don't
                           have to specify this when providing data via
                           `data`.
    :param errors_stream: an optional error stream that is used for
                          `wsgi.errors`.  Defaults to :data:`stderr`.
    :param multithread: controls `wsgi.multithread`.  Defaults to `False`.
    :param multiprocess: controls `wsgi.multiprocess`.  Defaults to `False`.
    :param run_once: controls `wsgi.run_once`.  Defaults to `False`.
    :param headers: an optional list or :class:`Headers` object of headers.
    :param data: a string or dict of form data.  See explanation above.
    :param environ_base: an optional dict of environment defaults.
    :param environ_overrides: an optional dict of environment overrides.
    :param charset: the charset used to encode unicode data.
    """

    #: the server protocol to use.  defaults to HTTP/1.1
    server_protocol = 'HTTP/1.1'

    #: the wsgi version to use.  defaults to (1, 0)
    wsgi_version = (1, 0)

    #: the default request class for :meth:`get_request`
    request_class = BaseRequest

    def __init__(self, path='/', base_url=None, query_string=None,
                 method='GET', input_stream=None, content_type=None,
                 content_length=None, errors_stream=None, multithread=False,
                 multiprocess=False, run_once=False, headers=None, data=None,
                 environ_base=None, environ_overrides=None, charset='utf-8'):
        path_s = make_literal_wrapper(path)
        if query_string is None and path_s('?') in path:
            path, query_string = path.split(path_s('?'), 1)
        self.charset = charset
        self.path = iri_to_uri(path)
        if base_url is not None:
            base_url = url_fix(iri_to_uri(base_url, charset), charset)
        self.base_url = base_url
        if isinstance(query_string, (bytes, text_type)):
            self.query_string = query_string
        else:
            if query_string is None:
                query_string = MultiDict()
            elif not isinstance(query_string, MultiDict):
                query_string = MultiDict(query_string)
            self.args = query_string
        self.method = method
        if headers is None:
            headers = Headers()
        elif not isinstance(headers, Headers):
            headers = Headers(headers)
        self.headers = headers
        if content_type is not None:
            self.content_type = content_type
        if errors_stream is None:
            errors_stream = sys.stderr
        self.errors_stream = errors_stream
        self.multithread = multithread
        self.multiprocess = multiprocess
        self.run_once = run_once
        self.environ_base = environ_base
        self.environ_overrides = environ_overrides
        self.input_stream = input_stream
        self.content_length = content_length
        self.closed = False

        if data:
            if input_stream is not None:
                raise TypeError('can\'t provide input stream and data')
            if isinstance(data, text_type):
                data = data.encode(self.charset)
            if isinstance(data, bytes):
                self.input_stream = BytesIO(data)
                if self.content_length is None:
                    self.content_length = len(data)
            else:
                for key, value in _iter_data(data):
                    if isinstance(value, (tuple, dict)) or \
                       hasattr(value, 'read'):
                        self._add_file_from_data(key, value)
                    else:
                        self.form.setlistdefault(key).append(value)

    def _add_file_from_data(self, key, value):
        """Called in the EnvironBuilder to add files from the data dict."""
        if isinstance(value, tuple):
            self.files.add_file(key, *value)
        elif isinstance(value, dict):
            from warnings import warn
            warn(DeprecationWarning('it\'s no longer possible to pass dicts '
                                    'as `data`.  Use tuples or FileStorage '
                                    'objects instead'), stacklevel=2)
            value = dict(value)
            mimetype = value.pop('mimetype', None)
            if mimetype is not None:
                value['content_type'] = mimetype
            self.files.add_file(key, **value)
        else:
            self.files.add_file(key, value)

    def _get_base_url(self):
        return url_unparse((self.url_scheme, self.host,
                            self.script_root, '', '')).rstrip('/') + '/'

    def _set_base_url(self, value):
        if value is None:
            scheme = 'http'
            netloc = 'localhost'
            script_root = ''
        else:
            scheme, netloc, script_root, qs, anchor = url_parse(value)
            if qs or anchor:
                raise ValueError('base url must not contain a query string '
                                 'or fragment')
        self.script_root = script_root.rstrip('/')
        self.host = netloc
        self.url_scheme = scheme

    base_url = property(_get_base_url, _set_base_url, doc='''
        The base URL is a URL that is used to extract the WSGI
        URL scheme, host (server name + server port) and the
        script root (`SCRIPT_NAME`).''')
    del _get_base_url, _set_base_url

    def _get_content_type(self):
        ct = self.headers.get('Content-Type')
        if ct is None and not self._input_stream:
            if self._files:
                return 'multipart/form-data'
            elif self._form:
                return 'application/x-www-form-urlencoded'
            return None
        return ct

    def _set_content_type(self, value):
        if value is None:
            self.headers.pop('Content-Type', None)
        else:
            self.headers['Content-Type'] = value

    content_type = property(_get_content_type, _set_content_type, doc='''
        The content type for the request.  Reflected from and to the
        :attr:`headers`.  Do not set if you set :attr:`files` or
        :attr:`form` for auto detection.''')
    del _get_content_type, _set_content_type

    def _get_content_length(self):
        return self.headers.get('Content-Length', type=int)

    def _set_content_length(self, value):
        if value is None:
            self.headers.pop('Content-Length', None)
        else:
            self.headers['Content-Length'] = str(value)

    content_length = property(_get_content_length, _set_content_length, doc='''
        The content length as integer.  Reflected from and to the
        :attr:`headers`.  Do not set if you set :attr:`files` or
        :attr:`form` for auto detection.''')
    del _get_content_length, _set_content_length

    def form_property(name, storage, doc):
        key = '_' + name

        def getter(self):
            if self._input_stream is not None:
                raise AttributeError('an input stream is defined')
            rv = getattr(self, key)
            if rv is None:
                rv = storage()
                setattr(self, key, rv)

            return rv

        def setter(self, value):
            self._input_stream = None
            setattr(self, key, value)
        return property(getter, setter, doc=doc)

    form = form_property('form', MultiDict, doc='''
        A :class:`MultiDict` of form values.''')
    files = form_property('files', FileMultiDict, doc='''
        A :class:`FileMultiDict` of uploaded files.  You can use the
        :meth:`~FileMultiDict.add_file` method to add new files to the
        dict.''')
    del form_property

    def _get_input_stream(self):
        return self._input_stream

    def _set_input_stream(self, value):
        self._input_stream = value
        self._form = self._files = None

    input_stream = property(_get_input_stream, _set_input_stream, doc='''
        An optional input stream.  If you set this it will clear
        :attr:`form` and :attr:`files`.''')
    del _get_input_stream, _set_input_stream

    def _get_query_string(self):
        if self._query_string is None:
            if self._args is not None:
                return url_encode(self._args, charset=self.charset)
            return ''
        return self._query_string

    def _set_query_string(self, value):
        self._query_string = value
        self._args = None

    query_string = property(_get_query_string, _set_query_string, doc='''
        The query string.  If you set this to a string :attr:`args` will
        no longer be available.''')
    del _get_query_string, _set_query_string

    def _get_args(self):
        if self._query_string is not None:
            raise AttributeError('a query string is defined')
        if self._args is None:
            self._args = MultiDict()
        return self._args

    def _set_args(self, value):
        self._query_string = None
        self._args = value

    args = property(_get_args, _set_args, doc='''
        The URL arguments as :class:`MultiDict`.''')
    del _get_args, _set_args

    @property
    def server_name(self):
        """The server name (read-only, use :attr:`host` to set)"""
        return self.host.split(':', 1)[0]

    @property
    def server_port(self):
        """The server port as integer (read-only, use :attr:`host` to set)"""
        pieces = self.host.split(':', 1)
        if len(pieces) == 2 and pieces[1].isdigit():
            return int(pieces[1])
        elif self.url_scheme == 'https':
            return 443
        return 80

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        """Closes all files.  If you put real :class:`file` objects into the
        :attr:`files` dict you can call this method to automatically close
        them all in one go.
        """
        if self.closed:
            return
        try:
            files = itervalues(self.files)
        except AttributeError:
            files = ()
        for f in files:
            try:
                f.close()
            except Exception:
                pass
        self.closed = True

    def get_environ(self):
        """Return the built environ."""
        input_stream = self.input_stream
        content_length = self.content_length
        content_type = self.content_type

        if input_stream is not None:
            start_pos = input_stream.tell()
            input_stream.seek(0, 2)
            end_pos = input_stream.tell()
            input_stream.seek(start_pos)
            content_length = end_pos - start_pos
        elif content_type == 'multipart/form-data':
            values = CombinedMultiDict([self.form, self.files])
            input_stream, content_length, boundary = \
                stream_encode_multipart(values, charset=self.charset)
            content_type += '; boundary="%s"' % boundary
        elif content_type == 'application/x-www-form-urlencoded':
            # XXX: py2v3 review
            values = url_encode(self.form, charset=self.charset)
            values = values.encode('ascii')
            content_length = len(values)
            input_stream = BytesIO(values)
        else:
            input_stream = _empty_stream

        result = {}
        if self.environ_base:
            result.update(self.environ_base)

        def _path_encode(x):
            return wsgi_encoding_dance(url_unquote(x, self.charset), self.charset)

        qs = wsgi_encoding_dance(self.query_string)

        result.update({
            'REQUEST_METHOD':       self.method,
            'SCRIPT_NAME':          _path_encode(self.script_root),
            'PATH_INFO':            _path_encode(self.path),
            'QUERY_STRING':         qs,
            'SERVER_NAME':          self.server_name,
            'SERVER_PORT':          str(self.server_port),
            'HTTP_HOST':            self.host,
            'SERVER_PROTOCOL':      self.server_protocol,
            'CONTENT_TYPE':         content_type or '',
            'CONTENT_LENGTH':       str(content_length or '0'),
            'wsgi.version':         self.wsgi_version,
            'wsgi.url_scheme':      self.url_scheme,
            'wsgi.input':           input_stream,
            'wsgi.errors':          self.errors_stream,
            'wsgi.multithread':     self.multithread,
            'wsgi.multiprocess':    self.multiprocess,
            'wsgi.run_once':        self.run_once
        })
        for key, value in self.headers.to_wsgi_list():
            result['HTTP_%s' % key.upper().replace('-', '_')] = value
        if self.environ_overrides:
            result.update(self.environ_overrides)
        return result

    def get_request(self, cls=None):
        """Returns a request with the data.  If the request class is not
        specified :attr:`request_class` is used.

        :param cls: The request wrapper to use.
        """
        if cls is None:
            cls = self.request_class
        return cls(self.get_environ())


class ClientRedirectError(Exception):

    """
    If a redirect loop is detected when using follow_redirects=True with
    the :cls:`Client`, then this exception is raised.
    """


class Client(object):

    """This class allows to send requests to a wrapped application.

    The response wrapper can be a class or factory function that takes
    three arguments: app_iter, status and headers.  The default response
    wrapper just returns a tuple.

    Example::

        class ClientResponse(BaseResponse):
            ...

        client = Client(MyApplication(), response_wrapper=ClientResponse)

    The use_cookies parameter indicates whether cookies should be stored and
    sent for subsequent requests. This is True by default, but passing False
    will disable this behaviour.

    If you want to request some subdomain of your application you may set
    `allow_subdomain_redirects` to `True` as if not no external redirects
    are allowed.

    .. versionadded:: 0.5
       `use_cookies` is new in this version.  Older versions did not provide
       builtin cookie support.
    """

    def __init__(self, application, response_wrapper=None, use_cookies=True,
                 allow_subdomain_redirects=False):
        self.application = application
        self.response_wrapper = response_wrapper
        if use_cookies:
            self.cookie_jar = _TestCookieJar()
        else:
            self.cookie_jar = None
        self.allow_subdomain_redirects = allow_subdomain_redirects

    def set_cookie(self, server_name, key, value='', max_age=None,
                   expires=None, path='/', domain=None, secure=None,
                   httponly=False, charset='utf-8'):
        """Sets a cookie in the client's cookie jar.  The server name
        is required and has to match the one that is also passed to
        the open call.
        """
        assert self.cookie_jar is not None, 'cookies disabled'
        header = dump_cookie(key, value, max_age, expires, path, domain,
                             secure, httponly, charset)
        environ = create_environ(path, base_url='http://' + server_name)
        headers = [('Set-Cookie', header)]
        self.cookie_jar.extract_wsgi(environ, headers)

    def delete_cookie(self, server_name, key, path='/', domain=None):
        """Deletes a cookie in the test client."""
        self.set_cookie(server_name, key, expires=0, max_age=0,
                        path=path, domain=domain)

    def run_wsgi_app(self, environ, buffered=False):
        """Runs the wrapped WSGI app with the given environment."""
        if self.cookie_jar is not None:
            self.cookie_jar.inject_wsgi(environ)
        rv = run_wsgi_app(self.application, environ, buffered=buffered)
        if self.cookie_jar is not None:
            self.cookie_jar.extract_wsgi(environ, rv[2])
        return rv

    def resolve_redirect(self, response, new_location, environ, buffered=False):
        """Resolves a single redirect and triggers the request again
        directly on this redirect client.
        """
        scheme, netloc, script_root, qs, anchor = url_parse(new_location)
        base_url = url_unparse((scheme, netloc, '', '', '')).rstrip('/') + '/'

        cur_server_name = netloc.split(':', 1)[0].split('.')
        real_server_name = get_host(environ).rsplit(':', 1)[0].split('.')

        if self.allow_subdomain_redirects:
            allowed = cur_server_name[-len(real_server_name):] == real_server_name
        else:
            allowed = cur_server_name == real_server_name

        if not allowed:
            raise RuntimeError('%r does not support redirect to '
                               'external targets' % self.__class__)

        status_code = int(response[1].split(None, 1)[0])
        if status_code == 307:
            method = environ['REQUEST_METHOD']
        else:
            method = 'GET'

        # For redirect handling we temporarily disable the response
        # wrapper.  This is not threadsafe but not a real concern
        # since the test client must not be shared anyways.
        old_response_wrapper = self.response_wrapper
        self.response_wrapper = None
        try:
            return self.open(path=script_root, base_url=base_url,
                             query_string=qs, as_tuple=True,
                             buffered=buffered, method=method)
        finally:
            self.response_wrapper = old_response_wrapper

    def open(self, *args, **kwargs):
        """Takes the same arguments as the :class:`EnvironBuilder` class with
        some additions:  You can provide a :class:`EnvironBuilder` or a WSGI
        environment as only argument instead of the :class:`EnvironBuilder`
        arguments and two optional keyword arguments (`as_tuple`, `buffered`)
        that change the type of the return value or the way the application is
        executed.

        .. versionchanged:: 0.5
           If a dict is provided as file in the dict for the `data` parameter
           the content type has to be called `content_type` now instead of
           `mimetype`.  This change was made for consistency with
           :class:`werkzeug.FileWrapper`.

            The `follow_redirects` parameter was added to :func:`open`.

        Additional parameters:

        :param as_tuple: Returns a tuple in the form ``(environ, result)``
        :param buffered: Set this to True to buffer the application run.
                         This will automatically close the application for
                         you as well.
        :param follow_redirects: Set this to True if the `Client` should
                                 follow HTTP redirects.
        """
        as_tuple = kwargs.pop('as_tuple', False)
        buffered = kwargs.pop('buffered', False)
        follow_redirects = kwargs.pop('follow_redirects', False)
        environ = None
        if not kwargs and len(args) == 1:
            if isinstance(args[0], EnvironBuilder):
                environ = args[0].get_environ()
            elif isinstance(args[0], dict):
                environ = args[0]
        if environ is None:
            builder = EnvironBuilder(*args, **kwargs)
            try:
                environ = builder.get_environ()
            finally:
                builder.close()

        response = self.run_wsgi_app(environ, buffered=buffered)

        # handle redirects
        redirect_chain = []
        while 1:
            status_code = int(response[1].split(None, 1)[0])
            if status_code not in (301, 302, 303, 305, 307) \
               or not follow_redirects:
                break
            new_location = response[2]['location']
            new_redirect_entry = (new_location, status_code)
            if new_redirect_entry in redirect_chain:
                raise ClientRedirectError('loop detected')
            redirect_chain.append(new_redirect_entry)
            environ, response = self.resolve_redirect(response, new_location,
                                                      environ,
                                                      buffered=buffered)

        if self.response_wrapper is not None:
            response = self.response_wrapper(*response)
        if as_tuple:
            return environ, response
        return response

    def get(self, *args, **kw):
        """Like open but method is enforced to GET."""
        kw['method'] = 'GET'
        return self.open(*args, **kw)

    def patch(self, *args, **kw):
        """Like open but method is enforced to PATCH."""
        kw['method'] = 'PATCH'
        return self.open(*args, **kw)

    def post(self, *args, **kw):
        """Like open but method is enforced to POST."""
        kw['method'] = 'POST'
        return self.open(*args, **kw)

    def head(self, *args, **kw):
        """Like open but method is enforced to HEAD."""
        kw['method'] = 'HEAD'
        return self.open(*args, **kw)

    def put(self, *args, **kw):
        """Like open but method is enforced to PUT."""
        kw['method'] = 'PUT'
        return self.open(*args, **kw)

    def delete(self, *args, **kw):
        """Like open but method is enforced to DELETE."""
        kw['method'] = 'DELETE'
        return self.open(*args, **kw)

    def options(self, *args, **kw):
        """Like open but method is enforced to OPTIONS."""
        kw['method'] = 'OPTIONS'
        return self.open(*args, **kw)

    def trace(self, *args, **kw):
        """Like open but method is enforced to TRACE."""
        kw['method'] = 'TRACE'
        return self.open(*args, **kw)

    def __repr__(self):
        return '<%s %r>' % (
            self.__class__.__name__,
            self.application
        )


def create_environ(*args, **kwargs):
    """Create a new WSGI environ dict based on the values passed.  The first
    parameter should be the path of the request which defaults to '/'.  The
    second one can either be an absolute path (in that case the host is
    localhost:80) or a full path to the request with scheme, netloc port and
    the path to the script.

    This accepts the same arguments as the :class:`EnvironBuilder`
    constructor.

    .. versionchanged:: 0.5
       This function is now a thin wrapper over :class:`EnvironBuilder` which
       was added in 0.5.  The `headers`, `environ_base`, `environ_overrides`
       and `charset` parameters were added.
    """
    builder = EnvironBuilder(*args, **kwargs)
    try:
        return builder.get_environ()
    finally:
        builder.close()


def run_wsgi_app(app, environ, buffered=False):
    """Return a tuple in the form (app_iter, status, headers) of the
    application output.  This works best if you pass it an application that
    returns an iterator all the time.

    Sometimes applications may use the `write()` callable returned
    by the `start_response` function.  This tries to resolve such edge
    cases automatically.  But if you don't get the expected output you
    should set `buffered` to `True` which enforces buffering.

    If passed an invalid WSGI application the behavior of this function is
    undefined.  Never pass non-conforming WSGI applications to this function.

    :param app: the application to execute.
    :param buffered: set to `True` to enforce buffering.
    :return: tuple in the form ``(app_iter, status, headers)``
    """
    environ = _get_environ(environ)
    response = []
    buffer = []

    def start_response(status, headers, exc_info=None):
        if exc_info is not None:
            reraise(*exc_info)
        response[:] = [status, headers]
        return buffer.append

    app_rv = app(environ, start_response)
    close_func = getattr(app_rv, 'close', None)
    app_iter = iter(app_rv)

    # when buffering we emit the close call early and convert the
    # application iterator into a regular list
    if buffered:
        try:
            app_iter = list(app_iter)
        finally:
            if close_func is not None:
                close_func()

    # otherwise we iterate the application iter until we have a response, chain
    # the already received data with the already collected data and wrap it in
    # a new `ClosingIterator` if we need to restore a `close` callable from the
    # original return value.
    else:
        while not response:
            buffer.append(next(app_iter))
        if buffer:
            app_iter = chain(buffer, app_iter)
        if close_func is not None and app_iter is not app_rv:
            app_iter = ClosingIterator(app_iter, close_func)

    return app_iter, response[0], Headers(response[1])
