# -*- coding: utf-8 -*-
"""
    werkzeug.contrib.lint
    ~~~~~~~~~~~~~~~~~~~~~

    .. versionadded:: 0.5

    This module provides a middleware that performs sanity checks of the WSGI
    application.  It checks that :pep:`333` is properly implemented and warns
    on some common HTTP errors such as non-empty responses for 304 status
    codes.

    This module provides a middleware, the :class:`LintMiddleware`.  Wrap your
    application with it and it will warn about common problems with WSGI and
    HTTP while your application is running.

    It's strongly recommended to use it during development.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from urlparse import urlparse
from warnings import warn

from werkzeug.datastructures import Headers
from werkzeug.http import is_entity_header
from werkzeug.wsgi import FileWrapper
from werkzeug._compat import string_types


class WSGIWarning(Warning):

    """Warning class for WSGI warnings."""


class HTTPWarning(Warning):

    """Warning class for HTTP warnings."""


def check_string(context, obj, stacklevel=3):
    if type(obj) is not str:
        warn(WSGIWarning('%s requires bytestrings, got %s' %
                         (context, obj.__class__.__name__)))


class InputStream(object):

    def __init__(self, stream):
        self._stream = stream

    def read(self, *args):
        if len(args) == 0:
            warn(WSGIWarning('wsgi does not guarantee an EOF marker on the '
                             'input stream, thus making calls to '
                             'wsgi.input.read() unsafe.  Conforming servers '
                             'may never return from this call.'),
                 stacklevel=2)
        elif len(args) != 1:
            warn(WSGIWarning('too many parameters passed to wsgi.input.read()'),
                 stacklevel=2)
        return self._stream.read(*args)

    def readline(self, *args):
        if len(args) == 0:
            warn(WSGIWarning('Calls to wsgi.input.readline() without arguments'
                             ' are unsafe.  Use wsgi.input.read() instead.'),
                 stacklevel=2)
        elif len(args) == 1:
            warn(WSGIWarning('wsgi.input.readline() was called with a size hint. '
                             'WSGI does not support this, although it\'s available '
                             'on all major servers.'),
                 stacklevel=2)
        else:
            raise TypeError('too many arguments passed to wsgi.input.readline()')
        return self._stream.readline(*args)

    def __iter__(self):
        try:
            return iter(self._stream)
        except TypeError:
            warn(WSGIWarning('wsgi.input is not iterable.'), stacklevel=2)
            return iter(())

    def close(self):
        warn(WSGIWarning('application closed the input stream!'),
             stacklevel=2)
        self._stream.close()


class ErrorStream(object):

    def __init__(self, stream):
        self._stream = stream

    def write(self, s):
        check_string('wsgi.error.write()', s)
        self._stream.write(s)

    def flush(self):
        self._stream.flush()

    def writelines(self, seq):
        for line in seq:
            self.write(seq)

    def close(self):
        warn(WSGIWarning('application closed the error stream!'),
             stacklevel=2)
        self._stream.close()


class GuardedWrite(object):

    def __init__(self, write, chunks):
        self._write = write
        self._chunks = chunks

    def __call__(self, s):
        check_string('write()', s)
        self._write.write(s)
        self._chunks.append(len(s))


class GuardedIterator(object):

    def __init__(self, iterator, headers_set, chunks):
        self._iterator = iterator
        self._next = iter(iterator).next
        self.closed = False
        self.headers_set = headers_set
        self.chunks = chunks

    def __iter__(self):
        return self

    def next(self):
        if self.closed:
            warn(WSGIWarning('iterated over closed app_iter'),
                 stacklevel=2)
        rv = self._next()
        if not self.headers_set:
            warn(WSGIWarning('Application returned before it '
                             'started the response'), stacklevel=2)
        check_string('application iterator items', rv)
        self.chunks.append(len(rv))
        return rv

    def close(self):
        self.closed = True
        if hasattr(self._iterator, 'close'):
            self._iterator.close()

        if self.headers_set:
            status_code, headers = self.headers_set
            bytes_sent = sum(self.chunks)
            content_length = headers.get('content-length', type=int)

            if status_code == 304:
                for key, value in headers:
                    key = key.lower()
                    if key not in ('expires', 'content-location') and \
                       is_entity_header(key):
                        warn(HTTPWarning('entity header %r found in 304 '
                                         'response' % key))
                if bytes_sent:
                    warn(HTTPWarning('304 responses must not have a body'))
            elif 100 <= status_code < 200 or status_code == 204:
                if content_length != 0:
                    warn(HTTPWarning('%r responses must have an empty '
                                     'content length') % status_code)
                if bytes_sent:
                    warn(HTTPWarning('%r responses must not have a body' %
                                     status_code))
            elif content_length is not None and content_length != bytes_sent:
                warn(WSGIWarning('Content-Length and the number of bytes '
                                 'sent to the client do not match.'))

    def __del__(self):
        if not self.closed:
            try:
                warn(WSGIWarning('Iterator was garbage collected before '
                                 'it was closed.'))
            except Exception:
                pass


class LintMiddleware(object):

    """This middleware wraps an application and warns on common errors.
    Among other thing it currently checks for the following problems:

    -   invalid status codes
    -   non-bytestrings sent to the WSGI server
    -   strings returned from the WSGI application
    -   non-empty conditional responses
    -   unquoted etags
    -   relative URLs in the Location header
    -   unsafe calls to wsgi.input
    -   unclosed iterators

    Detected errors are emitted using the standard Python :mod:`warnings`
    system and usually end up on :data:`stderr`.

    ::

        from werkzeug.contrib.lint import LintMiddleware
        app = LintMiddleware(app)

    :param app: the application to wrap
    """

    def __init__(self, app):
        self.app = app

    def check_environ(self, environ):
        if type(environ) is not dict:
            warn(WSGIWarning('WSGI environment is not a standard python dict.'),
                 stacklevel=4)
        for key in ('REQUEST_METHOD', 'SERVER_NAME', 'SERVER_PORT',
                    'wsgi.version', 'wsgi.input', 'wsgi.errors',
                    'wsgi.multithread', 'wsgi.multiprocess',
                    'wsgi.run_once'):
            if key not in environ:
                warn(WSGIWarning('required environment key %r not found'
                                 % key), stacklevel=3)
        if environ['wsgi.version'] != (1, 0):
            warn(WSGIWarning('environ is not a WSGI 1.0 environ'),
                 stacklevel=3)

        script_name = environ.get('SCRIPT_NAME', '')
        if script_name and script_name[:1] != '/':
            warn(WSGIWarning('SCRIPT_NAME does not start with a slash: %r'
                             % script_name), stacklevel=3)
        path_info = environ.get('PATH_INFO', '')
        if path_info[:1] != '/':
            warn(WSGIWarning('PATH_INFO does not start with a slash: %r'
                             % path_info), stacklevel=3)

    def check_start_response(self, status, headers, exc_info):
        check_string('status', status)
        status_code = status.split(None, 1)[0]
        if len(status_code) != 3 or not status_code.isdigit():
            warn(WSGIWarning('Status code must be three digits'), stacklevel=3)
        if len(status) < 4 or status[3] != ' ':
            warn(WSGIWarning('Invalid value for status %r.  Valid '
                             'status strings are three digits, a space '
                             'and a status explanation'), stacklevel=3)
        status_code = int(status_code)
        if status_code < 100:
            warn(WSGIWarning('status code < 100 detected'), stacklevel=3)

        if type(headers) is not list:
            warn(WSGIWarning('header list is not a list'), stacklevel=3)
        for item in headers:
            if type(item) is not tuple or len(item) != 2:
                warn(WSGIWarning('Headers must tuple 2-item tuples'),
                     stacklevel=3)
            name, value = item
            if type(name) is not str or type(value) is not str:
                warn(WSGIWarning('header items must be strings'),
                     stacklevel=3)
            if name.lower() == 'status':
                warn(WSGIWarning('The status header is not supported due to '
                                 'conflicts with the CGI spec.'),
                     stacklevel=3)

        if exc_info is not None and not isinstance(exc_info, tuple):
            warn(WSGIWarning('invalid value for exc_info'), stacklevel=3)

        headers = Headers(headers)
        self.check_headers(headers)

        return status_code, headers

    def check_headers(self, headers):
        etag = headers.get('etag')
        if etag is not None:
            if etag.startswith(('W/', 'w/')):
                if etag.startswith('w/'):
                    warn(HTTPWarning('weak etag indicator should be upcase.'),
                         stacklevel=4)
                etag = etag[2:]
            if not (etag[:1] == etag[-1:] == '"'):
                warn(HTTPWarning('unquoted etag emitted.'), stacklevel=4)

        location = headers.get('location')
        if location is not None:
            if not urlparse(location).netloc:
                warn(HTTPWarning('absolute URLs required for location header'),
                     stacklevel=4)

    def check_iterator(self, app_iter):
        if isinstance(app_iter, string_types):
            warn(WSGIWarning('application returned string.  Response will '
                             'send character for character to the client '
                             'which will kill the performance.  Return a '
                             'list or iterable instead.'), stacklevel=3)

    def __call__(self, *args, **kwargs):
        if len(args) != 2:
            warn(WSGIWarning('Two arguments to WSGI app required'), stacklevel=2)
        if kwargs:
            warn(WSGIWarning('No keyword arguments to WSGI app allowed'),
                 stacklevel=2)
        environ, start_response = args

        self.check_environ(environ)
        environ['wsgi.input'] = InputStream(environ['wsgi.input'])
        environ['wsgi.errors'] = ErrorStream(environ['wsgi.errors'])

        # hook our own file wrapper in so that applications will always
        # iterate to the end and we can check the content length
        environ['wsgi.file_wrapper'] = FileWrapper

        headers_set = []
        chunks = []

        def checking_start_response(*args, **kwargs):
            if len(args) not in (2, 3):
                warn(WSGIWarning('Invalid number of arguments: %s, expected '
                                 '2 or 3' % len(args), stacklevel=2))
            if kwargs:
                warn(WSGIWarning('no keyword arguments allowed.'))

            status, headers = args[:2]
            if len(args) == 3:
                exc_info = args[2]
            else:
                exc_info = None

            headers_set[:] = self.check_start_response(status, headers,
                                                       exc_info)
            return GuardedWrite(start_response(status, headers, exc_info),
                                chunks)

        app_iter = self.app(environ, checking_start_response)
        self.check_iterator(app_iter)
        return GuardedIterator(app_iter, headers_set, chunks)
