from __future__ import absolute_import
import errno
import logging
import sys
import warnings

from socket import error as SocketError, timeout as SocketTimeout
import socket

try:  # Python 3
    from queue import LifoQueue, Empty, Full
except ImportError:
    from Queue import LifoQueue, Empty, Full
    # Queue is imported for side effects on MS Windows
    import Queue as _unused_module_Queue  # noqa: unused


from .exceptions import (
    ClosedPoolError,
    ProtocolError,
    EmptyPoolError,
    HeaderParsingError,
    HostChangedError,
    LocationValueError,
    MaxRetryError,
    ProxyError,
    ReadTimeoutError,
    SSLError,
    TimeoutError,
    InsecureRequestWarning,
    NewConnectionError,
)
from .packages.ssl_match_hostname import CertificateError
from .packages import six
from .connection import (
    port_by_scheme,
    DummyConnection,
    HTTPConnection, HTTPSConnection, VerifiedHTTPSConnection,
    HTTPException, BaseSSLError,
)
from .request import RequestMethods
from .response import HTTPResponse

from .util.connection import is_connection_dropped
from .util.response import assert_header_parsing
from .util.retry import Retry
from .util.timeout import Timeout
from .util.url import get_host, Url


xrange = six.moves.xrange

log = logging.getLogger(__name__)

_Default = object()


# Pool objects
class ConnectionPool(object):
    """
    Base class for all connection pools, such as
    :class:`.HTTPConnectionPool` and :class:`.HTTPSConnectionPool`.
    """

    scheme = None
    QueueCls = LifoQueue

    def __init__(self, host, port=None):
        if not host:
            raise LocationValueError("No host specified.")

        # httplib doesn't like it when we include brackets in ipv6 addresses
        # Specifically, if we include brackets but also pass the port then
        # httplib crazily doubles up the square brackets on the Host header.
        # Instead, we need to make sure we never pass ``None`` as the port.
        # However, for backward compatibility reasons we can't actually
        # *assert* that.
        self.host = host.strip('[]')
        self.port = port

    def __str__(self):
        return '%s(host=%r, port=%r)' % (type(self).__name__,
                                         self.host, self.port)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        # Return False to re-raise any potential exceptions
        return False

    def close(self):
        """
        Close all pooled connections and disable the pool.
        """
        pass


# This is taken from http://hg.python.org/cpython/file/7aaba721ebc0/Lib/socket.py#l252
_blocking_errnos = set([errno.EAGAIN, errno.EWOULDBLOCK])


class HTTPConnectionPool(ConnectionPool, RequestMethods):
    """
    Thread-safe connection pool for one host.

    :param host:
        Host used for this HTTP Connection (e.g. "localhost"), passed into
        :class:`httplib.HTTPConnection`.

    :param port:
        Port used for this HTTP Connection (None is equivalent to 80), passed
        into :class:`httplib.HTTPConnection`.

    :param strict:
        Causes BadStatusLine to be raised if the status line can't be parsed
        as a valid HTTP/1.0 or 1.1 status line, passed into
        :class:`httplib.HTTPConnection`.

        .. note::
           Only works in Python 2. This parameter is ignored in Python 3.

    :param timeout:
        Socket timeout in seconds for each individual connection. This can
        be a float or integer, which sets the timeout for the HTTP request,
        or an instance of :class:`urllib3.util.Timeout` which gives you more
        fine-grained control over request timeouts. After the constructor has
        been parsed, this is always a `urllib3.util.Timeout` object.

    :param maxsize:
        Number of connections to save that can be reused. More than 1 is useful
        in multithreaded situations. If ``block`` is set to False, more
        connections will be created but they will not be saved once they've
        been used.

    :param block:
        If set to True, no more than ``maxsize`` connections will be used at
        a time. When no free connections are available, the call will block
        until a connection has been released. This is a useful side effect for
        particular multithreaded situations where one does not want to use more
        than maxsize connections per host to prevent flooding.

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.

    :param retries:
        Retry configuration to use by default with requests in this pool.

    :param _proxy:
        Parsed proxy URL, should not be used directly, instead, see
        :class:`urllib3.connectionpool.ProxyManager`"

    :param _proxy_headers:
        A dictionary with proxy headers, should not be used directly,
        instead, see :class:`urllib3.connectionpool.ProxyManager`"

    :param \**conn_kw:
        Additional parameters are used to create fresh :class:`urllib3.connection.HTTPConnection`,
        :class:`urllib3.connection.HTTPSConnection` instances.
    """

    scheme = 'http'
    ConnectionCls = HTTPConnection
    ResponseCls = HTTPResponse

    def __init__(self, host, port=None, strict=False,
                 timeout=Timeout.DEFAULT_TIMEOUT, maxsize=1, block=False,
                 headers=None, retries=None,
                 _proxy=None, _proxy_headers=None,
                 **conn_kw):
        ConnectionPool.__init__(self, host, port)
        RequestMethods.__init__(self, headers)

        self.strict = strict

        if not isinstance(timeout, Timeout):
            timeout = Timeout.from_float(timeout)

        if retries is None:
            retries = Retry.DEFAULT

        self.timeout = timeout
        self.retries = retries

        self.pool = self.QueueCls(maxsize)
        self.block = block

        self.proxy = _proxy
        self.proxy_headers = _proxy_headers or {}

        # Fill the queue up so that doing get() on it will block properly
        for _ in xrange(maxsize):
            self.pool.put(None)

        # These are mostly for testing and debugging purposes.
        self.num_connections = 0
        self.num_requests = 0
        self.conn_kw = conn_kw

        if self.proxy:
            # Enable Nagle's algorithm for proxies, to avoid packet fragmentation.
            # We cannot know if the user has added default socket options, so we cannot replace the
            # list.
            self.conn_kw.setdefault('socket_options', [])

    def _new_conn(self):
        """
        Return a fresh :class:`HTTPConnection`.
        """
        self.num_connections += 1
        log.info("Starting new HTTP connection (%d): %s",
                 self.num_connections, self.host)

        conn = self.ConnectionCls(host=self.host, port=self.port,
                                  timeout=self.timeout.connect_timeout,
                                  strict=self.strict, **self.conn_kw)
        return conn

    def _get_conn(self, timeout=None):
        """
        Get a connection. Will return a pooled connection if one is available.

        If no connections are available and :prop:`.block` is ``False``, then a
        fresh connection is returned.

        :param timeout:
            Seconds to wait before giving up and raising
            :class:`urllib3.exceptions.EmptyPoolError` if the pool is empty and
            :prop:`.block` is ``True``.
        """
        conn = None
        try:
            conn = self.pool.get(block=self.block, timeout=timeout)

        except AttributeError:  # self.pool is None
            raise ClosedPoolError(self, "Pool is closed.")

        except Empty:
            if self.block:
                raise EmptyPoolError(self,
                                     "Pool reached maximum size and no more "
                                     "connections are allowed.")
            pass  # Oh well, we'll create a new connection then

        # If this is a persistent connection, check if it got disconnected
        if conn and is_connection_dropped(conn):
            log.info("Resetting dropped connection: %s", self.host)
            conn.close()
            if getattr(conn, 'auto_open', 1) == 0:
                # This is a proxied connection that has been mutated by
                # httplib._tunnel() and cannot be reused (since it would
                # attempt to bypass the proxy)
                conn = None

        return conn or self._new_conn()

    def _put_conn(self, conn):
        """
        Put a connection back into the pool.

        :param conn:
            Connection object for the current host and port as returned by
            :meth:`._new_conn` or :meth:`._get_conn`.

        If the pool is already full, the connection is closed and discarded
        because we exceeded maxsize. If connections are discarded frequently,
        then maxsize should be increased.

        If the pool is closed, then the connection will be closed and discarded.
        """
        try:
            self.pool.put(conn, block=False)
            return  # Everything is dandy, done.
        except AttributeError:
            # self.pool is None.
            pass
        except Full:
            # This should never happen if self.block == True
            log.warning(
                "Connection pool is full, discarding connection: %s",
                self.host)

        # Connection never got put back into the pool, close it.
        if conn:
            conn.close()

    def _validate_conn(self, conn):
        """
        Called right before a request is made, after the socket is created.
        """
        pass

    def _prepare_proxy(self, conn):
        # Nothing to do for HTTP connections.
        pass

    def _get_timeout(self, timeout):
        """ Helper that always returns a :class:`urllib3.util.Timeout` """
        if timeout is _Default:
            return self.timeout.clone()

        if isinstance(timeout, Timeout):
            return timeout.clone()
        else:
            # User passed us an int/float. This is for backwards compatibility,
            # can be removed later
            return Timeout.from_float(timeout)

    def _raise_timeout(self, err, url, timeout_value):
        """Is the error actually a timeout? Will raise a ReadTimeout or pass"""

        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(self, url, "Read timed out. (read timeout=%s)" % timeout_value)

        # See the above comment about EAGAIN in Python 3. In Python 2 we have
        # to specifically catch it and throw the timeout error
        if hasattr(err, 'errno') and err.errno in _blocking_errnos:
            raise ReadTimeoutError(self, url, "Read timed out. (read timeout=%s)" % timeout_value)

        # Catch possible read timeouts thrown as SSL errors. If not the
        # case, rethrow the original. We need to do this because of:
        # http://bugs.python.org/issue10272
        if 'timed out' in str(err) or 'did not complete (read)' in str(err):  # Python 2.6
            raise ReadTimeoutError(self, url, "Read timed out. (read timeout=%s)" % timeout_value)

    def _make_request(self, conn, method, url, timeout=_Default, chunked=False,
                      **httplib_request_kw):
        """
        Perform a request on a given urllib connection object taken from our
        pool.

        :param conn:
            a connection from one of our connection pools

        :param timeout:
            Socket timeout in seconds for the request. This can be a
            float or integer, which will set the same timeout value for
            the socket connect and the socket read, or an instance of
            :class:`urllib3.util.Timeout`, which gives you more fine-grained
            control over your timeouts.
        """
        self.num_requests += 1

        timeout_obj = self._get_timeout(timeout)
        timeout_obj.start_connect()
        conn.timeout = timeout_obj.connect_timeout

        # Trigger any extra validation we need to do.
        try:
            self._validate_conn(conn)
        except (SocketTimeout, BaseSSLError) as e:
            # Py2 raises this as a BaseSSLError, Py3 raises it as socket timeout.
            self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)
            raise

        # conn.request() calls httplib.*.request, not the method in
        # urllib3.request. It also calls makefile (recv) on the socket.
        if chunked:
            conn.request_chunked(method, url, **httplib_request_kw)
        else:
            conn.request(method, url, **httplib_request_kw)

        # Reset the timeout for the recv() on the socket
        read_timeout = timeout_obj.read_timeout

        # App Engine doesn't have a sock attr
        if getattr(conn, 'sock', None):
            # In Python 3 socket.py will catch EAGAIN and return None when you
            # try and read into the file pointer created by http.client, which
            # instead raises a BadStatusLine exception. Instead of catching
            # the exception and assuming all BadStatusLine exceptions are read
            # timeouts, check for a zero timeout before making the request.
            if read_timeout == 0:
                raise ReadTimeoutError(
                    self, url, "Read timed out. (read timeout=%s)" % read_timeout)
            if read_timeout is Timeout.DEFAULT_TIMEOUT:
                conn.sock.settimeout(socket.getdefaulttimeout())
            else:  # None or a value
                conn.sock.settimeout(read_timeout)

        # Receive the response from the server
        try:
            try:  # Python 2.7, use buffering of HTTP responses
                httplib_response = conn.getresponse(buffering=True)
            except TypeError:  # Python 2.6 and older, Python 3
                try:
                    httplib_response = conn.getresponse()
                except Exception as e:
                    # Remove the TypeError from the exception chain in Python 3;
                    # otherwise it looks like a programming error was the cause.
                    six.raise_from(e, None)
        except (SocketTimeout, BaseSSLError, SocketError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
            raise

        # AppEngine doesn't have a version attr.
        http_version = getattr(conn, '_http_vsn_str', 'HTTP/?')
        log.debug("\"%s %s %s\" %s %s", method, url, http_version,
                  httplib_response.status, httplib_response.length)

        try:
            assert_header_parsing(httplib_response.msg)
        except HeaderParsingError as hpe:  # Platform-specific: Python 3
            log.warning(
                'Failed to parse headers (url=%s): %s',
                self._absolute_url(url), hpe, exc_info=True)

        return httplib_response

    def _absolute_url(self, path):
        return Url(scheme=self.scheme, host=self.host, port=self.port, path=path).url

    def close(self):
        """
        Close all pooled connections and disable the pool.
        """
        # Disable access to the pool
        old_pool, self.pool = self.pool, None

        try:
            while True:
                conn = old_pool.get(block=False)
                if conn:
                    conn.close()

        except Empty:
            pass  # Done.

    def is_same_host(self, url):
        """
        Check if the given ``url`` is a member of the same host as this
        connection pool.
        """
        if url.startswith('/'):
            return True

        # TODO: Add optional support for socket.gethostbyname checking.
        scheme, host, port = get_host(url)

        # Use explicit default port for comparison when none is given
        if self.port and not port:
            port = port_by_scheme.get(scheme)
        elif not self.port and port == port_by_scheme.get(scheme):
            port = None

        return (scheme, host, port) == (self.scheme, self.host, self.port)

    def urlopen(self, method, url, body=None, headers=None, retries=None,
                redirect=True, assert_same_host=True, timeout=_Default,
                pool_timeout=None, release_conn=None, chunked=False,
                **response_kw):
        """
        Get a connection from the pool and perform an HTTP request. This is the
        lowest level call for making a request, so you'll need to specify all
        the raw details.

        .. note::

           More commonly, it's appropriate to use a convenience method provided
           by :class:`.RequestMethods`, such as :meth:`request`.

        .. note::

           `release_conn` will only behave as expected if
           `preload_content=False` because we want to make
           `preload_content=False` the default behaviour someday soon without
           breaking backwards compatibility.

        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)

        :param body:
            Data to send in the request body (useful for creating
            POST requests, see HTTPConnectionPool.post_url for
            more convenience).

        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.

        :param retries:
            Configure the number of retries to allow before raising a
            :class:`~urllib3.exceptions.MaxRetryError` exception.

            Pass ``None`` to retry until you receive a response. Pass a
            :class:`~urllib3.util.retry.Retry` object for fine-grained control
            over different types of retries.
            Pass an integer number to retry connection errors that many times,
            but no other types of errors. Pass zero to never retry.

            If ``False``, then retries are disabled and any exception is raised
            immediately. Also, instead of raising a MaxRetryError on redirects,
            the redirect response will be returned.

        :type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.

        :param redirect:
            If True, automatically handle redirects (status codes 301, 302,
            303, 307, 308). Each redirect counts as a retry. Disabling retries
            will disable redirect, too.

        :param assert_same_host:
            If ``True``, will make sure that the host of the pool requests is
            consistent else will raise HostChangedError. When False, you can
            use the pool on an HTTP proxy and request foreign hosts.

        :param timeout:
            If specified, overrides the default timeout for this one
            request. It may be a float (in seconds) or an instance of
            :class:`urllib3.util.Timeout`.

        :param pool_timeout:
            If set and the pool is set to block=True, then this method will
            block for ``pool_timeout`` seconds and raise EmptyPoolError if no
            connection is available within the time period.

        :param release_conn:
            If False, then the urlopen call will not release the connection
            back into the pool once a response is received (but will release if
            you read the entire contents of the response such as when
            `preload_content=True`). This is useful if you're not preloading
            the response's content immediately. You will need to call
            ``r.release_conn()`` on the response ``r`` to return the connection
            back into the pool. If None, it takes the value of
            ``response_kw.get('preload_content', True)``.

        :param chunked:
            If True, urllib3 will send the body using chunked transfer
            encoding. Otherwise, urllib3 will send the body using the standard
            content-length form. Defaults to False.

        :param \**response_kw:
            Additional parameters are passed to
            :meth:`urllib3.response.HTTPResponse.from_httplib`
        """
        if headers is None:
            headers = self.headers

        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)

        if release_conn is None:
            release_conn = response_kw.get('preload_content', True)

        # Check host
        if assert_same_host and not self.is_same_host(url):
            raise HostChangedError(self, url, retries)

        conn = None

        # Track whether `conn` needs to be released before
        # returning/raising/recursing. Update this variable if necessary, and
        # leave `release_conn` constant throughout the function. That way, if
        # the function recurses, the original value of `release_conn` will be
        # passed down into the recursive call, and its value will be respected.
        #
        # See issue #651 [1] for details.
        #
        # [1] <https://github.com/shazow/urllib3/issues/651>
        release_this_conn = release_conn

        # Merge the proxy headers. Only do this in HTTP. We have to copy the
        # headers dict so we can safely change it without those changes being
        # reflected in anyone else's copy.
        if self.scheme == 'http':
            headers = headers.copy()
            headers.update(self.proxy_headers)

        # Must keep the exception bound to a separate variable or else Python 3
        # complains about UnboundLocalError.
        err = None

        # Keep track of whether we cleanly exited the except block. This
        # ensures we do proper cleanup in finally.
        clean_exit = False

        try:
            # Request a connection from the queue.
            timeout_obj = self._get_timeout(timeout)
            conn = self._get_conn(timeout=pool_timeout)

            conn.timeout = timeout_obj.connect_timeout

            is_new_proxy_conn = self.proxy is not None and not getattr(conn, 'sock', None)
            if is_new_proxy_conn:
                self._prepare_proxy(conn)

            # Make the request on the httplib connection object.
            httplib_response = self._make_request(conn, method, url,
                                                  timeout=timeout_obj,
                                                  body=body, headers=headers,
                                                  chunked=chunked)

            # If we're going to release the connection in ``finally:``, then
            # the response doesn't need to know about the connection. Otherwise
            # it will also try to release it and we'll have a double-release
            # mess.
            response_conn = conn if not release_conn else None

            # Import httplib's response into our own wrapper object
            response = self.ResponseCls.from_httplib(httplib_response,
                                                     pool=self,
                                                     connection=response_conn,
                                                     **response_kw)

            # Everything went great!
            clean_exit = True

        except Empty:
            # Timed out by queue.
            raise EmptyPoolError(self, "No pool connections are available.")

        except (BaseSSLError, CertificateError) as e:
            # Close the connection. If a connection is reused on which there
            # was a Certificate error, the next request will certainly raise
            # another Certificate error.
            clean_exit = False
            raise SSLError(e)

        except SSLError:
            # Treat SSLError separately from BaseSSLError to preserve
            # traceback.
            clean_exit = False
            raise

        except (TimeoutError, HTTPException, SocketError, ProtocolError) as e:
            # Discard the connection for these exceptions. It will be
            # be replaced during the next _get_conn() call.
            clean_exit = False

            if isinstance(e, (SocketError, NewConnectionError)) and self.proxy:
                e = ProxyError('Cannot connect to proxy.', e)
            elif isinstance(e, (SocketError, HTTPException)):
                e = ProtocolError('Connection aborted.', e)

            retries = retries.increment(method, url, error=e, _pool=self,
                                        _stacktrace=sys.exc_info()[2])
            retries.sleep()

            # Keep track of the error for the retry warning.
            err = e

        finally:
            if not clean_exit:
                # We hit some kind of exception, handled or otherwise. We need
                # to throw the connection away unless explicitly told not to.
                # Close the connection, set the variable to None, and make sure
                # we put the None back in the pool to avoid leaking it.
                conn = conn and conn.close()
                release_this_conn = True

            if release_this_conn:
                # Put the connection back to be reused. If the connection is
                # expired then it will be None, which will get replaced with a
                # fresh connection during _get_conn.
                self._put_conn(conn)

        if not conn:
            # Try again
            log.warning("Retrying (%r) after connection "
                        "broken by '%r': %s", retries, err, url)
            return self.urlopen(method, url, body, headers, retries,
                                redirect, assert_same_host,
                                timeout=timeout, pool_timeout=pool_timeout,
                                release_conn=release_conn, **response_kw)

        # Handle redirect?
        redirect_location = redirect and response.get_redirect_location()
        if redirect_location:
            if response.status == 303:
                method = 'GET'

            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_redirect:
                    # Release the connection for this response, since we're not
                    # returning it to be released manually.
                    response.release_conn()
                    raise
                return response

            log.info("Redirecting %s -> %s", url, redirect_location)
            return self.urlopen(
                method, redirect_location, body, headers,
                retries=retries, redirect=redirect,
                assert_same_host=assert_same_host,
                timeout=timeout, pool_timeout=pool_timeout,
                release_conn=release_conn, **response_kw)

        # Check if we should retry the HTTP response.
        if retries.is_forced_retry(method, status_code=response.status):
            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_status:
                    # Release the connection for this response, since we're not
                    # returning it to be released manually.
                    response.release_conn()
                    raise
                return response
            retries.sleep()
            log.info("Forced retry: %s", url)
            return self.urlopen(
                method, url, body, headers,
                retries=retries, redirect=redirect,
                assert_same_host=assert_same_host,
                timeout=timeout, pool_timeout=pool_timeout,
                release_conn=release_conn, **response_kw)

        return response


class HTTPSConnectionPool(HTTPConnectionPool):
    """
    Same as :class:`.HTTPConnectionPool`, but HTTPS.

    When Python is compiled with the :mod:`ssl` module, then
    :class:`.VerifiedHTTPSConnection` is used, which *can* verify certificates,
    instead of :class:`.HTTPSConnection`.

    :class:`.VerifiedHTTPSConnection` uses one of ``assert_fingerprint``,
    ``assert_hostname`` and ``host`` in this order to verify connections.
    If ``assert_hostname`` is False, no verification is done.

    The ``key_file``, ``cert_file``, ``cert_reqs``, ``ca_certs``,
    ``ca_cert_dir``, and ``ssl_version`` are only used if :mod:`ssl` is
    available and are fed into :meth:`urllib3.util.ssl_wrap_socket` to upgrade
    the connection socket into an SSL socket.
    """

    scheme = 'https'
    ConnectionCls = HTTPSConnection

    def __init__(self, host, port=None,
                 strict=False, timeout=Timeout.DEFAULT_TIMEOUT, maxsize=1,
                 block=False, headers=None, retries=None,
                 _proxy=None, _proxy_headers=None,
                 key_file=None, cert_file=None, cert_reqs=None,
                 ca_certs=None, ssl_version=None,
                 assert_hostname=None, assert_fingerprint=None,
                 ca_cert_dir=None, **conn_kw):

        HTTPConnectionPool.__init__(self, host, port, strict, timeout, maxsize,
                                    block, headers, retries, _proxy, _proxy_headers,
                                    **conn_kw)

        if ca_certs and cert_reqs is None:
            cert_reqs = 'CERT_REQUIRED'

        self.key_file = key_file
        self.cert_file = cert_file
        self.cert_reqs = cert_reqs
        self.ca_certs = ca_certs
        self.ca_cert_dir = ca_cert_dir
        self.ssl_version = ssl_version
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint

    def _prepare_conn(self, conn):
        """
        Prepare the ``connection`` for :meth:`urllib3.util.ssl_wrap_socket`
        and establish the tunnel if proxy is used.
        """

        if isinstance(conn, VerifiedHTTPSConnection):
            conn.set_cert(key_file=self.key_file,
                          cert_file=self.cert_file,
                          cert_reqs=self.cert_reqs,
                          ca_certs=self.ca_certs,
                          ca_cert_dir=self.ca_cert_dir,
                          assert_hostname=self.assert_hostname,
                          assert_fingerprint=self.assert_fingerprint)
            conn.ssl_version = self.ssl_version

        return conn

    def _prepare_proxy(self, conn):
        """
        Establish tunnel connection early, because otherwise httplib
        would improperly set Host: header to proxy's IP:port.
        """
        # Python 2.7+
        try:
            set_tunnel = conn.set_tunnel
        except AttributeError:  # Platform-specific: Python 2.6
            set_tunnel = conn._set_tunnel

        if sys.version_info <= (2, 6, 4) and not self.proxy_headers:  # Python 2.6.4 and older
            set_tunnel(self.host, self.port)
        else:
            set_tunnel(self.host, self.port, self.proxy_headers)

        conn.connect()

    def _new_conn(self):
        """
        Return a fresh :class:`httplib.HTTPSConnection`.
        """
        self.num_connections += 1
        log.info("Starting new HTTPS connection (%d): %s",
                 self.num_connections, self.host)

        if not self.ConnectionCls or self.ConnectionCls is DummyConnection:
            raise SSLError("Can't connect to HTTPS URL because the SSL "
                           "module is not available.")

        actual_host = self.host
        actual_port = self.port
        if self.proxy is not None:
            actual_host = self.proxy.host
            actual_port = self.proxy.port

        conn = self.ConnectionCls(host=actual_host, port=actual_port,
                                  timeout=self.timeout.connect_timeout,
                                  strict=self.strict, **self.conn_kw)

        return self._prepare_conn(conn)

    def _validate_conn(self, conn):
        """
        Called right before a request is made, after the socket is created.
        """
        super(HTTPSConnectionPool, self)._validate_conn(conn)

        # Force connect early to allow us to validate the connection.
        if not getattr(conn, 'sock', None):  # AppEngine might not have  `.sock`
            conn.connect()

        if not conn.is_verified:
            warnings.warn((
                'Unverified HTTPS request is being made. '
                'Adding certificate verification is strongly advised. See: '
                'https://urllib3.readthedocs.io/en/latest/security.html'),
                InsecureRequestWarning)


def connection_from_url(url, **kw):
    """
    Given a url, return an :class:`.ConnectionPool` instance of its host.

    This is a shortcut for not having to parse out the scheme, host, and port
    of the url before creating an :class:`.ConnectionPool` instance.

    :param url:
        Absolute URL string that must include the scheme. Port is optional.

    :param \**kw:
        Passes additional parameters to the constructor of the appropriate
        :class:`.ConnectionPool`. Useful for specifying things like
        timeout, maxsize, headers, etc.

    Example::

        >>> conn = connection_from_url('http://google.com/')
        >>> r = conn.request('GET', '/')
    """
    scheme, host, port = get_host(url)
    port = port or port_by_scheme.get(scheme, 80)
    if scheme == 'https':
        return HTTPSConnectionPool(host, port=port, **kw)
    else:
        return HTTPConnectionPool(host, port=port, **kw)
