from __future__ import absolute_import
# Base Exceptions


class HTTPError(Exception):
    "Base exception used by this module."
    pass


class HTTPWarning(Warning):
    "Base warning used by this module."
    pass


class PoolError(HTTPError):
    "Base exception for errors caused within a pool."
    def __init__(self, pool, message):
        self.pool = pool
        HTTPError.__init__(self, "%s: %s" % (pool, message))

    def __reduce__(self):
        # For pickling purposes.
        return self.__class__, (None, None)


class RequestError(PoolError):
    "Base exception for PoolErrors that have associated URLs."
    def __init__(self, pool, url, message):
        self.url = url
        PoolError.__init__(self, pool, message)

    def __reduce__(self):
        # For pickling purposes.
        return self.__class__, (None, self.url, None)


class SSLError(HTTPError):
    "Raised when SSL certificate fails in an HTTPS connection."
    pass


class ProxyError(HTTPError):
    "Raised when the connection to a proxy fails."
    pass


class DecodeError(HTTPError):
    "Raised when automatic decoding based on Content-Type fails."
    pass


class ProtocolError(HTTPError):
    "Raised when something unexpected happens mid-request/response."
    pass


#: Renamed to ProtocolError but aliased for backwards compatibility.
ConnectionError = ProtocolError


# Leaf Exceptions

class MaxRetryError(RequestError):
    """Raised when the maximum number of retries is exceeded.

    :param pool: The connection pool
    :type pool: :class:`~urllib3.connectionpool.HTTPConnectionPool`
    :param string url: The requested Url
    :param exceptions.Exception reason: The underlying error

    """

    def __init__(self, pool, url, reason=None):
        self.reason = reason

        message = "Max retries exceeded with url: %s (Caused by %r)" % (
            url, reason)

        RequestError.__init__(self, pool, url, message)


class HostChangedError(RequestError):
    "Raised when an existing pool gets a request for a foreign host."

    def __init__(self, pool, url, retries=3):
        message = "Tried to open a foreign host with url: %s" % url
        RequestError.__init__(self, pool, url, message)
        self.retries = retries


class TimeoutStateError(HTTPError):
    """ Raised when passing an invalid state to a timeout """
    pass


class TimeoutError(HTTPError):
    """ Raised when a socket timeout error occurs.

    Catching this error will catch both :exc:`ReadTimeoutErrors
    <ReadTimeoutError>` and :exc:`ConnectTimeoutErrors <ConnectTimeoutError>`.
    """
    pass


class ReadTimeoutError(TimeoutError, RequestError):
    "Raised when a socket timeout occurs while receiving data from a server"
    pass


# This timeout error does not have a URL attached and needs to inherit from the
# base HTTPError
class ConnectTimeoutError(TimeoutError):
    "Raised when a socket timeout occurs while connecting to a server"
    pass


class NewConnectionError(ConnectTimeoutError, PoolError):
    "Raised when we fail to establish a new connection. Usually ECONNREFUSED."
    pass


class EmptyPoolError(PoolError):
    "Raised when a pool runs out of connections and no more are allowed."
    pass


class ClosedPoolError(PoolError):
    "Raised when a request enters a pool after the pool has been closed."
    pass


class LocationValueError(ValueError, HTTPError):
    "Raised when there is something wrong with a given URL input."
    pass


class LocationParseError(LocationValueError):
    "Raised when get_host or similar fails to parse the URL input."

    def __init__(self, location):
        message = "Failed to parse: %s" % location
        HTTPError.__init__(self, message)

        self.location = location


class ResponseError(HTTPError):
    "Used as a container for an error reason supplied in a MaxRetryError."
    GENERIC_ERROR = 'too many error responses'
    SPECIFIC_ERROR = 'too many {status_code} error responses'


class SecurityWarning(HTTPWarning):
    "Warned when perfoming security reducing actions"
    pass


class SubjectAltNameWarning(SecurityWarning):
    "Warned when connecting to a host with a certificate missing a SAN."
    pass


class InsecureRequestWarning(SecurityWarning):
    "Warned when making an unverified HTTPS request."
    pass


class SystemTimeWarning(SecurityWarning):
    "Warned when system time is suspected to be wrong"
    pass


class InsecurePlatformWarning(SecurityWarning):
    "Warned when certain SSL configuration is not available on a platform."
    pass


class SNIMissingWarning(HTTPWarning):
    "Warned when making a HTTPS request without SNI available."
    pass


class DependencyWarning(HTTPWarning):
    """
    Warned when an attempt is made to import a module with missing optional
    dependencies.
    """
    pass


class ResponseNotChunked(ProtocolError, ValueError):
    "Response needs to be chunked in order to read it as chunks."
    pass


class ProxySchemeUnknown(AssertionError, ValueError):
    "ProxyManager does not support the supplied scheme"
    # TODO(t-8ch): Stop inheriting from AssertionError in v2.0.

    def __init__(self, scheme):
        message = "Not supported proxy scheme %s" % scheme
        super(ProxySchemeUnknown, self).__init__(message)


class HeaderParsingError(HTTPError):
    "Raised by assert_header_parsing, but we convert it to a log.warning statement."
    def __init__(self, defects, unparsed_data):
        message = '%s, unparsed data: %r' % (defects or 'Unknown', unparsed_data)
        super(HeaderParsingError, self).__init__(message)
