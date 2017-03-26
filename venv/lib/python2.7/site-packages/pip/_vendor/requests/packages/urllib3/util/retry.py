from __future__ import absolute_import
import time
import logging

from ..exceptions import (
    ConnectTimeoutError,
    MaxRetryError,
    ProtocolError,
    ReadTimeoutError,
    ResponseError,
)
from ..packages import six


log = logging.getLogger(__name__)


class Retry(object):
    """ Retry configuration.

    Each retry attempt will create a new Retry object with updated values, so
    they can be safely reused.

    Retries can be defined as a default for a pool::

        retries = Retry(connect=5, read=2, redirect=5)
        http = PoolManager(retries=retries)
        response = http.request('GET', 'http://example.com/')

    Or per-request (which overrides the default for the pool)::

        response = http.request('GET', 'http://example.com/', retries=Retry(10))

    Retries can be disabled by passing ``False``::

        response = http.request('GET', 'http://example.com/', retries=False)

    Errors will be wrapped in :class:`~urllib3.exceptions.MaxRetryError` unless
    retries are disabled, in which case the causing exception will be raised.

    :param int total:
        Total number of retries to allow. Takes precedence over other counts.

        Set to ``None`` to remove this constraint and fall back on other
        counts. It's a good idea to set this to some sensibly-high value to
        account for unexpected edge cases and avoid infinite retry loops.

        Set to ``0`` to fail on the first retry.

        Set to ``False`` to disable and imply ``raise_on_redirect=False``.

    :param int connect:
        How many connection-related errors to retry on.

        These are errors raised before the request is sent to the remote server,
        which we assume has not triggered the server to process the request.

        Set to ``0`` to fail on the first retry of this type.

    :param int read:
        How many times to retry on read errors.

        These errors are raised after the request was sent to the server, so the
        request may have side-effects.

        Set to ``0`` to fail on the first retry of this type.

    :param int redirect:
        How many redirects to perform. Limit this to avoid infinite redirect
        loops.

        A redirect is a HTTP response with a status code 301, 302, 303, 307 or
        308.

        Set to ``0`` to fail on the first retry of this type.

        Set to ``False`` to disable and imply ``raise_on_redirect=False``.

    :param iterable method_whitelist:
        Set of uppercased HTTP method verbs that we should retry on.

        By default, we only retry on methods which are considered to be
        idempotent (multiple requests with the same parameters end with the
        same state). See :attr:`Retry.DEFAULT_METHOD_WHITELIST`.

        Set to a ``False`` value to retry on any verb.

    :param iterable status_forcelist:
        A set of integer HTTP status codes that we should force a retry on.
        A retry is initiated if the request method is in ``method_whitelist``
        and the response status code is in ``status_forcelist``.

        By default, this is disabled with ``None``.

    :param float backoff_factor:
        A backoff factor to apply between attempts after the second try
        (most errors are resolved immediately by a second try without a
        delay). urllib3 will sleep for::

            {backoff factor} * (2 ^ ({number of total retries} - 1))

        seconds. If the backoff_factor is 0.1, then :func:`.sleep` will sleep
        for [0.0s, 0.2s, 0.4s, ...] between retries. It will never be longer
        than :attr:`Retry.BACKOFF_MAX`.

        By default, backoff is disabled (set to 0).

    :param bool raise_on_redirect: Whether, if the number of redirects is
        exhausted, to raise a MaxRetryError, or to return a response with a
        response code in the 3xx range.

    :param bool raise_on_status: Similar meaning to ``raise_on_redirect``:
        whether we should raise an exception, or return a response,
        if status falls in ``status_forcelist`` range and retries have
        been exhausted.
    """

    DEFAULT_METHOD_WHITELIST = frozenset([
        'HEAD', 'GET', 'PUT', 'DELETE', 'OPTIONS', 'TRACE'])

    #: Maximum backoff time.
    BACKOFF_MAX = 120

    def __init__(self, total=10, connect=None, read=None, redirect=None,
                 method_whitelist=DEFAULT_METHOD_WHITELIST, status_forcelist=None,
                 backoff_factor=0, raise_on_redirect=True, raise_on_status=True,
                 _observed_errors=0):

        self.total = total
        self.connect = connect
        self.read = read

        if redirect is False or total is False:
            redirect = 0
            raise_on_redirect = False

        self.redirect = redirect
        self.status_forcelist = status_forcelist or set()
        self.method_whitelist = method_whitelist
        self.backoff_factor = backoff_factor
        self.raise_on_redirect = raise_on_redirect
        self.raise_on_status = raise_on_status
        self._observed_errors = _observed_errors  # TODO: use .history instead?

    def new(self, **kw):
        params = dict(
            total=self.total,
            connect=self.connect, read=self.read, redirect=self.redirect,
            method_whitelist=self.method_whitelist,
            status_forcelist=self.status_forcelist,
            backoff_factor=self.backoff_factor,
            raise_on_redirect=self.raise_on_redirect,
            raise_on_status=self.raise_on_status,
            _observed_errors=self._observed_errors,
        )
        params.update(kw)
        return type(self)(**params)

    @classmethod
    def from_int(cls, retries, redirect=True, default=None):
        """ Backwards-compatibility for the old retries format."""
        if retries is None:
            retries = default if default is not None else cls.DEFAULT

        if isinstance(retries, Retry):
            return retries

        redirect = bool(redirect) and None
        new_retries = cls(retries, redirect=redirect)
        log.debug("Converted retries value: %r -> %r", retries, new_retries)
        return new_retries

    def get_backoff_time(self):
        """ Formula for computing the current backoff

        :rtype: float
        """
        if self._observed_errors <= 1:
            return 0

        backoff_value = self.backoff_factor * (2 ** (self._observed_errors - 1))
        return min(self.BACKOFF_MAX, backoff_value)

    def sleep(self):
        """ Sleep between retry attempts using an exponential backoff.

        By default, the backoff factor is 0 and this method will return
        immediately.
        """
        backoff = self.get_backoff_time()
        if backoff <= 0:
            return
        time.sleep(backoff)

    def _is_connection_error(self, err):
        """ Errors when we're fairly sure that the server did not receive the
        request, so it should be safe to retry.
        """
        return isinstance(err, ConnectTimeoutError)

    def _is_read_error(self, err):
        """ Errors that occur after the request has been started, so we should
        assume that the server began processing it.
        """
        return isinstance(err, (ReadTimeoutError, ProtocolError))

    def is_forced_retry(self, method, status_code):
        """ Is this method/status code retryable? (Based on method/codes whitelists)
        """
        if self.method_whitelist and method.upper() not in self.method_whitelist:
            return False

        return self.status_forcelist and status_code in self.status_forcelist

    def is_exhausted(self):
        """ Are we out of retries? """
        retry_counts = (self.total, self.connect, self.read, self.redirect)
        retry_counts = list(filter(None, retry_counts))
        if not retry_counts:
            return False

        return min(retry_counts) < 0

    def increment(self, method=None, url=None, response=None, error=None,
                  _pool=None, _stacktrace=None):
        """ Return a new Retry object with incremented retry counters.

        :param response: A response object, or None, if the server did not
            return a response.
        :type response: :class:`~urllib3.response.HTTPResponse`
        :param Exception error: An error encountered during the request, or
            None if the response was received successfully.

        :return: A new ``Retry`` object.
        """
        if self.total is False and error:
            # Disabled, indicate to re-raise the error.
            raise six.reraise(type(error), error, _stacktrace)

        total = self.total
        if total is not None:
            total -= 1

        _observed_errors = self._observed_errors
        connect = self.connect
        read = self.read
        redirect = self.redirect
        cause = 'unknown'

        if error and self._is_connection_error(error):
            # Connect retry?
            if connect is False:
                raise six.reraise(type(error), error, _stacktrace)
            elif connect is not None:
                connect -= 1
            _observed_errors += 1

        elif error and self._is_read_error(error):
            # Read retry?
            if read is False:
                raise six.reraise(type(error), error, _stacktrace)
            elif read is not None:
                read -= 1
            _observed_errors += 1

        elif response and response.get_redirect_location():
            # Redirect retry?
            if redirect is not None:
                redirect -= 1
            cause = 'too many redirects'

        else:
            # Incrementing because of a server error like a 500 in
            # status_forcelist and a the given method is in the whitelist
            _observed_errors += 1
            cause = ResponseError.GENERIC_ERROR
            if response and response.status:
                cause = ResponseError.SPECIFIC_ERROR.format(
                    status_code=response.status)

        new_retry = self.new(
            total=total,
            connect=connect, read=read, redirect=redirect,
            _observed_errors=_observed_errors)

        if new_retry.is_exhausted():
            raise MaxRetryError(_pool, url, error or ResponseError(cause))

        log.debug("Incremented Retry for (url='%s'): %r", url, new_retry)

        return new_retry

    def __repr__(self):
        return ('{cls.__name__}(total={self.total}, connect={self.connect}, '
                'read={self.read}, redirect={self.redirect})').format(
                    cls=type(self), self=self)


# For backwards compatibility (equivalent to pre-v1.9):
Retry.DEFAULT = Retry(3)
