from __future__ import absolute_import
import logging
import os
import warnings

from ..exceptions import (
    HTTPError,
    HTTPWarning,
    MaxRetryError,
    ProtocolError,
    TimeoutError,
    SSLError
)

from ..packages.six import BytesIO
from ..request import RequestMethods
from ..response import HTTPResponse
from ..util.timeout import Timeout
from ..util.retry import Retry

try:
    from google.appengine.api import urlfetch
except ImportError:
    urlfetch = None


log = logging.getLogger(__name__)


class AppEnginePlatformWarning(HTTPWarning):
    pass


class AppEnginePlatformError(HTTPError):
    pass


class AppEngineManager(RequestMethods):
    """
    Connection manager for Google App Engine sandbox applications.

    This manager uses the URLFetch service directly instead of using the
    emulated httplib, and is subject to URLFetch limitations as described in
    the App Engine documentation here:

        https://cloud.google.com/appengine/docs/python/urlfetch

    Notably it will raise an AppEnginePlatformError if:
        * URLFetch is not available.
        * If you attempt to use this on GAEv2 (Managed VMs), as full socket
          support is available.
        * If a request size is more than 10 megabytes.
        * If a response size is more than 32 megabtyes.
        * If you use an unsupported request method such as OPTIONS.

    Beyond those cases, it will raise normal urllib3 errors.
    """

    def __init__(self, headers=None, retries=None, validate_certificate=True):
        if not urlfetch:
            raise AppEnginePlatformError(
                "URLFetch is not available in this environment.")

        if is_prod_appengine_mvms():
            raise AppEnginePlatformError(
                "Use normal urllib3.PoolManager instead of AppEngineManager"
                "on Managed VMs, as using URLFetch is not necessary in "
                "this environment.")

        warnings.warn(
            "urllib3 is using URLFetch on Google App Engine sandbox instead "
            "of sockets. To use sockets directly instead of URLFetch see "
            "https://urllib3.readthedocs.io/en/latest/contrib.html.",
            AppEnginePlatformWarning)

        RequestMethods.__init__(self, headers)
        self.validate_certificate = validate_certificate

        self.retries = retries or Retry.DEFAULT

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Return False to re-raise any potential exceptions
        return False

    def urlopen(self, method, url, body=None, headers=None,
                retries=None, redirect=True, timeout=Timeout.DEFAULT_TIMEOUT,
                **response_kw):

        retries = self._get_retries(retries, redirect)

        try:
            response = urlfetch.fetch(
                url,
                payload=body,
                method=method,
                headers=headers or {},
                allow_truncated=False,
                follow_redirects=(
                    redirect and
                    retries.redirect != 0 and
                    retries.total),
                deadline=self._get_absolute_timeout(timeout),
                validate_certificate=self.validate_certificate,
            )
        except urlfetch.DeadlineExceededError as e:
            raise TimeoutError(self, e)

        except urlfetch.InvalidURLError as e:
            if 'too large' in str(e):
                raise AppEnginePlatformError(
                    "URLFetch request too large, URLFetch only "
                    "supports requests up to 10mb in size.", e)
            raise ProtocolError(e)

        except urlfetch.DownloadError as e:
            if 'Too many redirects' in str(e):
                raise MaxRetryError(self, url, reason=e)
            raise ProtocolError(e)

        except urlfetch.ResponseTooLargeError as e:
            raise AppEnginePlatformError(
                "URLFetch response too large, URLFetch only supports"
                "responses up to 32mb in size.", e)

        except urlfetch.SSLCertificateError as e:
            raise SSLError(e)

        except urlfetch.InvalidMethodError as e:
            raise AppEnginePlatformError(
                "URLFetch does not support method: %s" % method, e)

        http_response = self._urlfetch_response_to_http_response(
            response, **response_kw)

        # Check for redirect response
        if (http_response.get_redirect_location() and
                retries.raise_on_redirect and redirect):
            raise MaxRetryError(self, url, "too many redirects")

        # Check if we should retry the HTTP response.
        if retries.is_forced_retry(method, status_code=http_response.status):
            retries = retries.increment(
                method, url, response=http_response, _pool=self)
            log.info("Forced retry: %s", url)
            retries.sleep()
            return self.urlopen(
                method, url,
                body=body, headers=headers,
                retries=retries, redirect=redirect,
                timeout=timeout, **response_kw)

        return http_response

    def _urlfetch_response_to_http_response(self, urlfetch_resp, **response_kw):

        if is_prod_appengine():
            # Production GAE handles deflate encoding automatically, but does
            # not remove the encoding header.
            content_encoding = urlfetch_resp.headers.get('content-encoding')

            if content_encoding == 'deflate':
                del urlfetch_resp.headers['content-encoding']

        transfer_encoding = urlfetch_resp.headers.get('transfer-encoding')
        # We have a full response's content,
        # so let's make sure we don't report ourselves as chunked data.
        if transfer_encoding == 'chunked':
            encodings = transfer_encoding.split(",")
            encodings.remove('chunked')
            urlfetch_resp.headers['transfer-encoding'] = ','.join(encodings)

        return HTTPResponse(
            # In order for decoding to work, we must present the content as
            # a file-like object.
            body=BytesIO(urlfetch_resp.content),
            headers=urlfetch_resp.headers,
            status=urlfetch_resp.status_code,
            **response_kw
        )

    def _get_absolute_timeout(self, timeout):
        if timeout is Timeout.DEFAULT_TIMEOUT:
            return 5  # 5s is the default timeout for URLFetch.
        if isinstance(timeout, Timeout):
            if timeout._read is not timeout._connect:
                warnings.warn(
                    "URLFetch does not support granular timeout settings, "
                    "reverting to total timeout.", AppEnginePlatformWarning)
            return timeout.total
        return timeout

    def _get_retries(self, retries, redirect):
        if not isinstance(retries, Retry):
            retries = Retry.from_int(
                retries, redirect=redirect, default=self.retries)

        if retries.connect or retries.read or retries.redirect:
            warnings.warn(
                "URLFetch only supports total retries and does not "
                "recognize connect, read, or redirect retry parameters.",
                AppEnginePlatformWarning)

        return retries


def is_appengine():
    return (is_local_appengine() or
            is_prod_appengine() or
            is_prod_appengine_mvms())


def is_appengine_sandbox():
    return is_appengine() and not is_prod_appengine_mvms()


def is_local_appengine():
    return ('APPENGINE_RUNTIME' in os.environ and
            'Development/' in os.environ['SERVER_SOFTWARE'])


def is_prod_appengine():
    return ('APPENGINE_RUNTIME' in os.environ and
            'Google App Engine/' in os.environ['SERVER_SOFTWARE'] and
            not is_prod_appengine_mvms())


def is_prod_appengine_mvms():
    return os.environ.get('GAE_VM', False) == 'true'
