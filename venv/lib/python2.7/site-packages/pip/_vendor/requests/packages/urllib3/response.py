from __future__ import absolute_import
from contextlib import contextmanager
import zlib
import io
from socket import timeout as SocketTimeout
from socket import error as SocketError

from ._collections import HTTPHeaderDict
from .exceptions import (
    ProtocolError, DecodeError, ReadTimeoutError, ResponseNotChunked
)
from .packages.six import string_types as basestring, binary_type, PY3
from .packages.six.moves import http_client as httplib
from .connection import HTTPException, BaseSSLError
from .util.response import is_fp_closed, is_response_to_head


class DeflateDecoder(object):

    def __init__(self):
        self._first_try = True
        self._data = binary_type()
        self._obj = zlib.decompressobj()

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def decompress(self, data):
        if not data:
            return data

        if not self._first_try:
            return self._obj.decompress(data)

        self._data += data
        try:
            return self._obj.decompress(data)
        except zlib.error:
            self._first_try = False
            self._obj = zlib.decompressobj(-zlib.MAX_WBITS)
            try:
                return self.decompress(self._data)
            finally:
                self._data = None


class GzipDecoder(object):

    def __init__(self):
        self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def decompress(self, data):
        if not data:
            return data
        return self._obj.decompress(data)


def _get_decoder(mode):
    if mode == 'gzip':
        return GzipDecoder()

    return DeflateDecoder()


class HTTPResponse(io.IOBase):
    """
    HTTP Response container.

    Backwards-compatible to httplib's HTTPResponse but the response ``body`` is
    loaded and decoded on-demand when the ``data`` property is accessed.  This
    class is also compatible with the Python standard library's :mod:`io`
    module, and can hence be treated as a readable object in the context of that
    framework.

    Extra parameters for behaviour not present in httplib.HTTPResponse:

    :param preload_content:
        If True, the response's body will be preloaded during construction.

    :param decode_content:
        If True, attempts to decode specific content-encoding's based on headers
        (like 'gzip' and 'deflate') will be skipped and raw data will be used
        instead.

    :param original_response:
        When this HTTPResponse wrapper is generated from an httplib.HTTPResponse
        object, it's convenient to include the original for debug purposes. It's
        otherwise unused.
    """

    CONTENT_DECODERS = ['gzip', 'deflate']
    REDIRECT_STATUSES = [301, 302, 303, 307, 308]

    def __init__(self, body='', headers=None, status=0, version=0, reason=None,
                 strict=0, preload_content=True, decode_content=True,
                 original_response=None, pool=None, connection=None):

        if isinstance(headers, HTTPHeaderDict):
            self.headers = headers
        else:
            self.headers = HTTPHeaderDict(headers)
        self.status = status
        self.version = version
        self.reason = reason
        self.strict = strict
        self.decode_content = decode_content

        self._decoder = None
        self._body = None
        self._fp = None
        self._original_response = original_response
        self._fp_bytes_read = 0

        if body and isinstance(body, (basestring, binary_type)):
            self._body = body

        self._pool = pool
        self._connection = connection

        if hasattr(body, 'read'):
            self._fp = body

        # Are we using the chunked-style of transfer encoding?
        self.chunked = False
        self.chunk_left = None
        tr_enc = self.headers.get('transfer-encoding', '').lower()
        # Don't incur the penalty of creating a list and then discarding it
        encodings = (enc.strip() for enc in tr_enc.split(","))
        if "chunked" in encodings:
            self.chunked = True

        # If requested, preload the body.
        if preload_content and not self._body:
            self._body = self.read(decode_content=decode_content)

    def get_redirect_location(self):
        """
        Should we redirect and where to?

        :returns: Truthy redirect location string if we got a redirect status
            code and valid location. ``None`` if redirect status and no
            location. ``False`` if not a redirect status code.
        """
        if self.status in self.REDIRECT_STATUSES:
            return self.headers.get('location')

        return False

    def release_conn(self):
        if not self._pool or not self._connection:
            return

        self._pool._put_conn(self._connection)
        self._connection = None

    @property
    def data(self):
        # For backwords-compat with earlier urllib3 0.4 and earlier.
        if self._body:
            return self._body

        if self._fp:
            return self.read(cache_content=True)

    @property
    def connection(self):
        return self._connection

    def tell(self):
        """
        Obtain the number of bytes pulled over the wire so far. May differ from
        the amount of content returned by :meth:``HTTPResponse.read`` if bytes
        are encoded on the wire (e.g, compressed).
        """
        return self._fp_bytes_read

    def _init_decoder(self):
        """
        Set-up the _decoder attribute if necessar.
        """
        # Note: content-encoding value should be case-insensitive, per RFC 7230
        # Section 3.2
        content_encoding = self.headers.get('content-encoding', '').lower()
        if self._decoder is None and content_encoding in self.CONTENT_DECODERS:
            self._decoder = _get_decoder(content_encoding)

    def _decode(self, data, decode_content, flush_decoder):
        """
        Decode the data passed in and potentially flush the decoder.
        """
        try:
            if decode_content and self._decoder:
                data = self._decoder.decompress(data)
        except (IOError, zlib.error) as e:
            content_encoding = self.headers.get('content-encoding', '').lower()
            raise DecodeError(
                "Received response with content-encoding: %s, but "
                "failed to decode it." % content_encoding, e)

        if flush_decoder and decode_content:
            data += self._flush_decoder()

        return data

    def _flush_decoder(self):
        """
        Flushes the decoder. Should only be called if the decoder is actually
        being used.
        """
        if self._decoder:
            buf = self._decoder.decompress(b'')
            return buf + self._decoder.flush()

        return b''

    @contextmanager
    def _error_catcher(self):
        """
        Catch low-level python exceptions, instead re-raising urllib3
        variants, so that low-level exceptions are not leaked in the
        high-level api.

        On exit, release the connection back to the pool.
        """
        clean_exit = False

        try:
            try:
                yield

            except SocketTimeout:
                # FIXME: Ideally we'd like to include the url in the ReadTimeoutError but
                # there is yet no clean way to get at it from this context.
                raise ReadTimeoutError(self._pool, None, 'Read timed out.')

            except BaseSSLError as e:
                # FIXME: Is there a better way to differentiate between SSLErrors?
                if 'read operation timed out' not in str(e):  # Defensive:
                    # This shouldn't happen but just in case we're missing an edge
                    # case, let's avoid swallowing SSL errors.
                    raise

                raise ReadTimeoutError(self._pool, None, 'Read timed out.')

            except (HTTPException, SocketError) as e:
                # This includes IncompleteRead.
                raise ProtocolError('Connection broken: %r' % e, e)

            # If no exception is thrown, we should avoid cleaning up
            # unnecessarily.
            clean_exit = True
        finally:
            # If we didn't terminate cleanly, we need to throw away our
            # connection.
            if not clean_exit:
                # The response may not be closed but we're not going to use it
                # anymore so close it now to ensure that the connection is
                # released back to the pool.
                if self._original_response:
                    self._original_response.close()

                # Closing the response may not actually be sufficient to close
                # everything, so if we have a hold of the connection close that
                # too.
                if self._connection:
                    self._connection.close()

            # If we hold the original response but it's closed now, we should
            # return the connection back to the pool.
            if self._original_response and self._original_response.isclosed():
                self.release_conn()

    def read(self, amt=None, decode_content=None, cache_content=False):
        """
        Similar to :meth:`httplib.HTTPResponse.read`, but with two additional
        parameters: ``decode_content`` and ``cache_content``.

        :param amt:
            How much of the content to read. If specified, caching is skipped
            because it doesn't make sense to cache partial content as the full
            response.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.

        :param cache_content:
            If True, will save the returned data such that the same result is
            returned despite of the state of the underlying file object. This
            is useful if you want the ``.data`` property to continue working
            after having ``.read()`` the file object. (Overridden if ``amt`` is
            set.)
        """
        self._init_decoder()
        if decode_content is None:
            decode_content = self.decode_content

        if self._fp is None:
            return

        flush_decoder = False
        data = None

        with self._error_catcher():
            if amt is None:
                # cStringIO doesn't like amt=None
                data = self._fp.read()
                flush_decoder = True
            else:
                cache_content = False
                data = self._fp.read(amt)
                if amt != 0 and not data:  # Platform-specific: Buggy versions of Python.
                    # Close the connection when no data is returned
                    #
                    # This is redundant to what httplib/http.client _should_
                    # already do.  However, versions of python released before
                    # December 15, 2012 (http://bugs.python.org/issue16298) do
                    # not properly close the connection in all cases. There is
                    # no harm in redundantly calling close.
                    self._fp.close()
                    flush_decoder = True

        if data:
            self._fp_bytes_read += len(data)

            data = self._decode(data, decode_content, flush_decoder)

            if cache_content:
                self._body = data

        return data

    def stream(self, amt=2**16, decode_content=None):
        """
        A generator wrapper for the read() method. A call will block until
        ``amt`` bytes have been read from the connection or until the
        connection is closed.

        :param amt:
            How much of the content to read. The generator will return up to
            much data per iteration, but may return less. This is particularly
            likely when using compressed data. However, the empty string will
            never be returned.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
        """
        if self.chunked:
            for line in self.read_chunked(amt, decode_content=decode_content):
                yield line
        else:
            while not is_fp_closed(self._fp):
                data = self.read(amt=amt, decode_content=decode_content)

                if data:
                    yield data

    @classmethod
    def from_httplib(ResponseCls, r, **response_kw):
        """
        Given an :class:`httplib.HTTPResponse` instance ``r``, return a
        corresponding :class:`urllib3.response.HTTPResponse` object.

        Remaining parameters are passed to the HTTPResponse constructor, along
        with ``original_response=r``.
        """
        headers = r.msg

        if not isinstance(headers, HTTPHeaderDict):
            if PY3:  # Python 3
                headers = HTTPHeaderDict(headers.items())
            else:  # Python 2
                headers = HTTPHeaderDict.from_httplib(headers)

        # HTTPResponse objects in Python 3 don't have a .strict attribute
        strict = getattr(r, 'strict', 0)
        resp = ResponseCls(body=r,
                           headers=headers,
                           status=r.status,
                           version=r.version,
                           reason=r.reason,
                           strict=strict,
                           original_response=r,
                           **response_kw)
        return resp

    # Backwards-compatibility methods for httplib.HTTPResponse
    def getheaders(self):
        return self.headers

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    # Overrides from io.IOBase
    def close(self):
        if not self.closed:
            self._fp.close()

        if self._connection:
            self._connection.close()

    @property
    def closed(self):
        if self._fp is None:
            return True
        elif hasattr(self._fp, 'closed'):
            return self._fp.closed
        elif hasattr(self._fp, 'isclosed'):  # Python 2
            return self._fp.isclosed()
        else:
            return True

    def fileno(self):
        if self._fp is None:
            raise IOError("HTTPResponse has no file to get a fileno from")
        elif hasattr(self._fp, "fileno"):
            return self._fp.fileno()
        else:
            raise IOError("The file-like object this HTTPResponse is wrapped "
                          "around has no file descriptor")

    def flush(self):
        if self._fp is not None and hasattr(self._fp, 'flush'):
            return self._fp.flush()

    def readable(self):
        # This method is required for `io` module compatibility.
        return True

    def readinto(self, b):
        # This method is required for `io` module compatibility.
        temp = self.read(len(b))
        if len(temp) == 0:
            return 0
        else:
            b[:len(temp)] = temp
            return len(temp)

    def _update_chunk_length(self):
        # First, we'll figure out length of a chunk and then
        # we'll try to read it from socket.
        if self.chunk_left is not None:
            return
        line = self._fp.fp.readline()
        line = line.split(b';', 1)[0]
        try:
            self.chunk_left = int(line, 16)
        except ValueError:
            # Invalid chunked protocol response, abort.
            self.close()
            raise httplib.IncompleteRead(line)

    def _handle_chunk(self, amt):
        returned_chunk = None
        if amt is None:
            chunk = self._fp._safe_read(self.chunk_left)
            returned_chunk = chunk
            self._fp._safe_read(2)  # Toss the CRLF at the end of the chunk.
            self.chunk_left = None
        elif amt < self.chunk_left:
            value = self._fp._safe_read(amt)
            self.chunk_left = self.chunk_left - amt
            returned_chunk = value
        elif amt == self.chunk_left:
            value = self._fp._safe_read(amt)
            self._fp._safe_read(2)  # Toss the CRLF at the end of the chunk.
            self.chunk_left = None
            returned_chunk = value
        else:  # amt > self.chunk_left
            returned_chunk = self._fp._safe_read(self.chunk_left)
            self._fp._safe_read(2)  # Toss the CRLF at the end of the chunk.
            self.chunk_left = None
        return returned_chunk

    def read_chunked(self, amt=None, decode_content=None):
        """
        Similar to :meth:`HTTPResponse.read`, but with an additional
        parameter: ``decode_content``.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
        """
        self._init_decoder()
        # FIXME: Rewrite this method and make it a class with a better structured logic.
        if not self.chunked:
            raise ResponseNotChunked(
                "Response is not chunked. "
                "Header 'transfer-encoding: chunked' is missing.")

        # Don't bother reading the body of a HEAD request.
        if self._original_response and is_response_to_head(self._original_response):
            self._original_response.close()
            return

        with self._error_catcher():
            while True:
                self._update_chunk_length()
                if self.chunk_left == 0:
                    break
                chunk = self._handle_chunk(amt)
                decoded = self._decode(chunk, decode_content=decode_content,
                                       flush_decoder=False)
                if decoded:
                    yield decoded

            if decode_content:
                # On CPython and PyPy, we should never need to flush the
                # decoder. However, on Jython we *might* need to, so
                # lets defensively do it anyway.
                decoded = self._flush_decoder()
                if decoded:  # Platform-specific: Jython.
                    yield decoded

            # Chunk content ends with \r\n: discard it.
            while True:
                line = self._fp.fp.readline()
                if not line:
                    # Some sites may not end with '\r\n'.
                    break
                if line == b'\r\n':
                    break

            # We read everything; close the "file".
            if self._original_response:
                self._original_response.close()
