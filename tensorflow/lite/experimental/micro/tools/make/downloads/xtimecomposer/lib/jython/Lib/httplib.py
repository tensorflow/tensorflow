"""HTTP/1.1 client library

<intro stuff goes here>
<other stuff, too>

HTTPConnection goes through a number of "states", which define when a client
may legally make another request or fetch the response for a particular
request. This diagram details these state transitions:

    (null)
      |
      | HTTPConnection()
      v
    Idle
      |
      | putrequest()
      v
    Request-started
      |
      | ( putheader() )*  endheaders()
      v
    Request-sent
      |
      | response = getresponse()
      v
    Unread-response   [Response-headers-read]
      |\____________________
      |                     |
      | response.read()     | putrequest()
      v                     v
    Idle                  Req-started-unread-response
                     ______/|
                   /        |
   response.read() |        | ( putheader() )*  endheaders()
                   v        v
       Request-started    Req-sent-unread-response
                            |
                            | response.read()
                            v
                          Request-sent

This diagram presents the following rules:
  -- a second request may not be started until {response-headers-read}
  -- a response [object] cannot be retrieved until {request-sent}
  -- there is no differentiation between an unread response body and a
     partially read response body

Note: this enforcement is applied by the HTTPConnection class. The
      HTTPResponse class does not enforce this state machine, which
      implies sophisticated clients may accelerate the request/response
      pipeline. Caution should be taken, though: accelerating the states
      beyond the above pattern may imply knowledge of the server's
      connection-close behavior for certain requests. For example, it
      is impossible to tell whether the server will close the connection
      UNTIL the response headers have been read; this means that further
      requests cannot be placed into the pipeline until it is known that
      the server will NOT be closing the connection.

Logical State                  __state            __response
-------------                  -------            ----------
Idle                           _CS_IDLE           None
Request-started                _CS_REQ_STARTED    None
Request-sent                   _CS_REQ_SENT       None
Unread-response                _CS_IDLE           <response_class>
Req-started-unread-response    _CS_REQ_STARTED    <response_class>
Req-sent-unread-response       _CS_REQ_SENT       <response_class>
"""

import errno
import mimetools
import socket
from urlparse import urlsplit

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

__all__ = ["HTTP", "HTTPResponse", "HTTPConnection", "HTTPSConnection",
           "HTTPException", "NotConnected", "UnknownProtocol",
           "UnknownTransferEncoding", "UnimplementedFileMode",
           "IncompleteRead", "InvalidURL", "ImproperConnectionState",
           "CannotSendRequest", "CannotSendHeader", "ResponseNotReady",
           "BadStatusLine", "error", "responses"]

HTTP_PORT = 80
HTTPS_PORT = 443

_UNKNOWN = 'UNKNOWN'

# connection states
_CS_IDLE = 'Idle'
_CS_REQ_STARTED = 'Request-started'
_CS_REQ_SENT = 'Request-sent'

# status codes
# informational
CONTINUE = 100
SWITCHING_PROTOCOLS = 101
PROCESSING = 102

# successful
OK = 200
CREATED = 201
ACCEPTED = 202
NON_AUTHORITATIVE_INFORMATION = 203
NO_CONTENT = 204
RESET_CONTENT = 205
PARTIAL_CONTENT = 206
MULTI_STATUS = 207
IM_USED = 226

# redirection
MULTIPLE_CHOICES = 300
MOVED_PERMANENTLY = 301
FOUND = 302
SEE_OTHER = 303
NOT_MODIFIED = 304
USE_PROXY = 305
TEMPORARY_REDIRECT = 307

# client error
BAD_REQUEST = 400
UNAUTHORIZED = 401
PAYMENT_REQUIRED = 402
FORBIDDEN = 403
NOT_FOUND = 404
METHOD_NOT_ALLOWED = 405
NOT_ACCEPTABLE = 406
PROXY_AUTHENTICATION_REQUIRED = 407
REQUEST_TIMEOUT = 408
CONFLICT = 409
GONE = 410
LENGTH_REQUIRED = 411
PRECONDITION_FAILED = 412
REQUEST_ENTITY_TOO_LARGE = 413
REQUEST_URI_TOO_LONG = 414
UNSUPPORTED_MEDIA_TYPE = 415
REQUESTED_RANGE_NOT_SATISFIABLE = 416
EXPECTATION_FAILED = 417
UNPROCESSABLE_ENTITY = 422
LOCKED = 423
FAILED_DEPENDENCY = 424
UPGRADE_REQUIRED = 426

# server error
INTERNAL_SERVER_ERROR = 500
NOT_IMPLEMENTED = 501
BAD_GATEWAY = 502
SERVICE_UNAVAILABLE = 503
GATEWAY_TIMEOUT = 504
HTTP_VERSION_NOT_SUPPORTED = 505
INSUFFICIENT_STORAGE = 507
NOT_EXTENDED = 510

# Mapping status codes to official W3C names
responses = {
    100: 'Continue',
    101: 'Switching Protocols',

    200: 'OK',
    201: 'Created',
    202: 'Accepted',
    203: 'Non-Authoritative Information',
    204: 'No Content',
    205: 'Reset Content',
    206: 'Partial Content',

    300: 'Multiple Choices',
    301: 'Moved Permanently',
    302: 'Found',
    303: 'See Other',
    304: 'Not Modified',
    305: 'Use Proxy',
    306: '(Unused)',
    307: 'Temporary Redirect',

    400: 'Bad Request',
    401: 'Unauthorized',
    402: 'Payment Required',
    403: 'Forbidden',
    404: 'Not Found',
    405: 'Method Not Allowed',
    406: 'Not Acceptable',
    407: 'Proxy Authentication Required',
    408: 'Request Timeout',
    409: 'Conflict',
    410: 'Gone',
    411: 'Length Required',
    412: 'Precondition Failed',
    413: 'Request Entity Too Large',
    414: 'Request-URI Too Long',
    415: 'Unsupported Media Type',
    416: 'Requested Range Not Satisfiable',
    417: 'Expectation Failed',

    500: 'Internal Server Error',
    501: 'Not Implemented',
    502: 'Bad Gateway',
    503: 'Service Unavailable',
    504: 'Gateway Timeout',
    505: 'HTTP Version Not Supported',
}

# maximal amount of data to read at one time in _safe_read
MAXAMOUNT = 1048576

class HTTPMessage(mimetools.Message):

    def addheader(self, key, value):
        """Add header for field key handling repeats."""
        prev = self.dict.get(key)
        if prev is None:
            self.dict[key] = value
        else:
            combined = ", ".join((prev, value))
            self.dict[key] = combined

    def addcontinue(self, key, more):
        """Add more field data from a continuation line."""
        prev = self.dict[key]
        self.dict[key] = prev + "\n " + more

    def readheaders(self):
        """Read header lines.

        Read header lines up to the entirely blank line that terminates them.
        The (normally blank) line that ends the headers is skipped, but not
        included in the returned list.  If a non-header line ends the headers,
        (which is an error), an attempt is made to backspace over it; it is
        never included in the returned list.

        The variable self.status is set to the empty string if all went well,
        otherwise it is an error message.  The variable self.headers is a
        completely uninterpreted list of lines contained in the header (so
        printing them will reproduce the header exactly as it appears in the
        file).

        If multiple header fields with the same name occur, they are combined
        according to the rules in RFC 2616 sec 4.2:

        Appending each subsequent field-value to the first, each separated
        by a comma. The order in which header fields with the same field-name
        are received is significant to the interpretation of the combined
        field value.
        """
        # XXX The implementation overrides the readheaders() method of
        # rfc822.Message.  The base class design isn't amenable to
        # customized behavior here so the method here is a copy of the
        # base class code with a few small changes.

        self.dict = {}
        self.unixfrom = ''
        self.headers = hlist = []
        self.status = ''
        headerseen = ""
        firstline = 1
        startofline = unread = tell = None
        if hasattr(self.fp, 'unread'):
            unread = self.fp.unread
        elif self.seekable:
            tell = self.fp.tell
        while True:
            if tell:
                try:
                    startofline = tell()
                except IOError:
                    startofline = tell = None
                    self.seekable = 0
            line = self.fp.readline()
            if not line:
                self.status = 'EOF in headers'
                break
            # Skip unix From name time lines
            if firstline and line.startswith('From '):
                self.unixfrom = self.unixfrom + line
                continue
            firstline = 0
            if headerseen and line[0] in ' \t':
                # XXX Not sure if continuation lines are handled properly
                # for http and/or for repeating headers
                # It's a continuation line.
                hlist.append(line)
                self.addcontinue(headerseen, line.strip())
                continue
            elif self.iscomment(line):
                # It's a comment.  Ignore it.
                continue
            elif self.islast(line):
                # Note! No pushback here!  The delimiter line gets eaten.
                break
            headerseen = self.isheader(line)
            if headerseen:
                # It's a legal header line, save it.
                hlist.append(line)
                self.addheader(headerseen, line[len(headerseen)+1:].strip())
                continue
            else:
                # It's not a header line; throw it back and stop here.
                if not self.dict:
                    self.status = 'No headers'
                else:
                    self.status = 'Non-header line where header expected'
                # Try to undo the read.
                if unread:
                    unread(line)
                elif tell:
                    self.fp.seek(startofline)
                else:
                    self.status = self.status + '; bad seek'
                break

class HTTPResponse:

    # strict: If true, raise BadStatusLine if the status line can't be
    # parsed as a valid HTTP/1.0 or 1.1 status line.  By default it is
    # false because it prevents clients from talking to HTTP/0.9
    # servers.  Note that a response with a sufficiently corrupted
    # status line will look like an HTTP/0.9 response.

    # See RFC 2616 sec 19.6 and RFC 1945 sec 6 for details.

    def __init__(self, sock, debuglevel=0, strict=0, method=None):
        self.fp = sock.makefile('rb', 0)
        self.debuglevel = debuglevel
        self.strict = strict
        self._method = method

        self.msg = None

        # from the Status-Line of the response
        self.version = _UNKNOWN # HTTP-Version
        self.status = _UNKNOWN  # Status-Code
        self.reason = _UNKNOWN  # Reason-Phrase

        self.chunked = _UNKNOWN         # is "chunked" being used?
        self.chunk_left = _UNKNOWN      # bytes left to read in current chunk
        self.length = _UNKNOWN          # number of bytes left in response
        self.will_close = _UNKNOWN      # conn will close at end of response

    def _read_status(self):
        # Initialize with Simple-Response defaults
        line = self.fp.readline()
        if self.debuglevel > 0:
            print "reply:", repr(line)
        if not line:
            # Presumably, the server closed the connection before
            # sending a valid response.
            raise BadStatusLine(line)
        try:
            [version, status, reason] = line.split(None, 2)
        except ValueError:
            try:
                [version, status] = line.split(None, 1)
                reason = ""
            except ValueError:
                # empty version will cause next test to fail and status
                # will be treated as 0.9 response.
                version = ""
        if not version.startswith('HTTP/'):
            if self.strict:
                self.close()
                raise BadStatusLine(line)
            else:
                # assume it's a Simple-Response from an 0.9 server
                self.fp = LineAndFileWrapper(line, self.fp)
                return "HTTP/0.9", 200, ""

        # The status code is a three-digit number
        try:
            status = int(status)
            if status < 100 or status > 999:
                raise BadStatusLine(line)
        except ValueError:
            raise BadStatusLine(line)
        return version, status, reason

    def begin(self):
        if self.msg is not None:
            # we've already started reading the response
            return

        # read until we get a non-100 response
        while True:
            version, status, reason = self._read_status()
            if status != CONTINUE:
                break
            # skip the header from the 100 response
            while True:
                skip = self.fp.readline().strip()
                if not skip:
                    break
                if self.debuglevel > 0:
                    print "header:", skip

        self.status = status
        self.reason = reason.strip()
        if version == 'HTTP/1.0':
            self.version = 10
        elif version.startswith('HTTP/1.'):
            self.version = 11   # use HTTP/1.1 code for HTTP/1.x where x>=1
        elif version == 'HTTP/0.9':
            self.version = 9
        else:
            raise UnknownProtocol(version)

        if self.version == 9:
            self.length = None
            self.chunked = 0
            self.will_close = 1
            self.msg = HTTPMessage(StringIO())
            return

        self.msg = HTTPMessage(self.fp, 0)
        if self.debuglevel > 0:
            for hdr in self.msg.headers:
                print "header:", hdr,

        # don't let the msg keep an fp
        self.msg.fp = None

        # are we using the chunked-style of transfer encoding?
        tr_enc = self.msg.getheader('transfer-encoding')
        if tr_enc and tr_enc.lower() == "chunked":
            self.chunked = 1
            self.chunk_left = None
        else:
            self.chunked = 0

        # will the connection close at the end of the response?
        self.will_close = self._check_close()

        # do we have a Content-Length?
        # NOTE: RFC 2616, S4.4, #3 says we ignore this if tr_enc is "chunked"
        length = self.msg.getheader('content-length')
        if length and not self.chunked:
            try:
                self.length = int(length)
            except ValueError:
                self.length = None
        else:
            self.length = None

        # does the body have a fixed length? (of zero)
        if (status == NO_CONTENT or status == NOT_MODIFIED or
            100 <= status < 200 or      # 1xx codes
            self._method == 'HEAD'):
            self.length = 0

        # if the connection remains open, and we aren't using chunked, and
        # a content-length was not provided, then assume that the connection
        # WILL close.
        if not self.will_close and \
           not self.chunked and \
           self.length is None:
            self.will_close = 1

    def _check_close(self):
        conn = self.msg.getheader('connection')
        if self.version == 11:
            # An HTTP/1.1 proxy is assumed to stay open unless
            # explicitly closed.
            conn = self.msg.getheader('connection')
            if conn and "close" in conn.lower():
                return True
            return False

        # Some HTTP/1.0 implementations have support for persistent
        # connections, using rules different than HTTP/1.1.

        # For older HTTP, Keep-Alive indiciates persistent connection.
        if self.msg.getheader('keep-alive'):
            return False

        # At least Akamai returns a "Connection: Keep-Alive" header,
        # which was supposed to be sent by the client.
        if conn and "keep-alive" in conn.lower():
            return False

        # Proxy-Connection is a netscape hack.
        pconn = self.msg.getheader('proxy-connection')
        if pconn and "keep-alive" in pconn.lower():
            return False

        # otherwise, assume it will close
        return True

    def close(self):
        if self.fp:
            self.fp.close()
            self.fp = None

    def isclosed(self):
        # NOTE: it is possible that we will not ever call self.close(). This
        #       case occurs when will_close is TRUE, length is None, and we
        #       read up to the last byte, but NOT past it.
        #
        # IMPLIES: if will_close is FALSE, then self.close() will ALWAYS be
        #          called, meaning self.isclosed() is meaningful.
        return self.fp is None

    # XXX It would be nice to have readline and __iter__ for this, too.

    def read(self, amt=None):
        if self.fp is None:
            return ''

        if self.chunked:
            return self._read_chunked(amt)

        if amt is None:
            # unbounded read
            if self.length is None:
                s = self.fp.read()
            else:
                s = self._safe_read(self.length)
                self.length = 0
            self.close()        # we read everything
            return s

        if self.length is not None:
            if amt > self.length:
                # clip the read to the "end of response"
                amt = self.length

        # we do not use _safe_read() here because this may be a .will_close
        # connection, and the user is reading more bytes than will be provided
        # (for example, reading in 1k chunks)
        s = self.fp.read(amt)
        if self.length is not None:
            self.length -= len(s)

        return s

    def _read_chunked(self, amt):
        assert self.chunked != _UNKNOWN
        chunk_left = self.chunk_left
        value = ''

        # XXX This accumulates chunks by repeated string concatenation,
        # which is not efficient as the number or size of chunks gets big.
        while True:
            if chunk_left is None:
                line = self.fp.readline()
                i = line.find(';')
                if i >= 0:
                    line = line[:i] # strip chunk-extensions
                chunk_left = int(line, 16)
                if chunk_left == 0:
                    break
            if amt is None:
                value += self._safe_read(chunk_left)
            elif amt < chunk_left:
                value += self._safe_read(amt)
                self.chunk_left = chunk_left - amt
                return value
            elif amt == chunk_left:
                value += self._safe_read(amt)
                self._safe_read(2)  # toss the CRLF at the end of the chunk
                self.chunk_left = None
                return value
            else:
                value += self._safe_read(chunk_left)
                amt -= chunk_left

            # we read the whole chunk, get another
            self._safe_read(2)      # toss the CRLF at the end of the chunk
            chunk_left = None

        # read and discard trailer up to the CRLF terminator
        ### note: we shouldn't have any trailers!
        while True:
            line = self.fp.readline()
            if not line:
                # a vanishingly small number of sites EOF without
                # sending the trailer
                break
            if line == '\r\n':
                break

        # we read everything; close the "file"
        self.close()

        return value

    def _safe_read(self, amt):
        """Read the number of bytes requested, compensating for partial reads.

        Normally, we have a blocking socket, but a read() can be interrupted
        by a signal (resulting in a partial read).

        Note that we cannot distinguish between EOF and an interrupt when zero
        bytes have been read. IncompleteRead() will be raised in this
        situation.

        This function should be used when <amt> bytes "should" be present for
        reading. If the bytes are truly not available (due to EOF), then the
        IncompleteRead exception can be used to detect the problem.
        """
        s = []
        while amt > 0:
            chunk = self.fp.read(min(amt, MAXAMOUNT))
            if not chunk:
                raise IncompleteRead(s)
            s.append(chunk)
            amt -= len(chunk)
        return ''.join(s)

    def getheader(self, name, default=None):
        if self.msg is None:
            raise ResponseNotReady()
        return self.msg.getheader(name, default)

    def getheaders(self):
        """Return list of (header, value) tuples."""
        if self.msg is None:
            raise ResponseNotReady()
        return self.msg.items()


class HTTPConnection:

    _http_vsn = 11
    _http_vsn_str = 'HTTP/1.1'

    response_class = HTTPResponse
    default_port = HTTP_PORT
    auto_open = 1
    debuglevel = 0
    strict = 0

    def __init__(self, host, port=None, strict=None):
        self.sock = None
        self._buffer = []
        self.__response = None
        self.__state = _CS_IDLE
        self._method = None

        self._set_hostport(host, port)
        if strict is not None:
            self.strict = strict

    def _set_hostport(self, host, port):
        if port is None:
            i = host.rfind(':')
            j = host.rfind(']')         # ipv6 addresses have [...]
            if i > j:
                try:
                    port = int(host[i+1:])
                except ValueError:
                    raise InvalidURL("nonnumeric port: '%s'" % host[i+1:])
                host = host[:i]
            else:
                port = self.default_port
            if host and host[0] == '[' and host[-1] == ']':
                host = host[1:-1]
        self.host = host
        self.port = port

    def set_debuglevel(self, level):
        self.debuglevel = level

    def connect(self):
        """Connect to the host and port specified in __init__."""
        msg = "getaddrinfo returns an empty list"
        for res in socket.getaddrinfo(self.host, self.port, 0,
                                      socket.SOCK_STREAM):
            af, socktype, proto, canonname, sa = res
            try:
                self.sock = socket.socket(af, socktype, proto)
                if self.debuglevel > 0:
                    print "connect: (%s, %s)" % (self.host, self.port)
                self.sock.connect(sa)
            except socket.error, msg:
                if self.debuglevel > 0:
                    print 'connect fail:', (self.host, self.port)
                if self.sock:
                    self.sock.close()
                self.sock = None
                continue
            break
        if not self.sock:
            raise socket.error, msg

    def close(self):
        """Close the connection to the HTTP server."""
        if self.sock:
            self.sock.close()   # close it manually... there may be other refs
            self.sock = None
        if self.__response:
            self.__response.close()
            self.__response = None
        self.__state = _CS_IDLE

    def send(self, str):
        """Send `str' to the server."""
        if self.sock is None:
            if self.auto_open:
                self.connect()
            else:
                raise NotConnected()

        # send the data to the server. if we get a broken pipe, then close
        # the socket. we want to reconnect when somebody tries to send again.
        #
        # NOTE: we DO propagate the error, though, because we cannot simply
        #       ignore the error... the caller will know if they can retry.
        if self.debuglevel > 0:
            print "send:", repr(str)
        try:
            self.sock.sendall(str)
        except socket.error, v:
            if v[0] == 32:      # Broken pipe
                self.close()
            raise

    def _output(self, s):
        """Add a line of output to the current request buffer.

        Assumes that the line does *not* end with \\r\\n.
        """
        self._buffer.append(s)

    def _send_output(self):
        """Send the currently buffered request and clear the buffer.

        Appends an extra \\r\\n to the buffer.
        """
        self._buffer.extend(("", ""))
        msg = "\r\n".join(self._buffer)
        del self._buffer[:]
        self.send(msg)

    def putrequest(self, method, url, skip_host=0, skip_accept_encoding=0):
        """Send a request to the server.

        `method' specifies an HTTP request method, e.g. 'GET'.
        `url' specifies the object being requested, e.g. '/index.html'.
        `skip_host' if True does not add automatically a 'Host:' header
        `skip_accept_encoding' if True does not add automatically an
           'Accept-Encoding:' header
        """

        # if a prior response has been completed, then forget about it.
        if self.__response and self.__response.isclosed():
            self.__response = None


        # in certain cases, we cannot issue another request on this connection.
        # this occurs when:
        #   1) we are in the process of sending a request.   (_CS_REQ_STARTED)
        #   2) a response to a previous request has signalled that it is going
        #      to close the connection upon completion.
        #   3) the headers for the previous response have not been read, thus
        #      we cannot determine whether point (2) is true.   (_CS_REQ_SENT)
        #
        # if there is no prior response, then we can request at will.
        #
        # if point (2) is true, then we will have passed the socket to the
        # response (effectively meaning, "there is no prior response"), and
        # will open a new one when a new request is made.
        #
        # Note: if a prior response exists, then we *can* start a new request.
        #       We are not allowed to begin fetching the response to this new
        #       request, however, until that prior response is complete.
        #
        if self.__state == _CS_IDLE:
            self.__state = _CS_REQ_STARTED
        else:
            raise CannotSendRequest()

        # Save the method we use, we need it later in the response phase
        self._method = method
        if not url:
            url = '/'
        str = '%s %s %s' % (method, url, self._http_vsn_str)

        self._output(str)

        if self._http_vsn == 11:
            # Issue some standard headers for better HTTP/1.1 compliance

            if not skip_host:
                # this header is issued *only* for HTTP/1.1
                # connections. more specifically, this means it is
                # only issued when the client uses the new
                # HTTPConnection() class. backwards-compat clients
                # will be using HTTP/1.0 and those clients may be
                # issuing this header themselves. we should NOT issue
                # it twice; some web servers (such as Apache) barf
                # when they see two Host: headers

                # If we need a non-standard port,include it in the
                # header.  If the request is going through a proxy,
                # but the host of the actual URL, not the host of the
                # proxy.

                netloc = ''
                if url.startswith('http'):
                    nil, netloc, nil, nil, nil = urlsplit(url)

                if netloc:
                    try:
                        netloc_enc = netloc.encode("ascii")
                    except UnicodeEncodeError:
                        netloc_enc = netloc.encode("idna")
                    self.putheader('Host', netloc_enc)
                else:
                    try:
                        host_enc = self.host.encode("ascii")
                    except UnicodeEncodeError:
                        host_enc = self.host.encode("idna")
                    if self.port == HTTP_PORT:
                        self.putheader('Host', host_enc)
                    else:
                        self.putheader('Host', "%s:%s" % (host_enc, self.port))

            # note: we are assuming that clients will not attempt to set these
            #       headers since *this* library must deal with the
            #       consequences. this also means that when the supporting
            #       libraries are updated to recognize other forms, then this
            #       code should be changed (removed or updated).

            # we only want a Content-Encoding of "identity" since we don't
            # support encodings such as x-gzip or x-deflate.
            if not skip_accept_encoding:
                self.putheader('Accept-Encoding', 'identity')

            # we can accept "chunked" Transfer-Encodings, but no others
            # NOTE: no TE header implies *only* "chunked"
            #self.putheader('TE', 'chunked')

            # if TE is supplied in the header, then it must appear in a
            # Connection header.
            #self.putheader('Connection', 'TE')

        else:
            # For HTTP/1.0, the server will assume "not chunked"
            pass

    def putheader(self, header, value):
        """Send a request header line to the server.

        For example: h.putheader('Accept', 'text/html')
        """
        if self.__state != _CS_REQ_STARTED:
            raise CannotSendHeader()

        str = '%s: %s' % (header, value)
        self._output(str)

    def endheaders(self):
        """Indicate that the last header line has been sent to the server."""

        if self.__state == _CS_REQ_STARTED:
            self.__state = _CS_REQ_SENT
        else:
            raise CannotSendHeader()

        self._send_output()

    def request(self, method, url, body=None, headers={}):
        """Send a complete request to the server."""

        try:
            self._send_request(method, url, body, headers)
        except socket.error, v:
            # trap 'Broken pipe' if we're allowed to automatically reconnect
            if v[0] != 32 or not self.auto_open:
                raise
            # try one more time
            self._send_request(method, url, body, headers)

    def _send_request(self, method, url, body, headers):
        # honour explicitly requested Host: and Accept-Encoding headers
        header_names = dict.fromkeys([k.lower() for k in headers])
        skips = {}
        if 'host' in header_names:
            skips['skip_host'] = 1
        if 'accept-encoding' in header_names:
            skips['skip_accept_encoding'] = 1

        self.putrequest(method, url, **skips)

        if body and ('content-length' not in header_names):
            self.putheader('Content-Length', str(len(body)))
        for hdr, value in headers.iteritems():
            self.putheader(hdr, value)
        self.endheaders()

        if body:
            self.send(body)

    def getresponse(self):
        "Get the response from the server."

        # if a prior response has been completed, then forget about it.
        if self.__response and self.__response.isclosed():
            self.__response = None

        #
        # if a prior response exists, then it must be completed (otherwise, we
        # cannot read this response's header to determine the connection-close
        # behavior)
        #
        # note: if a prior response existed, but was connection-close, then the
        # socket and response were made independent of this HTTPConnection
        # object since a new request requires that we open a whole new
        # connection
        #
        # this means the prior response had one of two states:
        #   1) will_close: this connection was reset and the prior socket and
        #                  response operate independently
        #   2) persistent: the response was retained and we await its
        #                  isclosed() status to become true.
        #
        if self.__state != _CS_REQ_SENT or self.__response:
            raise ResponseNotReady()

        if self.debuglevel > 0:
            response = self.response_class(self.sock, self.debuglevel,
                                           strict=self.strict,
                                           method=self._method)
        else:
            response = self.response_class(self.sock, strict=self.strict,
                                           method=self._method)

        response.begin()
        assert response.will_close != _UNKNOWN
        self.__state = _CS_IDLE

        if response.will_close:
            # this effectively passes the connection to the response
            self.close()
        else:
            # remember this, so we can tell when it is complete
            self.__response = response

        return response

# The next several classes are used to define FakeSocket, a socket-like
# interface to an SSL connection.

# The primary complexity comes from faking a makefile() method.  The
# standard socket makefile() implementation calls dup() on the socket
# file descriptor.  As a consequence, clients can call close() on the
# parent socket and its makefile children in any order.  The underlying
# socket isn't closed until they are all closed.

# The implementation uses reference counting to keep the socket open
# until the last client calls close().  SharedSocket keeps track of
# the reference counting and SharedSocketClient provides an constructor
# and close() method that call incref() and decref() correctly.

class SharedSocket:

    def __init__(self, sock):
        self.sock = sock
        self._refcnt = 0

    def incref(self):
        self._refcnt += 1

    def decref(self):
        self._refcnt -= 1
        assert self._refcnt >= 0
        if self._refcnt == 0:
            self.sock.close()

    def __del__(self):
        self.sock.close()

class SharedSocketClient:

    def __init__(self, shared):
        self._closed = 0
        self._shared = shared
        self._shared.incref()
        self._sock = shared.sock

    def close(self):
        if not self._closed:
            self._shared.decref()
            self._closed = 1
            self._shared = None

class SSLFile(SharedSocketClient):
    """File-like object wrapping an SSL socket."""

    BUFSIZE = 8192

    def __init__(self, sock, ssl, bufsize=None):
        SharedSocketClient.__init__(self, sock)
        self._ssl = ssl
        self._buf = ''
        self._bufsize = bufsize or self.__class__.BUFSIZE

    def _read(self):
        buf = ''
        # put in a loop so that we retry on transient errors
        while True:
            try:
                buf = self._ssl.read(self._bufsize)
            except socket.sslerror, err:
                if (err[0] == socket.SSL_ERROR_WANT_READ
                    or err[0] == socket.SSL_ERROR_WANT_WRITE):
                    continue
                if (err[0] == socket.SSL_ERROR_ZERO_RETURN
                    or err[0] == socket.SSL_ERROR_EOF):
                    break
                raise
            except socket.error, err:
                if err[0] == errno.EINTR:
                    continue
                if err[0] == errno.EBADF:
                    # XXX socket was closed?
                    break
                raise
            else:
                break
        return buf

    def read(self, size=None):
        L = [self._buf]
        avail = len(self._buf)
        while size is None or avail < size:
            s = self._read()
            if s == '':
                break
            L.append(s)
            avail += len(s)
        all = "".join(L)
        if size is None:
            self._buf = ''
            return all
        else:
            self._buf = all[size:]
            return all[:size]

    def readline(self):
        L = [self._buf]
        self._buf = ''
        while 1:
            i = L[-1].find("\n")
            if i >= 0:
                break
            s = self._read()
            if s == '':
                break
            L.append(s)
        if i == -1:
            # loop exited because there is no more data
            return "".join(L)
        else:
            all = "".join(L)
            # XXX could do enough bookkeeping not to do a 2nd search
            i = all.find("\n") + 1
            line = all[:i]
            self._buf = all[i:]
            return line

    def readlines(self, sizehint=0):
        total = 0
        list = []
        while True:
            line = self.readline()
            if not line:
                break
            list.append(line)
            total += len(line)
            if sizehint and total >= sizehint:
                break
        return list

    def fileno(self):
        return self._sock.fileno()

    def __iter__(self):
        return self

    def next(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

class FakeSocket(SharedSocketClient):

    class _closedsocket:
        def __getattr__(self, name):
            raise error(9, 'Bad file descriptor')

    def __init__(self, sock, ssl):
        sock = SharedSocket(sock)
        SharedSocketClient.__init__(self, sock)
        self._ssl = ssl

    def close(self):
        SharedSocketClient.close(self)
        self._sock = self.__class__._closedsocket()

    def makefile(self, mode, bufsize=None):
        if mode != 'r' and mode != 'rb':
            raise UnimplementedFileMode()
        return SSLFile(self._shared, self._ssl, bufsize)

    def send(self, stuff, flags = 0):
        return self._ssl.write(stuff)

    sendall = send

    def recv(self, len = 1024, flags = 0):
        return self._ssl.read(len)

    def __getattr__(self, attr):
        return getattr(self._sock, attr)


class HTTPSConnection(HTTPConnection):
    "This class allows communication via SSL."

    default_port = HTTPS_PORT

    def __init__(self, host, port=None, key_file=None, cert_file=None,
                 strict=None):
        HTTPConnection.__init__(self, host, port, strict)
        self.key_file = key_file
        self.cert_file = cert_file

    def connect(self):
        "Connect to a host on a given (SSL) port."

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        ssl = socket.ssl(sock, self.key_file, self.cert_file)
        self.sock = FakeSocket(sock, ssl)


class HTTP:
    "Compatibility class with httplib.py from 1.5."

    _http_vsn = 10
    _http_vsn_str = 'HTTP/1.0'

    debuglevel = 0

    _connection_class = HTTPConnection

    def __init__(self, host='', port=None, strict=None):
        "Provide a default host, since the superclass requires one."

        # some joker passed 0 explicitly, meaning default port
        if port == 0:
            port = None

        # Note that we may pass an empty string as the host; this will throw
        # an error when we attempt to connect. Presumably, the client code
        # will call connect before then, with a proper host.
        self._setup(self._connection_class(host, port, strict))

    def _setup(self, conn):
        self._conn = conn

        # set up delegation to flesh out interface
        self.send = conn.send
        self.putrequest = conn.putrequest
        self.endheaders = conn.endheaders
        self.set_debuglevel = conn.set_debuglevel

        conn._http_vsn = self._http_vsn
        conn._http_vsn_str = self._http_vsn_str

        self.file = None

    def connect(self, host=None, port=None):
        "Accept arguments to set the host/port, since the superclass doesn't."

        if host is not None:
            self._conn._set_hostport(host, port)
        self._conn.connect()

    def getfile(self):
        "Provide a getfile, since the superclass' does not use this concept."
        return self.file

    def putheader(self, header, *values):
        "The superclass allows only one value argument."
        self._conn.putheader(header, '\r\n\t'.join(values))

    def getreply(self):
        """Compat definition since superclass does not define it.

        Returns a tuple consisting of:
        - server status code (e.g. '200' if all goes well)
        - server "reason" corresponding to status code
        - any RFC822 headers in the response from the server
        """
        try:
            response = self._conn.getresponse()
        except BadStatusLine, e:
            ### hmm. if getresponse() ever closes the socket on a bad request,
            ### then we are going to have problems with self.sock

            ### should we keep this behavior? do people use it?
            # keep the socket open (as a file), and return it
            self.file = self._conn.sock.makefile('rb', 0)

            # close our socket -- we want to restart after any protocol error
            self.close()

            self.headers = None
            return -1, e.line, None

        self.headers = response.msg
        self.file = response.fp
        return response.status, response.reason, response.msg

    def close(self):
        self._conn.close()

        # note that self.file == response.fp, which gets closed by the
        # superclass. just clear the object ref here.
        ### hmm. messy. if status==-1, then self.file is owned by us.
        ### well... we aren't explicitly closing, but losing this ref will
        ### do it
        self.file = None

if hasattr(socket, 'ssl'):
    class HTTPS(HTTP):
        """Compatibility with 1.5 httplib interface

        Python 1.5.2 did not have an HTTPS class, but it defined an
        interface for sending http requests that is also useful for
        https.
        """

        _connection_class = HTTPSConnection

        def __init__(self, host='', port=None, key_file=None, cert_file=None,
                     strict=None):
            # provide a default host, pass the X509 cert info

            # urf. compensate for bad input.
            if port == 0:
                port = None
            self._setup(self._connection_class(host, port, key_file,
                                               cert_file, strict))

            # we never actually use these for anything, but we keep them
            # here for compatibility with post-1.5.2 CVS.
            self.key_file = key_file
            self.cert_file = cert_file


class HTTPException(Exception):
    # Subclasses that define an __init__ must call Exception.__init__
    # or define self.args.  Otherwise, str() will fail.
    pass

class NotConnected(HTTPException):
    pass

class InvalidURL(HTTPException):
    pass

class UnknownProtocol(HTTPException):
    def __init__(self, version):
        self.args = version,
        self.version = version

class UnknownTransferEncoding(HTTPException):
    pass

class UnimplementedFileMode(HTTPException):
    pass

class IncompleteRead(HTTPException):
    def __init__(self, partial):
        self.args = partial,
        self.partial = partial

class ImproperConnectionState(HTTPException):
    pass

class CannotSendRequest(ImproperConnectionState):
    pass

class CannotSendHeader(ImproperConnectionState):
    pass

class ResponseNotReady(ImproperConnectionState):
    pass

class BadStatusLine(HTTPException):
    def __init__(self, line):
        self.args = line,
        self.line = line

# for backwards compatibility
error = HTTPException

class LineAndFileWrapper:
    """A limited file-like object for HTTP/0.9 responses."""

    # The status-line parsing code calls readline(), which normally
    # get the HTTP status line.  For a 0.9 response, however, this is
    # actually the first line of the body!  Clients need to get a
    # readable file object that contains that line.

    def __init__(self, line, file):
        self._line = line
        self._file = file
        self._line_consumed = 0
        self._line_offset = 0
        self._line_left = len(line)

    def __getattr__(self, attr):
        return getattr(self._file, attr)

    def _done(self):
        # called when the last byte is read from the line.  After the
        # call, all read methods are delegated to the underlying file
        # object.
        self._line_consumed = 1
        self.read = self._file.read
        self.readline = self._file.readline
        self.readlines = self._file.readlines

    def read(self, amt=None):
        if self._line_consumed:
            return self._file.read(amt)
        assert self._line_left
        if amt is None or amt > self._line_left:
            s = self._line[self._line_offset:]
            self._done()
            if amt is None:
                return s + self._file.read()
            else:
                return s + self._file.read(amt - len(s))
        else:
            assert amt <= self._line_left
            i = self._line_offset
            j = i + amt
            s = self._line[i:j]
            self._line_offset = j
            self._line_left -= amt
            if self._line_left == 0:
                self._done()
            return s

    def readline(self):
        if self._line_consumed:
            return self._file.readline()
        assert self._line_left
        s = self._line[self._line_offset:]
        self._done()
        return s

    def readlines(self, size=None):
        if self._line_consumed:
            return self._file.readlines(size)
        assert self._line_left
        L = [self._line[self._line_offset:]]
        self._done()
        if size is None:
            return L + self._file.readlines()
        else:
            return L + self._file.readlines(size)

def test():
    """Test this module.

    A hodge podge of tests collected here, because they have too many
    external dependencies for the regular test suite.
    """

    import sys
    import getopt
    opts, args = getopt.getopt(sys.argv[1:], 'd')
    dl = 0
    for o, a in opts:
        if o == '-d': dl = dl + 1
    host = 'www.python.org'
    selector = '/'
    if args[0:]: host = args[0]
    if args[1:]: selector = args[1]
    h = HTTP()
    h.set_debuglevel(dl)
    h.connect(host)
    h.putrequest('GET', selector)
    h.endheaders()
    status, reason, headers = h.getreply()
    print 'status =', status
    print 'reason =', reason
    print "read", len(h.getfile().read())
    print
    if headers:
        for header in headers.headers: print header.strip()
    print

    # minimal test that code to extract host from url works
    class HTTP11(HTTP):
        _http_vsn = 11
        _http_vsn_str = 'HTTP/1.1'

    h = HTTP11('www.python.org')
    h.putrequest('GET', 'http://www.python.org/~jeremy/')
    h.endheaders()
    h.getreply()
    h.close()

    if hasattr(socket, 'ssl'):

        for host, selector in (('sourceforge.net', '/projects/python'),
                               ):
            print "https://%s%s" % (host, selector)
            hs = HTTPS()
            hs.set_debuglevel(dl)
            hs.connect(host)
            hs.putrequest('GET', selector)
            hs.endheaders()
            status, reason, headers = hs.getreply()
            print 'status =', status
            print 'reason =', reason
            print "read", len(hs.getfile().read())
            print
            if headers:
                for header in headers.headers: print header.strip()
            print

if __name__ == '__main__':
    test()
