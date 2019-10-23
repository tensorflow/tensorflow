"""
This is an updated socket module for use on JVMs > 1.4; it is derived from the
old jython socket module.
The primary extra it provides is non-blocking support.

XXX Restrictions:

- Only INET sockets
- Can't do a very good gethostbyaddr() right...
AMAK: 20050527: added socket timeouts
AMAK: 20070515: Added non-blocking (asynchronous) support
AMAK: 20070515: Added client-side SSL support
AMAK: 20080513: Added support for options
"""

_defaulttimeout = None

import errno
import jarray
import string
import struct
import sys
import threading
import time
import types

# Java.io classes
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
# Java.io exceptions
import java.io.InterruptedIOException
import java.io.IOException

# Java.lang classes
import java.lang.String
# Java.lang exceptions
import java.lang.Exception

# Java.net classes
import java.net.DatagramPacket
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket
# Java.net exceptions
import java.net.BindException
import java.net.ConnectException
import java.net.NoRouteToHostException
import java.net.PortUnreachableException
import java.net.ProtocolException
import java.net.SocketException
import java.net.SocketTimeoutException
import java.net.UnknownHostException

# Java.nio classes
import java.nio.ByteBuffer
import java.nio.channels.DatagramChannel
import java.nio.channels.ServerSocketChannel
import java.nio.channels.SocketChannel
# Java.nio exceptions
import java.nio.channels.AlreadyConnectedException
import java.nio.channels.AsynchronousCloseException
import java.nio.channels.CancelledKeyException
import java.nio.channels.ClosedByInterruptException
import java.nio.channels.ClosedChannelException
import java.nio.channels.ClosedSelectorException
import java.nio.channels.ConnectionPendingException
import java.nio.channels.IllegalBlockingModeException
import java.nio.channels.IllegalSelectorException
import java.nio.channels.NoConnectionPendingException
import java.nio.channels.NonReadableChannelException
import java.nio.channels.NonWritableChannelException
import java.nio.channels.NotYetBoundException
import java.nio.channels.NotYetConnectedException
import java.nio.channels.UnresolvedAddressException
import java.nio.channels.UnsupportedAddressTypeException

# Javax.net.ssl classes
import javax.net.ssl.SSLSocketFactory
# Javax.net.ssl exceptions
javax.net.ssl.SSLException
javax.net.ssl.SSLHandshakeException
javax.net.ssl.SSLKeyException
javax.net.ssl.SSLPeerUnverifiedException
javax.net.ssl.SSLProtocolException

import org.python.core.io.DatagramSocketIO
import org.python.core.io.ServerSocketIO
import org.python.core.io.SocketIO
from org.python.core.Py import newString as asPyString

class error(Exception): pass
class herror(error): pass
class gaierror(error): pass
class timeout(error): pass
class sslerror(error): pass

ALL = None

_exception_map = {

# (<javaexception>, <circumstance>) : lambda: <code that raises the python equivalent>, or None to stub out as unmapped

(java.io.IOException, ALL)            : lambda: error(errno.ECONNRESET, 'Software caused connection abort'),
(java.io.InterruptedIOException, ALL) : lambda: timeout('timed out'),

(java.net.BindException, ALL)            : lambda: error(errno.EADDRINUSE, 'Address already in use'),
(java.net.ConnectException, ALL)         : lambda: error(errno.ECONNREFUSED, 'Connection refused'),
(java.net.NoRouteToHostException, ALL)   : None,
(java.net.PortUnreachableException, ALL) : None,
(java.net.ProtocolException, ALL)        : None,
(java.net.SocketException, ALL)          : None,
(java.net.SocketTimeoutException, ALL)   : lambda: timeout('timed out'),
(java.net.UnknownHostException, ALL)     : lambda: gaierror(errno.EGETADDRINFOFAILED, 'getaddrinfo failed'),

(java.nio.channels.AlreadyConnectedException, ALL)       : lambda: error(errno.EISCONN, 'Socket is already connected'),
(java.nio.channels.AsynchronousCloseException, ALL)      : None,
(java.nio.channels.CancelledKeyException, ALL)           : None,
(java.nio.channels.ClosedByInterruptException, ALL)      : None,
(java.nio.channels.ClosedChannelException, ALL)          : lambda: error(errno.EPIPE, 'Socket closed'),
(java.nio.channels.ClosedSelectorException, ALL)         : None,
(java.nio.channels.ConnectionPendingException, ALL)      : None,
(java.nio.channels.IllegalBlockingModeException, ALL)    : None,
(java.nio.channels.IllegalSelectorException, ALL)        : None,
(java.nio.channels.NoConnectionPendingException, ALL)    : None,
(java.nio.channels.NonReadableChannelException, ALL)     : None,
(java.nio.channels.NonWritableChannelException, ALL)     : None,
(java.nio.channels.NotYetBoundException, ALL)            : None,
(java.nio.channels.NotYetConnectedException, ALL)        : None,
(java.nio.channels.UnresolvedAddressException, ALL)      : lambda: gaierror(errno.EGETADDRINFOFAILED, 'getaddrinfo failed'),
(java.nio.channels.UnsupportedAddressTypeException, ALL) : None,

# These error codes are currently wrong: getting them correct is going to require
# some investigation. Cpython 2.6 introduced extensive SSL support.

(javax.net.ssl.SSLException, ALL)                        : lambda: sslerror(-1, 'SSL exception'),
(javax.net.ssl.SSLHandshakeException, ALL)               : lambda: sslerror(-1, 'SSL handshake exception'),
(javax.net.ssl.SSLKeyException, ALL)                     : lambda: sslerror(-1, 'SSL key exception'),
(javax.net.ssl.SSLPeerUnverifiedException, ALL)          : lambda: sslerror(-1, 'SSL peer unverified exception'),
(javax.net.ssl.SSLProtocolException, ALL)                : lambda: sslerror(-1, 'SSL protocol exception'),

}

def would_block_error(exc=None):
    return error(errno.EWOULDBLOCK, 'The socket operation could not complete without blocking')

def _map_exception(exc, circumstance=ALL):
#    print "Mapping exception: %s" % exc
    mapped_exception = _exception_map.get((exc.__class__, circumstance))
    if mapped_exception:
        exception = mapped_exception()
    else:
        exception = error(-1, 'Unmapped exception: %s' % exc)
    exception.java_exception = exc
    return exception

MODE_BLOCKING    = 'block'
MODE_NONBLOCKING = 'nonblock'
MODE_TIMEOUT     = 'timeout'

_permitted_modes = (MODE_BLOCKING, MODE_NONBLOCKING, MODE_TIMEOUT)

SHUT_RD   = 0
SHUT_WR   = 1
SHUT_RDWR = 2

AF_UNSPEC = 0
AF_INET = 2
AF_INET6 = 23

AI_PASSIVE=1
AI_CANONNAME=2

# For some reason, probably historical, SOCK_DGRAM and SOCK_STREAM are opposite values of what they are on cpython.
# I.E. The following is the way they are on cpython
# SOCK_STREAM    = 1
# SOCK_DGRAM     = 2
# At some point, we should probably switch them around, which *should* not affect anybody

SOCK_DGRAM     = 1
SOCK_STREAM    = 2
SOCK_RAW       = 3 # not supported
SOCK_RDM       = 4 # not supported
SOCK_SEQPACKET = 5 # not supported

SOL_SOCKET = 0xFFFF
IPPROTO_TCP = 6
IPPROTO_UDP = 17

SO_BROADCAST   = 1
SO_KEEPALIVE   = 2
SO_LINGER      = 4
SO_OOBINLINE   = 8
SO_RCVBUF      = 16
SO_REUSEADDR   = 32
SO_SNDBUF      = 64
SO_TIMEOUT     = 128

TCP_NODELAY    = 256

INADDR_ANY = "0.0.0.0"
INADDR_BROADCAST = "255.255.255.255"

# Options with negative constants are not supported
# They are being added here so that code that refers to them
# will not break with an AttributeError

SO_ACCEPTCONN       = -1
SO_DEBUG            = -2
SO_DONTROUTE        = -4
SO_ERROR            = -8
SO_EXCLUSIVEADDRUSE = -16
SO_RCVLOWAT         = -32
SO_RCVTIMEO         = -64
SO_REUSEPORT        = -128
SO_SNDLOWAT         = -256
SO_SNDTIMEO         = -512
SO_TYPE             = -1024
SO_USELOOPBACK      = -2048

__all__ = ['AF_UNSPEC', 'AF_INET', 'AF_INET6', 'AI_PASSIVE', 'SOCK_DGRAM',
        'SOCK_RAW', 'SOCK_RDM', 'SOCK_SEQPACKET', 'SOCK_STREAM', 'SOL_SOCKET',
        'SO_BROADCAST', 'SO_ERROR', 'SO_KEEPALIVE', 'SO_LINGER', 'SO_OOBINLINE',
        'SO_RCVBUF', 'SO_REUSEADDR', 'SO_SNDBUF', 'SO_TIMEOUT', 'TCP_NODELAY',
        'INADDR_ANY', 'INADDR_BROADCAST', 'IPPROTO_TCP', 'IPPROTO_UDP',
        'SocketType', 'error', 'herror', 'gaierror', 'timeout',
        'getfqdn', 'gethostbyaddr', 'gethostbyname', 'gethostname',
        'socket', 'getaddrinfo', 'getdefaulttimeout', 'setdefaulttimeout',
        'has_ipv6', 'htons', 'htonl', 'ntohs', 'ntohl',
        'SHUT_RD', 'SHUT_WR', 'SHUT_RDWR',
        ]

def _constant_to_name(const_value):
    sock_module = sys.modules['socket']
    try:
        for name in dir(sock_module):
            if getattr(sock_module, name) is const_value:
                return name
        return "Unknown"
    finally:
        sock_module = None

class _nio_impl:

    timeout = None
    mode = MODE_BLOCKING

    def getpeername(self):
        return (self.jsocket.getInetAddress().getHostAddress(), self.jsocket.getPort() )

    def config(self, mode, timeout):
        self.mode = mode
        if self.mode == MODE_BLOCKING:
            self.jchannel.configureBlocking(1)
        if self.mode == MODE_NONBLOCKING:
            self.jchannel.configureBlocking(0)
        if self.mode == MODE_TIMEOUT:
            self.jchannel.configureBlocking(1)
            self._timeout_millis = int(timeout*1000)
            self.jsocket.setSoTimeout(self._timeout_millis)

    def getsockopt(self, level, option):
        if self.options.has_key( (level, option) ):
            result = getattr(self.jsocket, "get%s" % self.options[ (level, option) ])()
            if option == SO_LINGER:
                if result == -1:
                    enabled, linger_time = 0, 0
                else:
                    enabled, linger_time = 1, result
                return struct.pack('ii', enabled, linger_time)
            return result
        else:
            raise error(errno.ENOPROTOOPT, "Socket option '%s' (level '%s') not supported on socket(%s)" % (_constant_to_name(option), _constant_to_name(level), str(self.jsocket)))

    def setsockopt(self, level, option, value):
        if self.options.has_key( (level, option) ):
            if option == SO_LINGER:
                values = struct.unpack('ii', value)
                self.jsocket.setSoLinger(*values)
            else:
                getattr(self.jsocket, "set%s" % self.options[ (level, option) ])(value)
        else:
            raise error(errno.ENOPROTOOPT, "Socket option '%s' (level '%s') not supported on socket(%s)" % (_constant_to_name(option), _constant_to_name(level), str(self.jsocket)))

    def close(self):
        self.jsocket.close()

    def getchannel(self):
        return self.jchannel

    def fileno(self):
        return self.socketio

class _client_socket_impl(_nio_impl):

    options = {
        (SOL_SOCKET,  SO_KEEPALIVE):   'KeepAlive',
        (SOL_SOCKET,  SO_LINGER):      'SoLinger',
        (SOL_SOCKET,  SO_OOBINLINE):   'OOBInline',
        (SOL_SOCKET,  SO_RCVBUF):      'ReceiveBufferSize',
        (SOL_SOCKET,  SO_REUSEADDR):   'ReuseAddress',
        (SOL_SOCKET,  SO_SNDBUF):      'SendBufferSize',
        (SOL_SOCKET,  SO_TIMEOUT):     'SoTimeout',
        (IPPROTO_TCP, TCP_NODELAY):    'TcpNoDelay',
    }

    def __init__(self, socket=None):
        if socket:
            self.jchannel = socket.getChannel()
            self.host = socket.getInetAddress().getHostAddress()
            self.port = socket.getPort()
        else:
            self.jchannel = java.nio.channels.SocketChannel.open()
            self.host = None
            self.port = None
        self.jsocket = self.jchannel.socket()
        self.socketio = org.python.core.io.SocketIO(self.jchannel, 'rw')

    def bind(self, host, port, reuse_addr):
        self.jsocket.setReuseAddress(reuse_addr)
        self.jsocket.bind(java.net.InetSocketAddress(host, port))

    def connect(self, host, port):
        self.host = host
        self.port = port
        if self.mode == MODE_TIMEOUT:
            self.jsocket.connect(java.net.InetSocketAddress(self.host, self.port), self._timeout_millis)
        else:
            self.jchannel.connect(java.net.InetSocketAddress(self.host, self.port))

    def finish_connect(self):
        return self.jchannel.finishConnect()

    def _do_read_net(self, buf):
        # Need two separate implementations because the java.nio APIs do not support timeouts
        return self.jsocket.getInputStream().read(buf)

    def _do_read_nio(self, buf):
        bytebuf = java.nio.ByteBuffer.wrap(buf)
        count = self.jchannel.read(bytebuf)
        return count

    def _do_write_net(self, buf):
        self.jsocket.getOutputStream().write(buf)
        return len(buf)

    def _do_write_nio(self, buf):
        bytebuf = java.nio.ByteBuffer.wrap(buf)
        count = self.jchannel.write(bytebuf)
        return count

    def read(self, buf):
        if self.mode == MODE_TIMEOUT:
            return self._do_read_net(buf)
        else:
            return self._do_read_nio(buf)

    def write(self, buf):
        if self.mode == MODE_TIMEOUT:
            return self._do_write_net(buf)
        else:
            return self._do_write_nio(buf)

    def shutdown(self, how):
        if how in (SHUT_RD, SHUT_RDWR):
            self.jsocket.shutdownInput()
        if how in (SHUT_WR, SHUT_RDWR):
            self.jsocket.shutdownOutput()

class _server_socket_impl(_nio_impl):

    options = {
        (SOL_SOCKET, SO_RCVBUF):      'ReceiveBufferSize',
        (SOL_SOCKET, SO_REUSEADDR):   'ReuseAddress',
        (SOL_SOCKET, SO_TIMEOUT):     'SoTimeout',
    }

    def __init__(self, host, port, backlog, reuse_addr):
        self.jchannel = java.nio.channels.ServerSocketChannel.open()
        self.jsocket = self.jchannel.socket()
        if host:
            bindaddr = java.net.InetSocketAddress(host, port)
        else:
            bindaddr = java.net.InetSocketAddress(port)
        self.jsocket.setReuseAddress(reuse_addr)
        self.jsocket.bind(bindaddr, backlog)
        self.socketio = org.python.core.io.ServerSocketIO(self.jchannel, 'rw')

    def accept(self):
        if self.mode in (MODE_BLOCKING, MODE_NONBLOCKING):
            new_cli_chan = self.jchannel.accept()
            if new_cli_chan != None:
                return _client_socket_impl(new_cli_chan.socket())
            else:
                return None
        else:
            # In timeout mode now
            new_cli_sock = self.jsocket.accept()
            return _client_socket_impl(new_cli_sock)

    def shutdown(self, how):
        # This is no-op on java, for server sockets.
        # What the user wants to achieve is achieved by calling close() on
        # java/jython. But we can't call that here because that would then
        # later cause the user explicit close() call to fail
        pass

class _datagram_socket_impl(_nio_impl):

    options = {
        (SOL_SOCKET, SO_BROADCAST):   'Broadcast',
        (SOL_SOCKET, SO_RCVBUF):      'ReceiveBufferSize',
        (SOL_SOCKET, SO_REUSEADDR):   'ReuseAddress',
        (SOL_SOCKET, SO_SNDBUF):      'SendBufferSize',
        (SOL_SOCKET, SO_TIMEOUT):     'SoTimeout',
    }

    def __init__(self, port=None, address=None, reuse_addr=0):
        self.jchannel = java.nio.channels.DatagramChannel.open()
        self.jsocket = self.jchannel.socket()
        if port is not None:
            if address is not None:
                local_address = java.net.InetSocketAddress(address, port)
            else:
                local_address = java.net.InetSocketAddress(port)
            self.jsocket.setReuseAddress(reuse_addr)
            self.jsocket.bind(local_address)
        self.socketio = org.python.core.io.DatagramSocketIO(self.jchannel, 'rw')

    def connect(self, host, port):
        self.jchannel.connect(java.net.InetSocketAddress(host, port))

    def disconnect(self):
        """
            Disconnect the datagram socket.
            cpython appears not to have this operation
        """
        self.jchannel.disconnect()

    def shutdown(self, how):
        # This is no-op on java, for datagram sockets.
        # What the user wants to achieve is achieved by calling close() on
        # java/jython. But we can't call that here because that would then
        # later cause the user explicit close() call to fail
        pass

    def _do_send_net(self, byte_array, socket_address, flags):
        # Need two separate implementations because the java.nio APIs do not support timeouts
        num_bytes = len(byte_array)
        if self.jsocket.isConnected() and socket_address is None:
            packet = java.net.DatagramPacket(byte_array, num_bytes)
        else:
            packet = java.net.DatagramPacket(byte_array, num_bytes, socket_address)
        self.jsocket.send(packet)
        return num_bytes

    def _do_send_nio(self, byte_array, socket_address, flags):
        byte_buf = java.nio.ByteBuffer.wrap(byte_array)
        if self.jchannel.isConnected() and socket_address is None:
            bytes_sent = self.jchannel.write(byte_buf)
        else:
            bytes_sent = self.jchannel.send(byte_buf, socket_address)
        return bytes_sent

    def sendto(self, byte_array, host, port, flags):
        socket_address = java.net.InetSocketAddress(host, port)
        if self.mode == MODE_TIMEOUT:
            return self._do_send_net(byte_array, socket_address, flags)
        else:
            return self._do_send_nio(byte_array, socket_address, flags)

    def send(self, byte_array, flags):
        if self.mode == MODE_TIMEOUT:
            return self._do_send_net(byte_array, None, flags)
        else:
            return self._do_send_nio(byte_array, None, flags)

    def _do_receive_net(self, return_source_address, num_bytes, flags):
        byte_array = jarray.zeros(num_bytes, 'b')
        packet = java.net.DatagramPacket(byte_array, num_bytes)
        self.jsocket.receive(packet)
        bytes_rcvd = packet.getLength()
        if bytes_rcvd < num_bytes:
            byte_array = byte_array[:bytes_rcvd]
        return_data = byte_array.tostring()
        if return_source_address:
            host = None
            if packet.getAddress():
                host = packet.getAddress().getHostAddress()
            port = packet.getPort()
            return return_data, (host, port)
        else:
            return return_data

    def _do_receive_nio(self, return_source_address, num_bytes, flags):
        byte_array = jarray.zeros(num_bytes, 'b')
        byte_buf = java.nio.ByteBuffer.wrap(byte_array)
        source_address = self.jchannel.receive(byte_buf)
        if source_address is None and not self.jchannel.isBlocking():
            raise would_block_error()
        byte_buf.flip() ; bytes_read = byte_buf.remaining()
        if bytes_read < num_bytes:
            byte_array = byte_array[:bytes_read]
        return_data = byte_array.tostring()
        if return_source_address:
            return return_data, (source_address.getAddress().getHostAddress(), source_address.getPort())
        else:
            return return_data

    def recvfrom(self, num_bytes, flags):
        if self.mode == MODE_TIMEOUT:
            return self._do_receive_net(1, num_bytes, flags)
        else:
            return self._do_receive_nio(1, num_bytes, flags)

    def recv(self, num_bytes, flags):
        if self.mode == MODE_TIMEOUT:
            return self._do_receive_net(0, num_bytes, flags)
        else:
            return self._do_receive_nio(0, num_bytes, flags)

# For now, we DO NOT have complete IPV6 support.
has_ipv6 = False

# Name and address functions

def _gethostbyaddr(name):
    # This is as close as I can get; at least the types are correct...
    addresses = java.net.InetAddress.getAllByName(gethostbyname(name))
    names = []
    addrs = []
    for addr in addresses:
        names.append(asPyString(addr.getHostName()))
        addrs.append(asPyString(addr.getHostAddress()))
    return (names, addrs)

def getfqdn(name=None):
    """
    Return a fully qualified domain name for name. If name is omitted or empty
    it is interpreted as the local host.  To find the fully qualified name,
    the hostname returned by gethostbyaddr() is checked, then aliases for the
    host, if available. The first name which includes a period is selected.
    In case no fully qualified domain name is available, the hostname is retur
    New in version 2.0.
    """
    if not name:
        name = gethostname()
    names, addrs = _gethostbyaddr(name)
    for a in names:
        if a.find(".") >= 0:
            return a
    return name

def gethostname():
    try:
        return asPyString(java.net.InetAddress.getLocalHost().getHostName())
    except java.lang.Exception, jlx:
        raise _map_exception(jlx)

def gethostbyname(name):
    try:
        return asPyString(java.net.InetAddress.getByName(name).getHostAddress())
    except java.lang.Exception, jlx:
        raise _map_exception(jlx)

def gethostbyaddr(name):
    names, addrs = _gethostbyaddr(name)
    return (names[0], names, addrs)

def getservbyname(servicename, protocolname=None):
    # http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=4071389
    # How complex is the structure of /etc/services?
    raise NotImplementedError("getservbyname not yet supported on jython.")

def getservbyport(port, protocolname=None):
    # Same situation as above
    raise NotImplementedError("getservbyport not yet supported on jython.")

def getprotobyname(protocolname=None):
    # Same situation as above
    raise NotImplementedError("getprotobyname not yet supported on jython.")

def _realsocket(family = AF_INET, type = SOCK_STREAM, protocol=0):
    assert family == AF_INET, "Only AF_INET sockets are currently supported on jython"
    assert type in (SOCK_DGRAM, SOCK_STREAM), "Only SOCK_STREAM and SOCK_DGRAM sockets are currently supported on jython"
    if type == SOCK_STREAM:
        if protocol != 0:
            assert protocol == IPPROTO_TCP, "Only IPPROTO_TCP supported on SOCK_STREAM sockets"
        return _tcpsocket()
    else:
        if protocol != 0:
            assert protocol == IPPROTO_UDP, "Only IPPROTO_UDP supported on SOCK_DGRAM sockets"
        return _udpsocket()

def getaddrinfo(host, port, family=AF_INET, socktype=None, proto=0, flags=None):
    try:
        if not family in [AF_INET, AF_INET6, AF_UNSPEC]:
            raise gaierror(errno.EIO, 'ai_family not supported')
        filter_fns = []
        filter_fns.append({
            AF_INET:   lambda x: isinstance(x, java.net.Inet4Address),
            AF_INET6:  lambda x: isinstance(x, java.net.Inet6Address),
            AF_UNSPEC: lambda x: isinstance(x, java.net.InetAddress),
        }[family])
        if host == "":
            host = java.net.InetAddress.getLocalHost().getHostName()
        passive_mode = flags is not None and flags & AI_PASSIVE
        canonname_mode = flags is not None and flags & AI_CANONNAME
        results = []
        for a in java.net.InetAddress.getAllByName(host):
            if len([f for f in filter_fns if f(a)]):
                family = {java.net.Inet4Address: AF_INET, java.net.Inet6Address: AF_INET6}[a.getClass()]
                if passive_mode and not canonname_mode:
                    canonname = ""
                else:
                    canonname = asPyString(a.getCanonicalHostName())
                if host is None and passive_mode and not canonname_mode:
                    sockname = INADDR_ANY
                else:
                    sockname = asPyString(a.getHostAddress())
                # TODO: Include flowinfo and scopeid in a 4-tuple for IPv6 addresses
                results.append((family, socktype, proto, canonname, (sockname, port)))
        return results
    except java.lang.Exception, jlx:
        raise _map_exception(jlx)

def getnameinfo(sock_addr, flags):
    raise NotImplementedError("getnameinfo not yet supported on jython.")

def getdefaulttimeout():
    return _defaulttimeout

def _calctimeoutvalue(value):
    if value is None:
        return None
    try:
        floatvalue = float(value)
    except:
        raise TypeError('Socket timeout value must be a number or None')
    if floatvalue < 0.0:
        raise ValueError("Socket timeout value cannot be negative")
    if floatvalue < 0.000001:
        return 0.0
    return floatvalue

def setdefaulttimeout(timeout):
    global _defaulttimeout
    try:
        _defaulttimeout = _calctimeoutvalue(timeout)
    finally:
        _nonblocking_api_mixin.timeout = _defaulttimeout

def htons(x): return x
def htonl(x): return x
def ntohs(x): return x
def ntohl(x): return x

def inet_pton(family, ip_string):
    try:
        ia = java.net.InetAddress.getByName(ip_string)
        bytes = []
        for byte in ia.getAddress():
            if byte < 0:
                bytes.append(byte+256)
            else:
                bytes.append(byte)
        return "".join([chr(byte) for byte in bytes])
    except java.lang.Exception, jlx:
        raise _map_exception(jlx)

def inet_ntop(family, packed_ip):
    try:
        jByteArray = jarray.array(packed_ip, 'b')
        ia = java.net.InetAddress.getByAddress(jByteArray)
        return ia.getHostAddress()
    except java.lang.Exception, jlx:
        raise _map_exception(jlx)

def inet_aton(ip_string):
    return inet_pton(AF_INET, ip_string)

def inet_ntoa(packed_ip):
    return inet_ntop(AF_INET, packed_ip)

class _nonblocking_api_mixin:

    mode = MODE_BLOCKING
    reference_count = 0
    close_lock = threading.Lock()

    def __init__(self):
        self.timeout = _defaulttimeout
        if self.timeout is not None:
            self.mode = MODE_TIMEOUT
        self.pending_options = {
            (SOL_SOCKET, SO_REUSEADDR):  0,
        }

    def gettimeout(self):
        return self.timeout

    def settimeout(self, timeout):
        self.timeout = _calctimeoutvalue(timeout)
        if self.timeout is None:
            self.mode = MODE_BLOCKING
        elif self.timeout < 0.000001:
            self.mode = MODE_NONBLOCKING
        else:
            self.mode = MODE_TIMEOUT
        self._config()

    def setblocking(self, flag):
        if flag:
            self.mode = MODE_BLOCKING
            self.timeout = None
        else:
            self.mode = MODE_NONBLOCKING
            self.timeout = 0.0
        self._config()

    def getblocking(self):
        return self.mode == MODE_BLOCKING

    def setsockopt(self, level, optname, value):
        try:
            if self.sock_impl:
                self.sock_impl.setsockopt(level, optname, value)
            else:
                self.pending_options[ (level, optname) ] = value
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def getsockopt(self, level, optname):
        try:
            if self.sock_impl:
                return self.sock_impl.getsockopt(level, optname)
            else:
                return self.pending_options.get( (level, optname), None)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def shutdown(self, how):
        assert how in (SHUT_RD, SHUT_WR, SHUT_RDWR)
        if not self.sock_impl:
            raise error(errno.ENOTCONN, "Transport endpoint is not connected")
        try:
            self.sock_impl.shutdown(how)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def close(self):
        try:
            if self.sock_impl:
                self.sock_impl.close()
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def _config(self):
        assert self.mode in _permitted_modes
        if self.sock_impl:
            self.sock_impl.config(self.mode, self.timeout)
            for level, optname in self.pending_options.keys():
                if optname != SO_REUSEADDR:
                    self.sock_impl.setsockopt(level, optname, self.pending_options[ (level, optname) ])

    def getchannel(self):
        if not self.sock_impl:
            return None
        return self.sock_impl.getchannel()

    def fileno(self):
        if not self.sock_impl:
            return None
        return self.sock_impl.fileno()

    def _get_jsocket(self):
        return self.sock_impl.jsocket

def _unpack_address_tuple(address_tuple):
    # TODO: Upgrade to support the 4-tuples used for IPv6 addresses
    # which include flowinfo and scope_id.
    # To be upgraded in synch with getaddrinfo
    error_message = "Address must be a tuple of (hostname, port)"
    if not isinstance(address_tuple, tuple) or \
            not isinstance(address_tuple[0], basestring) or \
            not isinstance(address_tuple[1], (int, long)):
        raise TypeError(error_message)
    hostname = address_tuple[0]
    if isinstance(hostname, unicode):
        # XXX: Should be encode('idna') (See CPython
        # socketmodule::getsockaddrarg), but Jython's idna support is
        # currently broken
        hostname = hostname.encode()
    hostname = hostname.strip()
    return hostname, address_tuple[1]

class _tcpsocket(_nonblocking_api_mixin):

    sock_impl = None
    istream = None
    ostream = None
    local_addr = None
    server = 0

    def __init__(self):
        _nonblocking_api_mixin.__init__(self)

    def bind(self, addr):
        assert not self.sock_impl
        assert not self.local_addr
        # Do the address format check
        _unpack_address_tuple(addr)
        self.local_addr = addr

    def listen(self, backlog):
        "This signifies a server socket"
        try:
            assert not self.sock_impl
            self.server = 1
            if self.local_addr:
                host, port = _unpack_address_tuple(self.local_addr)
            else:
                host, port = "", 0
            self.sock_impl = _server_socket_impl(host, port, backlog, self.pending_options[ (SOL_SOCKET, SO_REUSEADDR) ])
            self._config()
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def accept(self):
        "This signifies a server socket"
        try:
            if not self.sock_impl:
                self.listen()
            assert self.server
            new_sock = self.sock_impl.accept()
            if not new_sock:
                raise would_block_error()
            cliconn = _tcpsocket()
            cliconn.pending_options[ (SOL_SOCKET, SO_REUSEADDR) ] = new_sock.jsocket.getReuseAddress()
            cliconn.sock_impl = new_sock
            cliconn._setup()
            return cliconn, new_sock.getpeername()
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def _get_host_port(self, addr):
        host, port = _unpack_address_tuple(addr)
        if host == "":
            host = java.net.InetAddress.getLocalHost()
        return host, port

    def _do_connect(self, addr):
        try:
            assert not self.sock_impl
            host, port = self._get_host_port(addr)
            self.sock_impl = _client_socket_impl()
            if self.local_addr: # Has the socket been bound to a local address?
                bind_host, bind_port = _unpack_address_tuple(self.local_addr)
                self.sock_impl.bind(bind_host, bind_port, self.pending_options[ (SOL_SOCKET, SO_REUSEADDR) ])
            self._config() # Configure timeouts, etc, now that the socket exists
            self.sock_impl.connect(host, port)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def connect(self, addr):
        "This signifies a client socket"
        self._do_connect(addr)
        self._setup()

    def connect_ex(self, addr):
        "This signifies a client socket"
        if not self.sock_impl:
            self._do_connect(addr)
        if self.sock_impl.finish_connect():
            self._setup()
            if self.mode == MODE_NONBLOCKING:
                return errno.EISCONN
            return 0
        return errno.EINPROGRESS

    def _setup(self):
        if self.mode != MODE_NONBLOCKING:
            self.istream = self.sock_impl.jsocket.getInputStream()
            self.ostream = self.sock_impl.jsocket.getOutputStream()

    def recv(self, n):
        try:
            if not self.sock_impl: raise error(errno.ENOTCONN, 'Socket is not connected')
            if self.sock_impl.jchannel.isConnectionPending():
                self.sock_impl.jchannel.finishConnect()
            data = jarray.zeros(n, 'b')
            m = self.sock_impl.read(data)
            if m == -1:#indicates EOF has been reached, so we just return the empty string
                return ""
            elif m <= 0:
                if self.mode == MODE_NONBLOCKING:
                    raise would_block_error()
                return ""
            if m < n:
                data = data[:m]
            return data.tostring()
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def recvfrom(self, n):
        return self.recv(n), None

    def send(self, s):
        try:
            if not self.sock_impl: raise error(errno.ENOTCONN, 'Socket is not connected')
            if self.sock_impl.jchannel.isConnectionPending():
                self.sock_impl.jchannel.finishConnect()
            numwritten = self.sock_impl.write(s)
            if numwritten == 0 and self.mode == MODE_NONBLOCKING:
                raise would_block_error()
            return numwritten
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    sendall = send

    def getsockname(self):
        try:
            if not self.sock_impl:
                host, port = self.local_addr or ("", 0)
                host = java.net.InetAddress.getByName(host).getHostAddress()
            else:
                if self.server:
                    host = self.sock_impl.jsocket.getInetAddress().getHostAddress()
                else:
                    host = self.sock_impl.jsocket.getLocalAddress().getHostAddress()
                port = self.sock_impl.jsocket.getLocalPort()
            return (host, port)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def getpeername(self):
        try:
            assert self.sock_impl
            assert not self.server
            host = self.sock_impl.jsocket.getInetAddress().getHostAddress()
            port = self.sock_impl.jsocket.getPort()
            return (host, port)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def close(self):
        try:
            if self.istream:
                self.istream.close()
            if self.ostream:
                self.ostream.close()
            if self.sock_impl:
                self.sock_impl.close()
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)


class _udpsocket(_nonblocking_api_mixin):

    sock_impl = None
    addr = None

    def __init__(self):
        _nonblocking_api_mixin.__init__(self)

    def bind(self, addr):
        try:
            assert not self.sock_impl
            host, port = _unpack_address_tuple(addr)
            if host == "":
                host = INADDR_ANY
            host_address = java.net.InetAddress.getByName(host)
            self.sock_impl = _datagram_socket_impl(port, host_address, self.pending_options[ (SOL_SOCKET, SO_REUSEADDR) ])
            self._config()
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def _do_connect(self, addr):
        try:
            host, port = _unpack_address_tuple(addr)
            assert not self.addr
            self.addr = addr
            if not self.sock_impl:
                self.sock_impl = _datagram_socket_impl()
                self._config()
                self.sock_impl.connect(host, port)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def connect(self, addr):
        self._do_connect(addr)

    def connect_ex(self, addr):
        if not self.sock_impl:
            self._do_connect(addr)
        return 0

    def sendto(self, data, p1, p2=None):
        try:
            if not p2:
                flags, addr = 0, p1
            else:
                flags, addr = 0, p2
            if not self.sock_impl:
                self.sock_impl = _datagram_socket_impl()
                self._config()
            host, port = _unpack_address_tuple(addr)
            if host == "<broadcast>":
                host = INADDR_BROADCAST
            byte_array = java.lang.String(data).getBytes('iso-8859-1')
            result = self.sock_impl.sendto(byte_array, host, port, flags)
            return result
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def send(self, data, flags=None):
        if not self.addr: raise error(errno.ENOTCONN, "Socket is not connected")
        byte_array = java.lang.String(data).getBytes('iso-8859-1')
        return self.sock_impl.send(byte_array, flags)

    def recvfrom(self, num_bytes, flags=None):
        """
        There is some disagreement as to what the behaviour should be if
        a recvfrom operation is requested on an unbound socket.
        See the following links for more information
        http://bugs.jython.org/issue1005
        http://bugs.sun.com/view_bug.do?bug_id=6621689
        """
        try:
            # This is the old 2.1 behaviour
            #assert self.sock_impl
            # This is amak's preferred interpretation
            #raise error(errno.ENOTCONN, "Recvfrom on unbound udp socket meaningless operation")
            # And this is the option for cpython compatibility
            if not self.sock_impl:
                self.sock_impl = _datagram_socket_impl()
                self._config()
            return self.sock_impl.recvfrom(num_bytes, flags)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def recv(self, num_bytes, flags=None):
        if not self.sock_impl: raise error(errno.ENOTCONN, "Socket is not connected")
        try:
            return self.sock_impl.recv(num_bytes, flags)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def getsockname(self):
        try:
            assert self.sock_impl
            host = self.sock_impl.jsocket.getLocalAddress().getHostAddress()
            port = self.sock_impl.jsocket.getLocalPort()
            return (host, port)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def getpeername(self):
        try:
            assert self.sock
            host = self.sock_impl.jsocket.getInetAddress().getHostAddress()
            port = self.sock_impl.jsocket.getPort()
            return (host, port)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def __del__(self):
        self.close()

_socketmethods = (
    'bind', 'connect', 'connect_ex', 'fileno', 'listen',
    'getpeername', 'getsockname', 'getsockopt', 'setsockopt',
    'sendall', 'setblocking',
    'settimeout', 'gettimeout', 'shutdown', 'getchannel')

# All the method names that must be delegated to either the real socket
# object or the _closedsocket object.
_delegate_methods = ("recv", "recvfrom", "recv_into", "recvfrom_into",
                     "send", "sendto")

class _closedsocket(object):
    __slots__ = []
    def _dummy(*args):
        raise error(errno.EBADF, 'Bad file descriptor')
    # All _delegate_methods must also be initialized here.
    send = recv = recv_into = sendto = recvfrom = recvfrom_into = _dummy
    __getattr__ = _dummy

_active_sockets = set()

def _closeActiveSockets():
    for socket in _active_sockets.copy():
        try:
            socket.close()
        except error:
            msg = 'Problem closing socket: %s: %r' % (socket, sys.exc_info())
            print >> sys.stderr, msg

class _socketobject(object):

    __doc__ = _realsocket.__doc__

    __slots__ = ["_sock", "__weakref__"] + list(_delegate_methods)

    def __init__(self, family=AF_INET, type=SOCK_STREAM, proto=0, _sock=None):
        if _sock is None:
            _sock = _realsocket(family, type, proto)
            _sock.reference_count += 1
        elif isinstance(_sock, _nonblocking_api_mixin):
            _sock.reference_count += 1
        self._sock = _sock
        for method in _delegate_methods:
            meth = getattr(_sock, method, None)
            if meth:
                setattr(self, method, meth)
        _active_sockets.add(self)

    def close(self):
        try:
            _active_sockets.remove(self)
        except KeyError:
            pass
        _sock = self._sock
        if isinstance(_sock, _nonblocking_api_mixin):
            _sock.close_lock.acquire()
            try:
                _sock.reference_count -=1
                if not _sock.reference_count:
                    _sock.close()
                self._sock = _closedsocket()
                dummy = self._sock._dummy
                for method in _delegate_methods:
                    setattr(self, method, dummy)
                self.send = self.recv = self.sendto = self.recvfrom = \
                    self._sock._dummy
            finally:
                _sock.close_lock.release()
    #close.__doc__ = _realsocket.close.__doc__

    def accept(self):
        sock, addr = self._sock.accept()
        return _socketobject(_sock=sock), addr
    #accept.__doc__ = _realsocket.accept.__doc__

    def dup(self):
        """dup() -> socket object

        Return a new socket object connected to the same system resource."""
        _sock = self._sock
        if not isinstance(_sock, _nonblocking_api_mixin):
            return _socketobject(_sock=_sock)

        _sock.close_lock.acquire()
        try:
            duped = _socketobject(_sock=_sock)
        finally:
            _sock.close_lock.release()
        return duped

    def makefile(self, mode='r', bufsize=-1):
        """makefile([mode[, bufsize]]) -> file object

        Return a regular file object corresponding to the socket.  The mode
        and bufsize arguments are as for the built-in open() function."""
        _sock = self._sock
        if not isinstance(_sock, _nonblocking_api_mixin):
            return _fileobject(_sock, mode, bufsize)

        _sock.close_lock.acquire()
        try:
            fileobject = _fileobject(_sock, mode, bufsize)
        finally:
            _sock.close_lock.release()
        return fileobject

    family = property(lambda self: self._sock.family, doc="the socket family")
    type = property(lambda self: self._sock.type, doc="the socket type")
    proto = property(lambda self: self._sock.proto, doc="the socket protocol")

    _s = ("def %s(self, *args): return self._sock.%s(*args)\n\n"
          #"%s.__doc__ = _realsocket.%s.__doc__\n")
          )
    for _m in _socketmethods:
        #exec _s % (_m, _m, _m, _m)
        exec _s % (_m, _m)
    del _m, _s

socket = SocketType = _socketobject

class _fileobject(object):
    """Faux file object attached to a socket object."""

    default_bufsize = 8192
    name = "<socket>"

    __slots__ = ["mode", "bufsize", "softspace",
                 # "closed" is a property, see below
                 "_sock", "_rbufsize", "_wbufsize", "_rbuf", "_wbuf",
                 "_close"]

    def __init__(self, sock, mode='rb', bufsize=-1, close=False):
        self._sock = sock
        if isinstance(sock, _nonblocking_api_mixin):
            sock.reference_count += 1
        self.mode = mode # Not actually used in this version
        if bufsize < 0:
            bufsize = self.default_bufsize
        self.bufsize = bufsize
        self.softspace = False
        if bufsize == 0:
            self._rbufsize = 1
        elif bufsize == 1:
            self._rbufsize = self.default_bufsize
        else:
            self._rbufsize = bufsize
        self._wbufsize = bufsize
        self._rbuf = "" # A string
        self._wbuf = [] # A list of strings
        self._close = close

    def _getclosed(self):
        return self._sock is None
    closed = property(_getclosed, doc="True if the file is closed")

    def close(self):
        try:
            if self._sock:
                self.flush()
        finally:
            if self._sock:
                if isinstance(self._sock, _nonblocking_api_mixin):
                    self._sock.reference_count -= 1
                    if not self._sock.reference_count or self._close:
                        self._sock.close()
                elif self._close:
                    self._sock.close()
            self._sock = None

    def __del__(self):
        try:
            self.close()
        except:
            # close() may fail if __init__ didn't complete
            pass

    def flush(self):
        if self._wbuf:
            buffer = "".join(self._wbuf)
            self._wbuf = []
            self._sock.sendall(buffer)

    def fileno(self):
        return self._sock.fileno()

    def write(self, data):
        data = str(data) # XXX Should really reject non-string non-buffers
        if not data:
            return
        self._wbuf.append(data)
        if (self._wbufsize == 0 or
            self._wbufsize == 1 and '\n' in data or
            self._get_wbuf_len() >= self._wbufsize):
            self.flush()

    def writelines(self, list):
        # XXX We could do better here for very long lists
        # XXX Should really reject non-string non-buffers
        self._wbuf.extend(filter(None, map(str, list)))
        if (self._wbufsize <= 1 or
            self._get_wbuf_len() >= self._wbufsize):
            self.flush()

    def _get_wbuf_len(self):
        buf_len = 0
        for x in self._wbuf:
            buf_len += len(x)
        return buf_len

    def read(self, size=-1):
        data = self._rbuf
        if size < 0:
            # Read until EOF
            buffers = []
            if data:
                buffers.append(data)
            self._rbuf = ""
            if self._rbufsize <= 1:
                recv_size = self.default_bufsize
            else:
                recv_size = self._rbufsize
            while True:
                data = self._sock.recv(recv_size)
                if not data:
                    break
                buffers.append(data)
            return "".join(buffers)
        else:
            # Read until size bytes or EOF seen, whichever comes first
            buf_len = len(data)
            if buf_len >= size:
                self._rbuf = data[size:]
                return data[:size]
            buffers = []
            if data:
                buffers.append(data)
            self._rbuf = ""
            while True:
                left = size - buf_len
                recv_size = max(self._rbufsize, left)
                data = self._sock.recv(recv_size)
                if not data:
                    break
                buffers.append(data)
                n = len(data)
                if n >= left:
                    self._rbuf = data[left:]
                    buffers[-1] = data[:left]
                    break
                buf_len += n
            return "".join(buffers)

    def readline(self, size=-1):
        data = self._rbuf
        if size < 0:
            # Read until \n or EOF, whichever comes first
            if self._rbufsize <= 1:
                # Speed up unbuffered case
                assert data == ""
                buffers = []
                recv = self._sock.recv
                while data != "\n":
                    data = recv(1)
                    if not data:
                        break
                    buffers.append(data)
                return "".join(buffers)
            nl = data.find('\n')
            if nl >= 0:
                nl += 1
                self._rbuf = data[nl:]
                return data[:nl]
            buffers = []
            if data:
                buffers.append(data)
            self._rbuf = ""
            while True:
                data = self._sock.recv(self._rbufsize)
                if not data:
                    break
                buffers.append(data)
                nl = data.find('\n')
                if nl >= 0:
                    nl += 1
                    self._rbuf = data[nl:]
                    buffers[-1] = data[:nl]
                    break
            return "".join(buffers)
        else:
            # Read until size bytes or \n or EOF seen, whichever comes first
            nl = data.find('\n', 0, size)
            if nl >= 0:
                nl += 1
                self._rbuf = data[nl:]
                return data[:nl]
            buf_len = len(data)
            if buf_len >= size:
                self._rbuf = data[size:]
                return data[:size]
            buffers = []
            if data:
                buffers.append(data)
            self._rbuf = ""
            while True:
                data = self._sock.recv(self._rbufsize)
                if not data:
                    break
                buffers.append(data)
                left = size - buf_len
                nl = data.find('\n', 0, left)
                if nl >= 0:
                    nl += 1
                    self._rbuf = data[nl:]
                    buffers[-1] = data[:nl]
                    break
                n = len(data)
                if n >= left:
                    self._rbuf = data[left:]
                    buffers[-1] = data[:left]
                    break
                buf_len += n
            return "".join(buffers)

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

    # Iterator protocols

    def __iter__(self):
        return self

    def next(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line


# Define the SSL support

class ssl:

    def __init__(self, plain_sock, keyfile=None, certfile=None):
        try:
            self.ssl_sock = self._make_ssl_socket(plain_sock)
            self._in_buf = java.io.BufferedInputStream(self.ssl_sock.getInputStream())
            self._out_buf = java.io.BufferedOutputStream(self.ssl_sock.getOutputStream())
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def _make_ssl_socket(self, plain_socket, auto_close=0):
        java_net_socket = plain_socket._get_jsocket()
        assert isinstance(java_net_socket, java.net.Socket)
        host = java_net_socket.getInetAddress().getHostAddress()
        port = java_net_socket.getPort()
        factory = javax.net.ssl.SSLSocketFactory.getDefault();
        ssl_socket = factory.createSocket(java_net_socket, host, port, auto_close)
        ssl_socket.setEnabledCipherSuites(ssl_socket.getSupportedCipherSuites())
        ssl_socket.startHandshake()
        return ssl_socket

    def read(self, n=4096):
        try:
            data = jarray.zeros(n, 'b')
            m = self._in_buf.read(data, 0, n)
            if m <= 0:
                return ""
            if m < n:
                data = data[:m]
            return data.tostring()
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def write(self, s):
        try:
            self._out_buf.write(s)
            self._out_buf.flush()
            return len(s)
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def _get_server_cert(self):
        try:
            return self.ssl_sock.getSession().getPeerCertificates()[0]
        except java.lang.Exception, jlx:
            raise _map_exception(jlx)

    def server(self):
        cert = self._get_server_cert()
        return cert.getSubjectDN().toString()

    def issuer(self):
        cert = self._get_server_cert()
        return cert.getIssuerDN().toString()

_realssl = ssl
def ssl(sock, keyfile=None, certfile=None):
    if hasattr(sock, "_sock"):
        sock = sock._sock
    return _realssl(sock, keyfile, certfile)

def test():
    s = socket(AF_INET, SOCK_STREAM)
    s.connect(("", 80))
    s.send("GET / HTTP/1.0\r\n\r\n")
    while 1:
        data = s.recv(2000)
        print data
        if not data:
            break

if __name__ == '__main__':
    test()
