# -*- Mode: Python -*-
#   Id: asyncore.py,v 2.51 2000/09/07 22:29:26 rushing Exp
#   Author: Sam Rushing <rushing@nightmare.com>

# ======================================================================
# Copyright 1996 by Sam Rushing
#
#                         All Rights Reserved
#
# Permission to use, copy, modify, and distribute this software and
# its documentation for any purpose and without fee is hereby
# granted, provided that the above copyright notice appear in all
# copies and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of Sam
# Rushing not be used in advertising or publicity pertaining to
# distribution of the software without specific, written prior
# permission.
#
# SAM RUSHING DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN
# NO EVENT SHALL SAM RUSHING BE LIABLE FOR ANY SPECIAL, INDIRECT OR
# CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# ======================================================================

"""Basic infrastructure for asynchronous socket service clients and servers.

There are only two ways to have a program on a single processor do "more
than one thing at a time".  Multi-threaded programming is the simplest and
most popular way to do it, but there is another very different technique,
that lets you have nearly all the advantages of multi-threading, without
actually using multiple threads. it's really only practical if your program
is largely I/O bound. If your program is CPU bound, then pre-emptive
scheduled threads are probably what you really need. Network servers are
rarely CPU-bound, however.

If your operating system supports the select() system call in its I/O
library (and nearly all do), then you can use it to juggle multiple
communication channels at once; doing other work while your I/O is taking
place in the "background."  Although this strategy can seem strange and
complex, especially at first, it is in many ways easier to understand and
control than multi-threaded programming. The module documented here solves
many of the difficult problems for you, making the task of building
sophisticated high-performance network servers and clients a snap.
"""

import select
import socket
import sys
import time

import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, \
     ENOTCONN, ESHUTDOWN, EINTR, EISCONN, errorcode

try:
    socket_map
except NameError:
    socket_map = {}

class ExitNow(Exception):
    pass

def read(obj):
    try:
        obj.handle_read_event()
    except ExitNow:
        raise
    except:
        obj.handle_error()

def write(obj):
    try:
        obj.handle_write_event()
    except ExitNow:
        raise
    except:
        obj.handle_error()

def _exception (obj):
    try:
        obj.handle_expt_event()
    except ExitNow:
        raise
    except:
        obj.handle_error()

def readwrite(obj, flags):
    try:
        if flags & (select.POLLIN | select.POLLPRI):
            obj.handle_read_event()
        if flags & select.POLLOUT:
            obj.handle_write_event()
        if flags & (select.POLLERR | select.POLLHUP | select.POLLNVAL):
            obj.handle_expt_event()
    except ExitNow:
        raise
    except:
        obj.handle_error()

def poll(timeout=0.0, map=None):
    if map is None:
        map = socket_map
    if map:
        r = []; w = []; e = []
        for fd, obj in map.items():
            is_r = obj.readable()
            is_w = obj.writable()
            if is_r:
                r.append(fd)
            if is_w:
                w.append(fd)
            if is_r or is_w:
                e.append(fd)
        if [] == r == w == e:
            time.sleep(timeout)
        else:
            try:
                r, w, e = select.select(r, w, e, timeout)
            except select.error, err:
                if err[0] != EINTR:
                    raise
                else:
                    return

        for fd in r:
            obj = map.get(fd)
            if obj is None:
                continue
            read(obj)

        for fd in w:
            obj = map.get(fd)
            if obj is None:
                continue
            write(obj)

        for fd in e:
            obj = map.get(fd)
            if obj is None:
                continue
            _exception(obj)

def poll2(timeout=0.0, map=None):
    # Use the poll() support added to the select module in Python 2.0
    if map is None:
        map = socket_map
    if timeout is not None:
        # timeout is in milliseconds
        timeout = int(timeout*1000)
    pollster = select.poll()
    if map:
        for fd, obj in map.items():
            flags = 0
            if obj.readable():
                flags |= select.POLLIN | select.POLLPRI
            if obj.writable():
                flags |= select.POLLOUT
            if flags:
                # Only check for exceptions if object was either readable
                # or writable.
                flags |= select.POLLERR | select.POLLHUP | select.POLLNVAL
                pollster.register(fd, flags)
        try:
            r = pollster.poll(timeout)
        except select.error, err:
            if err[0] != EINTR:
                raise
            r = []
        for fd, flags in r:
            obj = map.get(fd)
            if obj is None:
                continue
            readwrite(obj, flags)

poll3 = poll2                           # Alias for backward compatibility

def loop(timeout=30.0, use_poll=True, map=None, count=None):
    if map is None:
        map = socket_map

    if use_poll and hasattr(select, 'poll'):
        poll_fun = poll2
    else:
        poll_fun = poll

    if count is None:
        while map:
            poll_fun(timeout, map)

    else:
        while map and count > 0:
            poll_fun(timeout, map)
            count = count - 1

class dispatcher:

    debug = False
    connected = False
    accepting = False
    closing = False
    addr = None

    def __init__(self, sock=None, map=None):
        if map is None:
            self._map = socket_map
        else:
            self._map = map

        if sock:
            self.set_socket(sock, map)
            # I think it should inherit this anyway
            self.socket.setblocking(0)
            self.connected = True
            # XXX Does the constructor require that the socket passed
            # be connected?
            try:
                self.addr = sock.getpeername()
            except socket.error:
                # The addr isn't crucial
                pass
        else:
            self.socket = None

    def __repr__(self):
        status = [self.__class__.__module__+"."+self.__class__.__name__]
        if self.accepting and self.addr:
            status.append('listening')
        elif self.connected:
            status.append('connected')
        if self.addr is not None:
            try:
                status.append('%s:%d' % self.addr)
            except TypeError:
                status.append(repr(self.addr))
        return '<%s at %#x>' % (' '.join(status), id(self))

    def add_channel(self, map=None):
        #self.log_info('adding channel %s' % self)
        if map is None:
            map = self._map
        map[self._fileno] = self

    def del_channel(self, map=None):
        fd = self._fileno
        if map is None:
            map = self._map
        if map.has_key(fd):
            #self.log_info('closing channel %d:%s' % (fd, self))
            del map[fd]
        self._fileno = None

    def create_socket(self, family, type):
        self.family_and_type = family, type
        self.socket = socket.socket(family, type)
        self.socket.setblocking(0)
        self._fileno = self.socket
        self.add_channel()

    def set_socket(self, sock, map=None):
        self.socket = sock
##        self.__dict__['socket'] = sock
        self._fileno = sock
        self.add_channel(map)

    def set_reuse_addr(self):
        # try to re-use a server port if possible
        try:
            self.socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR,
                self.socket.getsockopt(socket.SOL_SOCKET,
                                       socket.SO_REUSEADDR) | 1
                )
        except socket.error:
            pass

    # ==================================================
    # predicates for select()
    # these are used as filters for the lists of sockets
    # to pass to select().
    # ==================================================

    def readable(self):
        return True

    def writable(self):
        return True

    # ==================================================
    # socket object methods.
    # ==================================================

    def listen(self, num):
        self.accepting = True
        if os.name == 'nt' and num > 5:
            num = 1
        return self.socket.listen(num)

    def bind(self, addr):
        self.addr = addr
        return self.socket.bind(addr)

    def connect(self, address):
        self.connected = False
        err = self.socket.connect_ex(address)
        # XXX Should interpret Winsock return values
        if err in (EINPROGRESS, EALREADY, EWOULDBLOCK):
            return
        if err in (0, EISCONN):
            self.addr = address
            self.connected = True
            self.handle_connect()
        else:
            raise socket.error, (err, errorcode[err])

    def accept(self):
        # XXX can return either an address pair or None
        try:
            conn, addr = self.socket.accept()
            return conn, addr
        except socket.error, why:
            if why[0] == EWOULDBLOCK:
                pass
            else:
                raise

    def send(self, data):
        try:
            result = self.socket.send(data)
            return result
        except socket.error, why:
            if why[0] == EWOULDBLOCK:
                return 0
            else:
                raise
            return 0

    def recv(self, buffer_size):
        try:
            data = self.socket.recv(buffer_size)
            if not data:
                # a closed connection is indicated by signaling
                # a read condition, and having recv() return 0.
                self.handle_close()
                return ''
            else:
                return data
        except socket.error, why:
            # winsock sometimes throws ENOTCONN
            if why[0] in [ECONNRESET, ENOTCONN, ESHUTDOWN]:
                self.handle_close()
                return ''
            else:
                raise

    def close(self):
        self.del_channel()
        self.socket.close()

    # cheap inheritance, used to pass all other attribute
    # references to the underlying socket object.
    def __getattr__(self, attr):
        return getattr(self.socket, attr)

    # log and log_info may be overridden to provide more sophisticated
    # logging and warning methods. In general, log is for 'hit' logging
    # and 'log_info' is for informational, warning and error logging.

    def log(self, message):
        sys.stderr.write('log: %s\n' % str(message))

    def log_info(self, message, type='info'):
        if __debug__ or type != 'info':
            print '%s: %s' % (type, message)

    def handle_read_event(self):
        if self.accepting:
            # for an accepting socket, getting a read implies
            # that we are connected
            if not self.connected:
                self.connected = True
            self.handle_accept()
        elif not self.connected:
            self.handle_connect()
            self.connected = True
            self.handle_read()
        else:
            self.handle_read()

    def handle_write_event(self):
        # getting a write implies that we are connected
        if not self.connected:
            self.handle_connect()
            self.connected = True
        self.handle_write()

    def handle_expt_event(self):
        self.handle_expt()

    def handle_error(self):
        nil, t, v, tbinfo = compact_traceback()

        # sometimes a user repr method will crash.
        try:
            self_repr = repr(self)
        except:
            self_repr = '<__repr__(self) failed for object at %0x>' % id(self)

        self.log_info(
            'uncaptured python exception, closing channel %s (%s:%s %s)' % (
                self_repr,
                t,
                v,
                tbinfo
                ),
            'error'
            )
        self.close()

    def handle_expt(self):
        self.log_info('unhandled exception', 'warning')

    def handle_read(self):
        self.log_info('unhandled read event', 'warning')

    def handle_write(self):
        self.log_info('unhandled write event', 'warning')

    def handle_connect(self):
        self.log_info('unhandled connect event', 'warning')

    def handle_accept(self):
        self.log_info('unhandled accept event', 'warning')

    def handle_close(self):
        self.log_info('unhandled close event', 'warning')
        self.close()

# ---------------------------------------------------------------------------
# adds simple buffered output capability, useful for simple clients.
# [for more sophisticated usage use asynchat.async_chat]
# ---------------------------------------------------------------------------

class dispatcher_with_send(dispatcher):

    def __init__(self, sock=None, map=None):
        dispatcher.__init__(self, sock, map)
        self.out_buffer = ''

    def initiate_send(self):
        num_sent = 0
        num_sent = dispatcher.send(self, self.out_buffer[:512])
        self.out_buffer = self.out_buffer[num_sent:]

    def handle_write(self):
        self.initiate_send()

    def writable(self):
        return (not self.connected) or len(self.out_buffer)

    def send(self, data):
        if self.debug:
            self.log_info('sending %s' % repr(data))
        self.out_buffer = self.out_buffer + data
        self.initiate_send()

# ---------------------------------------------------------------------------
# used for debugging.
# ---------------------------------------------------------------------------

def compact_traceback():
    t, v, tb = sys.exc_info()
    tbinfo = []
    assert tb # Must have a traceback
    while tb:
        tbinfo.append((
            tb.tb_frame.f_code.co_filename,
            tb.tb_frame.f_code.co_name,
            str(tb.tb_lineno)
            ))
        tb = tb.tb_next

    # just to be safe
    del tb

    file, function, line = tbinfo[-1]
    info = ' '.join(['[%s|%s|%s]' % x for x in tbinfo])
    return (file, function, line), t, v, info

def close_all(map=None):
    if map is None:
        map = socket_map
    for x in map.values():
        x.socket.close()
    map.clear()

# Asynchronous File I/O:
#
# After a little research (reading man pages on various unixen, and
# digging through the linux kernel), I've determined that select()
# isn't meant for doing asynchronous file i/o.
# Heartening, though - reading linux/mm/filemap.c shows that linux
# supports asynchronous read-ahead.  So _MOST_ of the time, the data
# will be sitting in memory for us already when we go to read it.
#
# What other OS's (besides NT) support async file i/o?  [VMS?]
#
# Regardless, this is useful for pipes, and stdin/stdout...

if os.name == 'posix':
    import fcntl

    class file_wrapper:
        # here we override just enough to make a file
        # look like a socket for the purposes of asyncore.

        def __init__(self, fd):
            self.fd = fd

        def recv(self, *args):
            return os.read(self.fd, *args)

        def send(self, *args):
            return os.write(self.fd, *args)

        read = recv
        write = send

        def close(self):
            os.close(self.fd)

        def fileno(self):
            return self.fd

    class file_dispatcher(dispatcher):

        def __init__(self, fd, map=None):
            dispatcher.__init__(self, None, map)
            self.connected = True
            self.set_file(fd)
            # set it to non-blocking mode
            flags = fcntl.fcntl(fd, fcntl.F_GETFL, 0)
            flags = flags | os.O_NONBLOCK
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)

        def set_file(self, fd):
            self._fileno = fd
            self.socket = file_wrapper(fd)
            self.add_channel()
