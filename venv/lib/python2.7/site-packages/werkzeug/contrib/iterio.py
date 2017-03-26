# -*- coding: utf-8 -*-
r"""
    werkzeug.contrib.iterio
    ~~~~~~~~~~~~~~~~~~~~~~~

    This module implements a :class:`IterIO` that converts an iterator into
    a stream object and the other way round.  Converting streams into
    iterators requires the `greenlet`_ module.

    To convert an iterator into a stream all you have to do is to pass it
    directly to the :class:`IterIO` constructor.  In this example we pass it
    a newly created generator::

        def foo():
            yield "something\n"
            yield "otherthings"
        stream = IterIO(foo())
        print stream.read()         # read the whole iterator

    The other way round works a bit different because we have to ensure that
    the code execution doesn't take place yet.  An :class:`IterIO` call with a
    callable as first argument does two things.  The function itself is passed
    an :class:`IterIO` stream it can feed.  The object returned by the
    :class:`IterIO` constructor on the other hand is not an stream object but
    an iterator::

        def foo(stream):
            stream.write("some")
            stream.write("thing")
            stream.flush()
            stream.write("otherthing")
        iterator = IterIO(foo)
        print iterator.next()       # prints something
        print iterator.next()       # prints otherthing
        iterator.next()             # raises StopIteration

    .. _greenlet: http://codespeak.net/py/dist/greenlet.html

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
try:
    import greenlet
except ImportError:
    greenlet = None

from werkzeug._compat import implements_iterator


def _mixed_join(iterable, sentinel):
    """concatenate any string type in an intelligent way."""
    iterator = iter(iterable)
    first_item = next(iterator, sentinel)
    if isinstance(first_item, bytes):
        return first_item + b''.join(iterator)
    return first_item + u''.join(iterator)


def _newline(reference_string):
    if isinstance(reference_string, bytes):
        return b'\n'
    return u'\n'


@implements_iterator
class IterIO(object):

    """Instances of this object implement an interface compatible with the
    standard Python :class:`file` object.  Streams are either read-only or
    write-only depending on how the object is created.

    If the first argument is an iterable a file like object is returned that
    returns the contents of the iterable.  In case the iterable is empty
    read operations will return the sentinel value.

    If the first argument is a callable then the stream object will be
    created and passed to that function.  The caller itself however will
    not receive a stream but an iterable.  The function will be be executed
    step by step as something iterates over the returned iterable.  Each
    call to :meth:`flush` will create an item for the iterable.  If
    :meth:`flush` is called without any writes in-between the sentinel
    value will be yielded.

    Note for Python 3: due to the incompatible interface of bytes and
    streams you should set the sentinel value explicitly to an empty
    bytestring (``b''``) if you are expecting to deal with bytes as
    otherwise the end of the stream is marked with the wrong sentinel
    value.

    .. versionadded:: 0.9
       `sentinel` parameter was added.
    """

    def __new__(cls, obj, sentinel=''):
        try:
            iterator = iter(obj)
        except TypeError:
            return IterI(obj, sentinel)
        return IterO(iterator, sentinel)

    def __iter__(self):
        return self

    def tell(self):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        return self.pos

    def isatty(self):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        return False

    def seek(self, pos, mode=0):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        raise IOError(9, 'Bad file descriptor')

    def truncate(self, size=None):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        raise IOError(9, 'Bad file descriptor')

    def write(self, s):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        raise IOError(9, 'Bad file descriptor')

    def writelines(self, list):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        raise IOError(9, 'Bad file descriptor')

    def read(self, n=-1):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        raise IOError(9, 'Bad file descriptor')

    def readlines(self, sizehint=0):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        raise IOError(9, 'Bad file descriptor')

    def readline(self, length=None):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        raise IOError(9, 'Bad file descriptor')

    def flush(self):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        raise IOError(9, 'Bad file descriptor')

    def __next__(self):
        if self.closed:
            raise StopIteration()
        line = self.readline()
        if not line:
            raise StopIteration()
        return line


class IterI(IterIO):

    """Convert an stream into an iterator."""

    def __new__(cls, func, sentinel=''):
        if greenlet is None:
            raise RuntimeError('IterI requires greenlet support')
        stream = object.__new__(cls)
        stream._parent = greenlet.getcurrent()
        stream._buffer = []
        stream.closed = False
        stream.sentinel = sentinel
        stream.pos = 0

        def run():
            func(stream)
            stream.close()

        g = greenlet.greenlet(run, stream._parent)
        while 1:
            rv = g.switch()
            if not rv:
                return
            yield rv[0]

    def close(self):
        if not self.closed:
            self.closed = True
            self._flush_impl()

    def write(self, s):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        if s:
            self.pos += len(s)
            self._buffer.append(s)

    def writelines(self, list):
        for item in list:
            self.write(item)

    def flush(self):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        self._flush_impl()

    def _flush_impl(self):
        data = _mixed_join(self._buffer, self.sentinel)
        self._buffer = []
        if not data and self.closed:
            self._parent.switch()
        else:
            self._parent.switch((data,))


class IterO(IterIO):

    """Iter output.  Wrap an iterator and give it a stream like interface."""

    def __new__(cls, gen, sentinel=''):
        self = object.__new__(cls)
        self._gen = gen
        self._buf = None
        self.sentinel = sentinel
        self.closed = False
        self.pos = 0
        return self

    def __iter__(self):
        return self

    def _buf_append(self, string):
        '''Replace string directly without appending to an empty string,
        avoiding type issues.'''
        if not self._buf:
            self._buf = string
        else:
            self._buf += string

    def close(self):
        if not self.closed:
            self.closed = True
            if hasattr(self._gen, 'close'):
                self._gen.close()

    def seek(self, pos, mode=0):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        if mode == 1:
            pos += self.pos
        elif mode == 2:
            self.read()
            self.pos = min(self.pos, self.pos + pos)
            return
        elif mode != 0:
            raise IOError('Invalid argument')
        buf = []
        try:
            tmp_end_pos = len(self._buf)
            while pos > tmp_end_pos:
                item = self._gen.next()
                tmp_end_pos += len(item)
                buf.append(item)
        except StopIteration:
            pass
        if buf:
            self._buf_append(_mixed_join(buf, self.sentinel))
        self.pos = max(0, pos)

    def read(self, n=-1):
        if self.closed:
            raise ValueError('I/O operation on closed file')
        if n < 0:
            self._buf_append(_mixed_join(self._gen, self.sentinel))
            result = self._buf[self.pos:]
            self.pos += len(result)
            return result
        new_pos = self.pos + n
        buf = []
        try:
            tmp_end_pos = 0 if self._buf is None else len(self._buf)
            while new_pos > tmp_end_pos or (self._buf is None and not buf):
                item = next(self._gen)
                tmp_end_pos += len(item)
                buf.append(item)
        except StopIteration:
            pass
        if buf:
            self._buf_append(_mixed_join(buf, self.sentinel))

        if self._buf is None:
            return self.sentinel

        new_pos = max(0, new_pos)
        try:
            return self._buf[self.pos:new_pos]
        finally:
            self.pos = min(new_pos, len(self._buf))

    def readline(self, length=None):
        if self.closed:
            raise ValueError('I/O operation on closed file')

        nl_pos = -1
        if self._buf:
            nl_pos = self._buf.find(_newline(self._buf), self.pos)
        buf = []
        try:
            if self._buf is None:
                pos = self.pos
            else:
                pos = len(self._buf)
            while nl_pos < 0:
                item = next(self._gen)
                local_pos = item.find(_newline(item))
                buf.append(item)
                if local_pos >= 0:
                    nl_pos = pos + local_pos
                    break
                pos += len(item)
        except StopIteration:
            pass
        if buf:
            self._buf_append(_mixed_join(buf, self.sentinel))

        if self._buf is None:
            return self.sentinel

        if nl_pos < 0:
            new_pos = len(self._buf)
        else:
            new_pos = nl_pos + 1
        if length is not None and self.pos + length < new_pos:
            new_pos = self.pos + length
        try:
            return self._buf[self.pos:new_pos]
        finally:
            self.pos = min(new_pos, len(self._buf))

    def readlines(self, sizehint=0):
        total = 0
        lines = []
        line = self.readline()
        while line:
            lines.append(line)
            total += len(line)
            if 0 < sizehint <= total:
                break
            line = self.readline()
        return lines
