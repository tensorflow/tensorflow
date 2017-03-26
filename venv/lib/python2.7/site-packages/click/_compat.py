import re
import io
import os
import sys
import codecs
from weakref import WeakKeyDictionary


PY2 = sys.version_info[0] == 2
WIN = sys.platform.startswith('win')
DEFAULT_COLUMNS = 80


_ansi_re = re.compile('\033\[((?:\d|;)*)([a-zA-Z])')


def get_filesystem_encoding():
    return sys.getfilesystemencoding() or sys.getdefaultencoding()


def _make_text_stream(stream, encoding, errors):
    if encoding is None:
        encoding = get_best_encoding(stream)
    if errors is None:
        errors = 'replace'
    return _NonClosingTextIOWrapper(stream, encoding, errors,
                                    line_buffering=True)


def is_ascii_encoding(encoding):
    """Checks if a given encoding is ascii."""
    try:
        return codecs.lookup(encoding).name == 'ascii'
    except LookupError:
        return False


def get_best_encoding(stream):
    """Returns the default stream encoding if not found."""
    rv = getattr(stream, 'encoding', None) or sys.getdefaultencoding()
    if is_ascii_encoding(rv):
        return 'utf-8'
    return rv


class _NonClosingTextIOWrapper(io.TextIOWrapper):

    def __init__(self, stream, encoding, errors, **extra):
        self._stream = stream = _FixupStream(stream)
        io.TextIOWrapper.__init__(self, stream, encoding, errors, **extra)

    # The io module is a place where the Python 3 text behavior
    # was forced upon Python 2, so we need to unbreak
    # it to look like Python 2.
    if PY2:
        def write(self, x):
            if isinstance(x, str) or is_bytes(x):
                try:
                    self.flush()
                except Exception:
                    pass
                return self.buffer.write(str(x))
            return io.TextIOWrapper.write(self, x)

        def writelines(self, lines):
            for line in lines:
                self.write(line)

    def __del__(self):
        try:
            self.detach()
        except Exception:
            pass

    def isatty(self):
        # https://bitbucket.org/pypy/pypy/issue/1803
        return self._stream.isatty()


class _FixupStream(object):
    """The new io interface needs more from streams than streams
    traditionally implement.  As such, this fix-up code is necessary in
    some circumstances.
    """

    def __init__(self, stream):
        self._stream = stream

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def read1(self, size):
        f = getattr(self._stream, 'read1', None)
        if f is not None:
            return f(size)
        # We only dispatch to readline instead of read in Python 2 as we
        # do not want cause problems with the different implementation
        # of line buffering.
        if PY2:
            return self._stream.readline(size)
        return self._stream.read(size)

    def readable(self):
        x = getattr(self._stream, 'readable', None)
        if x is not None:
            return x()
        try:
            self._stream.read(0)
        except Exception:
            return False
        return True

    def writable(self):
        x = getattr(self._stream, 'writable', None)
        if x is not None:
            return x()
        try:
            self._stream.write('')
        except Exception:
            try:
                self._stream.write(b'')
            except Exception:
                return False
        return True

    def seekable(self):
        x = getattr(self._stream, 'seekable', None)
        if x is not None:
            return x()
        try:
            self._stream.seek(self._stream.tell())
        except Exception:
            return False
        return True


if PY2:
    text_type = unicode
    bytes = str
    raw_input = raw_input
    string_types = (str, unicode)
    iteritems = lambda x: x.iteritems()
    range_type = xrange

    def is_bytes(x):
        return isinstance(x, (buffer, bytearray))

    _identifier_re = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    # For Windows, we need to force stdout/stdin/stderr to binary if it's
    # fetched for that.  This obviously is not the most correct way to do
    # it as it changes global state.  Unfortunately, there does not seem to
    # be a clear better way to do it as just reopening the file in binary
    # mode does not change anything.
    #
    # An option would be to do what Python 3 does and to open the file as
    # binary only, patch it back to the system, and then use a wrapper
    # stream that converts newlines.  It's not quite clear what's the
    # correct option here.
    #
    # This code also lives in _winconsole for the fallback to the console
    # emulation stream.
    #
    # There are also Windows environments where the `msvcrt` module is not
    # available (which is why we use try-catch instead of the WIN variable
    # here), such as the Google App Engine development server on Windows. In
    # those cases there is just nothing we can do.
    try:
        import msvcrt
    except ImportError:
        set_binary_mode = lambda x: x
    else:
        def set_binary_mode(f):
            try:
                fileno = f.fileno()
            except Exception:
                pass
            else:
                msvcrt.setmode(fileno, os.O_BINARY)
            return f

    def isidentifier(x):
        return _identifier_re.search(x) is not None

    def get_binary_stdin():
        return set_binary_mode(sys.stdin)

    def get_binary_stdout():
        return set_binary_mode(sys.stdout)

    def get_binary_stderr():
        return set_binary_mode(sys.stderr)

    def get_text_stdin(encoding=None, errors=None):
        rv = _get_windows_console_stream(sys.stdin, encoding, errors)
        if rv is not None:
            return rv
        return _make_text_stream(sys.stdin, encoding, errors)

    def get_text_stdout(encoding=None, errors=None):
        rv = _get_windows_console_stream(sys.stdout, encoding, errors)
        if rv is not None:
            return rv
        return _make_text_stream(sys.stdout, encoding, errors)

    def get_text_stderr(encoding=None, errors=None):
        rv = _get_windows_console_stream(sys.stderr, encoding, errors)
        if rv is not None:
            return rv
        return _make_text_stream(sys.stderr, encoding, errors)

    def filename_to_ui(value):
        if isinstance(value, bytes):
            value = value.decode(get_filesystem_encoding(), 'replace')
        return value
else:
    import io
    text_type = str
    raw_input = input
    string_types = (str,)
    range_type = range
    isidentifier = lambda x: x.isidentifier()
    iteritems = lambda x: iter(x.items())

    def is_bytes(x):
        return isinstance(x, (bytes, memoryview, bytearray))

    def _is_binary_reader(stream, default=False):
        try:
            return isinstance(stream.read(0), bytes)
        except Exception:
            return default
            # This happens in some cases where the stream was already
            # closed.  In this case, we assume the default.

    def _is_binary_writer(stream, default=False):
        try:
            stream.write(b'')
        except Exception:
            try:
                stream.write('')
                return False
            except Exception:
                pass
            return default
        return True

    def _find_binary_reader(stream):
        # We need to figure out if the given stream is already binary.
        # This can happen because the official docs recommend detaching
        # the streams to get binary streams.  Some code might do this, so
        # we need to deal with this case explicitly.
        if _is_binary_reader(stream, False):
            return stream

        buf = getattr(stream, 'buffer', None)

        # Same situation here; this time we assume that the buffer is
        # actually binary in case it's closed.
        if buf is not None and _is_binary_reader(buf, True):
            return buf

    def _find_binary_writer(stream):
        # We need to figure out if the given stream is already binary.
        # This can happen because the official docs recommend detatching
        # the streams to get binary streams.  Some code might do this, so
        # we need to deal with this case explicitly.
        if _is_binary_writer(stream, False):
            return stream

        buf = getattr(stream, 'buffer', None)

        # Same situation here; this time we assume that the buffer is
        # actually binary in case it's closed.
        if buf is not None and _is_binary_writer(buf, True):
            return buf

    def _stream_is_misconfigured(stream):
        """A stream is misconfigured if its encoding is ASCII."""
        # If the stream does not have an encoding set, we assume it's set
        # to ASCII.  This appears to happen in certain unittest
        # environments.  It's not quite clear what the correct behavior is
        # but this at least will force Click to recover somehow.
        return is_ascii_encoding(getattr(stream, 'encoding', None) or 'ascii')

    def _is_compatible_text_stream(stream, encoding, errors):
        stream_encoding = getattr(stream, 'encoding', None)
        stream_errors = getattr(stream, 'errors', None)

        # Perfect match.
        if stream_encoding == encoding and stream_errors == errors:
            return True

        # Otherwise, it's only a compatible stream if we did not ask for
        # an encoding.
        if encoding is None:
            return stream_encoding is not None

        return False

    def _force_correct_text_reader(text_reader, encoding, errors):
        if _is_binary_reader(text_reader, False):
            binary_reader = text_reader
        else:
            # If there is no target encoding set, we need to verify that the
            # reader is not actually misconfigured.
            if encoding is None and not _stream_is_misconfigured(text_reader):
                return text_reader

            if _is_compatible_text_stream(text_reader, encoding, errors):
                return text_reader

            # If the reader has no encoding, we try to find the underlying
            # binary reader for it.  If that fails because the environment is
            # misconfigured, we silently go with the same reader because this
            # is too common to happen.  In that case, mojibake is better than
            # exceptions.
            binary_reader = _find_binary_reader(text_reader)
            if binary_reader is None:
                return text_reader

        # At this point, we default the errors to replace instead of strict
        # because nobody handles those errors anyways and at this point
        # we're so fundamentally fucked that nothing can repair it.
        if errors is None:
            errors = 'replace'
        return _make_text_stream(binary_reader, encoding, errors)

    def _force_correct_text_writer(text_writer, encoding, errors):
        if _is_binary_writer(text_writer, False):
            binary_writer = text_writer
        else:
            # If there is no target encoding set, we need to verify that the
            # writer is not actually misconfigured.
            if encoding is None and not _stream_is_misconfigured(text_writer):
                return text_writer

            if _is_compatible_text_stream(text_writer, encoding, errors):
                return text_writer

            # If the writer has no encoding, we try to find the underlying
            # binary writer for it.  If that fails because the environment is
            # misconfigured, we silently go with the same writer because this
            # is too common to happen.  In that case, mojibake is better than
            # exceptions.
            binary_writer = _find_binary_writer(text_writer)
            if binary_writer is None:
                return text_writer

        # At this point, we default the errors to replace instead of strict
        # because nobody handles those errors anyways and at this point
        # we're so fundamentally fucked that nothing can repair it.
        if errors is None:
            errors = 'replace'
        return _make_text_stream(binary_writer, encoding, errors)

    def get_binary_stdin():
        reader = _find_binary_reader(sys.stdin)
        if reader is None:
            raise RuntimeError('Was not able to determine binary '
                               'stream for sys.stdin.')
        return reader

    def get_binary_stdout():
        writer = _find_binary_writer(sys.stdout)
        if writer is None:
            raise RuntimeError('Was not able to determine binary '
                               'stream for sys.stdout.')
        return writer

    def get_binary_stderr():
        writer = _find_binary_writer(sys.stderr)
        if writer is None:
            raise RuntimeError('Was not able to determine binary '
                               'stream for sys.stderr.')
        return writer

    def get_text_stdin(encoding=None, errors=None):
        rv = _get_windows_console_stream(sys.stdin, encoding, errors)
        if rv is not None:
            return rv
        return _force_correct_text_reader(sys.stdin, encoding, errors)

    def get_text_stdout(encoding=None, errors=None):
        rv = _get_windows_console_stream(sys.stdout, encoding, errors)
        if rv is not None:
            return rv
        return _force_correct_text_writer(sys.stdout, encoding, errors)

    def get_text_stderr(encoding=None, errors=None):
        rv = _get_windows_console_stream(sys.stderr, encoding, errors)
        if rv is not None:
            return rv
        return _force_correct_text_writer(sys.stderr, encoding, errors)

    def filename_to_ui(value):
        if isinstance(value, bytes):
            value = value.decode(get_filesystem_encoding(), 'replace')
        else:
            value = value.encode('utf-8', 'surrogateescape') \
                .decode('utf-8', 'replace')
        return value


def get_streerror(e, default=None):
    if hasattr(e, 'strerror'):
        msg = e.strerror
    else:
        if default is not None:
            msg = default
        else:
            msg = str(e)
    if isinstance(msg, bytes):
        msg = msg.decode('utf-8', 'replace')
    return msg


def open_stream(filename, mode='r', encoding=None, errors='strict',
                atomic=False):
    # Standard streams first.  These are simple because they don't need
    # special handling for the atomic flag.  It's entirely ignored.
    if filename == '-':
        if 'w' in mode:
            if 'b' in mode:
                return get_binary_stdout(), False
            return get_text_stdout(encoding=encoding, errors=errors), False
        if 'b' in mode:
            return get_binary_stdin(), False
        return get_text_stdin(encoding=encoding, errors=errors), False

    # Non-atomic writes directly go out through the regular open functions.
    if not atomic:
        if encoding is None:
            return open(filename, mode), True
        return io.open(filename, mode, encoding=encoding, errors=errors), True

    # Some usability stuff for atomic writes
    if 'a' in mode:
        raise ValueError(
            'Appending to an existing file is not supported, because that '
            'would involve an expensive `copy`-operation to a temporary '
            'file. Open the file in normal `w`-mode and copy explicitly '
            'if that\'s what you\'re after.'
        )
    if 'x' in mode:
        raise ValueError('Use the `overwrite`-parameter instead.')
    if 'w' not in mode:
        raise ValueError('Atomic writes only make sense with `w`-mode.')

    # Atomic writes are more complicated.  They work by opening a file
    # as a proxy in the same folder and then using the fdopen
    # functionality to wrap it in a Python file.  Then we wrap it in an
    # atomic file that moves the file over on close.
    import tempfile
    fd, tmp_filename = tempfile.mkstemp(dir=os.path.dirname(filename),
                                        prefix='.__atomic-write')

    if encoding is not None:
        f = io.open(fd, mode, encoding=encoding, errors=errors)
    else:
        f = os.fdopen(fd, mode)

    return _AtomicFile(f, tmp_filename, filename), True


# Used in a destructor call, needs extra protection from interpreter cleanup.
if hasattr(os, 'replace'):
    _replace = os.replace
    _can_replace = True
else:
    _replace = os.rename
    _can_replace = not WIN


class _AtomicFile(object):

    def __init__(self, f, tmp_filename, real_filename):
        self._f = f
        self._tmp_filename = tmp_filename
        self._real_filename = real_filename
        self.closed = False

    @property
    def name(self):
        return self._real_filename

    def close(self, delete=False):
        if self.closed:
            return
        self._f.close()
        if not _can_replace:
            try:
                os.remove(self._real_filename)
            except OSError:
                pass
        _replace(self._tmp_filename, self._real_filename)
        self.closed = True

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close(delete=exc_type is not None)

    def __repr__(self):
        return repr(self._f)


auto_wrap_for_ansi = None
colorama = None
get_winterm_size = None


def strip_ansi(value):
    return _ansi_re.sub('', value)


def should_strip_ansi(stream=None, color=None):
    if color is None:
        if stream is None:
            stream = sys.stdin
        return not isatty(stream)
    return not color


# If we're on Windows, we provide transparent integration through
# colorama.  This will make ANSI colors through the echo function
# work automatically.
if WIN:
    # Windows has a smaller terminal
    DEFAULT_COLUMNS = 79

    from ._winconsole import _get_windows_console_stream

    def _get_argv_encoding():
        import locale
        return locale.getpreferredencoding()

    if PY2:
        def raw_input(prompt=''):
            sys.stderr.flush()
            if prompt:
                stdout = _default_text_stdout()
                stdout.write(prompt)
            stdin = _default_text_stdin()
            return stdin.readline().rstrip('\r\n')

    try:
        import colorama
    except ImportError:
        pass
    else:
        _ansi_stream_wrappers = WeakKeyDictionary()

        def auto_wrap_for_ansi(stream, color=None):
            """This function wraps a stream so that calls through colorama
            are issued to the win32 console API to recolor on demand.  It
            also ensures to reset the colors if a write call is interrupted
            to not destroy the console afterwards.
            """
            try:
                cached = _ansi_stream_wrappers.get(stream)
            except Exception:
                cached = None
            if cached is not None:
                return cached
            strip = should_strip_ansi(stream, color)
            ansi_wrapper = colorama.AnsiToWin32(stream, strip=strip)
            rv = ansi_wrapper.stream
            _write = rv.write

            def _safe_write(s):
                try:
                    return _write(s)
                except:
                    ansi_wrapper.reset_all()
                    raise

            rv.write = _safe_write
            try:
                _ansi_stream_wrappers[stream] = rv
            except Exception:
                pass
            return rv

        def get_winterm_size():
            win = colorama.win32.GetConsoleScreenBufferInfo(
                colorama.win32.STDOUT).srWindow
            return win.Right - win.Left, win.Bottom - win.Top
else:
    def _get_argv_encoding():
        return getattr(sys.stdin, 'encoding', None) or get_filesystem_encoding()

    _get_windows_console_stream = lambda *x: None


def term_len(x):
    return len(strip_ansi(x))


def isatty(stream):
    try:
        return stream.isatty()
    except Exception:
        return False


def _make_cached_stream_func(src_func, wrapper_func):
    cache = WeakKeyDictionary()
    def func():
        stream = src_func()
        try:
            rv = cache.get(stream)
        except Exception:
            rv = None
        if rv is not None:
            return rv
        rv = wrapper_func()
        try:
            cache[stream] = rv
        except Exception:
            pass
        return rv
    return func


_default_text_stdin = _make_cached_stream_func(
    lambda: sys.stdin, get_text_stdin)
_default_text_stdout = _make_cached_stream_func(
    lambda: sys.stdout, get_text_stdout)
_default_text_stderr = _make_cached_stream_func(
    lambda: sys.stderr, get_text_stderr)


binary_streams = {
    'stdin': get_binary_stdin,
    'stdout': get_binary_stdout,
    'stderr': get_binary_stderr,
}

text_streams = {
    'stdin': get_text_stdin,
    'stdout': get_text_stdout,
    'stderr': get_text_stderr,
}
