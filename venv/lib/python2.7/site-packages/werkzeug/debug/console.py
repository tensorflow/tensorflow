# -*- coding: utf-8 -*-
"""
    werkzeug.debug.console
    ~~~~~~~~~~~~~~~~~~~~~~

    Interactive console support.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD.
"""
import sys
import code
from types import CodeType

from werkzeug.utils import escape
from werkzeug.local import Local
from werkzeug.debug.repr import debug_repr, dump, helper


_local = Local()


class HTMLStringO(object):

    """A StringO version that HTML escapes on write."""

    def __init__(self):
        self._buffer = []

    def isatty(self):
        return False

    def close(self):
        pass

    def flush(self):
        pass

    def seek(self, n, mode=0):
        pass

    def readline(self):
        if len(self._buffer) == 0:
            return ''
        ret = self._buffer[0]
        del self._buffer[0]
        return ret

    def reset(self):
        val = ''.join(self._buffer)
        del self._buffer[:]
        return val

    def _write(self, x):
        if isinstance(x, bytes):
            x = x.decode('utf-8', 'replace')
        self._buffer.append(x)

    def write(self, x):
        self._write(escape(x))

    def writelines(self, x):
        self._write(escape(''.join(x)))


class ThreadedStream(object):

    """Thread-local wrapper for sys.stdout for the interactive console."""

    def push():
        if not isinstance(sys.stdout, ThreadedStream):
            sys.stdout = ThreadedStream()
        _local.stream = HTMLStringO()
    push = staticmethod(push)

    def fetch():
        try:
            stream = _local.stream
        except AttributeError:
            return ''
        return stream.reset()
    fetch = staticmethod(fetch)

    def displayhook(obj):
        try:
            stream = _local.stream
        except AttributeError:
            return _displayhook(obj)
        # stream._write bypasses escaping as debug_repr is
        # already generating HTML for us.
        if obj is not None:
            _local._current_ipy.locals['_'] = obj
            stream._write(debug_repr(obj))
    displayhook = staticmethod(displayhook)

    def __setattr__(self, name, value):
        raise AttributeError('read only attribute %s' % name)

    def __dir__(self):
        return dir(sys.__stdout__)

    def __getattribute__(self, name):
        if name == '__members__':
            return dir(sys.__stdout__)
        try:
            stream = _local.stream
        except AttributeError:
            stream = sys.__stdout__
        return getattr(stream, name)

    def __repr__(self):
        return repr(sys.__stdout__)


# add the threaded stream as display hook
_displayhook = sys.displayhook
sys.displayhook = ThreadedStream.displayhook


class _ConsoleLoader(object):

    def __init__(self):
        self._storage = {}

    def register(self, code, source):
        self._storage[id(code)] = source
        # register code objects of wrapped functions too.
        for var in code.co_consts:
            if isinstance(var, CodeType):
                self._storage[id(var)] = source

    def get_source_by_code(self, code):
        try:
            return self._storage[id(code)]
        except KeyError:
            pass


def _wrap_compiler(console):
    compile = console.compile

    def func(source, filename, symbol):
        code = compile(source, filename, symbol)
        console.loader.register(code, source)
        return code
    console.compile = func


class _InteractiveConsole(code.InteractiveInterpreter):

    def __init__(self, globals, locals):
        code.InteractiveInterpreter.__init__(self, locals)
        self.globals = dict(globals)
        self.globals['dump'] = dump
        self.globals['help'] = helper
        self.globals['__loader__'] = self.loader = _ConsoleLoader()
        self.more = False
        self.buffer = []
        _wrap_compiler(self)

    def runsource(self, source):
        source = source.rstrip() + '\n'
        ThreadedStream.push()
        prompt = self.more and '... ' or '>>> '
        try:
            source_to_eval = ''.join(self.buffer + [source])
            if code.InteractiveInterpreter.runsource(self,
                                                     source_to_eval, '<debugger>', 'single'):
                self.more = True
                self.buffer.append(source)
            else:
                self.more = False
                del self.buffer[:]
        finally:
            output = ThreadedStream.fetch()
        return prompt + source + output

    def runcode(self, code):
        try:
            eval(code, self.globals, self.locals)
        except Exception:
            self.showtraceback()

    def showtraceback(self):
        from werkzeug.debug.tbtools import get_current_traceback
        tb = get_current_traceback(skip=1)
        sys.stdout._write(tb.render_summary())

    def showsyntaxerror(self, filename=None):
        from werkzeug.debug.tbtools import get_current_traceback
        tb = get_current_traceback(skip=4)
        sys.stdout._write(tb.render_summary())

    def write(self, data):
        sys.stdout.write(data)


class Console(object):

    """An interactive console."""

    def __init__(self, globals=None, locals=None):
        if locals is None:
            locals = {}
        if globals is None:
            globals = {}
        self._ipy = _InteractiveConsole(globals, locals)

    def eval(self, code):
        _local._current_ipy = self._ipy
        old_sys_stdout = sys.stdout
        try:
            return self._ipy.runsource(code)
        finally:
            sys.stdout = old_sys_stdout
