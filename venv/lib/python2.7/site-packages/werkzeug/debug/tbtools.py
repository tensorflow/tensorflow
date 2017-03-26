# -*- coding: utf-8 -*-
"""
    werkzeug.debug.tbtools
    ~~~~~~~~~~~~~~~~~~~~~~

    This module provides various traceback related utility functions.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD.
"""
import re

import os
import sys
import json
import inspect
import traceback
import codecs
from tokenize import TokenError

from werkzeug.utils import cached_property, escape
from werkzeug.debug.console import Console
from werkzeug._compat import range_type, PY2, text_type, string_types, \
    to_native, to_unicode
from werkzeug.filesystem import get_filesystem_encoding


_coding_re = re.compile(br'coding[:=]\s*([-\w.]+)')
_line_re = re.compile(br'^(.*?)$(?m)')
_funcdef_re = re.compile(r'^(\s*def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)')
UTF8_COOKIE = b'\xef\xbb\xbf'

system_exceptions = (SystemExit, KeyboardInterrupt)
try:
    system_exceptions += (GeneratorExit,)
except NameError:
    pass


HEADER = u'''\
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
  "http://www.w3.org/TR/html4/loose.dtd">
<html>
  <head>
    <title>%(title)s // Werkzeug Debugger</title>
    <link rel="stylesheet" href="?__debugger__=yes&amp;cmd=resource&amp;f=style.css"
        type="text/css">
    <!-- We need to make sure this has a favicon so that the debugger does
         not by accident trigger a request to /favicon.ico which might
         change the application state. -->
    <link rel="shortcut icon"
        href="?__debugger__=yes&amp;cmd=resource&amp;f=console.png">
    <script src="?__debugger__=yes&amp;cmd=resource&amp;f=jquery.js"></script>
    <script src="?__debugger__=yes&amp;cmd=resource&amp;f=debugger.js"></script>
    <script type="text/javascript">
      var TRACEBACK = %(traceback_id)d,
          CONSOLE_MODE = %(console)s,
          EVALEX = %(evalex)s,
          EVALEX_TRUSTED = %(evalex_trusted)s,
          SECRET = "%(secret)s";
    </script>
  </head>
  <body>
    <div class="debugger">
'''
FOOTER = u'''\
      <div class="footer">
        Brought to you by <strong class="arthur">DON'T PANIC</strong>, your
        friendly Werkzeug powered traceback interpreter.
      </div>
    </div>

    <div class="pin-prompt">
      <div class="inner">
        <h3>Console Locked</h3>
        <p>
          The console is locked and needs to be unlocked by entering the PIN.
          You can find the PIN printed out on the standard output of your
          shell that runs the server.
        <form>
          <p>PIN:
            <input type=text name=pin size=14>
            <input type=submit name=btn value="Confirm Pin">
        </form>
      </div>
    </div>
  </body>
</html>
'''

PAGE_HTML = HEADER + u'''\
<h1>%(exception_type)s</h1>
<div class="detail">
  <p class="errormsg">%(exception)s</p>
</div>
<h2 class="traceback">Traceback <em>(most recent call last)</em></h2>
%(summary)s
<div class="plain">
  <form action="/?__debugger__=yes&amp;cmd=paste" method="post">
    <p>
      <input type="hidden" name="language" value="pytb">
      This is the Copy/Paste friendly version of the traceback.  <span
      class="pastemessage">You can also paste this traceback into
      a <a href="https://gist.github.com/">gist</a>:
      <input type="submit" value="create paste"></span>
    </p>
    <textarea cols="50" rows="10" name="code" readonly>%(plaintext)s</textarea>
  </form>
</div>
<div class="explanation">
  The debugger caught an exception in your WSGI application.  You can now
  look at the traceback which led to the error.  <span class="nojavascript">
  If you enable JavaScript you can also use additional features such as code
  execution (if the evalex feature is enabled), automatic pasting of the
  exceptions and much more.</span>
</div>
''' + FOOTER + '''
<!--

%(plaintext_cs)s

-->
'''

CONSOLE_HTML = HEADER + u'''\
<h1>Interactive Console</h1>
<div class="explanation">
In this console you can execute Python expressions in the context of the
application.  The initial namespace was created by the debugger automatically.
</div>
<div class="console"><div class="inner">The Console requires JavaScript.</div></div>
''' + FOOTER

SUMMARY_HTML = u'''\
<div class="%(classes)s">
  %(title)s
  <ul>%(frames)s</ul>
  %(description)s
</div>
'''

FRAME_HTML = u'''\
<div class="frame" id="frame-%(id)d">
  <h4>File <cite class="filename">"%(filename)s"</cite>,
      line <em class="line">%(lineno)s</em>,
      in <code class="function">%(function_name)s</code></h4>
  <div class="source">%(lines)s</div>
</div>
'''

SOURCE_LINE_HTML = u'''\
<tr class="%(classes)s">
  <td class=lineno>%(lineno)s</td>
  <td>%(code)s</td>
</tr>
'''


def render_console_html(secret, evalex_trusted=True):
    return CONSOLE_HTML % {
        'evalex':           'true',
        'evalex_trusted':   evalex_trusted and 'true' or 'false',
        'console':          'true',
        'title':            'Console',
        'secret':           secret,
        'traceback_id': -1
    }


def get_current_traceback(ignore_system_exceptions=False,
                          show_hidden_frames=False, skip=0):
    """Get the current exception info as `Traceback` object.  Per default
    calling this method will reraise system exceptions such as generator exit,
    system exit or others.  This behavior can be disabled by passing `False`
    to the function as first parameter.
    """
    exc_type, exc_value, tb = sys.exc_info()
    if ignore_system_exceptions and exc_type in system_exceptions:
        raise
    for x in range_type(skip):
        if tb.tb_next is None:
            break
        tb = tb.tb_next
    tb = Traceback(exc_type, exc_value, tb)
    if not show_hidden_frames:
        tb.filter_hidden_frames()
    return tb


class Line(object):
    """Helper for the source renderer."""
    __slots__ = ('lineno', 'code', 'in_frame', 'current')

    def __init__(self, lineno, code):
        self.lineno = lineno
        self.code = code
        self.in_frame = False
        self.current = False

    def classes(self):
        rv = ['line']
        if self.in_frame:
            rv.append('in-frame')
        if self.current:
            rv.append('current')
        return rv
    classes = property(classes)

    def render(self):
        return SOURCE_LINE_HTML % {
            'classes':      u' '.join(self.classes),
            'lineno':       self.lineno,
            'code':         escape(self.code)
        }


class Traceback(object):
    """Wraps a traceback."""

    def __init__(self, exc_type, exc_value, tb):
        self.exc_type = exc_type
        self.exc_value = exc_value
        if not isinstance(exc_type, str):
            exception_type = exc_type.__name__
            if exc_type.__module__ not in ('__builtin__', 'exceptions'):
                exception_type = exc_type.__module__ + '.' + exception_type
        else:
            exception_type = exc_type
        self.exception_type = exception_type

        # we only add frames to the list that are not hidden.  This follows
        # the the magic variables as defined by paste.exceptions.collector
        self.frames = []
        while tb:
            self.frames.append(Frame(exc_type, exc_value, tb))
            tb = tb.tb_next

    def filter_hidden_frames(self):
        """Remove the frames according to the paste spec."""
        if not self.frames:
            return

        new_frames = []
        hidden = False
        for frame in self.frames:
            hide = frame.hide
            if hide in ('before', 'before_and_this'):
                new_frames = []
                hidden = False
                if hide == 'before_and_this':
                    continue
            elif hide in ('reset', 'reset_and_this'):
                hidden = False
                if hide == 'reset_and_this':
                    continue
            elif hide in ('after', 'after_and_this'):
                hidden = True
                if hide == 'after_and_this':
                    continue
            elif hide or hidden:
                continue
            new_frames.append(frame)

        # if we only have one frame and that frame is from the codeop
        # module, remove it.
        if len(new_frames) == 1 and self.frames[0].module == 'codeop':
            del self.frames[:]

        # if the last frame is missing something went terrible wrong :(
        elif self.frames[-1] in new_frames:
            self.frames[:] = new_frames

    def is_syntax_error(self):
        """Is it a syntax error?"""
        return isinstance(self.exc_value, SyntaxError)
    is_syntax_error = property(is_syntax_error)

    def exception(self):
        """String representation of the exception."""
        buf = traceback.format_exception_only(self.exc_type, self.exc_value)
        rv = ''.join(buf).strip()
        return rv.decode('utf-8', 'replace') if PY2 else rv
    exception = property(exception)

    def log(self, logfile=None):
        """Log the ASCII traceback into a file object."""
        if logfile is None:
            logfile = sys.stderr
        tb = self.plaintext.rstrip() + u'\n'
        if PY2:
            tb = tb.encode('utf-8', 'replace')
        logfile.write(tb)

    def paste(self):
        """Create a paste and return the paste id."""
        data = json.dumps({
            'description': 'Werkzeug Internal Server Error',
            'public': False,
            'files': {
                'traceback.txt': {
                    'content': self.plaintext
                }
            }
        }).encode('utf-8')
        try:
            from urllib2 import urlopen
        except ImportError:
            from urllib.request import urlopen
        rv = urlopen('https://api.github.com/gists', data=data)
        resp = json.loads(rv.read().decode('utf-8'))
        rv.close()
        return {
            'url': resp['html_url'],
            'id': resp['id']
        }

    def render_summary(self, include_title=True):
        """Render the traceback for the interactive console."""
        title = ''
        frames = []
        classes = ['traceback']
        if not self.frames:
            classes.append('noframe-traceback')

        if include_title:
            if self.is_syntax_error:
                title = u'Syntax Error'
            else:
                title = u'Traceback <em>(most recent call last)</em>:'

        for frame in self.frames:
            frames.append(u'<li%s>%s' % (
                frame.info and u' title="%s"' % escape(frame.info) or u'',
                frame.render()
            ))

        if self.is_syntax_error:
            description_wrapper = u'<pre class=syntaxerror>%s</pre>'
        else:
            description_wrapper = u'<blockquote>%s</blockquote>'

        return SUMMARY_HTML % {
            'classes':      u' '.join(classes),
            'title':        title and u'<h3>%s</h3>' % title or u'',
            'frames':       u'\n'.join(frames),
            'description':  description_wrapper % escape(self.exception)
        }

    def render_full(self, evalex=False, secret=None,
                    evalex_trusted=True):
        """Render the Full HTML page with the traceback info."""
        exc = escape(self.exception)
        return PAGE_HTML % {
            'evalex':           evalex and 'true' or 'false',
            'evalex_trusted':   evalex_trusted and 'true' or 'false',
            'console':          'false',
            'title':            exc,
            'exception':        exc,
            'exception_type':   escape(self.exception_type),
            'summary':          self.render_summary(include_title=False),
            'plaintext':        escape(self.plaintext),
            'plaintext_cs':     re.sub('-{2,}', '-', self.plaintext),
            'traceback_id':     self.id,
            'secret':           secret
        }

    def generate_plaintext_traceback(self):
        """Like the plaintext attribute but returns a generator"""
        yield u'Traceback (most recent call last):'
        for frame in self.frames:
            yield u'  File "%s", line %s, in %s' % (
                frame.filename,
                frame.lineno,
                frame.function_name
            )
            yield u'    ' + frame.current_line.strip()
        yield self.exception

    def plaintext(self):
        return u'\n'.join(self.generate_plaintext_traceback())
    plaintext = cached_property(plaintext)

    id = property(lambda x: id(x))


class Frame(object):

    """A single frame in a traceback."""

    def __init__(self, exc_type, exc_value, tb):
        self.lineno = tb.tb_lineno
        self.function_name = tb.tb_frame.f_code.co_name
        self.locals = tb.tb_frame.f_locals
        self.globals = tb.tb_frame.f_globals

        fn = inspect.getsourcefile(tb) or inspect.getfile(tb)
        if fn[-4:] in ('.pyo', '.pyc'):
            fn = fn[:-1]
        # if it's a file on the file system resolve the real filename.
        if os.path.isfile(fn):
            fn = os.path.realpath(fn)
        self.filename = to_unicode(fn, get_filesystem_encoding())
        self.module = self.globals.get('__name__')
        self.loader = self.globals.get('__loader__')
        self.code = tb.tb_frame.f_code

        # support for paste's traceback extensions
        self.hide = self.locals.get('__traceback_hide__', False)
        info = self.locals.get('__traceback_info__')
        if info is not None:
            try:
                info = text_type(info)
            except UnicodeError:
                info = str(info).decode('utf-8', 'replace')
        self.info = info

    def render(self):
        """Render a single frame in a traceback."""
        return FRAME_HTML % {
            'id':               self.id,
            'filename':         escape(self.filename),
            'lineno':           self.lineno,
            'function_name':    escape(self.function_name),
            'lines':            self.render_line_context(),
        }

    def render_line_context(self):
        before, current, after = self.get_context_lines()
        rv = []

        def render_line(line, cls):
            line = line.expandtabs().rstrip()
            stripped_line = line.strip()
            prefix = len(line) - len(stripped_line)
            rv.append(
                '<pre class="line %s"><span class="ws">%s</span>%s</pre>' % (
                    cls, ' ' * prefix, escape(stripped_line) or ' '))

        for line in before:
            render_line(line, 'before')
        render_line(current, 'current')
        for line in after:
            render_line(line, 'after')

        return '\n'.join(rv)

    def get_annotated_lines(self):
        """Helper function that returns lines with extra information."""
        lines = [Line(idx + 1, x) for idx, x in enumerate(self.sourcelines)]

        # find function definition and mark lines
        if hasattr(self.code, 'co_firstlineno'):
            lineno = self.code.co_firstlineno - 1
            while lineno > 0:
                if _funcdef_re.match(lines[lineno].code):
                    break
                lineno -= 1
            try:
                offset = len(inspect.getblock([x.code + '\n' for x
                                               in lines[lineno:]]))
            except TokenError:
                offset = 0
            for line in lines[lineno:lineno + offset]:
                line.in_frame = True

        # mark current line
        try:
            lines[self.lineno - 1].current = True
        except IndexError:
            pass

        return lines

    def eval(self, code, mode='single'):
        """Evaluate code in the context of the frame."""
        if isinstance(code, string_types):
            if PY2 and isinstance(code, unicode):  # noqa
                code = UTF8_COOKIE + code.encode('utf-8')
            code = compile(code, '<interactive>', mode)
        return eval(code, self.globals, self.locals)

    @cached_property
    def sourcelines(self):
        """The sourcecode of the file as list of unicode strings."""
        # get sourcecode from loader or file
        source = None
        if self.loader is not None:
            try:
                if hasattr(self.loader, 'get_source'):
                    source = self.loader.get_source(self.module)
                elif hasattr(self.loader, 'get_source_by_code'):
                    source = self.loader.get_source_by_code(self.code)
            except Exception:
                # we munch the exception so that we don't cause troubles
                # if the loader is broken.
                pass

        if source is None:
            try:
                f = open(to_native(self.filename, get_filesystem_encoding()),
                         mode='rb')
            except IOError:
                return []
            try:
                source = f.read()
            finally:
                f.close()

        # already unicode?  return right away
        if isinstance(source, text_type):
            return source.splitlines()

        # yes. it should be ascii, but we don't want to reject too many
        # characters in the debugger if something breaks
        charset = 'utf-8'
        if source.startswith(UTF8_COOKIE):
            source = source[3:]
        else:
            for idx, match in enumerate(_line_re.finditer(source)):
                match = _coding_re.search(match.group())
                if match is not None:
                    charset = match.group(1)
                    break
                if idx > 1:
                    break

        # on broken cookies we fall back to utf-8 too
        charset = to_native(charset)
        try:
            codecs.lookup(charset)
        except LookupError:
            charset = 'utf-8'

        return source.decode(charset, 'replace').splitlines()

    def get_context_lines(self, context=5):
        before = self.sourcelines[self.lineno - context - 1:self.lineno - 1]
        past = self.sourcelines[self.lineno:self.lineno + context]
        return (
            before,
            self.current_line,
            past,
        )

    @property
    def current_line(self):
        try:
            return self.sourcelines[self.lineno - 1]
        except IndexError:
            return u''

    @cached_property
    def console(self):
        return Console(self.globals, self.locals)

    id = property(lambda x: id(x))
