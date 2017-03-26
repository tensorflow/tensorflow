# -*- coding: utf-8 -*-
"""
    werkzeug.debug
    ~~~~~~~~~~~~~~

    WSGI application traceback debugger.

    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import os
import re
import sys
import uuid
import json
import time
import getpass
import hashlib
import mimetypes
from itertools import chain
from os.path import join, dirname, basename, isfile
from werkzeug.wrappers import BaseRequest as Request, BaseResponse as Response
from werkzeug.http import parse_cookie
from werkzeug.debug.tbtools import get_current_traceback, render_console_html
from werkzeug.debug.console import Console
from werkzeug.security import gen_salt
from werkzeug._internal import _log
from werkzeug._compat import text_type


# DEPRECATED
#: import this here because it once was documented as being available
#: from this module.  In case there are users left ...
from werkzeug.debug.repr import debug_repr  # noqa


# A week
PIN_TIME = 60 * 60 * 24 * 7


def hash_pin(pin):
    if isinstance(pin, text_type):
        pin = pin.encode('utf-8', 'replace')
    return hashlib.md5(pin + b'shittysalt').hexdigest()[:12]


_machine_id = None


def get_machine_id():
    global _machine_id
    rv = _machine_id
    if rv is not None:
        return rv

    def _generate():
        # Potential sources of secret information on linux.  The machine-id
        # is stable across boots, the boot id is not
        for filename in '/etc/machine-id', '/proc/sys/kernel/random/boot_id':
            try:
                with open(filename, 'rb') as f:
                    return f.readline().strip()
            except IOError:
                continue

        # On OS X we can use the computer's serial number assuming that
        # ioreg exists and can spit out that information.
        try:
            # Also catch import errors: subprocess may not be available, e.g.
            # Google App Engine
            # See https://github.com/pallets/werkzeug/issues/925
            from subprocess import Popen, PIPE
            dump = Popen(['ioreg', '-c', 'IOPlatformExpertDevice', '-d', '2'],
                         stdout=PIPE).communicate()[0]
            match = re.search(b'"serial-number" = <([^>]+)', dump)
            if match is not None:
                return match.group(1)
        except (OSError, ImportError):
            pass

        # On Windows we can use winreg to get the machine guid
        wr = None
        try:
            import winreg as wr
        except ImportError:
            try:
                import _winreg as wr
            except ImportError:
                pass
        if wr is not None:
            try:
                with wr.OpenKey(wr.HKEY_LOCAL_MACHINE,
                                'SOFTWARE\\Microsoft\\Cryptography', 0,
                                wr.KEY_READ | wr.KEY_WOW64_64KEY) as rk:
                    return wr.QueryValueEx(rk, 'MachineGuid')[0]
            except WindowsError:
                pass

    _machine_id = rv = _generate()
    return rv


class _ConsoleFrame(object):

    """Helper class so that we can reuse the frame console code for the
    standalone console.
    """

    def __init__(self, namespace):
        self.console = Console(namespace)
        self.id = 0


def get_pin_and_cookie_name(app):
    """Given an application object this returns a semi-stable 9 digit pin
    code and a random key.  The hope is that this is stable between
    restarts to not make debugging particularly frustrating.  If the pin
    was forcefully disabled this returns `None`.

    Second item in the resulting tuple is the cookie name for remembering.
    """
    pin = os.environ.get('WERKZEUG_DEBUG_PIN')
    rv = None
    num = None

    # Pin was explicitly disabled
    if pin == 'off':
        return None, None

    # Pin was provided explicitly
    if pin is not None and pin.replace('-', '').isdigit():
        # If there are separators in the pin, return it directly
        if '-' in pin:
            rv = pin
        else:
            num = pin

    modname = getattr(app, '__module__',
                      getattr(app.__class__, '__module__'))

    try:
        # `getpass.getuser()` imports the `pwd` module,
        # which does not exist in the Google App Engine sandbox.
        username = getpass.getuser()
    except ImportError:
        username = None

    mod = sys.modules.get(modname)

    # This information only exists to make the cookie unique on the
    # computer, not as a security feature.
    probably_public_bits = [
        username,
        modname,
        getattr(app, '__name__', getattr(app.__class__, '__name__')),
        getattr(mod, '__file__', None),
    ]

    # This information is here to make it harder for an attacker to
    # guess the cookie name.  They are unlikely to be contained anywhere
    # within the unauthenticated debug page.
    private_bits = [
        str(uuid.getnode()),
        get_machine_id(),
    ]

    h = hashlib.md5()
    for bit in chain(probably_public_bits, private_bits):
        if not bit:
            continue
        if isinstance(bit, text_type):
            bit = bit.encode('utf-8')
        h.update(bit)
    h.update(b'cookiesalt')

    cookie_name = '__wzd' + h.hexdigest()[:20]

    # If we need to generate a pin we salt it a bit more so that we don't
    # end up with the same value and generate out 9 digits
    if num is None:
        h.update(b'pinsalt')
        num = ('%09d' % int(h.hexdigest(), 16))[:9]

    # Format the pincode in groups of digits for easier remembering if
    # we don't have a result yet.
    if rv is None:
        for group_size in 5, 4, 3:
            if len(num) % group_size == 0:
                rv = '-'.join(num[x:x + group_size].rjust(group_size, '0')
                              for x in range(0, len(num), group_size))
                break
        else:
            rv = num

    return rv, cookie_name


class DebuggedApplication(object):
    """Enables debugging support for a given application::

        from werkzeug.debug import DebuggedApplication
        from myapp import app
        app = DebuggedApplication(app, evalex=True)

    The `evalex` keyword argument allows evaluating expressions in a
    traceback's frame context.

    .. versionadded:: 0.9
       The `lodgeit_url` parameter was deprecated.

    :param app: the WSGI application to run debugged.
    :param evalex: enable exception evaluation feature (interactive
                   debugging).  This requires a non-forking server.
    :param request_key: The key that points to the request object in ths
                        environment.  This parameter is ignored in current
                        versions.
    :param console_path: the URL for a general purpose console.
    :param console_init_func: the function that is executed before starting
                              the general purpose console.  The return value
                              is used as initial namespace.
    :param show_hidden_frames: by default hidden traceback frames are skipped.
                               You can show them by setting this parameter
                               to `True`.
    :param pin_security: can be used to disable the pin based security system.
    :param pin_logging: enables the logging of the pin system.
    """

    def __init__(self, app, evalex=False, request_key='werkzeug.request',
                 console_path='/console', console_init_func=None,
                 show_hidden_frames=False, lodgeit_url=None,
                 pin_security=True, pin_logging=True):
        if lodgeit_url is not None:
            from warnings import warn
            warn(DeprecationWarning('Werkzeug now pastes into gists.'))
        if not console_init_func:
            console_init_func = None
        self.app = app
        self.evalex = evalex
        self.frames = {}
        self.tracebacks = {}
        self.request_key = request_key
        self.console_path = console_path
        self.console_init_func = console_init_func
        self.show_hidden_frames = show_hidden_frames
        self.secret = gen_salt(20)
        self._failed_pin_auth = 0

        self.pin_logging = pin_logging
        if pin_security:
            # Print out the pin for the debugger on standard out.
            if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' and \
               pin_logging:
                _log('warning', ' * Debugger is active!')
                if self.pin is None:
                    _log('warning', ' * Debugger pin disabled.  '
                         'DEBUGGER UNSECURED!')
                else:
                    _log('info', ' * Debugger pin code: %s' % self.pin)
        else:
            self.pin = None

    def _get_pin(self):
        if not hasattr(self, '_pin'):
            self._pin, self._pin_cookie = get_pin_and_cookie_name(self.app)
        return self._pin

    def _set_pin(self, value):
        self._pin = value

    pin = property(_get_pin, _set_pin)
    del _get_pin, _set_pin

    @property
    def pin_cookie_name(self):
        """The name of the pin cookie."""
        if not hasattr(self, '_pin_cookie'):
            self._pin, self._pin_cookie = get_pin_and_cookie_name(self.app)
        return self._pin_cookie

    def debug_application(self, environ, start_response):
        """Run the application and conserve the traceback frames."""
        app_iter = None
        try:
            app_iter = self.app(environ, start_response)
            for item in app_iter:
                yield item
            if hasattr(app_iter, 'close'):
                app_iter.close()
        except Exception:
            if hasattr(app_iter, 'close'):
                app_iter.close()
            traceback = get_current_traceback(
                skip=1, show_hidden_frames=self.show_hidden_frames,
                ignore_system_exceptions=True)
            for frame in traceback.frames:
                self.frames[frame.id] = frame
            self.tracebacks[traceback.id] = traceback

            try:
                start_response('500 INTERNAL SERVER ERROR', [
                    ('Content-Type', 'text/html; charset=utf-8'),
                    # Disable Chrome's XSS protection, the debug
                    # output can cause false-positives.
                    ('X-XSS-Protection', '0'),
                ])
            except Exception:
                # if we end up here there has been output but an error
                # occurred.  in that situation we can do nothing fancy any
                # more, better log something into the error log and fall
                # back gracefully.
                environ['wsgi.errors'].write(
                    'Debugging middleware caught exception in streamed '
                    'response at a point where response headers were already '
                    'sent.\n')
            else:
                is_trusted = bool(self.check_pin_trust(environ))
                yield traceback.render_full(evalex=self.evalex,
                                            evalex_trusted=is_trusted,
                                            secret=self.secret) \
                    .encode('utf-8', 'replace')

            traceback.log(environ['wsgi.errors'])

    def execute_command(self, request, command, frame):
        """Execute a command in a console."""
        return Response(frame.console.eval(command), mimetype='text/html')

    def display_console(self, request):
        """Display a standalone shell."""
        if 0 not in self.frames:
            if self.console_init_func is None:
                ns = {}
            else:
                ns = dict(self.console_init_func())
            ns.setdefault('app', self.app)
            self.frames[0] = _ConsoleFrame(ns)
        is_trusted = bool(self.check_pin_trust(request.environ))
        return Response(render_console_html(secret=self.secret,
                                            evalex_trusted=is_trusted),
                        mimetype='text/html')

    def paste_traceback(self, request, traceback):
        """Paste the traceback and return a JSON response."""
        rv = traceback.paste()
        return Response(json.dumps(rv), mimetype='application/json')

    def get_resource(self, request, filename):
        """Return a static resource from the shared folder."""
        filename = join(dirname(__file__), 'shared', basename(filename))
        if isfile(filename):
            mimetype = mimetypes.guess_type(filename)[0] \
                or 'application/octet-stream'
            f = open(filename, 'rb')
            try:
                return Response(f.read(), mimetype=mimetype)
            finally:
                f.close()
        return Response('Not Found', status=404)

    def check_pin_trust(self, environ):
        """Checks if the request passed the pin test.  This returns `True` if the
        request is trusted on a pin/cookie basis and returns `False` if not.
        Additionally if the cookie's stored pin hash is wrong it will return
        `None` so that appropriate action can be taken.
        """
        if self.pin is None:
            return True
        val = parse_cookie(environ).get(self.pin_cookie_name)
        if not val or '|' not in val:
            return False
        ts, pin_hash = val.split('|', 1)
        if not ts.isdigit():
            return False
        if pin_hash != hash_pin(self.pin):
            return None
        return (time.time() - PIN_TIME) < int(ts)

    def _fail_pin_auth(self):
        time.sleep(self._failed_pin_auth > 5 and 5.0 or 0.5)
        self._failed_pin_auth += 1

    def pin_auth(self, request):
        """Authenticates with the pin."""
        exhausted = False
        auth = False
        trust = self.check_pin_trust(request.environ)

        # If the trust return value is `None` it means that the cookie is
        # set but the stored pin hash value is bad.  This means that the
        # pin was changed.  In this case we count a bad auth and unset the
        # cookie.  This way it becomes harder to guess the cookie name
        # instead of the pin as we still count up failures.
        bad_cookie = False
        if trust is None:
            self._fail_pin_auth()
            bad_cookie = True

        # If we're trusted, we're authenticated.
        elif trust:
            auth = True

        # If we failed too many times, then we're locked out.
        elif self._failed_pin_auth > 10:
            exhausted = True

        # Otherwise go through pin based authentication
        else:
            entered_pin = request.args.get('pin')
            if entered_pin.strip().replace('-', '') == \
               self.pin.replace('-', ''):
                self._failed_pin_auth = 0
                auth = True
            else:
                self._fail_pin_auth()

        rv = Response(json.dumps({
            'auth': auth,
            'exhausted': exhausted,
        }), mimetype='application/json')
        if auth:
            rv.set_cookie(self.pin_cookie_name, '%s|%s' % (
                int(time.time()),
                hash_pin(self.pin)
            ), httponly=True)
        elif bad_cookie:
            rv.delete_cookie(self.pin_cookie_name)
        return rv

    def log_pin_request(self):
        """Log the pin if needed."""
        if self.pin_logging and self.pin is not None:
            _log('info', ' * To enable the debugger you need to '
                 'enter the security pin:')
            _log('info', ' * Debugger pin code: %s' % self.pin)
        return Response('')

    def __call__(self, environ, start_response):
        """Dispatch the requests."""
        # important: don't ever access a function here that reads the incoming
        # form data!  Otherwise the application won't have access to that data
        # any more!
        request = Request(environ)
        response = self.debug_application
        if request.args.get('__debugger__') == 'yes':
            cmd = request.args.get('cmd')
            arg = request.args.get('f')
            secret = request.args.get('s')
            traceback = self.tracebacks.get(request.args.get('tb', type=int))
            frame = self.frames.get(request.args.get('frm', type=int))
            if cmd == 'resource' and arg:
                response = self.get_resource(request, arg)
            elif cmd == 'paste' and traceback is not None and \
                    secret == self.secret:
                response = self.paste_traceback(request, traceback)
            elif cmd == 'pinauth' and secret == self.secret:
                response = self.pin_auth(request)
            elif cmd == 'printpin' and secret == self.secret:
                response = self.log_pin_request()
            elif self.evalex and cmd is not None and frame is not None \
                    and self.secret == secret and \
                    self.check_pin_trust(environ):
                response = self.execute_command(request, cmd, frame)
        elif self.evalex and self.console_path is not None and \
                request.path == self.console_path:
            response = self.display_console(request)
        return response(environ, start_response)
