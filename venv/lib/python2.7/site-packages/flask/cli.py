# -*- coding: utf-8 -*-
"""
    flask.cli
    ~~~~~~~~~

    A simple command line application to run flask apps.

    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""

import os
import sys
from threading import Lock, Thread
from functools import update_wrapper

import click

from ._compat import iteritems, reraise
from .helpers import get_debug_flag
from . import __version__

class NoAppException(click.UsageError):
    """Raised if an application cannot be found or loaded."""


def find_best_app(module):
    """Given a module instance this tries to find the best possible
    application in the module or raises an exception.
    """
    from . import Flask

    # Search for the most common names first.
    for attr_name in 'app', 'application':
        app = getattr(module, attr_name, None)
        if app is not None and isinstance(app, Flask):
            return app

    # Otherwise find the only object that is a Flask instance.
    matches = [v for k, v in iteritems(module.__dict__)
               if isinstance(v, Flask)]

    if len(matches) == 1:
        return matches[0]
    raise NoAppException('Failed to find application in module "%s".  Are '
                         'you sure it contains a Flask application?  Maybe '
                         'you wrapped it in a WSGI middleware or you are '
                         'using a factory function.' % module.__name__)


def prepare_exec_for_file(filename):
    """Given a filename this will try to calculate the python path, add it
    to the search path and return the actual module name that is expected.
    """
    module = []

    # Chop off file extensions or package markers
    if os.path.split(filename)[1] == '__init__.py':
        filename = os.path.dirname(filename)
    elif filename.endswith('.py'):
        filename = filename[:-3]
    else:
        raise NoAppException('The file provided (%s) does exist but is not a '
                             'valid Python file.  This means that it cannot '
                             'be used as application.  Please change the '
                             'extension to .py' % filename)
    filename = os.path.realpath(filename)

    dirpath = filename
    while 1:
        dirpath, extra = os.path.split(dirpath)
        module.append(extra)
        if not os.path.isfile(os.path.join(dirpath, '__init__.py')):
            break

    sys.path.insert(0, dirpath)
    return '.'.join(module[::-1])


def locate_app(app_id):
    """Attempts to locate the application."""
    __traceback_hide__ = True
    if ':' in app_id:
        module, app_obj = app_id.split(':', 1)
    else:
        module = app_id
        app_obj = None

    try:
        __import__(module)
    except ImportError:
        raise NoAppException('The file/path provided (%s) does not appear to '
                             'exist.  Please verify the path is correct.  If '
                             'app is not on PYTHONPATH, ensure the extension '
                             'is .py' % module)
    mod = sys.modules[module]
    if app_obj is None:
        app = find_best_app(mod)
    else:
        app = getattr(mod, app_obj, None)
        if app is None:
            raise RuntimeError('Failed to find application in module "%s"'
                               % module)

    return app


def find_default_import_path():
    app = os.environ.get('FLASK_APP')
    if app is None:
        return
    if os.path.isfile(app):
        return prepare_exec_for_file(app)
    return app


def get_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    message = 'Flask %(version)s\nPython %(python_version)s'
    click.echo(message % {
        'version': __version__,
        'python_version': sys.version,
    }, color=ctx.color)
    ctx.exit()

version_option = click.Option(['--version'],
                              help='Show the flask version',
                              expose_value=False,
                              callback=get_version,
                              is_flag=True, is_eager=True)

class DispatchingApp(object):
    """Special application that dispatches to a Flask application which
    is imported by name in a background thread.  If an error happens
    it is recorded and shown as part of the WSGI handling which in case
    of the Werkzeug debugger means that it shows up in the browser.
    """

    def __init__(self, loader, use_eager_loading=False):
        self.loader = loader
        self._app = None
        self._lock = Lock()
        self._bg_loading_exc_info = None
        if use_eager_loading:
            self._load_unlocked()
        else:
            self._load_in_background()

    def _load_in_background(self):
        def _load_app():
            __traceback_hide__ = True
            with self._lock:
                try:
                    self._load_unlocked()
                except Exception:
                    self._bg_loading_exc_info = sys.exc_info()
        t = Thread(target=_load_app, args=())
        t.start()

    def _flush_bg_loading_exception(self):
        __traceback_hide__ = True
        exc_info = self._bg_loading_exc_info
        if exc_info is not None:
            self._bg_loading_exc_info = None
            reraise(*exc_info)

    def _load_unlocked(self):
        __traceback_hide__ = True
        self._app = rv = self.loader()
        self._bg_loading_exc_info = None
        return rv

    def __call__(self, environ, start_response):
        __traceback_hide__ = True
        if self._app is not None:
            return self._app(environ, start_response)
        self._flush_bg_loading_exception()
        with self._lock:
            if self._app is not None:
                rv = self._app
            else:
                rv = self._load_unlocked()
            return rv(environ, start_response)


class ScriptInfo(object):
    """Help object to deal with Flask applications.  This is usually not
    necessary to interface with as it's used internally in the dispatching
    to click.  In future versions of Flask this object will most likely play
    a bigger role.  Typically it's created automatically by the
    :class:`FlaskGroup` but you can also manually create it and pass it
    onwards as click object.
    """

    def __init__(self, app_import_path=None, create_app=None):
        if create_app is None:
            if app_import_path is None:
                app_import_path = find_default_import_path()
            self.app_import_path = app_import_path
        else:
            app_import_path = None

        #: Optionally the import path for the Flask application.
        self.app_import_path = app_import_path
        #: Optionally a function that is passed the script info to create
        #: the instance of the application.
        self.create_app = create_app
        #: A dictionary with arbitrary data that can be associated with
        #: this script info.
        self.data = {}
        self._loaded_app = None

    def load_app(self):
        """Loads the Flask app (if not yet loaded) and returns it.  Calling
        this multiple times will just result in the already loaded app to
        be returned.
        """
        __traceback_hide__ = True
        if self._loaded_app is not None:
            return self._loaded_app
        if self.create_app is not None:
            rv = self.create_app(self)
        else:
            if not self.app_import_path:
                raise NoAppException(
                    'Could not locate Flask application. You did not provide '
                    'the FLASK_APP environment variable.\n\nFor more '
                    'information see '
                    'http://flask.pocoo.org/docs/latest/quickstart/')
            rv = locate_app(self.app_import_path)
        debug = get_debug_flag()
        if debug is not None:
            rv.debug = debug
        self._loaded_app = rv
        return rv


pass_script_info = click.make_pass_decorator(ScriptInfo, ensure=True)


def with_appcontext(f):
    """Wraps a callback so that it's guaranteed to be executed with the
    script's application context.  If callbacks are registered directly
    to the ``app.cli`` object then they are wrapped with this function
    by default unless it's disabled.
    """
    @click.pass_context
    def decorator(__ctx, *args, **kwargs):
        with __ctx.ensure_object(ScriptInfo).load_app().app_context():
            return __ctx.invoke(f, *args, **kwargs)
    return update_wrapper(decorator, f)


class AppGroup(click.Group):
    """This works similar to a regular click :class:`~click.Group` but it
    changes the behavior of the :meth:`command` decorator so that it
    automatically wraps the functions in :func:`with_appcontext`.

    Not to be confused with :class:`FlaskGroup`.
    """

    def command(self, *args, **kwargs):
        """This works exactly like the method of the same name on a regular
        :class:`click.Group` but it wraps callbacks in :func:`with_appcontext`
        unless it's disabled by passing ``with_appcontext=False``.
        """
        wrap_for_ctx = kwargs.pop('with_appcontext', True)
        def decorator(f):
            if wrap_for_ctx:
                f = with_appcontext(f)
            return click.Group.command(self, *args, **kwargs)(f)
        return decorator

    def group(self, *args, **kwargs):
        """This works exactly like the method of the same name on a regular
        :class:`click.Group` but it defaults the group class to
        :class:`AppGroup`.
        """
        kwargs.setdefault('cls', AppGroup)
        return click.Group.group(self, *args, **kwargs)


class FlaskGroup(AppGroup):
    """Special subclass of the :class:`AppGroup` group that supports
    loading more commands from the configured Flask app.  Normally a
    developer does not have to interface with this class but there are
    some very advanced use cases for which it makes sense to create an
    instance of this.

    For information as of why this is useful see :ref:`custom-scripts`.

    :param add_default_commands: if this is True then the default run and
                                 shell commands wil be added.
    :param add_version_option: adds the ``--version`` option.
    :param create_app: an optional callback that is passed the script info
                       and returns the loaded app.
    """

    def __init__(self, add_default_commands=True, create_app=None,
                 add_version_option=True, **extra):
        params = list(extra.pop('params', None) or ())

        if add_version_option:
            params.append(version_option)

        AppGroup.__init__(self, params=params, **extra)
        self.create_app = create_app

        if add_default_commands:
            self.add_command(run_command)
            self.add_command(shell_command)

        self._loaded_plugin_commands = False

    def _load_plugin_commands(self):
        if self._loaded_plugin_commands:
            return
        try:
            import pkg_resources
        except ImportError:
            self._loaded_plugin_commands = True
            return

        for ep in pkg_resources.iter_entry_points('flask.commands'):
            self.add_command(ep.load(), ep.name)
        self._loaded_plugin_commands = True

    def get_command(self, ctx, name):
        self._load_plugin_commands()

        # We load built-in commands first as these should always be the
        # same no matter what the app does.  If the app does want to
        # override this it needs to make a custom instance of this group
        # and not attach the default commands.
        #
        # This also means that the script stays functional in case the
        # application completely fails.
        rv = AppGroup.get_command(self, ctx, name)
        if rv is not None:
            return rv

        info = ctx.ensure_object(ScriptInfo)
        try:
            rv = info.load_app().cli.get_command(ctx, name)
            if rv is not None:
                return rv
        except NoAppException:
            pass

    def list_commands(self, ctx):
        self._load_plugin_commands()

        # The commands available is the list of both the application (if
        # available) plus the builtin commands.
        rv = set(click.Group.list_commands(self, ctx))
        info = ctx.ensure_object(ScriptInfo)
        try:
            rv.update(info.load_app().cli.list_commands(ctx))
        except Exception:
            # Here we intentionally swallow all exceptions as we don't
            # want the help page to break if the app does not exist.
            # If someone attempts to use the command we try to create
            # the app again and this will give us the error.
            pass
        return sorted(rv)

    def main(self, *args, **kwargs):
        obj = kwargs.get('obj')
        if obj is None:
            obj = ScriptInfo(create_app=self.create_app)
        kwargs['obj'] = obj
        kwargs.setdefault('auto_envvar_prefix', 'FLASK')
        return AppGroup.main(self, *args, **kwargs)


@click.command('run', short_help='Runs a development server.')
@click.option('--host', '-h', default='127.0.0.1',
              help='The interface to bind to.')
@click.option('--port', '-p', default=5000,
              help='The port to bind to.')
@click.option('--reload/--no-reload', default=None,
              help='Enable or disable the reloader.  By default the reloader '
              'is active if debug is enabled.')
@click.option('--debugger/--no-debugger', default=None,
              help='Enable or disable the debugger.  By default the debugger '
              'is active if debug is enabled.')
@click.option('--eager-loading/--lazy-loader', default=None,
              help='Enable or disable eager loading.  By default eager '
              'loading is enabled if the reloader is disabled.')
@click.option('--with-threads/--without-threads', default=False,
              help='Enable or disable multithreading.')
@pass_script_info
def run_command(info, host, port, reload, debugger, eager_loading,
                with_threads):
    """Runs a local development server for the Flask application.

    This local server is recommended for development purposes only but it
    can also be used for simple intranet deployments.  By default it will
    not support any sort of concurrency at all to simplify debugging.  This
    can be changed with the --with-threads option which will enable basic
    multithreading.

    The reloader and debugger are by default enabled if the debug flag of
    Flask is enabled and disabled otherwise.
    """
    from werkzeug.serving import run_simple

    debug = get_debug_flag()
    if reload is None:
        reload = bool(debug)
    if debugger is None:
        debugger = bool(debug)
    if eager_loading is None:
        eager_loading = not reload

    app = DispatchingApp(info.load_app, use_eager_loading=eager_loading)

    # Extra startup messages.  This depends a bit on Werkzeug internals to
    # not double execute when the reloader kicks in.
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        # If we have an import path we can print it out now which can help
        # people understand what's being served.  If we do not have an
        # import path because the app was loaded through a callback then
        # we won't print anything.
        if info.app_import_path is not None:
            print(' * Serving Flask app "%s"' % info.app_import_path)
        if debug is not None:
            print(' * Forcing debug mode %s' % (debug and 'on' or 'off'))

    run_simple(host, port, app, use_reloader=reload,
               use_debugger=debugger, threaded=with_threads)


@click.command('shell', short_help='Runs a shell in the app context.')
@with_appcontext
def shell_command():
    """Runs an interactive Python shell in the context of a given
    Flask application.  The application will populate the default
    namespace of this shell according to it's configuration.

    This is useful for executing small snippets of management code
    without having to manually configuring the application.
    """
    import code
    from flask.globals import _app_ctx_stack
    app = _app_ctx_stack.top.app
    banner = 'Python %s on %s\nApp: %s%s\nInstance: %s' % (
        sys.version,
        sys.platform,
        app.import_name,
        app.debug and ' [debug]' or '',
        app.instance_path,
    )
    ctx = {}

    # Support the regular Python interpreter startup script if someone
    # is using it.
    startup = os.environ.get('PYTHONSTARTUP')
    if startup and os.path.isfile(startup):
        with open(startup, 'r') as f:
            eval(compile(f.read(), startup, 'exec'), ctx)

    ctx.update(app.make_shell_context())

    code.interact(banner=banner, local=ctx)


cli = FlaskGroup(help="""\
This shell command acts as general utility script for Flask applications.

It loads the application configured (through the FLASK_APP environment
variable) and then provides commands either provided by the application or
Flask itself.

The most useful commands are the "run" and "shell" command.

Example usage:

\b
  %(prefix)s%(cmd)s FLASK_APP=hello.py
  %(prefix)s%(cmd)s FLASK_DEBUG=1
  %(prefix)sflask run
""" % {
    'cmd': os.name == 'posix' and 'export' or 'set',
    'prefix': os.name == 'posix' and '$ ' or '',
})


def main(as_module=False):
    this_module = __package__ + '.cli'
    args = sys.argv[1:]

    if as_module:
        if sys.version_info >= (2, 7):
            name = 'python -m ' + this_module.rsplit('.', 1)[0]
        else:
            name = 'python -m ' + this_module

        # This module is always executed as "python -m flask.run" and as such
        # we need to ensure that we restore the actual command line so that
        # the reloader can properly operate.
        sys.argv = ['-m', this_module] + sys.argv[1:]
    else:
        name = None

    cli.main(args=args, prog_name=name)


if __name__ == '__main__':
    main(as_module=True)
