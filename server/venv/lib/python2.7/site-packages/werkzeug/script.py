# -*- coding: utf-8 -*-
r'''
    werkzeug.script
    ~~~~~~~~~~~~~~~

    .. admonition:: Deprecated Functionality

       ``werkzeug.script`` is deprecated without replacement functionality.
       Python's command line support improved greatly with :mod:`argparse`
       and a bunch of alternative modules.

    Most of the time you have recurring tasks while writing an application
    such as starting up an interactive python interpreter with some prefilled
    imports, starting the development server, initializing the database or
    something similar.

    For that purpose werkzeug provides the `werkzeug.script` module which
    helps you writing such scripts.


    Basic Usage
    -----------

    The following snippet is roughly the same in every werkzeug script::

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        from werkzeug import script

        # actions go here

        if __name__ == '__main__':
            script.run()

    Starting this script now does nothing because no actions are defined.
    An action is a function in the same module starting with ``"action_"``
    which takes a number of arguments where every argument has a default.  The
    type of the default value specifies the type of the argument.

    Arguments can then be passed by position or using ``--name=value`` from
    the shell.

    Because a runserver and shell command is pretty common there are two
    factory functions that create such commands::

        def make_app():
            from yourapplication import YourApplication
            return YourApplication(...)

        action_runserver = script.make_runserver(make_app, use_reloader=True)
        action_shell = script.make_shell(lambda: {'app': make_app()})


    Using The Scripts
    -----------------

    The script from above can be used like this from the shell now:

    .. sourcecode:: text

        $ ./manage.py --help
        $ ./manage.py runserver localhost 8080 --debugger --no-reloader
        $ ./manage.py runserver -p 4000
        $ ./manage.py shell

    As you can see it's possible to pass parameters as positional arguments
    or as named parameters, pretty much like Python function calls.


    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
'''
from __future__ import print_function

import sys
import inspect
import getopt
from os.path import basename
from werkzeug._compat import iteritems


argument_types = {
    bool:       'boolean',
    str:        'string',
    int:        'integer',
    float:      'float'
}


converters = {
    'boolean': lambda x: x.lower() in ('1', 'true', 'yes', 'on'),
    'string':   str,
    'integer':  int,
    'float':    float
}


def run(namespace=None, action_prefix='action_', args=None):
    """Run the script.  Participating actions are looked up in the caller's
    namespace if no namespace is given, otherwise in the dict provided.
    Only items that start with action_prefix are processed as actions.  If
    you want to use all items in the namespace provided as actions set
    action_prefix to an empty string.

    :param namespace: An optional dict where the functions are looked up in.
                      By default the local namespace of the caller is used.
    :param action_prefix: The prefix for the functions.  Everything else
                          is ignored.
    :param args: the arguments for the function.  If not specified
                 :data:`sys.argv` without the first argument is used.
    """
    if namespace is None:
        namespace = sys._getframe(1).f_locals
    actions = find_actions(namespace, action_prefix)

    if args is None:
        args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        return print_usage(actions)
    elif args[0] not in actions:
        fail('Unknown action \'%s\'' % args[0])

    arguments = {}
    types = {}
    key_to_arg = {}
    long_options = []
    formatstring = ''
    func, doc, arg_def = actions[args.pop(0)]
    for idx, (arg, shortcut, default, option_type) in enumerate(arg_def):
        real_arg = arg.replace('-', '_')
        if shortcut:
            formatstring += shortcut
            if not isinstance(default, bool):
                formatstring += ':'
            key_to_arg['-' + shortcut] = real_arg
        long_options.append(isinstance(default, bool) and arg or arg + '=')
        key_to_arg['--' + arg] = real_arg
        key_to_arg[idx] = real_arg
        types[real_arg] = option_type
        arguments[real_arg] = default

    try:
        optlist, posargs = getopt.gnu_getopt(args, formatstring, long_options)
    except getopt.GetoptError as e:
        fail(str(e))

    specified_arguments = set()
    for key, value in enumerate(posargs):
        try:
            arg = key_to_arg[key]
        except IndexError:
            fail('Too many parameters')
        specified_arguments.add(arg)
        try:
            arguments[arg] = converters[types[arg]](value)
        except ValueError:
            fail('Invalid value for argument %s (%s): %s' % (key, arg, value))

    for key, value in optlist:
        arg = key_to_arg[key]
        if arg in specified_arguments:
            fail('Argument \'%s\' is specified twice' % arg)
        if types[arg] == 'boolean':
            if arg.startswith('no_'):
                value = 'no'
            else:
                value = 'yes'
        try:
            arguments[arg] = converters[types[arg]](value)
        except ValueError:
            fail('Invalid value for \'%s\': %s' % (key, value))

    newargs = {}
    for k, v in iteritems(arguments):
        newargs[k.startswith('no_') and k[3:] or k] = v
    arguments = newargs
    return func(**arguments)


def fail(message, code=-1):
    """Fail with an error."""
    print('Error: %s' % message, file=sys.stderr)
    sys.exit(code)


def find_actions(namespace, action_prefix):
    """Find all the actions in the namespace."""
    actions = {}
    for key, value in iteritems(namespace):
        if key.startswith(action_prefix):
            actions[key[len(action_prefix):]] = analyse_action(value)
    return actions


def print_usage(actions):
    """Print the usage information.  (Help screen)"""
    actions = sorted(iteritems(actions))
    print('usage: %s <action> [<options>]' % basename(sys.argv[0]))
    print('       %s --help' % basename(sys.argv[0]))
    print()
    print('actions:')
    for name, (func, doc, arguments) in actions:
        print('  %s:' % name)
        for line in doc.splitlines():
            print('    %s' % line)
        if arguments:
            print()
        for arg, shortcut, default, argtype in arguments:
            if isinstance(default, bool):
                print('    %s' % (
                    (shortcut and '-%s, ' % shortcut or '') + '--' + arg
                ))
            else:
                print('    %-30s%-10s%s' % (
                    (shortcut and '-%s, ' % shortcut or '') + '--' + arg,
                    argtype, default
                ))
        print()


def analyse_action(func):
    """Analyse a function."""
    description = inspect.getdoc(func) or 'undocumented action'
    arguments = []
    args, varargs, kwargs, defaults = inspect.getargspec(func)
    if varargs or kwargs:
        raise TypeError('variable length arguments for action not allowed.')
    if len(args) != len(defaults or ()):
        raise TypeError('not all arguments have proper definitions')

    for idx, (arg, definition) in enumerate(zip(args, defaults or ())):
        if arg.startswith('_'):
            raise TypeError('arguments may not start with an underscore')
        if not isinstance(definition, tuple):
            shortcut = None
            default = definition
        else:
            shortcut, default = definition
        argument_type = argument_types[type(default)]
        if isinstance(default, bool) and default is True:
            arg = 'no-' + arg
        arguments.append((arg.replace('_', '-'), shortcut,
                          default, argument_type))
    return func, description, arguments


def make_shell(init_func=None, banner=None, use_ipython=True):
    """Returns an action callback that spawns a new interactive
    python shell.

    :param init_func: an optional initialization function that is
                      called before the shell is started.  The return
                      value of this function is the initial namespace.
    :param banner: the banner that is displayed before the shell.  If
                   not specified a generic banner is used instead.
    :param use_ipython: if set to `True` ipython is used if available.
    """
    if banner is None:
        banner = 'Interactive Werkzeug Shell'
    if init_func is None:
        init_func = dict

    def action(ipython=use_ipython):
        """Start a new interactive python session."""
        namespace = init_func()
        if ipython:
            try:
                try:
                    from IPython.frontend.terminal.embed import InteractiveShellEmbed
                    sh = InteractiveShellEmbed(banner1=banner)
                except ImportError:
                    from IPython.Shell import IPShellEmbed
                    sh = IPShellEmbed(banner=banner)
            except ImportError:
                pass
            else:
                sh(global_ns={}, local_ns=namespace)
                return
        from code import interact
        interact(banner, local=namespace)
    return action


def make_runserver(app_factory, hostname='localhost', port=5000,
                   use_reloader=False, use_debugger=False, use_evalex=True,
                   threaded=False, processes=1, static_files=None,
                   extra_files=None, ssl_context=None):
    """Returns an action callback that spawns a new development server.

    .. versionadded:: 0.5
       `static_files` and `extra_files` was added.

    ..versionadded:: 0.6.1
       `ssl_context` was added.

    :param app_factory: a function that returns a new WSGI application.
    :param hostname: the default hostname the server should listen on.
    :param port: the default port of the server.
    :param use_reloader: the default setting for the reloader.
    :param use_evalex: the default setting for the evalex flag of the debugger.
    :param threaded: the default threading setting.
    :param processes: the default number of processes to start.
    :param static_files: optional dict of static files.
    :param extra_files: optional list of extra files to track for reloading.
    :param ssl_context: optional SSL context for running server in HTTPS mode.
    """
    def action(hostname=('h', hostname), port=('p', port),
               reloader=use_reloader, debugger=use_debugger,
               evalex=use_evalex, threaded=threaded, processes=processes):
        """Start a new development server."""
        from werkzeug.serving import run_simple
        app = app_factory()
        run_simple(hostname, port, app,
                   use_reloader=reloader, use_debugger=debugger,
                   use_evalex=evalex, extra_files=extra_files,
                   reloader_interval=1, threaded=threaded, processes=processes,
                   static_files=static_files, ssl_context=ssl_context)
    return action
