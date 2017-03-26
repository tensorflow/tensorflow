import os
import sys
import shutil
import tempfile
import contextlib

from ._compat import iteritems, PY2


# If someone wants to vendor click, we want to ensure the
# correct package is discovered.  Ideally we could use a
# relative import here but unfortunately Python does not
# support that.
clickpkg = sys.modules[__name__.rsplit('.', 1)[0]]


if PY2:
    from cStringIO import StringIO
else:
    import io
    from ._compat import _find_binary_reader


class EchoingStdin(object):

    def __init__(self, input, output):
        self._input = input
        self._output = output

    def __getattr__(self, x):
        return getattr(self._input, x)

    def _echo(self, rv):
        self._output.write(rv)
        return rv

    def read(self, n=-1):
        return self._echo(self._input.read(n))

    def readline(self, n=-1):
        return self._echo(self._input.readline(n))

    def readlines(self):
        return [self._echo(x) for x in self._input.readlines()]

    def __iter__(self):
        return iter(self._echo(x) for x in self._input)

    def __repr__(self):
        return repr(self._input)


def make_input_stream(input, charset):
    # Is already an input stream.
    if hasattr(input, 'read'):
        if PY2:
            return input
        rv = _find_binary_reader(input)
        if rv is not None:
            return rv
        raise TypeError('Could not find binary reader for input stream.')

    if input is None:
        input = b''
    elif not isinstance(input, bytes):
        input = input.encode(charset)
    if PY2:
        return StringIO(input)
    return io.BytesIO(input)


class Result(object):
    """Holds the captured result of an invoked CLI script."""

    def __init__(self, runner, output_bytes, exit_code, exception,
                 exc_info=None):
        #: The runner that created the result
        self.runner = runner
        #: The output as bytes.
        self.output_bytes = output_bytes
        #: The exit code as integer.
        self.exit_code = exit_code
        #: The exception that happend if one did.
        self.exception = exception
        #: The traceback
        self.exc_info = exc_info

    @property
    def output(self):
        """The output as unicode string."""
        return self.output_bytes.decode(self.runner.charset, 'replace') \
            .replace('\r\n', '\n')

    def __repr__(self):
        return '<Result %s>' % (
            self.exception and repr(self.exception) or 'okay',
        )


class CliRunner(object):
    """The CLI runner provides functionality to invoke a Click command line
    script for unittesting purposes in a isolated environment.  This only
    works in single-threaded systems without any concurrency as it changes the
    global interpreter state.

    :param charset: the character set for the input and output data.  This is
                    UTF-8 by default and should not be changed currently as
                    the reporting to Click only works in Python 2 properly.
    :param env: a dictionary with environment variables for overriding.
    :param echo_stdin: if this is set to `True`, then reading from stdin writes
                       to stdout.  This is useful for showing examples in
                       some circumstances.  Note that regular prompts
                       will automatically echo the input.
    """

    def __init__(self, charset=None, env=None, echo_stdin=False):
        if charset is None:
            charset = 'utf-8'
        self.charset = charset
        self.env = env or {}
        self.echo_stdin = echo_stdin

    def get_default_prog_name(self, cli):
        """Given a command object it will return the default program name
        for it.  The default is the `name` attribute or ``"root"`` if not
        set.
        """
        return cli.name or 'root'

    def make_env(self, overrides=None):
        """Returns the environment overrides for invoking a script."""
        rv = dict(self.env)
        if overrides:
            rv.update(overrides)
        return rv

    @contextlib.contextmanager
    def isolation(self, input=None, env=None, color=False):
        """A context manager that sets up the isolation for invoking of a
        command line tool.  This sets up stdin with the given input data
        and `os.environ` with the overrides from the given dictionary.
        This also rebinds some internals in Click to be mocked (like the
        prompt functionality).

        This is automatically done in the :meth:`invoke` method.

        .. versionadded:: 4.0
           The ``color`` parameter was added.

        :param input: the input stream to put into sys.stdin.
        :param env: the environment overrides as dictionary.
        :param color: whether the output should contain color codes. The
                      application can still override this explicitly.
        """
        input = make_input_stream(input, self.charset)

        old_stdin = sys.stdin
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_forced_width = clickpkg.formatting.FORCED_WIDTH
        clickpkg.formatting.FORCED_WIDTH = 80

        env = self.make_env(env)

        if PY2:
            sys.stdout = sys.stderr = bytes_output = StringIO()
            if self.echo_stdin:
                input = EchoingStdin(input, bytes_output)
        else:
            bytes_output = io.BytesIO()
            if self.echo_stdin:
                input = EchoingStdin(input, bytes_output)
            input = io.TextIOWrapper(input, encoding=self.charset)
            sys.stdout = sys.stderr = io.TextIOWrapper(
                bytes_output, encoding=self.charset)

        sys.stdin = input

        def visible_input(prompt=None):
            sys.stdout.write(prompt or '')
            val = input.readline().rstrip('\r\n')
            sys.stdout.write(val + '\n')
            sys.stdout.flush()
            return val

        def hidden_input(prompt=None):
            sys.stdout.write((prompt or '') + '\n')
            sys.stdout.flush()
            return input.readline().rstrip('\r\n')

        def _getchar(echo):
            char = sys.stdin.read(1)
            if echo:
                sys.stdout.write(char)
                sys.stdout.flush()
            return char

        default_color = color
        def should_strip_ansi(stream=None, color=None):
            if color is None:
                return not default_color
            return not color

        old_visible_prompt_func = clickpkg.termui.visible_prompt_func
        old_hidden_prompt_func = clickpkg.termui.hidden_prompt_func
        old__getchar_func = clickpkg.termui._getchar
        old_should_strip_ansi = clickpkg.utils.should_strip_ansi
        clickpkg.termui.visible_prompt_func = visible_input
        clickpkg.termui.hidden_prompt_func = hidden_input
        clickpkg.termui._getchar = _getchar
        clickpkg.utils.should_strip_ansi = should_strip_ansi

        old_env = {}
        try:
            for key, value in iteritems(env):
                old_env[key] = os.environ.get(key)
                if value is None:
                    try:
                        del os.environ[key]
                    except Exception:
                        pass
                else:
                    os.environ[key] = value
            yield bytes_output
        finally:
            for key, value in iteritems(old_env):
                if value is None:
                    try:
                        del os.environ[key]
                    except Exception:
                        pass
                else:
                    os.environ[key] = value
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.stdin = old_stdin
            clickpkg.termui.visible_prompt_func = old_visible_prompt_func
            clickpkg.termui.hidden_prompt_func = old_hidden_prompt_func
            clickpkg.termui._getchar = old__getchar_func
            clickpkg.utils.should_strip_ansi = old_should_strip_ansi
            clickpkg.formatting.FORCED_WIDTH = old_forced_width

    def invoke(self, cli, args=None, input=None, env=None,
               catch_exceptions=True, color=False, **extra):
        """Invokes a command in an isolated environment.  The arguments are
        forwarded directly to the command line script, the `extra` keyword
        arguments are passed to the :meth:`~clickpkg.Command.main` function of
        the command.

        This returns a :class:`Result` object.

        .. versionadded:: 3.0
           The ``catch_exceptions`` parameter was added.

        .. versionchanged:: 3.0
           The result object now has an `exc_info` attribute with the
           traceback if available.

        .. versionadded:: 4.0
           The ``color`` parameter was added.

        :param cli: the command to invoke
        :param args: the arguments to invoke
        :param input: the input data for `sys.stdin`.
        :param env: the environment overrides.
        :param catch_exceptions: Whether to catch any other exceptions than
                                 ``SystemExit``.
        :param extra: the keyword arguments to pass to :meth:`main`.
        :param color: whether the output should contain color codes. The
                      application can still override this explicitly.
        """
        exc_info = None
        with self.isolation(input=input, env=env, color=color) as out:
            exception = None
            exit_code = 0

            try:
                cli.main(args=args or (),
                         prog_name=self.get_default_prog_name(cli), **extra)
            except SystemExit as e:
                if e.code != 0:
                    exception = e

                exc_info = sys.exc_info()

                exit_code = e.code
                if not isinstance(exit_code, int):
                    sys.stdout.write(str(exit_code))
                    sys.stdout.write('\n')
                    exit_code = 1
            except Exception as e:
                if not catch_exceptions:
                    raise
                exception = e
                exit_code = -1
                exc_info = sys.exc_info()
            finally:
                sys.stdout.flush()
                output = out.getvalue()

        return Result(runner=self,
                      output_bytes=output,
                      exit_code=exit_code,
                      exception=exception,
                      exc_info=exc_info)

    @contextlib.contextmanager
    def isolated_filesystem(self):
        """A context manager that creates a temporary folder and changes
        the current working directory to it for isolated filesystem tests.
        """
        cwd = os.getcwd()
        t = tempfile.mkdtemp()
        os.chdir(t)
        try:
            yield t
        finally:
            os.chdir(cwd)
            try:
                shutil.rmtree(t)
            except (OSError, IOError):
                pass
