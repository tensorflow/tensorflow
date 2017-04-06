from ._compat import PY2, filename_to_ui, get_text_stderr
from .utils import echo


class ClickException(Exception):
    """An exception that Click can handle and show to the user."""

    #: The exit code for this exception
    exit_code = 1

    def __init__(self, message):
        if PY2:
            if message is not None:
                message = message.encode('utf-8')
        Exception.__init__(self, message)
        self.message = message

    def format_message(self):
        return self.message

    def show(self, file=None):
        if file is None:
            file = get_text_stderr()
        echo('Error: %s' % self.format_message(), file=file)


class UsageError(ClickException):
    """An internal exception that signals a usage error.  This typically
    aborts any further handling.

    :param message: the error message to display.
    :param ctx: optionally the context that caused this error.  Click will
                fill in the context automatically in some situations.
    """
    exit_code = 2

    def __init__(self, message, ctx=None):
        ClickException.__init__(self, message)
        self.ctx = ctx

    def show(self, file=None):
        if file is None:
            file = get_text_stderr()
        color = None
        if self.ctx is not None:
            color = self.ctx.color
            echo(self.ctx.get_usage() + '\n', file=file, color=color)
        echo('Error: %s' % self.format_message(), file=file, color=color)


class BadParameter(UsageError):
    """An exception that formats out a standardized error message for a
    bad parameter.  This is useful when thrown from a callback or type as
    Click will attach contextual information to it (for instance, which
    parameter it is).

    .. versionadded:: 2.0

    :param param: the parameter object that caused this error.  This can
                  be left out, and Click will attach this info itself
                  if possible.
    :param param_hint: a string that shows up as parameter name.  This
                       can be used as alternative to `param` in cases
                       where custom validation should happen.  If it is
                       a string it's used as such, if it's a list then
                       each item is quoted and separated.
    """

    def __init__(self, message, ctx=None, param=None,
                 param_hint=None):
        UsageError.__init__(self, message, ctx)
        self.param = param
        self.param_hint = param_hint

    def format_message(self):
        if self.param_hint is not None:
            param_hint = self.param_hint
        elif self.param is not None:
            param_hint = self.param.opts or [self.param.human_readable_name]
        else:
            return 'Invalid value: %s' % self.message
        if isinstance(param_hint, (tuple, list)):
            param_hint = ' / '.join('"%s"' % x for x in param_hint)
        return 'Invalid value for %s: %s' % (param_hint, self.message)


class MissingParameter(BadParameter):
    """Raised if click required an option or argument but it was not
    provided when invoking the script.

    .. versionadded:: 4.0

    :param param_type: a string that indicates the type of the parameter.
                       The default is to inherit the parameter type from
                       the given `param`.  Valid values are ``'parameter'``,
                       ``'option'`` or ``'argument'``.
    """

    def __init__(self, message=None, ctx=None, param=None,
                 param_hint=None, param_type=None):
        BadParameter.__init__(self, message, ctx, param, param_hint)
        self.param_type = param_type

    def format_message(self):
        if self.param_hint is not None:
            param_hint = self.param_hint
        elif self.param is not None:
            param_hint = self.param.opts or [self.param.human_readable_name]
        else:
            param_hint = None
        if isinstance(param_hint, (tuple, list)):
            param_hint = ' / '.join('"%s"' % x for x in param_hint)

        param_type = self.param_type
        if param_type is None and self.param is not None:
            param_type = self.param.param_type_name

        msg = self.message
        if self.param is not None:
            msg_extra = self.param.type.get_missing_message(self.param)
            if msg_extra:
                if msg:
                    msg += '.  ' + msg_extra
                else:
                    msg = msg_extra

        return 'Missing %s%s%s%s' % (
            param_type,
            param_hint and ' %s' % param_hint or '',
            msg and '.  ' or '.',
            msg or '',
        )


class NoSuchOption(UsageError):
    """Raised if click attempted to handle an option that does not
    exist.

    .. versionadded:: 4.0
    """

    def __init__(self, option_name, message=None, possibilities=None,
                 ctx=None):
        if message is None:
            message = 'no such option: %s' % option_name
        UsageError.__init__(self, message, ctx)
        self.option_name = option_name
        self.possibilities = possibilities

    def format_message(self):
        bits = [self.message]
        if self.possibilities:
            if len(self.possibilities) == 1:
                bits.append('Did you mean %s?' % self.possibilities[0])
            else:
                possibilities = sorted(self.possibilities)
                bits.append('(Possible options: %s)' % ', '.join(possibilities))
        return '  '.join(bits)


class BadOptionUsage(UsageError):
    """Raised if an option is generally supplied but the use of the option
    was incorrect.  This is for instance raised if the number of arguments
    for an option is not correct.

    .. versionadded:: 4.0
    """

    def __init__(self, message, ctx=None):
        UsageError.__init__(self, message, ctx)


class BadArgumentUsage(UsageError):
    """Raised if an argument is generally supplied but the use of the argument
    was incorrect.  This is for instance raised if the number of values
    for an argument is not correct.

    .. versionadded:: 6.0
    """

    def __init__(self, message, ctx=None):
        UsageError.__init__(self, message, ctx)


class FileError(ClickException):
    """Raised if a file cannot be opened."""

    def __init__(self, filename, hint=None):
        ui_filename = filename_to_ui(filename)
        if hint is None:
            hint = 'unknown error'
        ClickException.__init__(self, hint)
        self.ui_filename = ui_filename
        self.filename = filename

    def format_message(self):
        return 'Could not open file %s: %s' % (self.ui_filename, self.message)


class Abort(RuntimeError):
    """An internal signalling exception that signals Click to abort."""
