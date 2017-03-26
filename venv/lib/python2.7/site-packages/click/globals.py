from threading import local


_local = local()


def get_current_context(silent=False):
    """Returns the current click context.  This can be used as a way to
    access the current context object from anywhere.  This is a more implicit
    alternative to the :func:`pass_context` decorator.  This function is
    primarily useful for helpers such as :func:`echo` which might be
    interested in changing it's behavior based on the current context.

    To push the current context, :meth:`Context.scope` can be used.

    .. versionadded:: 5.0

    :param silent: is set to `True` the return value is `None` if no context
                   is available.  The default behavior is to raise a
                   :exc:`RuntimeError`.
    """
    try:
        return getattr(_local, 'stack')[-1]
    except (AttributeError, IndexError):
        if not silent:
            raise RuntimeError('There is no active click context.')


def push_context(ctx):
    """Pushes a new context to the current stack."""
    _local.__dict__.setdefault('stack', []).append(ctx)


def pop_context():
    """Removes the top level from the stack."""
    _local.stack.pop()


def resolve_color_default(color=None):
    """"Internal helper to get the default value of the color flag.  If a
    value is passed it's returned unchanged, otherwise it's looked up from
    the current context.
    """
    if color is not None:
        return color
    ctx = get_current_context(silent=True)
    if ctx is not None:
        return ctx.color
