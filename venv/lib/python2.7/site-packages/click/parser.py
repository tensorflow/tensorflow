# -*- coding: utf-8 -*-
"""
    click.parser
    ~~~~~~~~~~~~

    This module started out as largely a copy paste from the stdlib's
    optparse module with the features removed that we do not need from
    optparse because we implement them in Click on a higher level (for
    instance type handling, help formatting and a lot more).

    The plan is to remove more and more from here over time.

    The reason this is a different module and not optparse from the stdlib
    is that there are differences in 2.x and 3.x about the error messages
    generated and optparse in the stdlib uses gettext for no good reason
    and might cause us issues.
"""
import re
from collections import deque
from .exceptions import UsageError, NoSuchOption, BadOptionUsage, \
     BadArgumentUsage


def _unpack_args(args, nargs_spec):
    """Given an iterable of arguments and an iterable of nargs specifications,
    it returns a tuple with all the unpacked arguments at the first index
    and all remaining arguments as the second.

    The nargs specification is the number of arguments that should be consumed
    or `-1` to indicate that this position should eat up all the remainders.

    Missing items are filled with `None`.
    """
    args = deque(args)
    nargs_spec = deque(nargs_spec)
    rv = []
    spos = None

    def _fetch(c):
        try:
            if spos is None:
                return c.popleft()
            else:
                return c.pop()
        except IndexError:
            return None

    while nargs_spec:
        nargs = _fetch(nargs_spec)
        if nargs == 1:
            rv.append(_fetch(args))
        elif nargs > 1:
            x = [_fetch(args) for _ in range(nargs)]
            # If we're reversed, we're pulling in the arguments in reverse,
            # so we need to turn them around.
            if spos is not None:
                x.reverse()
            rv.append(tuple(x))
        elif nargs < 0:
            if spos is not None:
                raise TypeError('Cannot have two nargs < 0')
            spos = len(rv)
            rv.append(None)

    # spos is the position of the wildcard (star).  If it's not `None`,
    # we fill it with the remainder.
    if spos is not None:
        rv[spos] = tuple(args)
        args = []
        rv[spos + 1:] = reversed(rv[spos + 1:])

    return tuple(rv), list(args)


def _error_opt_args(nargs, opt):
    if nargs == 1:
        raise BadOptionUsage('%s option requires an argument' % opt)
    raise BadOptionUsage('%s option requires %d arguments' % (opt, nargs))


def split_opt(opt):
    first = opt[:1]
    if first.isalnum():
        return '', opt
    if opt[1:2] == first:
        return opt[:2], opt[2:]
    return first, opt[1:]


def normalize_opt(opt, ctx):
    if ctx is None or ctx.token_normalize_func is None:
        return opt
    prefix, opt = split_opt(opt)
    return prefix + ctx.token_normalize_func(opt)


def split_arg_string(string):
    """Given an argument string this attempts to split it into small parts."""
    rv = []
    for match in re.finditer(r"('([^'\\]*(?:\\.[^'\\]*)*)'"
                             r'|"([^"\\]*(?:\\.[^"\\]*)*)"'
                             r'|\S+)\s*', string, re.S):
        arg = match.group().strip()
        if arg[:1] == arg[-1:] and arg[:1] in '"\'':
            arg = arg[1:-1].encode('ascii', 'backslashreplace') \
                .decode('unicode-escape')
        try:
            arg = type(string)(arg)
        except UnicodeError:
            pass
        rv.append(arg)
    return rv


class Option(object):

    def __init__(self, opts, dest, action=None, nargs=1, const=None, obj=None):
        self._short_opts = []
        self._long_opts = []
        self.prefixes = set()

        for opt in opts:
            prefix, value = split_opt(opt)
            if not prefix:
                raise ValueError('Invalid start character for option (%s)'
                                 % opt)
            self.prefixes.add(prefix[0])
            if len(prefix) == 1 and len(value) == 1:
                self._short_opts.append(opt)
            else:
                self._long_opts.append(opt)
                self.prefixes.add(prefix)

        if action is None:
            action = 'store'

        self.dest = dest
        self.action = action
        self.nargs = nargs
        self.const = const
        self.obj = obj

    @property
    def takes_value(self):
        return self.action in ('store', 'append')

    def process(self, value, state):
        if self.action == 'store':
            state.opts[self.dest] = value
        elif self.action == 'store_const':
            state.opts[self.dest] = self.const
        elif self.action == 'append':
            state.opts.setdefault(self.dest, []).append(value)
        elif self.action == 'append_const':
            state.opts.setdefault(self.dest, []).append(self.const)
        elif self.action == 'count':
            state.opts[self.dest] = state.opts.get(self.dest, 0) + 1
        else:
            raise ValueError('unknown action %r' % self.action)
        state.order.append(self.obj)


class Argument(object):

    def __init__(self, dest, nargs=1, obj=None):
        self.dest = dest
        self.nargs = nargs
        self.obj = obj

    def process(self, value, state):
        if self.nargs > 1:
            holes = sum(1 for x in value if x is None)
            if holes == len(value):
                value = None
            elif holes != 0:
                raise BadArgumentUsage('argument %s takes %d values'
                                       % (self.dest, self.nargs))
        state.opts[self.dest] = value
        state.order.append(self.obj)


class ParsingState(object):

    def __init__(self, rargs):
        self.opts = {}
        self.largs = []
        self.rargs = rargs
        self.order = []


class OptionParser(object):
    """The option parser is an internal class that is ultimately used to
    parse options and arguments.  It's modelled after optparse and brings
    a similar but vastly simplified API.  It should generally not be used
    directly as the high level Click classes wrap it for you.

    It's not nearly as extensible as optparse or argparse as it does not
    implement features that are implemented on a higher level (such as
    types or defaults).

    :param ctx: optionally the :class:`~click.Context` where this parser
                should go with.
    """

    def __init__(self, ctx=None):
        #: The :class:`~click.Context` for this parser.  This might be
        #: `None` for some advanced use cases.
        self.ctx = ctx
        #: This controls how the parser deals with interspersed arguments.
        #: If this is set to `False`, the parser will stop on the first
        #: non-option.  Click uses this to implement nested subcommands
        #: safely.
        self.allow_interspersed_args = True
        #: This tells the parser how to deal with unknown options.  By
        #: default it will error out (which is sensible), but there is a
        #: second mode where it will ignore it and continue processing
        #: after shifting all the unknown options into the resulting args.
        self.ignore_unknown_options = False
        if ctx is not None:
            self.allow_interspersed_args = ctx.allow_interspersed_args
            self.ignore_unknown_options = ctx.ignore_unknown_options
        self._short_opt = {}
        self._long_opt = {}
        self._opt_prefixes = set(['-', '--'])
        self._args = []

    def add_option(self, opts, dest, action=None, nargs=1, const=None,
                   obj=None):
        """Adds a new option named `dest` to the parser.  The destination
        is not inferred (unlike with optparse) and needs to be explicitly
        provided.  Action can be any of ``store``, ``store_const``,
        ``append``, ``appnd_const`` or ``count``.

        The `obj` can be used to identify the option in the order list
        that is returned from the parser.
        """
        if obj is None:
            obj = dest
        opts = [normalize_opt(opt, self.ctx) for opt in opts]
        option = Option(opts, dest, action=action, nargs=nargs,
                        const=const, obj=obj)
        self._opt_prefixes.update(option.prefixes)
        for opt in option._short_opts:
            self._short_opt[opt] = option
        for opt in option._long_opts:
            self._long_opt[opt] = option

    def add_argument(self, dest, nargs=1, obj=None):
        """Adds a positional argument named `dest` to the parser.

        The `obj` can be used to identify the option in the order list
        that is returned from the parser.
        """
        if obj is None:
            obj = dest
        self._args.append(Argument(dest=dest, nargs=nargs, obj=obj))

    def parse_args(self, args):
        """Parses positional arguments and returns ``(values, args, order)``
        for the parsed options and arguments as well as the leftover
        arguments if there are any.  The order is a list of objects as they
        appear on the command line.  If arguments appear multiple times they
        will be memorized multiple times as well.
        """
        state = ParsingState(args)
        try:
            self._process_args_for_options(state)
            self._process_args_for_args(state)
        except UsageError:
            if self.ctx is None or not self.ctx.resilient_parsing:
                raise
        return state.opts, state.largs, state.order

    def _process_args_for_args(self, state):
        pargs, args = _unpack_args(state.largs + state.rargs,
                                   [x.nargs for x in self._args])

        for idx, arg in enumerate(self._args):
            arg.process(pargs[idx], state)

        state.largs = args
        state.rargs = []

    def _process_args_for_options(self, state):
        while state.rargs:
            arg = state.rargs.pop(0)
            arglen = len(arg)
            # Double dashes always handled explicitly regardless of what
            # prefixes are valid.
            if arg == '--':
                return
            elif arg[:1] in self._opt_prefixes and arglen > 1:
                self._process_opts(arg, state)
            elif self.allow_interspersed_args:
                state.largs.append(arg)
            else:
                state.rargs.insert(0, arg)
                return

        # Say this is the original argument list:
        # [arg0, arg1, ..., arg(i-1), arg(i), arg(i+1), ..., arg(N-1)]
        #                            ^
        # (we are about to process arg(i)).
        #
        # Then rargs is [arg(i), ..., arg(N-1)] and largs is a *subset* of
        # [arg0, ..., arg(i-1)] (any options and their arguments will have
        # been removed from largs).
        #
        # The while loop will usually consume 1 or more arguments per pass.
        # If it consumes 1 (eg. arg is an option that takes no arguments),
        # then after _process_arg() is done the situation is:
        #
        #   largs = subset of [arg0, ..., arg(i)]
        #   rargs = [arg(i+1), ..., arg(N-1)]
        #
        # If allow_interspersed_args is false, largs will always be
        # *empty* -- still a subset of [arg0, ..., arg(i-1)], but
        # not a very interesting subset!

    def _match_long_opt(self, opt, explicit_value, state):
        if opt not in self._long_opt:
            possibilities = [word for word in self._long_opt
                             if word.startswith(opt)]
            raise NoSuchOption(opt, possibilities=possibilities)

        option = self._long_opt[opt]
        if option.takes_value:
            # At this point it's safe to modify rargs by injecting the
            # explicit value, because no exception is raised in this
            # branch.  This means that the inserted value will be fully
            # consumed.
            if explicit_value is not None:
                state.rargs.insert(0, explicit_value)

            nargs = option.nargs
            if len(state.rargs) < nargs:
                _error_opt_args(nargs, opt)
            elif nargs == 1:
                value = state.rargs.pop(0)
            else:
                value = tuple(state.rargs[:nargs])
                del state.rargs[:nargs]

        elif explicit_value is not None:
            raise BadOptionUsage('%s option does not take a value' % opt)

        else:
            value = None

        option.process(value, state)

    def _match_short_opt(self, arg, state):
        stop = False
        i = 1
        prefix = arg[0]
        unknown_options = []

        for ch in arg[1:]:
            opt = normalize_opt(prefix + ch, self.ctx)
            option = self._short_opt.get(opt)
            i += 1

            if not option:
                if self.ignore_unknown_options:
                    unknown_options.append(ch)
                    continue
                raise NoSuchOption(opt)
            if option.takes_value:
                # Any characters left in arg?  Pretend they're the
                # next arg, and stop consuming characters of arg.
                if i < len(arg):
                    state.rargs.insert(0, arg[i:])
                    stop = True

                nargs = option.nargs
                if len(state.rargs) < nargs:
                    _error_opt_args(nargs, opt)
                elif nargs == 1:
                    value = state.rargs.pop(0)
                else:
                    value = tuple(state.rargs[:nargs])
                    del state.rargs[:nargs]

            else:
                value = None

            option.process(value, state)

            if stop:
                break

        # If we got any unknown options we re-combinate the string of the
        # remaining options and re-attach the prefix, then report that
        # to the state as new larg.  This way there is basic combinatorics
        # that can be achieved while still ignoring unknown arguments.
        if self.ignore_unknown_options and unknown_options:
            state.largs.append(prefix + ''.join(unknown_options))

    def _process_opts(self, arg, state):
        explicit_value = None
        # Long option handling happens in two parts.  The first part is
        # supporting explicitly attached values.  In any case, we will try
        # to long match the option first.
        if '=' in arg:
            long_opt, explicit_value = arg.split('=', 1)
        else:
            long_opt = arg
        norm_long_opt = normalize_opt(long_opt, self.ctx)

        # At this point we will match the (assumed) long option through
        # the long option matching code.  Note that this allows options
        # like "-foo" to be matched as long options.
        try:
            self._match_long_opt(norm_long_opt, explicit_value, state)
        except NoSuchOption:
            # At this point the long option matching failed, and we need
            # to try with short options.  However there is a special rule
            # which says, that if we have a two character options prefix
            # (applies to "--foo" for instance), we do not dispatch to the
            # short option code and will instead raise the no option
            # error.
            if arg[:2] not in self._opt_prefixes:
                return self._match_short_opt(arg, state)
            if not self.ignore_unknown_options:
                raise
            state.largs.append(arg)
