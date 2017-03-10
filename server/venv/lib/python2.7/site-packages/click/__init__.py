# -*- coding: utf-8 -*-
"""
    click
    ~~~~~

    Click is a simple Python module that wraps the stdlib's optparse to make
    writing command line scripts fun.  Unlike other modules, it's based around
    a simple API that does not come with too much magic and is composable.

    In case optparse ever gets removed from the stdlib, it will be shipped by
    this module.

    :copyright: (c) 2014 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""

# Core classes
from .core import Context, BaseCommand, Command, MultiCommand, Group, \
     CommandCollection, Parameter, Option, Argument

# Globals
from .globals import get_current_context

# Decorators
from .decorators import pass_context, pass_obj, make_pass_decorator, \
     command, group, argument, option, confirmation_option, \
     password_option, version_option, help_option

# Types
from .types import ParamType, File, Path, Choice, IntRange, Tuple, \
     STRING, INT, FLOAT, BOOL, UUID, UNPROCESSED

# Utilities
from .utils import echo, get_binary_stream, get_text_stream, open_file, \
     format_filename, get_app_dir, get_os_args

# Terminal functions
from .termui import prompt, confirm, get_terminal_size, echo_via_pager, \
     progressbar, clear, style, unstyle, secho, edit, launch, getchar, \
     pause

# Exceptions
from .exceptions import ClickException, UsageError, BadParameter, \
     FileError, Abort, NoSuchOption, BadOptionUsage, BadArgumentUsage, \
     MissingParameter

# Formatting
from .formatting import HelpFormatter, wrap_text

# Parsing
from .parser import OptionParser


__all__ = [
    # Core classes
    'Context', 'BaseCommand', 'Command', 'MultiCommand', 'Group',
    'CommandCollection', 'Parameter', 'Option', 'Argument',

    # Globals
    'get_current_context',

    # Decorators
    'pass_context', 'pass_obj', 'make_pass_decorator', 'command', 'group',
    'argument', 'option', 'confirmation_option', 'password_option',
    'version_option', 'help_option',

    # Types
    'ParamType', 'File', 'Path', 'Choice', 'IntRange', 'Tuple', 'STRING',
    'INT', 'FLOAT', 'BOOL', 'UUID', 'UNPROCESSED',

    # Utilities
    'echo', 'get_binary_stream', 'get_text_stream', 'open_file',
    'format_filename', 'get_app_dir', 'get_os_args',

    # Terminal functions
    'prompt', 'confirm', 'get_terminal_size', 'echo_via_pager',
    'progressbar', 'clear', 'style', 'unstyle', 'secho', 'edit', 'launch',
    'getchar', 'pause',

    # Exceptions
    'ClickException', 'UsageError', 'BadParameter', 'FileError',
    'Abort', 'NoSuchOption', 'BadOptionUsage', 'BadArgumentUsage',
    'MissingParameter',

    # Formatting
    'HelpFormatter', 'wrap_text',

    # Parsing
    'OptionParser',
]


# Controls if click should emit the warning about the use of unicode
# literals.
disable_unicode_literals_warning = False


__version__ = '6.7'
