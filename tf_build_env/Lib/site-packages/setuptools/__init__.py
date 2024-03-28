"""Extensions to the 'distutils' for large or complex distributions"""

import functools
import os
import re
import warnings

import _distutils_hack.override  # noqa: F401

import distutils.core
from distutils.errors import DistutilsOptionError
from distutils.util import convert_path as _convert_path

from ._deprecation_warning import SetuptoolsDeprecationWarning

import setuptools.version
from setuptools.extension import Extension
from setuptools.dist import Distribution
from setuptools.depends import Require
from setuptools.discovery import PackageFinder, PEP420PackageFinder
from . import monkey
from . import logging


__all__ = [
    'setup',
    'Distribution',
    'Command',
    'Extension',
    'Require',
    'SetuptoolsDeprecationWarning',
    'find_packages',
    'find_namespace_packages',
]

__version__ = setuptools.version.__version__

bootstrap_install_from = None


find_packages = PackageFinder.find
find_namespace_packages = PEP420PackageFinder.find


def _install_setup_requires(attrs):
    # Note: do not use `setuptools.Distribution` directly, as
    # our PEP 517 backend patch `distutils.core.Distribution`.
    class MinimalDistribution(distutils.core.Distribution):
        """
        A minimal version of a distribution for supporting the
        fetch_build_eggs interface.
        """

        def __init__(self, attrs):
            _incl = 'dependency_links', 'setup_requires'
            filtered = {k: attrs[k] for k in set(_incl) & set(attrs)}
            super().__init__(filtered)
            # Prevent accidentally triggering discovery with incomplete set of attrs
            self.set_defaults._disable()

        def _get_project_config_files(self, filenames=None):
            """Ignore ``pyproject.toml``, they are not related to setup_requires"""
            try:
                cfg, toml = super()._split_standard_project_metadata(filenames)
                return cfg, ()
            except Exception:
                return filenames, ()

        def finalize_options(self):
            """
            Disable finalize_options to avoid building the working set.
            Ref #2158.
            """

    dist = MinimalDistribution(attrs)

    # Honor setup.cfg's options.
    dist.parse_config_files(ignore_option_errors=True)
    if dist.setup_requires:
        dist.fetch_build_eggs(dist.setup_requires)


def setup(**attrs):
    # Make sure we have any requirements needed to interpret 'attrs'.
    logging.configure()
    _install_setup_requires(attrs)
    return distutils.core.setup(**attrs)


setup.__doc__ = distutils.core.setup.__doc__


_Command = monkey.get_unpatched(distutils.core.Command)


class Command(_Command):
    """
    Setuptools internal actions are organized using a *command design pattern*.
    This means that each action (or group of closely related actions) executed during
    the build should be implemented as a ``Command`` subclass.

    These commands are abstractions and do not necessarily correspond to a command that
    can (or should) be executed via a terminal, in a CLI fashion (although historically
    they would).

    When creating a new command from scratch, custom defined classes **SHOULD** inherit
    from ``setuptools.Command`` and implement a few mandatory methods.
    Between these mandatory methods, are listed:

    .. method:: initialize_options(self)

        Set or (reset) all options/attributes/caches used by the command
        to their default values. Note that these values may be overwritten during
        the build.

    .. method:: finalize_options(self)

        Set final values for all options/attributes used by the command.
        Most of the time, each option/attribute/cache should only be set if it does not
        have any value yet (e.g. ``if self.attr is None: self.attr = val``).

    .. method:: run(self)

        Execute the actions intended by the command.
        (Side effects **SHOULD** only take place when ``run`` is executed,
        for example, creating new files or writing to the terminal output).

    A useful analogy for command classes is to think of them as subroutines with local
    variables called "options".  The options are "declared" in ``initialize_options()``
    and "defined" (given their final values, aka "finalized") in ``finalize_options()``,
    both of which must be defined by every command class. The "body" of the subroutine,
    (where it does all the work) is the ``run()`` method.
    Between ``initialize_options()`` and ``finalize_options()``, ``setuptools`` may set
    the values for options/attributes based on user's input (or circumstance),
    which means that the implementation should be careful to not overwrite values in
    ``finalize_options`` unless necessary.

    Please note that other commands (or other parts of setuptools) may also overwrite
    the values of the command's options/attributes multiple times during the build
    process.
    Therefore it is important to consistently implement ``initialize_options()`` and
    ``finalize_options()``. For example, all derived attributes (or attributes that
    depend on the value of other attributes) **SHOULD** be recomputed in
    ``finalize_options``.

    When overwriting existing commands, custom defined classes **MUST** abide by the
    same APIs implemented by the original class. They also **SHOULD** inherit from the
    original class.
    """

    command_consumes_arguments = False

    def __init__(self, dist, **kw):
        """
        Construct the command for dist, updating
        vars(self) with any keyword parameters.
        """
        super().__init__(dist)
        vars(self).update(kw)

    def _ensure_stringlike(self, option, what, default=None):
        val = getattr(self, option)
        if val is None:
            setattr(self, option, default)
            return default
        elif not isinstance(val, str):
            raise DistutilsOptionError(
                "'%s' must be a %s (got `%s`)" % (option, what, val)
            )
        return val

    def ensure_string_list(self, option):
        r"""Ensure that 'option' is a list of strings.  If 'option' is
        currently a string, we split it either on /,\s*/ or /\s+/, so
        "foo bar baz", "foo,bar,baz", and "foo,   bar baz" all become
        ["foo", "bar", "baz"].

        ..
           TODO: This method seems to be similar to the one in ``distutils.cmd``
           Probably it is just here for backward compatibility with old Python versions?

        :meta private:
        """
        val = getattr(self, option)
        if val is None:
            return
        elif isinstance(val, str):
            setattr(self, option, re.split(r',\s*|\s+', val))
        else:
            if isinstance(val, list):
                ok = all(isinstance(v, str) for v in val)
            else:
                ok = False
            if not ok:
                raise DistutilsOptionError(
                    "'%s' must be a list of strings (got %r)" % (option, val)
                )

    def reinitialize_command(self, command, reinit_subcommands=0, **kw):
        cmd = _Command.reinitialize_command(self, command, reinit_subcommands)
        vars(cmd).update(kw)
        return cmd


def _find_all_simple(path):
    """
    Find all files under 'path'
    """
    results = (
        os.path.join(base, file)
        for base, dirs, files in os.walk(path, followlinks=True)
        for file in files
    )
    return filter(os.path.isfile, results)


def findall(dir=os.curdir):
    """
    Find all files under 'dir' and return the list of full filenames.
    Unless dir is '.', return full filenames with dir prepended.
    """
    files = _find_all_simple(dir)
    if dir == os.curdir:
        make_rel = functools.partial(os.path.relpath, start=dir)
        files = map(make_rel, files)
    return list(files)


@functools.wraps(_convert_path)
def convert_path(pathname):
    from inspect import cleandoc

    msg = """
    The function `convert_path` is considered internal and not part of the public API.
    Its direct usage by 3rd-party packages is considered deprecated and the function
    may be removed in the future.
    """
    warnings.warn(cleandoc(msg), SetuptoolsDeprecationWarning)
    return _convert_path(pathname)


class sic(str):
    """Treat this string as-is (https://en.wikipedia.org/wiki/Sic)"""


# Apply monkey patches
monkey.patch_all()
