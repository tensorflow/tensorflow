"""distutils.util

Miscellaneous utility functions -- anything that doesn't fit into
one of the other *util.py modules.
"""

import importlib.util
import os
import re
import string
import subprocess
import sys
import sysconfig
import functools

from distutils.errors import DistutilsPlatformError, DistutilsByteCompileError
from distutils.dep_util import newer
from distutils.spawn import spawn
from distutils import log


def get_host_platform():
    """
    Return a string that identifies the current platform. Use this
    function to distinguish platform-specific build directories and
    platform-specific built distributions.
    """

    # This function initially exposed platforms as defined in Python 3.9
    # even with older Python versions when distutils was split out.
    # Now it delegates to stdlib sysconfig, but maintains compatibility.

    if sys.version_info < (3, 8):
        if os.name == 'nt':
            if '(arm)' in sys.version.lower():
                return 'win-arm32'
            if '(arm64)' in sys.version.lower():
                return 'win-arm64'

    if sys.version_info < (3, 9):
        if os.name == "posix" and hasattr(os, 'uname'):
            osname, host, release, version, machine = os.uname()
            if osname[:3] == "aix":
                from .py38compat import aix_platform

                return aix_platform(osname, version, release)

    return sysconfig.get_platform()


def get_platform():
    if os.name == 'nt':
        TARGET_TO_PLAT = {
            'x86': 'win32',
            'x64': 'win-amd64',
            'arm': 'win-arm32',
            'arm64': 'win-arm64',
        }
        target = os.environ.get('VSCMD_ARG_TGT_ARCH')
        return TARGET_TO_PLAT.get(target) or get_host_platform()
    return get_host_platform()


if sys.platform == 'darwin':
    _syscfg_macosx_ver = None  # cache the version pulled from sysconfig
MACOSX_VERSION_VAR = 'MACOSX_DEPLOYMENT_TARGET'


def _clear_cached_macosx_ver():
    """For testing only. Do not call."""
    global _syscfg_macosx_ver
    _syscfg_macosx_ver = None


def get_macosx_target_ver_from_syscfg():
    """Get the version of macOS latched in the Python interpreter configuration.
    Returns the version as a string or None if can't obtain one. Cached."""
    global _syscfg_macosx_ver
    if _syscfg_macosx_ver is None:
        from distutils import sysconfig

        ver = sysconfig.get_config_var(MACOSX_VERSION_VAR) or ''
        if ver:
            _syscfg_macosx_ver = ver
    return _syscfg_macosx_ver


def get_macosx_target_ver():
    """Return the version of macOS for which we are building.

    The target version defaults to the version in sysconfig latched at time
    the Python interpreter was built, unless overridden by an environment
    variable. If neither source has a value, then None is returned"""

    syscfg_ver = get_macosx_target_ver_from_syscfg()
    env_ver = os.environ.get(MACOSX_VERSION_VAR)

    if env_ver:
        # Validate overridden version against sysconfig version, if have both.
        # Ensure that the deployment target of the build process is not less
        # than 10.3 if the interpreter was built for 10.3 or later.  This
        # ensures extension modules are built with correct compatibility
        # values, specifically LDSHARED which can use
        # '-undefined dynamic_lookup' which only works on >= 10.3.
        if (
            syscfg_ver
            and split_version(syscfg_ver) >= [10, 3]
            and split_version(env_ver) < [10, 3]
        ):
            my_msg = (
                '$' + MACOSX_VERSION_VAR + ' mismatch: '
                'now "%s" but "%s" during configure; '
                'must use 10.3 or later' % (env_ver, syscfg_ver)
            )
            raise DistutilsPlatformError(my_msg)
        return env_ver
    return syscfg_ver


def split_version(s):
    """Convert a dot-separated string into a list of numbers for comparisons"""
    return [int(n) for n in s.split('.')]


def convert_path(pathname):
    """Return 'pathname' as a name that will work on the native filesystem,
    i.e. split it on '/' and put it back together again using the current
    directory separator.  Needed because filenames in the setup script are
    always supplied in Unix style, and have to be converted to the local
    convention before we can actually use them in the filesystem.  Raises
    ValueError on non-Unix-ish systems if 'pathname' either starts or
    ends with a slash.
    """
    if os.sep == '/':
        return pathname
    if not pathname:
        return pathname
    if pathname[0] == '/':
        raise ValueError("path '%s' cannot be absolute" % pathname)
    if pathname[-1] == '/':
        raise ValueError("path '%s' cannot end with '/'" % pathname)

    paths = pathname.split('/')
    while '.' in paths:
        paths.remove('.')
    if not paths:
        return os.curdir
    return os.path.join(*paths)


# convert_path ()


def change_root(new_root, pathname):
    """Return 'pathname' with 'new_root' prepended.  If 'pathname' is
    relative, this is equivalent to "os.path.join(new_root,pathname)".
    Otherwise, it requires making 'pathname' relative and then joining the
    two, which is tricky on DOS/Windows and Mac OS.
    """
    if os.name == 'posix':
        if not os.path.isabs(pathname):
            return os.path.join(new_root, pathname)
        else:
            return os.path.join(new_root, pathname[1:])

    elif os.name == 'nt':
        (drive, path) = os.path.splitdrive(pathname)
        if path[0] == '\\':
            path = path[1:]
        return os.path.join(new_root, path)

    raise DistutilsPlatformError(f"nothing known about platform '{os.name}'")


@functools.lru_cache()
def check_environ():
    """Ensure that 'os.environ' has all the environment variables we
    guarantee that users can use in config files, command-line options,
    etc.  Currently this includes:
      HOME - user's home directory (Unix only)
      PLAT - description of the current platform, including hardware
             and OS (see 'get_platform()')
    """
    if os.name == 'posix' and 'HOME' not in os.environ:
        try:
            import pwd

            os.environ['HOME'] = pwd.getpwuid(os.getuid())[5]
        except (ImportError, KeyError):
            # bpo-10496: if the current user identifier doesn't exist in the
            # password database, do nothing
            pass

    if 'PLAT' not in os.environ:
        os.environ['PLAT'] = get_platform()


def subst_vars(s, local_vars):
    """
    Perform variable substitution on 'string'.
    Variables are indicated by format-style braces ("{var}").
    Variable is substituted by the value found in the 'local_vars'
    dictionary or in 'os.environ' if it's not in 'local_vars'.
    'os.environ' is first checked/augmented to guarantee that it contains
    certain values: see 'check_environ()'.  Raise ValueError for any
    variables not found in either 'local_vars' or 'os.environ'.
    """
    check_environ()
    lookup = dict(os.environ)
    lookup.update((name, str(value)) for name, value in local_vars.items())
    try:
        return _subst_compat(s).format_map(lookup)
    except KeyError as var:
        raise ValueError(f"invalid variable {var}")


def _subst_compat(s):
    """
    Replace shell/Perl-style variable substitution with
    format-style. For compatibility.
    """

    def _subst(match):
        return f'{{{match.group(1)}}}'

    repl = re.sub(r'\$([a-zA-Z_][a-zA-Z_0-9]*)', _subst, s)
    if repl != s:
        import warnings

        warnings.warn(
            "shell/Perl-style substitions are deprecated",
            DeprecationWarning,
        )
    return repl


def grok_environment_error(exc, prefix="error: "):
    # Function kept for backward compatibility.
    # Used to try clever things with EnvironmentErrors,
    # but nowadays str(exception) produces good messages.
    return prefix + str(exc)


# Needed by 'split_quoted()'
_wordchars_re = _squote_re = _dquote_re = None


def _init_regex():
    global _wordchars_re, _squote_re, _dquote_re
    _wordchars_re = re.compile(r'[^\\\'\"%s ]*' % string.whitespace)
    _squote_re = re.compile(r"'(?:[^'\\]|\\.)*'")
    _dquote_re = re.compile(r'"(?:[^"\\]|\\.)*"')


def split_quoted(s):
    """Split a string up according to Unix shell-like rules for quotes and
    backslashes.  In short: words are delimited by spaces, as long as those
    spaces are not escaped by a backslash, or inside a quoted string.
    Single and double quotes are equivalent, and the quote characters can
    be backslash-escaped.  The backslash is stripped from any two-character
    escape sequence, leaving only the escaped character.  The quote
    characters are stripped from any quoted string.  Returns a list of
    words.
    """

    # This is a nice algorithm for splitting up a single string, since it
    # doesn't require character-by-character examination.  It was a little
    # bit of a brain-bender to get it working right, though...
    if _wordchars_re is None:
        _init_regex()

    s = s.strip()
    words = []
    pos = 0

    while s:
        m = _wordchars_re.match(s, pos)
        end = m.end()
        if end == len(s):
            words.append(s[:end])
            break

        if s[end] in string.whitespace:
            # unescaped, unquoted whitespace: now
            # we definitely have a word delimiter
            words.append(s[:end])
            s = s[end:].lstrip()
            pos = 0

        elif s[end] == '\\':
            # preserve whatever is being escaped;
            # will become part of the current word
            s = s[:end] + s[end + 1 :]
            pos = end + 1

        else:
            if s[end] == "'":  # slurp singly-quoted string
                m = _squote_re.match(s, end)
            elif s[end] == '"':  # slurp doubly-quoted string
                m = _dquote_re.match(s, end)
            else:
                raise RuntimeError("this can't happen (bad char '%c')" % s[end])

            if m is None:
                raise ValueError("bad string (mismatched %s quotes?)" % s[end])

            (beg, end) = m.span()
            s = s[:beg] + s[beg + 1 : end - 1] + s[end:]
            pos = m.end() - 2

        if pos >= len(s):
            words.append(s)
            break

    return words


# split_quoted ()


def execute(func, args, msg=None, verbose=0, dry_run=0):
    """Perform some action that affects the outside world (eg.  by
    writing to the filesystem).  Such actions are special because they
    are disabled by the 'dry_run' flag.  This method takes care of all
    that bureaucracy for you; all you have to do is supply the
    function to call and an argument tuple for it (to embody the
    "external action" being performed), and an optional message to
    print.
    """
    if msg is None:
        msg = "{}{!r}".format(func.__name__, args)
        if msg[-2:] == ',)':  # correct for singleton tuple
            msg = msg[0:-2] + ')'

    log.info(msg)
    if not dry_run:
        func(*args)


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value {!r}".format(val))


def byte_compile(  # noqa: C901
    py_files,
    optimize=0,
    force=0,
    prefix=None,
    base_dir=None,
    verbose=1,
    dry_run=0,
    direct=None,
):
    """Byte-compile a collection of Python source files to .pyc
    files in a __pycache__ subdirectory.  'py_files' is a list
    of files to compile; any files that don't end in ".py" are silently
    skipped.  'optimize' must be one of the following:
      0 - don't optimize
      1 - normal optimization (like "python -O")
      2 - extra optimization (like "python -OO")
    If 'force' is true, all files are recompiled regardless of
    timestamps.

    The source filename encoded in each bytecode file defaults to the
    filenames listed in 'py_files'; you can modify these with 'prefix' and
    'basedir'.  'prefix' is a string that will be stripped off of each
    source filename, and 'base_dir' is a directory name that will be
    prepended (after 'prefix' is stripped).  You can supply either or both
    (or neither) of 'prefix' and 'base_dir', as you wish.

    If 'dry_run' is true, doesn't actually do anything that would
    affect the filesystem.

    Byte-compilation is either done directly in this interpreter process
    with the standard py_compile module, or indirectly by writing a
    temporary script and executing it.  Normally, you should let
    'byte_compile()' figure out to use direct compilation or not (see
    the source for details).  The 'direct' flag is used by the script
    generated in indirect mode; unless you know what you're doing, leave
    it set to None.
    """

    # nothing is done if sys.dont_write_bytecode is True
    if sys.dont_write_bytecode:
        raise DistutilsByteCompileError('byte-compiling is disabled.')

    # First, if the caller didn't force us into direct or indirect mode,
    # figure out which mode we should be in.  We take a conservative
    # approach: choose direct mode *only* if the current interpreter is
    # in debug mode and optimize is 0.  If we're not in debug mode (-O
    # or -OO), we don't know which level of optimization this
    # interpreter is running with, so we can't do direct
    # byte-compilation and be certain that it's the right thing.  Thus,
    # always compile indirectly if the current interpreter is in either
    # optimize mode, or if either optimization level was requested by
    # the caller.
    if direct is None:
        direct = __debug__ and optimize == 0

    # "Indirect" byte-compilation: write a temporary script and then
    # run it with the appropriate flags.
    if not direct:
        try:
            from tempfile import mkstemp

            (script_fd, script_name) = mkstemp(".py")
        except ImportError:
            from tempfile import mktemp

            (script_fd, script_name) = None, mktemp(".py")
        log.info("writing byte-compilation script '%s'", script_name)
        if not dry_run:
            if script_fd is not None:
                script = os.fdopen(script_fd, "w")
            else:
                script = open(script_name, "w")

            with script:
                script.write(
                    """\
from distutils.util import byte_compile
files = [
"""
                )

                # XXX would be nice to write absolute filenames, just for
                # safety's sake (script should be more robust in the face of
                # chdir'ing before running it).  But this requires abspath'ing
                # 'prefix' as well, and that breaks the hack in build_lib's
                # 'byte_compile()' method that carefully tacks on a trailing
                # slash (os.sep really) to make sure the prefix here is "just
                # right".  This whole prefix business is rather delicate -- the
                # problem is that it's really a directory, but I'm treating it
                # as a dumb string, so trailing slashes and so forth matter.

                script.write(",\n".join(map(repr, py_files)) + "]\n")
                script.write(
                    """
byte_compile(files, optimize=%r, force=%r,
             prefix=%r, base_dir=%r,
             verbose=%r, dry_run=0,
             direct=1)
"""
                    % (optimize, force, prefix, base_dir, verbose)
                )

        cmd = [sys.executable]
        cmd.extend(subprocess._optim_args_from_interpreter_flags())
        cmd.append(script_name)
        spawn(cmd, dry_run=dry_run)
        execute(os.remove, (script_name,), "removing %s" % script_name, dry_run=dry_run)

    # "Direct" byte-compilation: use the py_compile module to compile
    # right here, right now.  Note that the script generated in indirect
    # mode simply calls 'byte_compile()' in direct mode, a weird sort of
    # cross-process recursion.  Hey, it works!
    else:
        from py_compile import compile

        for file in py_files:
            if file[-3:] != ".py":
                # This lets us be lazy and not filter filenames in
                # the "install_lib" command.
                continue

            # Terminology from the py_compile module:
            #   cfile - byte-compiled file
            #   dfile - purported source filename (same as 'file' by default)
            if optimize >= 0:
                opt = '' if optimize == 0 else optimize
                cfile = importlib.util.cache_from_source(file, optimization=opt)
            else:
                cfile = importlib.util.cache_from_source(file)
            dfile = file
            if prefix:
                if file[: len(prefix)] != prefix:
                    raise ValueError(
                        "invalid prefix: filename %r doesn't start with %r"
                        % (file, prefix)
                    )
                dfile = dfile[len(prefix) :]
            if base_dir:
                dfile = os.path.join(base_dir, dfile)

            cfile_base = os.path.basename(cfile)
            if direct:
                if force or newer(file, cfile):
                    log.info("byte-compiling %s to %s", file, cfile_base)
                    if not dry_run:
                        compile(file, cfile, dfile)
                else:
                    log.debug("skipping byte-compilation of %s to %s", file, cfile_base)


def rfc822_escape(header):
    """Return a version of the string escaped for inclusion in an
    RFC-822 header, by ensuring there are 8 spaces space after each newline.
    """
    lines = header.split('\n')
    sep = '\n' + 8 * ' '
    return sep.join(lines)
