"""distutils.util

Miscellaneous utility functions -- anything that doesn't fit into
one of the other *util.py modules.
"""

__revision__ = "$Id: util.py 59116 2007-11-22 10:14:26Z ronald.oussoren $"

import sys, os, string, re
from distutils.errors import DistutilsPlatformError
from distutils.dep_util import newer
from distutils.spawn import spawn
from distutils import log

def get_platform ():
    """Return a string that identifies the current platform.  This is used
    mainly to distinguish platform-specific build directories and
    platform-specific built distributions.  Typically includes the OS name
    and version and the architecture (as supplied by 'os.uname()'),
    although the exact information included depends on the OS; eg. for IRIX
    the architecture isn't particularly important (IRIX only runs on SGI
    hardware), but for Linux the kernel version isn't particularly
    important.

    Examples of returned values:
       linux-i586
       linux-alpha (?)
       solaris-2.6-sun4u
       irix-5.3
       irix64-6.2

    For non-POSIX platforms, currently just returns 'sys.platform'.
    """
    if os.name != "posix" or not hasattr(os, 'uname'):
        # XXX what about the architecture? NT is Intel or Alpha,
        # Mac OS is M68k or PPC, etc.
        return sys.platform

    # Try to distinguish various flavours of Unix

    (osname, host, release, version, machine) = os.uname()

    # Convert the OS name to lowercase, remove '/' characters
    # (to accommodate BSD/OS), and translate spaces (for "Power Macintosh")
    osname = string.lower(osname)
    osname = string.replace(osname, '/', '')
    machine = string.replace(machine, ' ', '_')
    machine = string.replace(machine, '/', '-')

    if osname[:5] == "linux":
        # At least on Linux/Intel, 'machine' is the processor --
        # i386, etc.
        # XXX what about Alpha, SPARC, etc?
        return  "%s-%s" % (osname, machine)
    elif osname[:5] == "sunos":
        if release[0] >= "5":           # SunOS 5 == Solaris 2
            osname = "solaris"
            release = "%d.%s" % (int(release[0]) - 3, release[2:])
        # fall through to standard osname-release-machine representation
    elif osname[:4] == "irix":              # could be "irix64"!
        return "%s-%s" % (osname, release)
    elif osname[:3] == "aix":
        return "%s-%s.%s" % (osname, version, release)
    elif osname[:6] == "cygwin":
        osname = "cygwin"
        rel_re = re.compile (r'[\d.]+')
        m = rel_re.match(release)
        if m:
            release = m.group()
    elif osname[:6] == "darwin":
        #
        # For our purposes, we'll assume that the system version from
        # distutils' perspective is what MACOSX_DEPLOYMENT_TARGET is set
        # to. This makes the compatibility story a bit more sane because the
        # machine is going to compile and link as if it were
        # MACOSX_DEPLOYMENT_TARGET.
        from distutils.sysconfig import get_config_vars
        cfgvars = get_config_vars()

        macver = os.environ.get('MACOSX_DEPLOYMENT_TARGET')
        if not macver:
            macver = cfgvars.get('MACOSX_DEPLOYMENT_TARGET')

        if not macver:
            # Get the system version. Reading this plist is a documented
            # way to get the system version (see the documentation for
            # the Gestalt Manager)
            try:
                f = open('/System/Library/CoreServices/SystemVersion.plist')
            except IOError:
                # We're on a plain darwin box, fall back to the default
                # behaviour.
                pass
            else:
                m = re.search(
                        r'<key>ProductUserVisibleVersion</key>\s*' +
                        r'<string>(.*?)</string>', f.read())
                f.close()
                if m is not None:
                    macver = '.'.join(m.group(1).split('.')[:2])
                # else: fall back to the default behaviour

        if macver:
            from distutils.sysconfig import get_config_vars
            release = macver
            osname = "macosx"


            if (release + '.') >= '10.4.' and \
                    get_config_vars().get('UNIVERSALSDK', '').strip():
                # The universal build will build fat binaries, but not on
                # systems before 10.4
                machine = 'fat'

            elif machine in ('PowerPC', 'Power_Macintosh'):
                # Pick a sane name for the PPC architecture.
                machine = 'ppc'

    return "%s-%s-%s" % (osname, release, machine)

# get_platform ()


def convert_path (pathname):
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
        raise ValueError, "path '%s' cannot be absolute" % pathname
    if pathname[-1] == '/':
        raise ValueError, "path '%s' cannot end with '/'" % pathname

    paths = string.split(pathname, '/')
    while '.' in paths:
        paths.remove('.')
    if not paths:
        return os.curdir
    return apply(os.path.join, paths)

# convert_path ()


def change_root (new_root, pathname):
    """Return 'pathname' with 'new_root' prepended.  If 'pathname' is
    relative, this is equivalent to "os.path.join(new_root,pathname)".
    Otherwise, it requires making 'pathname' relative and then joining the
    two, which is tricky on DOS/Windows and Mac OS.
    """
    os_name = os._name if sys.platform.startswith('java') else os.name
    if os_name == 'posix':
        if not os.path.isabs(pathname):
            return os.path.join(new_root, pathname)
        else:
            return os.path.join(new_root, pathname[1:])

    elif os_name == 'nt':
        (drive, path) = os.path.splitdrive(pathname)
        if path[0] == '\\':
            path = path[1:]
        return os.path.join(new_root, path)

    elif os_name == 'os2':
        (drive, path) = os.path.splitdrive(pathname)
        if path[0] == os.sep:
            path = path[1:]
        return os.path.join(new_root, path)

    elif os_name == 'mac':
        if not os.path.isabs(pathname):
            return os.path.join(new_root, pathname)
        else:
            # Chop off volume name from start of path
            elements = string.split(pathname, ":", 1)
            pathname = ":" + elements[1]
            return os.path.join(new_root, pathname)

    else:
        raise DistutilsPlatformError, \
              "nothing known about platform '%s'" % os_name


_environ_checked = 0
def check_environ ():
    """Ensure that 'os.environ' has all the environment variables we
    guarantee that users can use in config files, command-line options,
    etc.  Currently this includes:
      HOME - user's home directory (Unix only)
      PLAT - description of the current platform, including hardware
             and OS (see 'get_platform()')
    """
    global _environ_checked
    if _environ_checked:
        return

    if os.name == 'posix' and not os.environ.has_key('HOME'):
        import pwd
        os.environ['HOME'] = pwd.getpwuid(os.getuid())[5]

    if not os.environ.has_key('PLAT'):
        os.environ['PLAT'] = get_platform()

    _environ_checked = 1


def subst_vars (s, local_vars):
    """Perform shell/Perl-style variable substitution on 'string'.  Every
    occurrence of '$' followed by a name is considered a variable, and
    variable is substituted by the value found in the 'local_vars'
    dictionary, or in 'os.environ' if it's not in 'local_vars'.
    'os.environ' is first checked/augmented to guarantee that it contains
    certain values: see 'check_environ()'.  Raise ValueError for any
    variables not found in either 'local_vars' or 'os.environ'.
    """
    check_environ()
    def _subst (match, local_vars=local_vars):
        var_name = match.group(1)
        if local_vars.has_key(var_name):
            return str(local_vars[var_name])
        else:
            return os.environ[var_name]

    try:
        return re.sub(r'\$([a-zA-Z_][a-zA-Z_0-9]*)', _subst, s)
    except KeyError, var:
        raise ValueError, "invalid variable '$%s'" % var

# subst_vars ()


def grok_environment_error (exc, prefix="error: "):
    """Generate a useful error message from an EnvironmentError (IOError or
    OSError) exception object.  Handles Python 1.5.1 and 1.5.2 styles, and
    does what it can to deal with exception objects that don't have a
    filename (which happens when the error is due to a two-file operation,
    such as 'rename()' or 'link()'.  Returns the error message as a string
    prefixed with 'prefix'.
    """
    # check for Python 1.5.2-style {IO,OS}Error exception objects
    if hasattr(exc, 'filename') and hasattr(exc, 'strerror'):
        if exc.filename:
            error = prefix + "%s: %s" % (exc.filename, exc.strerror)
        else:
            # two-argument functions in posix module don't
            # include the filename in the exception object!
            error = prefix + "%s" % exc.strerror
    else:
        error = prefix + str(exc[-1])

    return error


# Needed by 'split_quoted()'
_wordchars_re = _squote_re = _dquote_re = None
def _init_regex():
    global _wordchars_re, _squote_re, _dquote_re
    _wordchars_re = re.compile(r'[^\\\'\"%s ]*' % string.whitespace)
    _squote_re = re.compile(r"'(?:[^'\\]|\\.)*'")
    _dquote_re = re.compile(r'"(?:[^"\\]|\\.)*"')

def split_quoted (s):
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
    if _wordchars_re is None: _init_regex()

    s = string.strip(s)
    words = []
    pos = 0

    while s:
        m = _wordchars_re.match(s, pos)
        end = m.end()
        if end == len(s):
            words.append(s[:end])
            break

        if s[end] in string.whitespace: # unescaped, unquoted whitespace: now
            words.append(s[:end])       # we definitely have a word delimiter
            s = string.lstrip(s[end:])
            pos = 0

        elif s[end] == '\\':            # preserve whatever is being escaped;
                                        # will become part of the current word
            s = s[:end] + s[end+1:]
            pos = end+1

        else:
            if s[end] == "'":           # slurp singly-quoted string
                m = _squote_re.match(s, end)
            elif s[end] == '"':         # slurp doubly-quoted string
                m = _dquote_re.match(s, end)
            else:
                raise RuntimeError, \
                      "this can't happen (bad char '%c')" % s[end]

            if m is None:
                raise ValueError, \
                      "bad string (mismatched %s quotes?)" % s[end]

            (beg, end) = m.span()
            s = s[:beg] + s[beg+1:end-1] + s[end:]
            pos = m.end() - 2

        if pos >= len(s):
            words.append(s)
            break

    return words

# split_quoted ()


def execute (func, args, msg=None, verbose=0, dry_run=0):
    """Perform some action that affects the outside world (eg.  by
    writing to the filesystem).  Such actions are special because they
    are disabled by the 'dry_run' flag.  This method takes care of all
    that bureaucracy for you; all you have to do is supply the
    function to call and an argument tuple for it (to embody the
    "external action" being performed), and an optional message to
    print.
    """
    if msg is None:
        msg = "%s%r" % (func.__name__, args)
        if msg[-2:] == ',)':        # correct for singleton tuple
            msg = msg[0:-2] + ')'

    log.info(msg)
    if not dry_run:
        apply(func, args)


def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = string.lower(val)
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError, "invalid truth value %r" % (val,)


def byte_compile (py_files,
                  optimize=0, force=0,
                  prefix=None, base_dir=None,
                  verbose=1, dry_run=0,
                  direct=None):
    """Byte-compile a collection of Python source files to either .pyc
    or .pyo files in the same directory.  'py_files' is a list of files
    to compile; any files that don't end in ".py" are silently skipped.
    'optimize' must be one of the following:
      0 - don't optimize (generate .pyc)
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
        direct = (__debug__ and optimize == 0)

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

            script.write("""\
from distutils.util import byte_compile
files = [
""")

            # XXX would be nice to write absolute filenames, just for
            # safety's sake (script should be more robust in the face of
            # chdir'ing before running it).  But this requires abspath'ing
            # 'prefix' as well, and that breaks the hack in build_lib's
            # 'byte_compile()' method that carefully tacks on a trailing
            # slash (os.sep really) to make sure the prefix here is "just
            # right".  This whole prefix business is rather delicate -- the
            # problem is that it's really a directory, but I'm treating it
            # as a dumb string, so trailing slashes and so forth matter.

            #py_files = map(os.path.abspath, py_files)
            #if prefix:
            #    prefix = os.path.abspath(prefix)

            script.write(string.join(map(repr, py_files), ",\n") + "]\n")
            script.write("""
byte_compile(files, optimize=%r, force=%r,
             prefix=%r, base_dir=%r,
             verbose=%r, dry_run=0,
             direct=1)
""" % (optimize, force, prefix, base_dir, verbose))

            script.close()

        cmd = [sys.executable, script_name]
        if optimize == 1:
            cmd.insert(1, "-O")
        elif optimize == 2:
            cmd.insert(1, "-OO")
        spawn(cmd, dry_run=dry_run)
        execute(os.remove, (script_name,), "removing %s" % script_name,
                dry_run=dry_run)

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
            if sys.platform.startswith('java'):
                cfile = file[:-3] + '$py.class'
            else:
                cfile = file + (__debug__ and "c" or "o")
            dfile = file
            if prefix:
                if file[:len(prefix)] != prefix:
                    raise ValueError, \
                          ("invalid prefix: filename %r doesn't start with %r"
                           % (file, prefix))
                dfile = dfile[len(prefix):]
            if base_dir:
                dfile = os.path.join(base_dir, dfile)

            cfile_base = os.path.basename(cfile)
            if direct:
                if force or newer(file, cfile):
                    log.info("byte-compiling %s to %s", file, cfile_base)
                    if not dry_run:
                        compile(file, cfile, dfile)
                else:
                    log.debug("skipping byte-compilation of %s to %s",
                              file, cfile_base)

# byte_compile ()

def rfc822_escape (header):
    """Return a version of the string escaped for inclusion in an
    RFC-822 header, by ensuring there are 8 spaces space after each newline.
    """
    lines = string.split(header, '\n')
    lines = map(string.strip, lines)
    header = string.join(lines, '\n' + 8*' ')
    return header
