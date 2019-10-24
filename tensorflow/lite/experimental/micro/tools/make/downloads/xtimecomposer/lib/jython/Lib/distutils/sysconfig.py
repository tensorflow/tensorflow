"""Provide access to Python's configuration information.  The specific
configuration variables available depend heavily on the platform and
configuration.  The values may be retrieved using
get_config_var(name), and the list of variables is available via
get_config_vars().keys().  Additional convenience functions are also
available.

Written by:   Fred L. Drake, Jr.
Email:        <fdrake@acm.org>
"""

__revision__ = "$Id: sysconfig.py 52234 2006-10-08 17:50:26Z ronald.oussoren $"

import os
import re
import string
import sys

from distutils.errors import DistutilsPlatformError

# These are needed in a couple of spots, so just compute them once.
PREFIX = os.path.normpath(sys.prefix)
EXEC_PREFIX = os.path.normpath(sys.exec_prefix)

# python_build: (Boolean) if true, we're either building Python or
# building an extension with an un-installed Python, so we use
# different (hard-wired) directories.

argv0_path = os.path.dirname(os.path.abspath(sys.executable))
landmark = os.path.join(argv0_path, "Modules", "Setup")

python_build = os.path.isfile(landmark)

del landmark


def get_python_version():
    """Return a string containing the major and minor Python version,
    leaving off the patchlevel.  Sample return values could be '1.5'
    or '2.2'.
    """
    return sys.version[:3]


def get_python_inc(plat_specific=0, prefix=None):
    """Return the directory containing installed Python header files.

    If 'plat_specific' is false (the default), this is the path to the
    non-platform-specific header files, i.e. Python.h and so on;
    otherwise, this is the path to platform-specific header files
    (namely pyconfig.h).

    If 'prefix' is supplied, use it instead of sys.prefix or
    sys.exec_prefix -- i.e., ignore 'plat_specific'.
    """
    if prefix is None:
        prefix = plat_specific and EXEC_PREFIX or PREFIX
    if os.name == "posix":
        if python_build:
            base = os.path.dirname(os.path.abspath(sys.executable))
            if plat_specific:
                inc_dir = base
            else:
                inc_dir = os.path.join(base, "Include")
                if not os.path.exists(inc_dir):
                    inc_dir = os.path.join(os.path.dirname(base), "Include")
            return inc_dir
        return os.path.join(prefix, "include", "python" + get_python_version())
    elif os.name == "nt":
        return os.path.join(prefix, "include")
    elif os.name == "mac":
        if plat_specific:
            return os.path.join(prefix, "Mac", "Include")
        else:
            return os.path.join(prefix, "Include")
    elif os.name == "os2" or os.name == "java":
        return os.path.join(prefix, "Include")
    else:
        raise DistutilsPlatformError(
            "I don't know where Python installs its C header files "
            "on platform '%s'" % os.name)


def get_python_lib(plat_specific=0, standard_lib=0, prefix=None):
    """Return the directory containing the Python library (standard or
    site additions).

    If 'plat_specific' is true, return the directory containing
    platform-specific modules, i.e. any module from a non-pure-Python
    module distribution; otherwise, return the platform-shared library
    directory.  If 'standard_lib' is true, return the directory
    containing standard Python library modules; otherwise, return the
    directory for site-specific modules.

    If 'prefix' is supplied, use it instead of sys.prefix or
    sys.exec_prefix -- i.e., ignore 'plat_specific'.
    """
    if prefix is None:
        prefix = plat_specific and EXEC_PREFIX or PREFIX

    if os.name == "posix":
        libpython = os.path.join(prefix,
                                 "lib", "python" + get_python_version())
        if standard_lib:
            return libpython
        else:
            return os.path.join(libpython, "site-packages")

    elif os.name == "nt":
        if standard_lib:
            return os.path.join(prefix, "Lib")
        else:
            if get_python_version() < "2.2":
                return prefix
            else:
                return os.path.join(PREFIX, "Lib", "site-packages")

    elif os.name == "mac":
        if plat_specific:
            if standard_lib:
                return os.path.join(prefix, "Lib", "lib-dynload")
            else:
                return os.path.join(prefix, "Lib", "site-packages")
        else:
            if standard_lib:
                return os.path.join(prefix, "Lib")
            else:
                return os.path.join(prefix, "Lib", "site-packages")

    elif os.name == "os2" or os.name == "java":
        if standard_lib:
            return os.path.join(PREFIX, "Lib")
        else:
            return os.path.join(PREFIX, "Lib", "site-packages")

    else:
        raise DistutilsPlatformError(
            "I don't know where Python installs its library "
            "on platform '%s'" % os.name)


def customize_compiler(compiler):
    """Do any platform-specific customization of a CCompiler instance.

    Mainly needed on Unix, so we can plug in the information that
    varies across Unices and is stored in Python's Makefile.
    """
    if compiler.compiler_type == "unix":
        (cc, cxx, opt, cflags, ccshared, ldshared, so_ext) = \
            get_config_vars('CC', 'CXX', 'OPT', 'CFLAGS',
                            'CCSHARED', 'LDSHARED', 'SO')

        if os.environ.has_key('CC'):
            cc = os.environ['CC']
        if os.environ.has_key('CXX'):
            cxx = os.environ['CXX']
        if os.environ.has_key('LDSHARED'):
            ldshared = os.environ['LDSHARED']
        if os.environ.has_key('CPP'):
            cpp = os.environ['CPP']
        else:
            cpp = cc + " -E"           # not always
        if os.environ.has_key('LDFLAGS'):
            ldshared = ldshared + ' ' + os.environ['LDFLAGS']
        if os.environ.has_key('CFLAGS'):
            cflags = opt + ' ' + os.environ['CFLAGS']
            ldshared = ldshared + ' ' + os.environ['CFLAGS']
        if os.environ.has_key('CPPFLAGS'):
            cpp = cpp + ' ' + os.environ['CPPFLAGS']
            cflags = cflags + ' ' + os.environ['CPPFLAGS']
            ldshared = ldshared + ' ' + os.environ['CPPFLAGS']

        cc_cmd = cc + ' ' + cflags
        compiler.set_executables(
            preprocessor=cpp,
            compiler=cc_cmd,
            compiler_so=cc_cmd + ' ' + ccshared,
            compiler_cxx=cxx,
            linker_so=ldshared,
            linker_exe=cc)

        compiler.shared_lib_extension = so_ext


def get_config_h_filename():
    """Return full pathname of installed pyconfig.h file."""
    if python_build:
        inc_dir = argv0_path
    else:
        inc_dir = get_python_inc(plat_specific=1)
    if get_python_version() < '2.2':
        config_h = 'config.h'
    else:
        # The name of the config.h file changed in 2.2
        config_h = 'pyconfig.h'
    return os.path.join(inc_dir, config_h)


def get_makefile_filename():
    """Return full pathname of installed Makefile from the Python build."""
    if python_build:
        return os.path.join(os.path.dirname(sys.executable), "Makefile")
    lib_dir = get_python_lib(plat_specific=1, standard_lib=1)
    return os.path.join(lib_dir, "config", "Makefile")


def parse_config_h(fp, g=None):
    """Parse a config.h-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    if g is None:
        g = {}
    define_rx = re.compile("#define ([A-Z][A-Za-z0-9_]+) (.*)\n")
    undef_rx = re.compile("/[*] #undef ([A-Z][A-Za-z0-9_]+) [*]/\n")
    #
    while 1:
        line = fp.readline()
        if not line:
            break
        m = define_rx.match(line)
        if m:
            n, v = m.group(1, 2)
            try: v = int(v)
            except ValueError: pass
            g[n] = v
        else:
            m = undef_rx.match(line)
            if m:
                g[m.group(1)] = 0
    return g


# Regexes needed for parsing Makefile (and similar syntaxes,
# like old-style Setup files).
_variable_rx = re.compile("([a-zA-Z][a-zA-Z0-9_]+)\s*=\s*(.*)")
_findvar1_rx = re.compile(r"\$\(([A-Za-z][A-Za-z0-9_]*)\)")
_findvar2_rx = re.compile(r"\${([A-Za-z][A-Za-z0-9_]*)}")

def parse_makefile(fn, g=None):
    """Parse a Makefile-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    from distutils.text_file import TextFile
    fp = TextFile(fn, strip_comments=1, skip_blanks=1, join_lines=1)

    if g is None:
        g = {}
    done = {}
    notdone = {}

    while 1:
        line = fp.readline()
        if line is None:                # eof
            break
        m = _variable_rx.match(line)
        if m:
            n, v = m.group(1, 2)
            v = string.strip(v)
            if "$" in v:
                notdone[n] = v
            else:
                try: v = int(v)
                except ValueError: pass
                done[n] = v

    # do variable interpolation here
    while notdone:
        for name in notdone.keys():
            value = notdone[name]
            m = _findvar1_rx.search(value) or _findvar2_rx.search(value)
            if m:
                n = m.group(1)
                found = True
                if done.has_key(n):
                    item = str(done[n])
                elif notdone.has_key(n):
                    # get it on a subsequent round
                    found = False
                elif os.environ.has_key(n):
                    # do it like make: fall back to environment
                    item = os.environ[n]
                else:
                    done[n] = item = ""
                if found:
                    after = value[m.end():]
                    value = value[:m.start()] + item + after
                    if "$" in after:
                        notdone[name] = value
                    else:
                        try: value = int(value)
                        except ValueError:
                            done[name] = string.strip(value)
                        else:
                            done[name] = value
                        del notdone[name]
            else:
                # bogus variable reference; just drop it since we can't deal
                del notdone[name]

    fp.close()

    # save the results in the global dictionary
    g.update(done)
    return g


def expand_makefile_vars(s, vars):
    """Expand Makefile-style variables -- "${foo}" or "$(foo)" -- in
    'string' according to 'vars' (a dictionary mapping variable names to
    values).  Variables not present in 'vars' are silently expanded to the
    empty string.  The variable values in 'vars' should not contain further
    variable expansions; if 'vars' is the output of 'parse_makefile()',
    you're fine.  Returns a variable-expanded version of 's'.
    """

    # This algorithm does multiple expansion, so if vars['foo'] contains
    # "${bar}", it will expand ${foo} to ${bar}, and then expand
    # ${bar}... and so forth.  This is fine as long as 'vars' comes from
    # 'parse_makefile()', which takes care of such expansions eagerly,
    # according to make's variable expansion semantics.

    while 1:
        m = _findvar1_rx.search(s) or _findvar2_rx.search(s)
        if m:
            (beg, end) = m.span()
            s = s[0:beg] + vars.get(m.group(1)) + s[end:]
        else:
            break
    return s


_config_vars = None

def _init_posix():
    """Initialize the module as appropriate for POSIX systems."""
    g = {}
    # load the installed Makefile:
    try:
        filename = get_makefile_filename()
        parse_makefile(filename, g)
    except IOError, msg:
        my_msg = "invalid Python installation: unable to open %s" % filename
        if hasattr(msg, "strerror"):
            my_msg = my_msg + " (%s)" % msg.strerror

        raise DistutilsPlatformError(my_msg)

    # load the installed pyconfig.h:
    try:
        filename = get_config_h_filename()
        parse_config_h(file(filename), g)
    except IOError, msg:
        my_msg = "invalid Python installation: unable to open %s" % filename
        if hasattr(msg, "strerror"):
            my_msg = my_msg + " (%s)" % msg.strerror

        raise DistutilsPlatformError(my_msg)

    # On MacOSX we need to check the setting of the environment variable
    # MACOSX_DEPLOYMENT_TARGET: configure bases some choices on it so
    # it needs to be compatible.
    # If it isn't set we set it to the configure-time value
    if sys.platform == 'darwin' and g.has_key('MACOSX_DEPLOYMENT_TARGET'):
        cfg_target = g['MACOSX_DEPLOYMENT_TARGET']
        cur_target = os.getenv('MACOSX_DEPLOYMENT_TARGET', '')
        if cur_target == '':
            cur_target = cfg_target
            os.putenv('MACOSX_DEPLOYMENT_TARGET', cfg_target)
        elif map(int, cfg_target.split('.')) > map(int, cur_target.split('.')):
            my_msg = ('$MACOSX_DEPLOYMENT_TARGET mismatch: now "%s" but "%s" during configure'
                % (cur_target, cfg_target))
            raise DistutilsPlatformError(my_msg)

    # On AIX, there are wrong paths to the linker scripts in the Makefile
    # -- these paths are relative to the Python source, but when installed
    # the scripts are in another directory.
    if python_build:
        g['LDSHARED'] = g['BLDSHARED']

    elif get_python_version() < '2.1':
        # The following two branches are for 1.5.2 compatibility.
        if sys.platform == 'aix4':          # what about AIX 3.x ?
            # Linker script is in the config directory, not in Modules as the
            # Makefile says.
            python_lib = get_python_lib(standard_lib=1)
            ld_so_aix = os.path.join(python_lib, 'config', 'ld_so_aix')
            python_exp = os.path.join(python_lib, 'config', 'python.exp')

            g['LDSHARED'] = "%s %s -bI:%s" % (ld_so_aix, g['CC'], python_exp)

        elif sys.platform == 'beos':
            # Linker script is in the config directory.  In the Makefile it is
            # relative to the srcdir, which after installation no longer makes
            # sense.
            python_lib = get_python_lib(standard_lib=1)
            linkerscript_path = string.split(g['LDSHARED'])[0]
            linkerscript_name = os.path.basename(linkerscript_path)
            linkerscript = os.path.join(python_lib, 'config',
                                        linkerscript_name)

            # XXX this isn't the right place to do this: adding the Python
            # library to the link, if needed, should be in the "build_ext"
            # command.  (It's also needed for non-MS compilers on Windows, and
            # it's taken care of for them by the 'build_ext.get_libraries()'
            # method.)
            g['LDSHARED'] = ("%s -L%s/lib -lpython%s" %
                             (linkerscript, PREFIX, get_python_version()))

    global _config_vars
    _config_vars = g


def _init_nt():
    """Initialize the module as appropriate for NT"""
    g = {}
    # set basic install directories
    g['LIBDEST'] = get_python_lib(plat_specific=0, standard_lib=1)
    g['BINLIBDEST'] = get_python_lib(plat_specific=1, standard_lib=1)

    # XXX hmmm.. a normal install puts include files here
    g['INCLUDEPY'] = get_python_inc(plat_specific=0)

    g['SO'] = '.pyd'
    g['EXE'] = ".exe"

    global _config_vars
    _config_vars = g


def _init_mac():
    """Initialize the module as appropriate for Macintosh systems"""
    g = {}
    # set basic install directories
    g['LIBDEST'] = get_python_lib(plat_specific=0, standard_lib=1)
    g['BINLIBDEST'] = get_python_lib(plat_specific=1, standard_lib=1)

    # XXX hmmm.. a normal install puts include files here
    g['INCLUDEPY'] = get_python_inc(plat_specific=0)

    import MacOS
    if not hasattr(MacOS, 'runtimemodel'):
        g['SO'] = '.ppc.slb'
    else:
        g['SO'] = '.%s.slb' % MacOS.runtimemodel

    # XXX are these used anywhere?
    g['install_lib'] = os.path.join(EXEC_PREFIX, "Lib")
    g['install_platlib'] = os.path.join(EXEC_PREFIX, "Mac", "Lib")

    # These are used by the extension module build
    g['srcdir'] = ':'
    global _config_vars
    _config_vars = g


def _init_os2():
    """Initialize the module as appropriate for OS/2"""
    g = {}
    # set basic install directories
    g['LIBDEST'] = get_python_lib(plat_specific=0, standard_lib=1)
    g['BINLIBDEST'] = get_python_lib(plat_specific=1, standard_lib=1)

    # XXX hmmm.. a normal install puts include files here
    g['INCLUDEPY'] = get_python_inc(plat_specific=0)

    g['SO'] = '.pyd'
    g['EXE'] = ".exe"

    global _config_vars
    _config_vars = g


def _init_jython():
    """Initialize the module as appropriate for Jython"""
    # Stub out some values that build_ext expects; they don't matter
    # anyway
    _init_os2()


def get_config_vars(*args):
    """With no arguments, return a dictionary of all configuration
    variables relevant for the current platform.  Generally this includes
    everything needed to build extensions and install both pure modules and
    extensions.  On Unix, this means every variable defined in Python's
    installed Makefile; on Windows and Mac OS it's a much smaller set.

    With arguments, return a list of values that result from looking up
    each argument in the configuration variable dictionary.
    """
    global _config_vars
    if _config_vars is None:
        if sys.platform.startswith('java'):
            # Jython might pose as a different os.name, but we always
            # want _init_jython regardless
            func = _init_jython
        else:
            func = globals().get("_init_" + os.name)
        if func:
            func()
        else:
            _config_vars = {}

        # Normalized versions of prefix and exec_prefix are handy to have;
        # in fact, these are the standard versions used most places in the
        # Distutils.
        _config_vars['prefix'] = PREFIX
        _config_vars['exec_prefix'] = EXEC_PREFIX

        if sys.platform == 'darwin':
            kernel_version = os.uname()[2] # Kernel version (8.4.3)
            major_version = int(kernel_version.split('.')[0])

            if major_version < 8:
                # On Mac OS X before 10.4, check if -arch and -isysroot
                # are in CFLAGS or LDFLAGS and remove them if they are.
                # This is needed when building extensions on a 10.3 system
                # using a universal build of python.
                for key in ('LDFLAGS', 'BASECFLAGS',
                        # a number of derived variables. These need to be
                        # patched up as well.
                        'CFLAGS', 'PY_CFLAGS', 'BLDSHARED'):

                    flags = _config_vars[key]
                    flags = re.sub('-arch\s+\w+\s', ' ', flags)
                    flags = re.sub('-isysroot [^ \t]*', ' ', flags)
                    _config_vars[key] = flags

    if args:
        vals = []
        for name in args:
            vals.append(_config_vars.get(name))
        return vals
    else:
        return _config_vars

def get_config_var(name):
    """Return the value of a single variable using the dictionary
    returned by 'get_config_vars()'.  Equivalent to
    get_config_vars().get(name)
    """
    return get_config_vars().get(name)
