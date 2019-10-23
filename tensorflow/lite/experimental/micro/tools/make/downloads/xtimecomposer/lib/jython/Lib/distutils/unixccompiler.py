"""distutils.unixccompiler

Contains the UnixCCompiler class, a subclass of CCompiler that handles
the "typical" Unix-style command-line C compiler:
  * macros defined with -Dname[=value]
  * macros undefined with -Uname
  * include search directories specified with -Idir
  * libraries specified with -lllib
  * library search directories specified with -Ldir
  * compile handled by 'cc' (or similar) executable with -c option:
    compiles .c to .o
  * link static library handled by 'ar' command (possibly with 'ranlib')
  * link shared library handled by 'cc -shared'
"""

__revision__ = "$Id: unixccompiler.py 54954 2007-04-25 06:42:41Z neal.norwitz $"

import os, sys
from types import StringType, NoneType
from copy import copy

from distutils import sysconfig
from distutils.dep_util import newer
from distutils.ccompiler import \
     CCompiler, gen_preprocess_options, gen_lib_options
from distutils.errors import \
     DistutilsExecError, CompileError, LibError, LinkError
from distutils import log

# XXX Things not currently handled:
#   * optimization/debug/warning flags; we just use whatever's in Python's
#     Makefile and live with it.  Is this adequate?  If not, we might
#     have to have a bunch of subclasses GNUCCompiler, SGICCompiler,
#     SunCCompiler, and I suspect down that road lies madness.
#   * even if we don't know a warning flag from an optimization flag,
#     we need some way for outsiders to feed preprocessor/compiler/linker
#     flags in to us -- eg. a sysadmin might want to mandate certain flags
#     via a site config file, or a user might want to set something for
#     compiling this module distribution only via the setup.py command
#     line, whatever.  As long as these options come from something on the
#     current system, they can be as system-dependent as they like, and we
#     should just happily stuff them into the preprocessor/compiler/linker
#     options and carry on.

def _darwin_compiler_fixup(compiler_so, cc_args):
    """
    This function will strip '-isysroot PATH' and '-arch ARCH' from the
    compile flags if the user has specified one them in extra_compile_flags.

    This is needed because '-arch ARCH' adds another architecture to the
    build, without a way to remove an architecture. Furthermore GCC will
    barf if multiple '-isysroot' arguments are present.
    """
    stripArch = stripSysroot = 0

    compiler_so = list(compiler_so)
    kernel_version = os.uname()[2] # 8.4.3
    major_version = int(kernel_version.split('.')[0])

    if major_version < 8:
        # OSX before 10.4.0, these don't support -arch and -isysroot at
        # all.
        stripArch = stripSysroot = True
    else:
        stripArch = '-arch' in cc_args
        stripSysroot = '-isysroot' in cc_args

    if stripArch:
        while 1:
            try:
                index = compiler_so.index('-arch')
                # Strip this argument and the next one:
                del compiler_so[index:index+2]
            except ValueError:
                break

    if stripSysroot:
        try:
            index = compiler_so.index('-isysroot')
            # Strip this argument and the next one:
            del compiler_so[index:index+2]
        except ValueError:
            pass

    # Check if the SDK that is used during compilation actually exists,
    # the universal build requires the usage of a universal SDK and not all
    # users have that installed by default.
    sysroot = None
    if '-isysroot' in cc_args:
        idx = cc_args.index('-isysroot')
        sysroot = cc_args[idx+1]
    elif '-isysroot' in compiler_so:
        idx = compiler_so.index('-isysroot')
        sysroot = compiler_so[idx+1]

    if sysroot and not os.path.isdir(sysroot):
        log.warn("Compiling with an SDK that doesn't seem to exist: %s",
                sysroot)
        log.warn("Please check your Xcode installation")

    return compiler_so

class UnixCCompiler(CCompiler):

    compiler_type = 'unix'

    # These are used by CCompiler in two places: the constructor sets
    # instance attributes 'preprocessor', 'compiler', etc. from them, and
    # 'set_executable()' allows any of these to be set.  The defaults here
    # are pretty generic; they will probably have to be set by an outsider
    # (eg. using information discovered by the sysconfig about building
    # Python extensions).
    executables = {'preprocessor' : None,
                   'compiler'     : ["cc"],
                   'compiler_so'  : ["cc"],
                   'compiler_cxx' : ["cc"],
                   'linker_so'    : ["cc", "-shared"],
                   'linker_exe'   : ["cc"],
                   'archiver'     : ["ar", "-cr"],
                   'ranlib'       : None,
                  }

    if sys.platform[:6] == "darwin":
        executables['ranlib'] = ["ranlib"]

    # Needed for the filename generation methods provided by the base
    # class, CCompiler.  NB. whoever instantiates/uses a particular
    # UnixCCompiler instance should set 'shared_lib_ext' -- we set a
    # reasonable common default here, but it's not necessarily used on all
    # Unices!

    src_extensions = [".c",".C",".cc",".cxx",".cpp",".m"]
    obj_extension = ".o"
    static_lib_extension = ".a"
    shared_lib_extension = ".so"
    dylib_lib_extension = ".dylib"
    static_lib_format = shared_lib_format = dylib_lib_format = "lib%s%s"
    if sys.platform == "cygwin":
        exe_extension = ".exe"

    def preprocess(self, source,
                   output_file=None, macros=None, include_dirs=None,
                   extra_preargs=None, extra_postargs=None):
        ignore, macros, include_dirs = \
            self._fix_compile_args(None, macros, include_dirs)
        pp_opts = gen_preprocess_options(macros, include_dirs)
        pp_args = self.preprocessor + pp_opts
        if output_file:
            pp_args.extend(['-o', output_file])
        if extra_preargs:
            pp_args[:0] = extra_preargs
        if extra_postargs:
            pp_args.extend(extra_postargs)
        pp_args.append(source)

        # We need to preprocess: either we're being forced to, or we're
        # generating output to stdout, or there's a target output file and
        # the source file is newer than the target (or the target doesn't
        # exist).
        if self.force or output_file is None or newer(source, output_file):
            if output_file:
                self.mkpath(os.path.dirname(output_file))
            try:
                self.spawn(pp_args)
            except DistutilsExecError, msg:
                raise CompileError, msg

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        compiler_so = self.compiler_so
        if sys.platform == 'darwin':
            compiler_so = _darwin_compiler_fixup(compiler_so, cc_args + extra_postargs)
        try:
            self.spawn(compiler_so + cc_args + [src, '-o', obj] +
                       extra_postargs)
        except DistutilsExecError, msg:
            raise CompileError, msg

    def create_static_lib(self, objects, output_libname,
                          output_dir=None, debug=0, target_lang=None):
        objects, output_dir = self._fix_object_args(objects, output_dir)

        output_filename = \
            self.library_filename(output_libname, output_dir=output_dir)

        if self._need_link(objects, output_filename):
            self.mkpath(os.path.dirname(output_filename))
            self.spawn(self.archiver +
                       [output_filename] +
                       objects + self.objects)

            # Not many Unices required ranlib anymore -- SunOS 4.x is, I
            # think the only major Unix that does.  Maybe we need some
            # platform intelligence here to skip ranlib if it's not
            # needed -- or maybe Python's configure script took care of
            # it for us, hence the check for leading colon.
            if self.ranlib:
                try:
                    self.spawn(self.ranlib + [output_filename])
                except DistutilsExecError, msg:
                    raise LibError, msg
        else:
            log.debug("skipping %s (up-to-date)", output_filename)

    def link(self, target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):
        objects, output_dir = self._fix_object_args(objects, output_dir)
        libraries, library_dirs, runtime_library_dirs = \
            self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)

        lib_opts = gen_lib_options(self, library_dirs, runtime_library_dirs,
                                   libraries)
        if type(output_dir) not in (StringType, NoneType):
            raise TypeError, "'output_dir' must be a string or None"
        if output_dir is not None:
            output_filename = os.path.join(output_dir, output_filename)

        if self._need_link(objects, output_filename):
            ld_args = (objects + self.objects +
                       lib_opts + ['-o', output_filename])
            if debug:
                ld_args[:0] = ['-g']
            if extra_preargs:
                ld_args[:0] = extra_preargs
            if extra_postargs:
                ld_args.extend(extra_postargs)
            self.mkpath(os.path.dirname(output_filename))
            try:
                if target_desc == CCompiler.EXECUTABLE:
                    linker = self.linker_exe[:]
                else:
                    linker = self.linker_so[:]
                if target_lang == "c++" and self.compiler_cxx:
                    # skip over environment variable settings if /usr/bin/env
                    # is used to set up the linker's environment.
                    # This is needed on OSX. Note: this assumes that the
                    # normal and C++ compiler have the same environment
                    # settings.
                    i = 0
                    if os.path.basename(linker[0]) == "env":
                        i = 1
                        while '=' in linker[i]:
                            i = i + 1

                    linker[i] = self.compiler_cxx[i]

                if sys.platform == 'darwin':
                    linker = _darwin_compiler_fixup(linker, ld_args)

                self.spawn(linker + ld_args)
            except DistutilsExecError, msg:
                raise LinkError, msg
        else:
            log.debug("skipping %s (up-to-date)", output_filename)

    # -- Miscellaneous methods -----------------------------------------
    # These are all used by the 'gen_lib_options() function, in
    # ccompiler.py.

    def library_dir_option(self, dir):
        return "-L" + dir

    def runtime_library_dir_option(self, dir):
        # XXX Hackish, at the very least.  See Python bug #445902:
        # http://sourceforge.net/tracker/index.php
        #   ?func=detail&aid=445902&group_id=5470&atid=105470
        # Linkers on different platforms need different options to
        # specify that directories need to be added to the list of
        # directories searched for dependencies when a dynamic library
        # is sought.  GCC has to be told to pass the -R option through
        # to the linker, whereas other compilers just know this.
        # Other compilers may need something slightly different.  At
        # this time, there's no way to determine this information from
        # the configuration data stored in the Python installation, so
        # we use this hack.
        compiler = os.path.basename(sysconfig.get_config_var("CC"))
        if sys.platform[:6] == "darwin":
            # MacOSX's linker doesn't understand the -R flag at all
            return "-L" + dir
        elif sys.platform[:5] == "hp-ux":
            return "+s -L" + dir
        elif sys.platform[:7] == "irix646" or sys.platform[:6] == "osf1V5":
            return ["-rpath", dir]
        elif compiler[:3] == "gcc" or compiler[:3] == "g++":
            return "-Wl,-R" + dir
        else:
            return "-R" + dir

    def library_option(self, lib):
        return "-l" + lib

    def find_library_file(self, dirs, lib, debug=0):
        shared_f = self.library_filename(lib, lib_type='shared')
        dylib_f = self.library_filename(lib, lib_type='dylib')
        static_f = self.library_filename(lib, lib_type='static')

        for dir in dirs:
            shared = os.path.join(dir, shared_f)
            dylib = os.path.join(dir, dylib_f)
            static = os.path.join(dir, static_f)
            # We're second-guessing the linker here, with not much hard
            # data to go on: GCC seems to prefer the shared library, so I'm
            # assuming that *all* Unix C compilers do.  And of course I'm
            # ignoring even GCC's "-static" option.  So sue me.
            if os.path.exists(dylib):
                return dylib
            elif os.path.exists(shared):
                return shared
            elif os.path.exists(static):
                return static

        # Oops, didn't find it in *any* of 'dirs'
        return None
