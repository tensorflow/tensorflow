"""distutils.emxccompiler

Provides the EMXCCompiler class, a subclass of UnixCCompiler that
handles the EMX port of the GNU C compiler to OS/2.
"""

# issues:
#
# * OS/2 insists that DLLs can have names no longer than 8 characters
#   We put export_symbols in a def-file, as though the DLL can have
#   an arbitrary length name, but truncate the output filename.
#
# * only use OMF objects and use LINK386 as the linker (-Zomf)
#
# * always build for multithreading (-Zmt) as the accompanying OS/2 port
#   of Python is only distributed with threads enabled.
#
# tested configurations:
#
# * EMX gcc 2.81/EMX 0.9d fix03

__revision__ = "$Id: emxccompiler.py 34786 2003-12-02 12:17:59Z aimacintyre $"

import os,sys,copy
from distutils.ccompiler import gen_preprocess_options, gen_lib_options
from distutils.unixccompiler import UnixCCompiler
from distutils.file_util import write_file
from distutils.errors import DistutilsExecError, CompileError, UnknownFileError
from distutils import log

class EMXCCompiler (UnixCCompiler):

    compiler_type = 'emx'
    obj_extension = ".obj"
    static_lib_extension = ".lib"
    shared_lib_extension = ".dll"
    static_lib_format = "%s%s"
    shared_lib_format = "%s%s"
    res_extension = ".res"      # compiled resource file
    exe_extension = ".exe"

    def __init__ (self,
                  verbose=0,
                  dry_run=0,
                  force=0):

        UnixCCompiler.__init__ (self, verbose, dry_run, force)

        (status, details) = check_config_h()
        self.debug_print("Python's GCC status: %s (details: %s)" %
                         (status, details))
        if status is not CONFIG_H_OK:
            self.warn(
                "Python's pyconfig.h doesn't seem to support your compiler.  " +
                ("Reason: %s." % details) +
                "Compiling may fail because of undefined preprocessor macros.")

        (self.gcc_version, self.ld_version) = \
            get_versions()
        self.debug_print(self.compiler_type + ": gcc %s, ld %s\n" %
                         (self.gcc_version,
                          self.ld_version) )

        # Hard-code GCC because that's what this is all about.
        # XXX optimization, warnings etc. should be customizable.
        self.set_executables(compiler='gcc -Zomf -Zmt -O3 -fomit-frame-pointer -mprobe -Wall',
                             compiler_so='gcc -Zomf -Zmt -O3 -fomit-frame-pointer -mprobe -Wall',
                             linker_exe='gcc -Zomf -Zmt -Zcrtdll',
                             linker_so='gcc -Zomf -Zmt -Zcrtdll -Zdll')

        # want the gcc library statically linked (so that we don't have
        # to distribute a version dependent on the compiler we have)
        self.dll_libraries=["gcc"]

    # __init__ ()

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        if ext == '.rc':
            # gcc requires '.rc' compiled to binary ('.res') files !!!
            try:
                self.spawn(["rc", "-r", src])
            except DistutilsExecError, msg:
                raise CompileError, msg
        else: # for other files use the C-compiler
            try:
                self.spawn(self.compiler_so + cc_args + [src, '-o', obj] +
                           extra_postargs)
            except DistutilsExecError, msg:
                raise CompileError, msg

    def link (self,
              target_desc,
              objects,
              output_filename,
              output_dir=None,
              libraries=None,
              library_dirs=None,
              runtime_library_dirs=None,
              export_symbols=None,
              debug=0,
              extra_preargs=None,
              extra_postargs=None,
              build_temp=None,
              target_lang=None):

        # use separate copies, so we can modify the lists
        extra_preargs = copy.copy(extra_preargs or [])
        libraries = copy.copy(libraries or [])
        objects = copy.copy(objects or [])

        # Additional libraries
        libraries.extend(self.dll_libraries)

        # handle export symbols by creating a def-file
        # with executables this only works with gcc/ld as linker
        if ((export_symbols is not None) and
            (target_desc != self.EXECUTABLE)):
            # (The linker doesn't do anything if output is up-to-date.
            # So it would probably better to check if we really need this,
            # but for this we had to insert some unchanged parts of
            # UnixCCompiler, and this is not what we want.)

            # we want to put some files in the same directory as the
            # object files are, build_temp doesn't help much
            # where are the object files
            temp_dir = os.path.dirname(objects[0])
            # name of dll to give the helper files the same base name
            (dll_name, dll_extension) = os.path.splitext(
                os.path.basename(output_filename))

            # generate the filenames for these files
            def_file = os.path.join(temp_dir, dll_name + ".def")

            # Generate .def file
            contents = [
                "LIBRARY %s INITINSTANCE TERMINSTANCE" % \
                os.path.splitext(os.path.basename(output_filename))[0],
                "DATA MULTIPLE NONSHARED",
                "EXPORTS"]
            for sym in export_symbols:
                contents.append('  "%s"' % sym)
            self.execute(write_file, (def_file, contents),
                         "writing %s" % def_file)

            # next add options for def-file and to creating import libraries
            # for gcc/ld the def-file is specified as any other object files
            objects.append(def_file)

        #end: if ((export_symbols is not None) and
        #        (target_desc != self.EXECUTABLE or self.linker_dll == "gcc")):

        # who wants symbols and a many times larger output file
        # should explicitly switch the debug mode on
        # otherwise we let dllwrap/ld strip the output file
        # (On my machine: 10KB < stripped_file < ??100KB
        #   unstripped_file = stripped_file + XXX KB
        #  ( XXX=254 for a typical python extension))
        if not debug:
            extra_preargs.append("-s")

        UnixCCompiler.link(self,
                           target_desc,
                           objects,
                           output_filename,
                           output_dir,
                           libraries,
                           library_dirs,
                           runtime_library_dirs,
                           None, # export_symbols, we do this in our def-file
                           debug,
                           extra_preargs,
                           extra_postargs,
                           build_temp,
                           target_lang)

    # link ()

    # -- Miscellaneous methods -----------------------------------------

    # override the object_filenames method from CCompiler to
    # support rc and res-files
    def object_filenames (self,
                          source_filenames,
                          strip_dir=0,
                          output_dir=''):
        if output_dir is None: output_dir = ''
        obj_names = []
        for src_name in source_filenames:
            # use normcase to make sure '.rc' is really '.rc' and not '.RC'
            (base, ext) = os.path.splitext (os.path.normcase(src_name))
            if ext not in (self.src_extensions + ['.rc']):
                raise UnknownFileError, \
                      "unknown file type '%s' (from '%s')" % \
                      (ext, src_name)
            if strip_dir:
                base = os.path.basename (base)
            if ext == '.rc':
                # these need to be compiled to object files
                obj_names.append (os.path.join (output_dir,
                                            base + self.res_extension))
            else:
                obj_names.append (os.path.join (output_dir,
                                            base + self.obj_extension))
        return obj_names

    # object_filenames ()

    # override the find_library_file method from UnixCCompiler
    # to deal with file naming/searching differences
    def find_library_file(self, dirs, lib, debug=0):
        shortlib = '%s.lib' % lib
        longlib = 'lib%s.lib' % lib    # this form very rare

        # get EMX's default library directory search path
        try:
            emx_dirs = os.environ['LIBRARY_PATH'].split(';')
        except KeyError:
            emx_dirs = []

        for dir in dirs + emx_dirs:
            shortlibp = os.path.join(dir, shortlib)
            longlibp = os.path.join(dir, longlib)
            if os.path.exists(shortlibp):
                return shortlibp
            elif os.path.exists(longlibp):
                return longlibp

        # Oops, didn't find it in *any* of 'dirs'
        return None

# class EMXCCompiler


# Because these compilers aren't configured in Python's pyconfig.h file by
# default, we should at least warn the user if he is using a unmodified
# version.

CONFIG_H_OK = "ok"
CONFIG_H_NOTOK = "not ok"
CONFIG_H_UNCERTAIN = "uncertain"

def check_config_h():

    """Check if the current Python installation (specifically, pyconfig.h)
    appears amenable to building extensions with GCC.  Returns a tuple
    (status, details), where 'status' is one of the following constants:
      CONFIG_H_OK
        all is well, go ahead and compile
      CONFIG_H_NOTOK
        doesn't look good
      CONFIG_H_UNCERTAIN
        not sure -- unable to read pyconfig.h
    'details' is a human-readable string explaining the situation.

    Note there are two ways to conclude "OK": either 'sys.version' contains
    the string "GCC" (implying that this Python was built with GCC), or the
    installed "pyconfig.h" contains the string "__GNUC__".
    """

    # XXX since this function also checks sys.version, it's not strictly a
    # "pyconfig.h" check -- should probably be renamed...

    from distutils import sysconfig
    import string
    # if sys.version contains GCC then python was compiled with
    # GCC, and the pyconfig.h file should be OK
    if string.find(sys.version,"GCC") >= 0:
        return (CONFIG_H_OK, "sys.version mentions 'GCC'")

    fn = sysconfig.get_config_h_filename()
    try:
        # It would probably better to read single lines to search.
        # But we do this only once, and it is fast enough
        f = open(fn)
        s = f.read()
        f.close()

    except IOError, exc:
        # if we can't read this file, we cannot say it is wrong
        # the compiler will complain later about this file as missing
        return (CONFIG_H_UNCERTAIN,
                "couldn't read '%s': %s" % (fn, exc.strerror))

    else:
        # "pyconfig.h" contains an "#ifdef __GNUC__" or something similar
        if string.find(s,"__GNUC__") >= 0:
            return (CONFIG_H_OK, "'%s' mentions '__GNUC__'" % fn)
        else:
            return (CONFIG_H_NOTOK, "'%s' does not mention '__GNUC__'" % fn)


def get_versions():
    """ Try to find out the versions of gcc and ld.
        If not possible it returns None for it.
    """
    from distutils.version import StrictVersion
    from distutils.spawn import find_executable
    import re

    gcc_exe = find_executable('gcc')
    if gcc_exe:
        out = os.popen(gcc_exe + ' -dumpversion','r')
        out_string = out.read()
        out.close()
        result = re.search('(\d+\.\d+\.\d+)',out_string)
        if result:
            gcc_version = StrictVersion(result.group(1))
        else:
            gcc_version = None
    else:
        gcc_version = None
    # EMX ld has no way of reporting version number, and we use GCC
    # anyway - so we can link OMF DLLs
    ld_version = None
    return (gcc_version, ld_version)
