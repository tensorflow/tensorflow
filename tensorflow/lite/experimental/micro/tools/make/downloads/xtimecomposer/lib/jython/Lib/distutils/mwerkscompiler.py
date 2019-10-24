"""distutils.mwerkscompiler

Contains MWerksCompiler, an implementation of the abstract CCompiler class
for MetroWerks CodeWarrior on the Macintosh. Needs work to support CW on
Windows."""

# This module should be kept compatible with Python 2.1.

__revision__ = "$Id: mwerkscompiler.py 37828 2004-11-10 22:23:15Z loewis $"

import sys, os, string
from types import *
from distutils.errors import \
     DistutilsExecError, DistutilsPlatformError, \
     CompileError, LibError, LinkError
from distutils.ccompiler import \
     CCompiler, gen_preprocess_options, gen_lib_options
import distutils.util
import distutils.dir_util
from distutils import log
import mkcwproject

class MWerksCompiler (CCompiler) :
    """Concrete class that implements an interface to MetroWerks CodeWarrior,
       as defined by the CCompiler abstract class."""

    compiler_type = 'mwerks'

    # Just set this so CCompiler's constructor doesn't barf.  We currently
    # don't use the 'set_executables()' bureaucracy provided by CCompiler,
    # as it really isn't necessary for this sort of single-compiler class.
    # Would be nice to have a consistent interface with UnixCCompiler,
    # though, so it's worth thinking about.
    executables = {}

    # Private class data (need to distinguish C from C++ source for compiler)
    _c_extensions = ['.c']
    _cpp_extensions = ['.cc', '.cpp', '.cxx']
    _rc_extensions = ['.r']
    _exp_extension = '.exp'

    # Needed for the filename generation methods provided by the
    # base class, CCompiler.
    src_extensions = (_c_extensions + _cpp_extensions +
                      _rc_extensions)
    res_extension = '.rsrc'
    obj_extension = '.obj' # Not used, really
    static_lib_extension = '.lib'
    shared_lib_extension = '.slb'
    static_lib_format = shared_lib_format = '%s%s'
    exe_extension = ''


    def __init__ (self,
                  verbose=0,
                  dry_run=0,
                  force=0):

        CCompiler.__init__ (self, verbose, dry_run, force)


    def compile (self,
                 sources,
                 output_dir=None,
                 macros=None,
                 include_dirs=None,
                 debug=0,
                 extra_preargs=None,
                 extra_postargs=None,
                 depends=None):
        (output_dir, macros, include_dirs) = \
           self._fix_compile_args (output_dir, macros, include_dirs)
        self.__sources = sources
        self.__macros = macros
        self.__include_dirs = include_dirs
        # Don't need extra_preargs and extra_postargs for CW
        return []

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
        # First fixup.
        (objects, output_dir) = self._fix_object_args (objects, output_dir)
        (libraries, library_dirs, runtime_library_dirs) = \
            self._fix_lib_args (libraries, library_dirs, runtime_library_dirs)

        # First examine a couple of options for things that aren't implemented yet
        if not target_desc in (self.SHARED_LIBRARY, self.SHARED_OBJECT):
            raise DistutilsPlatformError, 'Can only make SHARED_LIBRARY or SHARED_OBJECT targets on the Mac'
        if runtime_library_dirs:
            raise DistutilsPlatformError, 'Runtime library dirs not implemented yet'
        if extra_preargs or extra_postargs:
            raise DistutilsPlatformError, 'Runtime library dirs not implemented yet'
        if len(export_symbols) != 1:
            raise DistutilsPlatformError, 'Need exactly one export symbol'
        # Next there are various things for which we need absolute pathnames.
        # This is because we (usually) create the project in a subdirectory of
        # where we are now, and keeping the paths relative is too much work right
        # now.
        sources = map(self._filename_to_abs, self.__sources)
        include_dirs = map(self._filename_to_abs, self.__include_dirs)
        if objects:
            objects = map(self._filename_to_abs, objects)
        else:
            objects = []
        if build_temp:
            build_temp = self._filename_to_abs(build_temp)
        else:
            build_temp = os.curdir()
        if output_dir:
            output_filename = os.path.join(output_dir, output_filename)
        # The output filename needs special handling: splitting it into dir and
        # filename part. Actually I'm not sure this is really needed, but it
        # can't hurt.
        output_filename = self._filename_to_abs(output_filename)
        output_dir, output_filename = os.path.split(output_filename)
        # Now we need the short names of a couple of things for putting them
        # into the project.
        if output_filename[-8:] == '.ppc.slb':
            basename = output_filename[:-8]
        elif output_filename[-11:] == '.carbon.slb':
            basename = output_filename[:-11]
        else:
            basename = os.path.strip(output_filename)[0]
        projectname = basename + '.mcp'
        targetname = basename
        xmlname = basename + '.xml'
        exportname = basename + '.mcp.exp'
        prefixname = 'mwerks_%s_config.h'%basename
        # Create the directories we need
        distutils.dir_util.mkpath(build_temp, dry_run=self.dry_run)
        distutils.dir_util.mkpath(output_dir, dry_run=self.dry_run)
        # And on to filling in the parameters for the project builder
        settings = {}
        settings['mac_exportname'] = exportname
        settings['mac_outputdir'] = output_dir
        settings['mac_dllname'] = output_filename
        settings['mac_targetname'] = targetname
        settings['sysprefix'] = sys.prefix
        settings['mac_sysprefixtype'] = 'Absolute'
        sourcefilenames = []
        sourcefiledirs = []
        for filename in sources + objects:
            dirname, filename = os.path.split(filename)
            sourcefilenames.append(filename)
            if not dirname in sourcefiledirs:
                sourcefiledirs.append(dirname)
        settings['sources'] = sourcefilenames
        settings['libraries'] = libraries
        settings['extrasearchdirs'] = sourcefiledirs + include_dirs + library_dirs
        if self.dry_run:
            print 'CALLING LINKER IN', os.getcwd()
            for key, value in settings.items():
                print '%20.20s %s'%(key, value)
            return
        # Build the export file
        exportfilename = os.path.join(build_temp, exportname)
        log.debug("\tCreate export file %s", exportfilename)
        fp = open(exportfilename, 'w')
        fp.write('%s\n'%export_symbols[0])
        fp.close()
        # Generate the prefix file, if needed, and put it in the settings
        if self.__macros:
            prefixfilename = os.path.join(os.getcwd(), os.path.join(build_temp, prefixname))
            fp = open(prefixfilename, 'w')
            fp.write('#include "mwerks_shcarbon_config.h"\n')
            for name, value in self.__macros:
                if value is None:
                    fp.write('#define %s\n'%name)
                else:
                    fp.write('#define %s %s\n'%(name, value))
            fp.close()
            settings['prefixname'] = prefixname

        # Build the XML file. We need the full pathname (only lateron, really)
        # because we pass this pathname to CodeWarrior in an AppleEvent, and CW
        # doesn't have a clue about our working directory.
        xmlfilename = os.path.join(os.getcwd(), os.path.join(build_temp, xmlname))
        log.debug("\tCreate XML file %s", xmlfilename)
        xmlbuilder = mkcwproject.cwxmlgen.ProjectBuilder(settings)
        xmlbuilder.generate()
        xmldata = settings['tmp_projectxmldata']
        fp = open(xmlfilename, 'w')
        fp.write(xmldata)
        fp.close()
        # Generate the project. Again a full pathname.
        projectfilename = os.path.join(os.getcwd(), os.path.join(build_temp, projectname))
        log.debug('\tCreate project file %s', projectfilename)
        mkcwproject.makeproject(xmlfilename, projectfilename)
        # And build it
        log.debug('\tBuild project')
        mkcwproject.buildproject(projectfilename)

    def _filename_to_abs(self, filename):
        # Some filenames seem to be unix-like. Convert to Mac names.
##        if '/' in filename and ':' in filename:
##           raise DistutilsPlatformError, 'Filename may be Unix or Mac style: %s'%filename
##        if '/' in filename:
##           filename = macurl2path(filename)
        filename = distutils.util.convert_path(filename)
        if not os.path.isabs(filename):
            curdir = os.getcwd()
            filename = os.path.join(curdir, filename)
        # Finally remove .. components
        components = string.split(filename, ':')
        for i in range(1, len(components)):
            if components[i] == '..':
                components[i] = ''
        return string.join(components, ':')

    def library_dir_option (self, dir):
        """Return the compiler option to add 'dir' to the list of
        directories searched for libraries.
        """
        return # XXXX Not correct...

    def runtime_library_dir_option (self, dir):
        """Return the compiler option to add 'dir' to the list of
        directories searched for runtime libraries.
        """
        # Nothing needed or Mwerks/Mac.
        return

    def library_option (self, lib):
        """Return the compiler option to add 'dir' to the list of libraries
        linked into the shared library or executable.
        """
        return

    def find_library_file (self, dirs, lib, debug=0):
        """Search the specified list of directories for a static or shared
        library file 'lib' and return the full path to that file.  If
        'debug' true, look for a debugging version (if that makes sense on
        the current platform).  Return None if 'lib' wasn't found in any of
        the specified directories.
        """
        return 0
