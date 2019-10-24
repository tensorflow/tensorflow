"""distutils.command.build

Implements the Distutils 'build' command."""

# This module should be kept compatible with Python 2.1.

__revision__ = "$Id: build.py 37828 2004-11-10 22:23:15Z loewis $"

import sys, os
from distutils.core import Command
from distutils.util import get_platform


def show_compilers ():
    from distutils.ccompiler import show_compilers
    show_compilers()


class build (Command):

    description = "build everything needed to install"

    user_options = [
        ('build-base=', 'b',
         "base directory for build library"),
        ('build-purelib=', None,
         "build directory for platform-neutral distributions"),
        ('build-platlib=', None,
         "build directory for platform-specific distributions"),
        ('build-lib=', None,
         "build directory for all distribution (defaults to either " +
         "build-purelib or build-platlib"),
        ('build-scripts=', None,
         "build directory for scripts"),
        ('build-temp=', 't',
         "temporary build directory"),
        ('compiler=', 'c',
         "specify the compiler type"),
        ('debug', 'g',
         "compile extensions and libraries with debugging information"),
        ('force', 'f',
         "forcibly build everything (ignore file timestamps)"),
        ('executable=', 'e',
         "specify final destination interpreter path (build.py)"),
        ]

    boolean_options = ['debug', 'force']

    help_options = [
        ('help-compiler', None,
         "list available compilers", show_compilers),
        ]

    def initialize_options (self):
        self.build_base = 'build'
        # these are decided only after 'build_base' has its final value
        # (unless overridden by the user or client)
        self.build_purelib = None
        self.build_platlib = None
        self.build_lib = None
        self.build_temp = None
        self.build_scripts = None
        self.compiler = None
        self.debug = None
        self.force = 0
        self.executable = None

    def finalize_options (self):

        plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])

        # 'build_purelib' and 'build_platlib' just default to 'lib' and
        # 'lib.<plat>' under the base build directory.  We only use one of
        # them for a given distribution, though --
        if self.build_purelib is None:
            self.build_purelib = os.path.join(self.build_base, 'lib')
        if self.build_platlib is None:
            self.build_platlib = os.path.join(self.build_base,
                                              'lib' + plat_specifier)

        # 'build_lib' is the actual directory that we will use for this
        # particular module distribution -- if user didn't supply it, pick
        # one of 'build_purelib' or 'build_platlib'.
        if self.build_lib is None:
            if self.distribution.ext_modules:
                self.build_lib = self.build_platlib
            else:
                self.build_lib = self.build_purelib

        # 'build_temp' -- temporary directory for compiler turds,
        # "build/temp.<plat>"
        if self.build_temp is None:
            self.build_temp = os.path.join(self.build_base,
                                           'temp' + plat_specifier)
        if self.build_scripts is None:
            self.build_scripts = os.path.join(self.build_base,
                                              'scripts-' + sys.version[0:3])

        if self.executable is None:
            self.executable = os.path.normpath(sys.executable)
    # finalize_options ()


    def run (self):

        # Run all relevant sub-commands.  This will be some subset of:
        #  - build_py      - pure Python modules
        #  - build_clib    - standalone C libraries
        #  - build_ext     - Python extensions
        #  - build_scripts - (Python) scripts
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)


    # -- Predicates for the sub-command list ---------------------------

    def has_pure_modules (self):
        return self.distribution.has_pure_modules()

    def has_c_libraries (self):
        return self.distribution.has_c_libraries()

    def has_ext_modules (self):
        return self.distribution.has_ext_modules()

    def has_scripts (self):
        return self.distribution.has_scripts()


    sub_commands = [('build_py',      has_pure_modules),
                    ('build_clib',    has_c_libraries),
                    ('build_ext',     has_ext_modules),
                    ('build_scripts', has_scripts),
                   ]

# class build
