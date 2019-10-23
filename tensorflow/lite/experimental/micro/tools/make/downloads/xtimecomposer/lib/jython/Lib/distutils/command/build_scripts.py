"""distutils.command.build_scripts

Implements the Distutils 'build_scripts' command."""

# This module should be kept compatible with Python 2.1.

__revision__ = "$Id: build_scripts.py 59668 2008-01-02 18:59:36Z guido.van.rossum $"

import sys, os, re
from stat import ST_MODE
from distutils import sysconfig
from distutils.core import Command
from distutils.dep_util import newer
from distutils.util import convert_path
from distutils import log

# check if Python is called on the first line with this expression
first_line_re = re.compile('^#!.*python[0-9.]*([ \t].*)?$')

class build_scripts (Command):

    description = "\"build\" scripts (copy and fixup #! line)"

    user_options = [
        ('build-dir=', 'd', "directory to \"build\" (copy) to"),
        ('force', 'f', "forcibly build everything (ignore file timestamps"),
        ('executable=', 'e', "specify final destination interpreter path"),
        ]

    boolean_options = ['force']


    def initialize_options (self):
        self.build_dir = None
        self.scripts = None
        self.force = None
        self.executable = None
        self.outfiles = None

    def finalize_options (self):
        self.set_undefined_options('build',
                                   ('build_scripts', 'build_dir'),
                                   ('force', 'force'),
                                   ('executable', 'executable'))
        self.scripts = self.distribution.scripts

    def get_source_files(self):
        return self.scripts

    def run (self):
        if not self.scripts:
            return
        self.copy_scripts()


    def copy_scripts (self):
        """Copy each script listed in 'self.scripts'; if it's marked as a
        Python script in the Unix way (first line matches 'first_line_re',
        ie. starts with "\#!" and contains "python"), then adjust the first
        line to refer to the current Python interpreter as we copy.
        """
        self.mkpath(self.build_dir)
        outfiles = []
        for script in self.scripts:
            adjust = 0
            script = convert_path(script)
            outfile = os.path.join(self.build_dir, os.path.basename(script))
            outfiles.append(outfile)

            if not self.force and not newer(script, outfile):
                log.debug("not copying %s (up-to-date)", script)
                continue

            # Always open the file, but ignore failures in dry-run mode --
            # that way, we'll get accurate feedback if we can read the
            # script.
            try:
                f = open(script, "r")
            except IOError:
                if not self.dry_run:
                    raise
                f = None
            else:
                first_line = f.readline()
                if not first_line:
                    self.warn("%s is an empty file (skipping)" % script)
                    continue

                match = first_line_re.match(first_line)
                if match:
                    adjust = 1
                    post_interp = match.group(1) or ''

            if adjust:
                log.info("copying and adjusting %s -> %s", script,
                         self.build_dir)
                if not sysconfig.python_build:
                    executable = self.executable
                else:
                    executable = os.path.join(
                        sysconfig.get_config_var("BINDIR"),
                        "python" + sysconfig.get_config_var("EXE"))
                executable = fix_jython_executable(executable, post_interp)
                if not self.dry_run:
                    outf = open(outfile, "w")
                    outf.write("#!%s%s\n" %
                               (executable,
                                post_interp))
                    outf.writelines(f.readlines())
                    outf.close()
                if f:
                    f.close()
            else:
                if f:
                    f.close()
                self.copy_file(script, outfile)

        if hasattr(os, 'chmod'):
            for file in outfiles:
                if self.dry_run:
                    log.info("changing mode of %s", file)
                else:
                    oldmode = os.stat(file)[ST_MODE] & 07777
                    newmode = (oldmode | 0555) & 07777
                    if newmode != oldmode:
                        log.info("changing mode of %s from %o to %o",
                                 file, oldmode, newmode)
                        os.chmod(file, newmode)

    # copy_scripts ()

# class build_scripts


def is_sh(executable):
    """Determine if the specified executable is a .sh (contains a #! line)"""
    try:
        fp = open(executable)
        magic = fp.read(2)
        fp.close()
    except IOError, OSError:
        return executable
    return magic == '#!'


def fix_jython_executable(executable, options):
    if sys.platform.startswith('java') and is_sh(executable):
        # Workaround Jython's sys.executable being a .sh (an invalid
        # shebang line interpreter)
        if options:
            # Can't apply the workaround, leave it broken
            log.warn("WARNING: Unable to adapt shebang line for Jython,"
                             " the following script is NOT executable\n"
                     "         see http://bugs.jython.org/issue1112 for"
                             " more information.")
        else:
            return '/usr/bin/env %s' % executable
    return executable
