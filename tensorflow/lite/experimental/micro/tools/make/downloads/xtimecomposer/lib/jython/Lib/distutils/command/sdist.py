"""distutils.command.sdist

Implements the Distutils 'sdist' command (create a source distribution)."""

# This module should be kept compatible with Python 2.1.

__revision__ = "$Id: sdist.py 61268 2008-03-06 07:14:26Z martin.v.loewis $"

import sys, os, string
from types import *
from glob import glob
from distutils.core import Command
from distutils import dir_util, dep_util, file_util, archive_util
from distutils.text_file import TextFile
from distutils.errors import *
from distutils.filelist import FileList
from distutils import log


def show_formats ():
    """Print all possible values for the 'formats' option (used by
    the "--help-formats" command-line option).
    """
    from distutils.fancy_getopt import FancyGetopt
    from distutils.archive_util import ARCHIVE_FORMATS
    formats=[]
    for format in ARCHIVE_FORMATS.keys():
        formats.append(("formats=" + format, None,
                        ARCHIVE_FORMATS[format][2]))
    formats.sort()
    pretty_printer = FancyGetopt(formats)
    pretty_printer.print_help(
        "List of available source distribution formats:")

class sdist (Command):

    description = "create a source distribution (tarball, zip file, etc.)"

    user_options = [
        ('template=', 't',
         "name of manifest template file [default: MANIFEST.in]"),
        ('manifest=', 'm',
         "name of manifest file [default: MANIFEST]"),
        ('use-defaults', None,
         "include the default file set in the manifest "
         "[default; disable with --no-defaults]"),
        ('no-defaults', None,
         "don't include the default file set"),
        ('prune', None,
         "specifically exclude files/directories that should not be "
         "distributed (build tree, RCS/CVS dirs, etc.) "
         "[default; disable with --no-prune]"),
        ('no-prune', None,
         "don't automatically exclude anything"),
        ('manifest-only', 'o',
         "just regenerate the manifest and then stop "
         "(implies --force-manifest)"),
        ('force-manifest', 'f',
         "forcibly regenerate the manifest and carry on as usual"),
        ('formats=', None,
         "formats for source distribution (comma-separated list)"),
        ('keep-temp', 'k',
         "keep the distribution tree around after creating " +
         "archive file(s)"),
        ('dist-dir=', 'd',
         "directory to put the source distribution archive(s) in "
         "[default: dist]"),
        ]

    boolean_options = ['use-defaults', 'prune',
                       'manifest-only', 'force-manifest',
                       'keep-temp']

    help_options = [
        ('help-formats', None,
         "list available distribution formats", show_formats),
        ]

    negative_opt = {'no-defaults': 'use-defaults',
                    'no-prune': 'prune' }

    default_format = { 'posix': 'gztar',
                       'java': 'gztar',
                       'nt': 'zip' }

    def initialize_options (self):
        # 'template' and 'manifest' are, respectively, the names of
        # the manifest template and manifest file.
        self.template = None
        self.manifest = None

        # 'use_defaults': if true, we will include the default file set
        # in the manifest
        self.use_defaults = 1
        self.prune = 1

        self.manifest_only = 0
        self.force_manifest = 0

        self.formats = None
        self.keep_temp = 0
        self.dist_dir = None

        self.archive_files = None


    def finalize_options (self):
        if self.manifest is None:
            self.manifest = "MANIFEST"
        if self.template is None:
            self.template = "MANIFEST.in"

        self.ensure_string_list('formats')
        if self.formats is None:
            try:
                self.formats = [self.default_format[os.name]]
            except KeyError:
                raise DistutilsPlatformError, \
                      "don't know how to create source distributions " + \
                      "on platform %s" % os.name

        bad_format = archive_util.check_archive_formats(self.formats)
        if bad_format:
            raise DistutilsOptionError, \
                  "unknown archive format '%s'" % bad_format

        if self.dist_dir is None:
            self.dist_dir = "dist"


    def run (self):

        # 'filelist' contains the list of files that will make up the
        # manifest
        self.filelist = FileList()

        # Ensure that all required meta-data is given; warn if not (but
        # don't die, it's not *that* serious!)
        self.check_metadata()

        # Do whatever it takes to get the list of files to process
        # (process the manifest template, read an existing manifest,
        # whatever).  File list is accumulated in 'self.filelist'.
        self.get_file_list()

        # If user just wanted us to regenerate the manifest, stop now.
        if self.manifest_only:
            return

        # Otherwise, go ahead and create the source distribution tarball,
        # or zipfile, or whatever.
        self.make_distribution()


    def check_metadata (self):
        """Ensure that all required elements of meta-data (name, version,
        URL, (author and author_email) or (maintainer and
        maintainer_email)) are supplied by the Distribution object; warn if
        any are missing.
        """
        metadata = self.distribution.metadata

        missing = []
        for attr in ('name', 'version', 'url'):
            if not (hasattr(metadata, attr) and getattr(metadata, attr)):
                missing.append(attr)

        if missing:
            self.warn("missing required meta-data: " +
                      string.join(missing, ", "))

        if metadata.author:
            if not metadata.author_email:
                self.warn("missing meta-data: if 'author' supplied, " +
                          "'author_email' must be supplied too")
        elif metadata.maintainer:
            if not metadata.maintainer_email:
                self.warn("missing meta-data: if 'maintainer' supplied, " +
                          "'maintainer_email' must be supplied too")
        else:
            self.warn("missing meta-data: either (author and author_email) " +
                      "or (maintainer and maintainer_email) " +
                      "must be supplied")

    # check_metadata ()


    def get_file_list (self):
        """Figure out the list of files to include in the source
        distribution, and put it in 'self.filelist'.  This might involve
        reading the manifest template (and writing the manifest), or just
        reading the manifest, or just using the default file set -- it all
        depends on the user's options and the state of the filesystem.
        """

        # If we have a manifest template, see if it's newer than the
        # manifest; if so, we'll regenerate the manifest.
        template_exists = os.path.isfile(self.template)
        if template_exists:
            template_newer = dep_util.newer(self.template, self.manifest)

        # The contents of the manifest file almost certainly depend on the
        # setup script as well as the manifest template -- so if the setup
        # script is newer than the manifest, we'll regenerate the manifest
        # from the template.  (Well, not quite: if we already have a
        # manifest, but there's no template -- which will happen if the
        # developer elects to generate a manifest some other way -- then we
        # can't regenerate the manifest, so we don't.)
        self.debug_print("checking if %s newer than %s" %
                         (self.distribution.script_name, self.manifest))
        setup_newer = dep_util.newer(self.distribution.script_name,
                                     self.manifest)

        # cases:
        #   1) no manifest, template exists: generate manifest
        #      (covered by 2a: no manifest == template newer)
        #   2) manifest & template exist:
        #      2a) template or setup script newer than manifest:
        #          regenerate manifest
        #      2b) manifest newer than both:
        #          do nothing (unless --force or --manifest-only)
        #   3) manifest exists, no template:
        #      do nothing (unless --force or --manifest-only)
        #   4) no manifest, no template: generate w/ warning ("defaults only")

        manifest_outofdate = (template_exists and
                              (template_newer or setup_newer))
        force_regen = self.force_manifest or self.manifest_only
        manifest_exists = os.path.isfile(self.manifest)
        neither_exists = (not template_exists and not manifest_exists)

        # Regenerate the manifest if necessary (or if explicitly told to)
        if manifest_outofdate or neither_exists or force_regen:
            if not template_exists:
                self.warn(("manifest template '%s' does not exist " +
                           "(using default file list)") %
                          self.template)
            self.filelist.findall()

            if self.use_defaults:
                self.add_defaults()
            if template_exists:
                self.read_template()
            if self.prune:
                self.prune_file_list()

            self.filelist.sort()
            self.filelist.remove_duplicates()
            self.write_manifest()

        # Don't regenerate the manifest, just read it in.
        else:
            self.read_manifest()

    # get_file_list ()


    def add_defaults (self):
        """Add all the default files to self.filelist:
          - README or README.txt
          - setup.py
          - test/test*.py
          - all pure Python modules mentioned in setup script
          - all C sources listed as part of extensions or C libraries
            in the setup script (doesn't catch C headers!)
        Warns if (README or README.txt) or setup.py are missing; everything
        else is optional.
        """

        standards = [('README', 'README.txt'), self.distribution.script_name]
        for fn in standards:
            if type(fn) is TupleType:
                alts = fn
                got_it = 0
                for fn in alts:
                    if os.path.exists(fn):
                        got_it = 1
                        self.filelist.append(fn)
                        break

                if not got_it:
                    self.warn("standard file not found: should have one of " +
                              string.join(alts, ', '))
            else:
                if os.path.exists(fn):
                    self.filelist.append(fn)
                else:
                    self.warn("standard file '%s' not found" % fn)

        optional = ['test/test*.py', 'setup.cfg']
        for pattern in optional:
            files = filter(os.path.isfile, glob(pattern))
            if files:
                self.filelist.extend(files)

        if self.distribution.has_pure_modules():
            build_py = self.get_finalized_command('build_py')
            self.filelist.extend(build_py.get_source_files())

        if self.distribution.has_ext_modules():
            build_ext = self.get_finalized_command('build_ext')
            self.filelist.extend(build_ext.get_source_files())

        if self.distribution.has_c_libraries():
            build_clib = self.get_finalized_command('build_clib')
            self.filelist.extend(build_clib.get_source_files())

        if self.distribution.has_scripts():
            build_scripts = self.get_finalized_command('build_scripts')
            self.filelist.extend(build_scripts.get_source_files())

    # add_defaults ()


    def read_template (self):
        """Read and parse manifest template file named by self.template.

        (usually "MANIFEST.in") The parsing and processing is done by
        'self.filelist', which updates itself accordingly.
        """
        log.info("reading manifest template '%s'", self.template)
        template = TextFile(self.template,
                            strip_comments=1,
                            skip_blanks=1,
                            join_lines=1,
                            lstrip_ws=1,
                            rstrip_ws=1,
                            collapse_join=1)

        while 1:
            line = template.readline()
            if line is None:            # end of file
                break

            try:
                self.filelist.process_template_line(line)
            except DistutilsTemplateError, msg:
                self.warn("%s, line %d: %s" % (template.filename,
                                               template.current_line,
                                               msg))

    # read_template ()


    def prune_file_list (self):
        """Prune off branches that might slip into the file list as created
        by 'read_template()', but really don't belong there:
          * the build tree (typically "build")
          * the release tree itself (only an issue if we ran "sdist"
            previously with --keep-temp, or it aborted)
          * any RCS, CVS, .svn, .hg, .git, .bzr, _darcs directories
        """
        build = self.get_finalized_command('build')
        base_dir = self.distribution.get_fullname()

        self.filelist.exclude_pattern(None, prefix=build.build_base)
        self.filelist.exclude_pattern(None, prefix=base_dir)
        self.filelist.exclude_pattern(r'(^|/)(RCS|CVS|\.svn|\.hg|\.git|\.bzr|_darcs)/.*', is_regex=1)


    def write_manifest (self):
        """Write the file list in 'self.filelist' (presumably as filled in
        by 'add_defaults()' and 'read_template()') to the manifest file
        named by 'self.manifest'.
        """
        self.execute(file_util.write_file,
                     (self.manifest, self.filelist.files),
                     "writing manifest file '%s'" % self.manifest)

    # write_manifest ()


    def read_manifest (self):
        """Read the manifest file (named by 'self.manifest') and use it to
        fill in 'self.filelist', the list of files to include in the source
        distribution.
        """
        log.info("reading manifest file '%s'", self.manifest)
        manifest = open(self.manifest)
        try:
            while 1:
                line = manifest.readline()
                if line == '':              # end of file
                    break
                if line[-1] == '\n':
                    line = line[0:-1]
                self.filelist.append(line)
        finally:
            manifest.close()

    # read_manifest ()


    def make_release_tree (self, base_dir, files):
        """Create the directory tree that will become the source
        distribution archive.  All directories implied by the filenames in
        'files' are created under 'base_dir', and then we hard link or copy
        (if hard linking is unavailable) those files into place.
        Essentially, this duplicates the developer's source tree, but in a
        directory named after the distribution, containing only the files
        to be distributed.
        """
        # Create all the directories under 'base_dir' necessary to
        # put 'files' there; the 'mkpath()' is just so we don't die
        # if the manifest happens to be empty.
        self.mkpath(base_dir)
        dir_util.create_tree(base_dir, files, dry_run=self.dry_run)

        # And walk over the list of files, either making a hard link (if
        # os.link exists) to each one that doesn't already exist in its
        # corresponding location under 'base_dir', or copying each file
        # that's out-of-date in 'base_dir'.  (Usually, all files will be
        # out-of-date, because by default we blow away 'base_dir' when
        # we're done making the distribution archives.)

        if hasattr(os, 'link'):        # can make hard links on this system
            link = 'hard'
            msg = "making hard links in %s..." % base_dir
        else:                           # nope, have to copy
            link = None
            msg = "copying files to %s..." % base_dir

        if not files:
            log.warn("no files to distribute -- empty manifest?")
        else:
            log.info(msg)
        for file in files:
            if not os.path.isfile(file):
                log.warn("'%s' not a regular file -- skipping" % file)
            else:
                dest = os.path.join(base_dir, file)
                self.copy_file(file, dest, link=link)

        self.distribution.metadata.write_pkg_info(base_dir)

    # make_release_tree ()

    def make_distribution (self):
        """Create the source distribution(s).  First, we create the release
        tree with 'make_release_tree()'; then, we create all required
        archive files (according to 'self.formats') from the release tree.
        Finally, we clean up by blowing away the release tree (unless
        'self.keep_temp' is true).  The list of archive files created is
        stored so it can be retrieved later by 'get_archive_files()'.
        """
        # Don't warn about missing meta-data here -- should be (and is!)
        # done elsewhere.
        base_dir = self.distribution.get_fullname()
        base_name = os.path.join(self.dist_dir, base_dir)

        self.make_release_tree(base_dir, self.filelist.files)
        archive_files = []              # remember names of files we create
        for fmt in self.formats:
            file = self.make_archive(base_name, fmt, base_dir=base_dir)
            archive_files.append(file)
            self.distribution.dist_files.append(('sdist', '', file))

        self.archive_files = archive_files

        if not self.keep_temp:
            dir_util.remove_tree(base_dir, dry_run=self.dry_run)

    def get_archive_files (self):
        """Return the list of archive files created when the command
        was run, or None if the command hasn't run yet.
        """
        return self.archive_files

# class sdist
