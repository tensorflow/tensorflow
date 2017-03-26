from distutils import log
import distutils.command.sdist as orig
import os
import sys
import io
import contextlib

import six

from .py36compat import sdist_add_defaults

import pkg_resources

_default_revctrl = list


def walk_revctrl(dirname=''):
    """Find all files under revision control"""
    for ep in pkg_resources.iter_entry_points('setuptools.file_finders'):
        for item in ep.load()(dirname):
            yield item


class sdist(sdist_add_defaults, orig.sdist):
    """Smart sdist that finds anything supported by revision control"""

    user_options = [
        ('formats=', None,
         "formats for source distribution (comma-separated list)"),
        ('keep-temp', 'k',
         "keep the distribution tree around after creating " +
         "archive file(s)"),
        ('dist-dir=', 'd',
         "directory to put the source distribution archive(s) in "
         "[default: dist]"),
    ]

    negative_opt = {}

    READMES = 'README', 'README.rst', 'README.txt'

    def run(self):
        self.run_command('egg_info')
        ei_cmd = self.get_finalized_command('egg_info')
        self.filelist = ei_cmd.filelist
        self.filelist.append(os.path.join(ei_cmd.egg_info, 'SOURCES.txt'))
        self.check_readme()

        # Run sub commands
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)

        # Call check_metadata only if no 'check' command
        # (distutils <= 2.6)
        import distutils.command

        if 'check' not in distutils.command.__all__:
            self.check_metadata()

        self.make_distribution()

        dist_files = getattr(self.distribution, 'dist_files', [])
        for file in self.archive_files:
            data = ('sdist', '', file)
            if data not in dist_files:
                dist_files.append(data)

    def initialize_options(self):
        orig.sdist.initialize_options(self)

        self._default_to_gztar()

    def _default_to_gztar(self):
        # only needed on Python prior to 3.6.
        if sys.version_info >= (3, 6, 0, 'beta', 1):
            return
        self.formats = ['gztar']

    def make_distribution(self):
        """
        Workaround for #516
        """
        with self._remove_os_link():
            orig.sdist.make_distribution(self)

    @staticmethod
    @contextlib.contextmanager
    def _remove_os_link():
        """
        In a context, remove and restore os.link if it exists
        """

        class NoValue:
            pass

        orig_val = getattr(os, 'link', NoValue)
        try:
            del os.link
        except Exception:
            pass
        try:
            yield
        finally:
            if orig_val is not NoValue:
                setattr(os, 'link', orig_val)

    def __read_template_hack(self):
        # This grody hack closes the template file (MANIFEST.in) if an
        #  exception occurs during read_template.
        # Doing so prevents an error when easy_install attempts to delete the
        #  file.
        try:
            orig.sdist.read_template(self)
        except Exception:
            _, _, tb = sys.exc_info()
            tb.tb_next.tb_frame.f_locals['template'].close()
            raise

    # Beginning with Python 2.7.2, 3.1.4, and 3.2.1, this leaky file handle
    #  has been fixed, so only override the method if we're using an earlier
    #  Python.
    has_leaky_handle = (
        sys.version_info < (2, 7, 2)
        or (3, 0) <= sys.version_info < (3, 1, 4)
        or (3, 2) <= sys.version_info < (3, 2, 1)
    )
    if has_leaky_handle:
        read_template = __read_template_hack

    def _add_defaults_python(self):
        """getting python files"""
        if self.distribution.has_pure_modules():
            build_py = self.get_finalized_command('build_py')
            self.filelist.extend(build_py.get_source_files())
            # This functionality is incompatible with include_package_data, and
            # will in fact create an infinite recursion if include_package_data
            # is True.  Use of include_package_data will imply that
            # distutils-style automatic handling of package_data is disabled
            if not self.distribution.include_package_data:
                for _, src_dir, _, filenames in build_py.data_files:
                    self.filelist.extend([os.path.join(src_dir, filename)
                                          for filename in filenames])

    def _add_defaults_data_files(self):
        try:
            if six.PY2:
                sdist_add_defaults._add_defaults_data_files(self)
            else:
                super()._add_defaults_data_files()
        except TypeError:
            log.warn("data_files contains unexpected objects")

    def check_readme(self):
        for f in self.READMES:
            if os.path.exists(f):
                return
        else:
            self.warn(
                "standard file not found: should have one of " +
                ', '.join(self.READMES)
            )

    def make_release_tree(self, base_dir, files):
        orig.sdist.make_release_tree(self, base_dir, files)

        # Save any egg_info command line options used to create this sdist
        dest = os.path.join(base_dir, 'setup.cfg')
        if hasattr(os, 'link') and os.path.exists(dest):
            # unlink and re-copy, since it might be hard-linked, and
            # we don't want to change the source version
            os.unlink(dest)
            self.copy_file('setup.cfg', dest)

        self.get_finalized_command('egg_info').save_version_info(dest)

    def _manifest_is_not_generated(self):
        # check for special comment used in 2.7.1 and higher
        if not os.path.isfile(self.manifest):
            return False

        with io.open(self.manifest, 'rb') as fp:
            first_line = fp.readline()
        return (first_line !=
                '# file GENERATED by distutils, do NOT edit\n'.encode())

    def read_manifest(self):
        """Read the manifest file (named by 'self.manifest') and use it to
        fill in 'self.filelist', the list of files to include in the source
        distribution.
        """
        log.info("reading manifest file '%s'", self.manifest)
        manifest = open(self.manifest, 'rb')
        for line in manifest:
            # The manifest must contain UTF-8. See #303.
            if six.PY3:
                try:
                    line = line.decode('UTF-8')
                except UnicodeDecodeError:
                    log.warn("%r not UTF-8 decodable -- skipping" % line)
                    continue
            # ignore comments and blank lines
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            self.filelist.append(line)
        manifest.close()
