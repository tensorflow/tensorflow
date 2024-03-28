from functools import partial
from glob import glob
from distutils.util import convert_path
import distutils.command.build_py as orig
import os
import fnmatch
import textwrap
import io
import distutils.errors
import itertools
import stat
import warnings
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from setuptools._deprecation_warning import SetuptoolsDeprecationWarning
from setuptools.extern.more_itertools import unique_everseen


def make_writable(target):
    os.chmod(target, os.stat(target).st_mode | stat.S_IWRITE)


class build_py(orig.build_py):
    """Enhanced 'build_py' command that includes data files with packages

    The data files are specified via a 'package_data' argument to 'setup()'.
    See 'setuptools.dist.Distribution' for more details.

    Also, this version of the 'build_py' command allows you to specify both
    'py_modules' and 'packages' in the same setup operation.
    """
    editable_mode: bool = False
    existing_egg_info_dir: Optional[str] = None  #: Private API, internal use only.

    def finalize_options(self):
        orig.build_py.finalize_options(self)
        self.package_data = self.distribution.package_data
        self.exclude_package_data = self.distribution.exclude_package_data or {}
        if 'data_files' in self.__dict__:
            del self.__dict__['data_files']
        self.__updated_files = []

    def copy_file(self, infile, outfile, preserve_mode=1, preserve_times=1,
                  link=None, level=1):
        # Overwrite base class to allow using links
        if link:
            infile = str(Path(infile).resolve())
            outfile = str(Path(outfile).resolve())
        return super().copy_file(infile, outfile, preserve_mode, preserve_times,
                                 link, level)

    def run(self):
        """Build modules, packages, and copy data files to build directory"""
        if not (self.py_modules or self.packages) or self.editable_mode:
            return

        if self.py_modules:
            self.build_modules()

        if self.packages:
            self.build_packages()
            self.build_package_data()

        # Only compile actual .py files, using our base class' idea of what our
        # output files are.
        self.byte_compile(orig.build_py.get_outputs(self, include_bytecode=0))

    def __getattr__(self, attr):
        "lazily compute data files"
        if attr == 'data_files':
            self.data_files = self._get_data_files()
            return self.data_files
        return orig.build_py.__getattr__(self, attr)

    def build_module(self, module, module_file, package):
        outfile, copied = orig.build_py.build_module(self, module, module_file, package)
        if copied:
            self.__updated_files.append(outfile)
        return outfile, copied

    def _get_data_files(self):
        """Generate list of '(package,src_dir,build_dir,filenames)' tuples"""
        self.analyze_manifest()
        return list(map(self._get_pkg_data_files, self.packages or ()))

    def get_data_files_without_manifest(self):
        """
        Generate list of ``(package,src_dir,build_dir,filenames)`` tuples,
        but without triggering any attempt to analyze or build the manifest.
        """
        # Prevent eventual errors from unset `manifest_files`
        # (that would otherwise be set by `analyze_manifest`)
        self.__dict__.setdefault('manifest_files', {})
        return list(map(self._get_pkg_data_files, self.packages or ()))

    def _get_pkg_data_files(self, package):
        # Locate package source directory
        src_dir = self.get_package_dir(package)

        # Compute package build directory
        build_dir = os.path.join(*([self.build_lib] + package.split('.')))

        # Strip directory from globbed filenames
        filenames = [
            os.path.relpath(file, src_dir)
            for file in self.find_data_files(package, src_dir)
        ]
        return package, src_dir, build_dir, filenames

    def find_data_files(self, package, src_dir):
        """Return filenames for package's data files in 'src_dir'"""
        patterns = self._get_platform_patterns(
            self.package_data,
            package,
            src_dir,
        )
        globs_expanded = map(partial(glob, recursive=True), patterns)
        # flatten the expanded globs into an iterable of matches
        globs_matches = itertools.chain.from_iterable(globs_expanded)
        glob_files = filter(os.path.isfile, globs_matches)
        files = itertools.chain(
            self.manifest_files.get(package, []),
            glob_files,
        )
        return self.exclude_data_files(package, src_dir, files)

    def get_outputs(self, include_bytecode=1) -> List[str]:
        """See :class:`setuptools.commands.build.SubCommand`"""
        if self.editable_mode:
            return list(self.get_output_mapping().keys())
        return super().get_outputs(include_bytecode)

    def get_output_mapping(self) -> Dict[str, str]:
        """See :class:`setuptools.commands.build.SubCommand`"""
        mapping = itertools.chain(
            self._get_package_data_output_mapping(),
            self._get_module_mapping(),
        )
        return dict(sorted(mapping, key=lambda x: x[0]))

    def _get_module_mapping(self) -> Iterator[Tuple[str, str]]:
        """Iterate over all modules producing (dest, src) pairs."""
        for (package, module, module_file) in self.find_all_modules():
            package = package.split('.')
            filename = self.get_module_outfile(self.build_lib, package, module)
            yield (filename, module_file)

    def _get_package_data_output_mapping(self) -> Iterator[Tuple[str, str]]:
        """Iterate over package data producing (dest, src) pairs."""
        for package, src_dir, build_dir, filenames in self.data_files:
            for filename in filenames:
                target = os.path.join(build_dir, filename)
                srcfile = os.path.join(src_dir, filename)
                yield (target, srcfile)

    def build_package_data(self):
        """Copy data files into build directory"""
        for target, srcfile in self._get_package_data_output_mapping():
            self.mkpath(os.path.dirname(target))
            _outf, _copied = self.copy_file(srcfile, target)
            make_writable(target)

    def analyze_manifest(self):
        self.manifest_files = mf = {}
        if not self.distribution.include_package_data:
            return
        src_dirs = {}
        for package in self.packages or ():
            # Locate package source directory
            src_dirs[assert_relative(self.get_package_dir(package))] = package

        if (
            getattr(self, 'existing_egg_info_dir', None)
            and Path(self.existing_egg_info_dir, "SOURCES.txt").exists()
        ):
            egg_info_dir = self.existing_egg_info_dir
            manifest = Path(egg_info_dir, "SOURCES.txt")
            files = manifest.read_text(encoding="utf-8").splitlines()
        else:
            self.run_command('egg_info')
            ei_cmd = self.get_finalized_command('egg_info')
            egg_info_dir = ei_cmd.egg_info
            files = ei_cmd.filelist.files

        check = _IncludePackageDataAbuse()
        for path in self._filter_build_files(files, egg_info_dir):
            d, f = os.path.split(assert_relative(path))
            prev = None
            oldf = f
            while d and d != prev and d not in src_dirs:
                prev = d
                d, df = os.path.split(d)
                f = os.path.join(df, f)
            if d in src_dirs:
                if f == oldf:
                    if check.is_module(f):
                        continue  # it's a module, not data
                else:
                    importable = check.importable_subpackage(src_dirs[d], f)
                    if importable:
                        check.warn(importable)
                mf.setdefault(src_dirs[d], []).append(path)

    def _filter_build_files(self, files: Iterable[str], egg_info: str) -> Iterator[str]:
        """
        ``build_meta`` may try to create egg_info outside of the project directory,
        and this can be problematic for certain plugins (reported in issue #3500).

        Extensions might also include between their sources files created on the
        ``build_lib`` and ``build_temp`` directories.

        This function should filter this case of invalid files out.
        """
        build = self.get_finalized_command("build")
        build_dirs = (egg_info, self.build_lib, build.build_temp, build.build_base)
        norm_dirs = [os.path.normpath(p) for p in build_dirs if p]

        for file in files:
            norm_path = os.path.normpath(file)
            if not os.path.isabs(file) or all(d not in norm_path for d in norm_dirs):
                yield file

    def get_data_files(self):
        pass  # Lazily compute data files in _get_data_files() function.

    def check_package(self, package, package_dir):
        """Check namespace packages' __init__ for declare_namespace"""
        try:
            return self.packages_checked[package]
        except KeyError:
            pass

        init_py = orig.build_py.check_package(self, package, package_dir)
        self.packages_checked[package] = init_py

        if not init_py or not self.distribution.namespace_packages:
            return init_py

        for pkg in self.distribution.namespace_packages:
            if pkg == package or pkg.startswith(package + '.'):
                break
        else:
            return init_py

        with io.open(init_py, 'rb') as f:
            contents = f.read()
        if b'declare_namespace' not in contents:
            raise distutils.errors.DistutilsError(
                "Namespace package problem: %s is a namespace package, but "
                "its\n__init__.py does not call declare_namespace()! Please "
                'fix it.\n(See the setuptools manual under '
                '"Namespace Packages" for details.)\n"' % (package,)
            )
        return init_py

    def initialize_options(self):
        self.packages_checked = {}
        orig.build_py.initialize_options(self)
        self.editable_mode = False
        self.existing_egg_info_dir = None

    def get_package_dir(self, package):
        res = orig.build_py.get_package_dir(self, package)
        if self.distribution.src_root is not None:
            return os.path.join(self.distribution.src_root, res)
        return res

    def exclude_data_files(self, package, src_dir, files):
        """Filter filenames for package's data files in 'src_dir'"""
        files = list(files)
        patterns = self._get_platform_patterns(
            self.exclude_package_data,
            package,
            src_dir,
        )
        match_groups = (fnmatch.filter(files, pattern) for pattern in patterns)
        # flatten the groups of matches into an iterable of matches
        matches = itertools.chain.from_iterable(match_groups)
        bad = set(matches)
        keepers = (fn for fn in files if fn not in bad)
        # ditch dupes
        return list(unique_everseen(keepers))

    @staticmethod
    def _get_platform_patterns(spec, package, src_dir):
        """
        yield platform-specific path patterns (suitable for glob
        or fn_match) from a glob-based spec (such as
        self.package_data or self.exclude_package_data)
        matching package in src_dir.
        """
        raw_patterns = itertools.chain(
            spec.get('', []),
            spec.get(package, []),
        )
        return (
            # Each pattern has to be converted to a platform-specific path
            os.path.join(src_dir, convert_path(pattern))
            for pattern in raw_patterns
        )


def assert_relative(path):
    if not os.path.isabs(path):
        return path
    from distutils.errors import DistutilsSetupError

    msg = (
        textwrap.dedent(
            """
        Error: setup script specifies an absolute path:

            %s

        setup() arguments must *always* be /-separated paths relative to the
        setup.py directory, *never* absolute paths.
        """
        ).lstrip()
        % path
    )
    raise DistutilsSetupError(msg)


class _IncludePackageDataAbuse:
    """Inform users that package or module is included as 'data file'"""

    MESSAGE = """\
    Installing {importable!r} as data is deprecated, please list it in `packages`.
    !!\n\n
    ############################
    # Package would be ignored #
    ############################
    Python recognizes {importable!r} as an importable package,
    but it is not listed in the `packages` configuration of setuptools.

    {importable!r} has been automatically added to the distribution only
    because it may contain data files, but this behavior is likely to change
    in future versions of setuptools (and therefore is considered deprecated).

    Please make sure that {importable!r} is included as a package by using
    the `packages` configuration field or the proper discovery methods
    (for example by using `find_namespace_packages(...)`/`find_namespace:`
    instead of `find_packages(...)`/`find:`).

    You can read more about "package discovery" and "data files" on setuptools
    documentation page.
    \n\n!!
    """

    def __init__(self):
        self._already_warned = set()

    def is_module(self, file):
        return file.endswith(".py") and file[:-len(".py")].isidentifier()

    def importable_subpackage(self, parent, file):
        pkg = Path(file).parent
        parts = list(itertools.takewhile(str.isidentifier, pkg.parts))
        if parts:
            return ".".join([parent, *parts])
        return None

    def warn(self, importable):
        if importable not in self._already_warned:
            msg = textwrap.dedent(self.MESSAGE).format(importable=importable)
            warnings.warn(msg, SetuptoolsDeprecationWarning, stacklevel=2)
            self._already_warned.add(importable)
