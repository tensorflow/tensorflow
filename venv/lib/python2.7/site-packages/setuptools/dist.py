__all__ = ['Distribution']

import re
import os
import warnings
import numbers
import distutils.log
import distutils.core
import distutils.cmd
import distutils.dist
from distutils.errors import (DistutilsOptionError, DistutilsPlatformError,
    DistutilsSetupError)
from distutils.util import rfc822_escape

import six
from six.moves import map
import packaging

from setuptools.depends import Require
from setuptools import windows_support
from setuptools.monkey import get_unpatched
from setuptools.config import parse_configuration
import pkg_resources
from .py36compat import Distribution_parse_config_files


def _get_unpatched(cls):
    warnings.warn("Do not call this function", DeprecationWarning)
    return get_unpatched(cls)


# Based on Python 3.5 version
def write_pkg_file(self, file):
    """Write the PKG-INFO format data to a file object.
    """
    version = '1.0'
    if (self.provides or self.requires or self.obsoletes or
            self.classifiers or self.download_url):
        version = '1.1'
    # Setuptools specific for PEP 345
    if hasattr(self, 'python_requires'):
        version = '1.2'

    file.write('Metadata-Version: %s\n' % version)
    file.write('Name: %s\n' % self.get_name())
    file.write('Version: %s\n' % self.get_version())
    file.write('Summary: %s\n' % self.get_description())
    file.write('Home-page: %s\n' % self.get_url())
    file.write('Author: %s\n' % self.get_contact())
    file.write('Author-email: %s\n' % self.get_contact_email())
    file.write('License: %s\n' % self.get_license())
    if self.download_url:
        file.write('Download-URL: %s\n' % self.download_url)

    long_desc = rfc822_escape(self.get_long_description())
    file.write('Description: %s\n' % long_desc)

    keywords = ','.join(self.get_keywords())
    if keywords:
        file.write('Keywords: %s\n' % keywords)

    self._write_list(file, 'Platform', self.get_platforms())
    self._write_list(file, 'Classifier', self.get_classifiers())

    # PEP 314
    self._write_list(file, 'Requires', self.get_requires())
    self._write_list(file, 'Provides', self.get_provides())
    self._write_list(file, 'Obsoletes', self.get_obsoletes())

    # Setuptools specific for PEP 345
    if hasattr(self, 'python_requires'):
        file.write('Requires-Python: %s\n' % self.python_requires)


# from Python 3.4
def write_pkg_info(self, base_dir):
    """Write the PKG-INFO file into the release tree.
    """
    with open(os.path.join(base_dir, 'PKG-INFO'), 'w',
              encoding='UTF-8') as pkg_info:
        self.write_pkg_file(pkg_info)


sequence = tuple, list


def check_importable(dist, attr, value):
    try:
        ep = pkg_resources.EntryPoint.parse('x=' + value)
        assert not ep.extras
    except (TypeError, ValueError, AttributeError, AssertionError):
        raise DistutilsSetupError(
            "%r must be importable 'module:attrs' string (got %r)"
            % (attr, value)
        )


def assert_string_list(dist, attr, value):
    """Verify that value is a string list or None"""
    try:
        assert ''.join(value) != value
    except (TypeError, ValueError, AttributeError, AssertionError):
        raise DistutilsSetupError(
            "%r must be a list of strings (got %r)" % (attr, value)
        )


def check_nsp(dist, attr, value):
    """Verify that namespace packages are valid"""
    ns_packages = value
    assert_string_list(dist, attr, ns_packages)
    for nsp in ns_packages:
        if not dist.has_contents_for(nsp):
            raise DistutilsSetupError(
                "Distribution contains no modules or packages for " +
                "namespace package %r" % nsp
            )
        parent, sep, child = nsp.rpartition('.')
        if parent and parent not in ns_packages:
            distutils.log.warn(
                "WARNING: %r is declared as a package namespace, but %r"
                " is not: please correct this in setup.py", nsp, parent
            )


def check_extras(dist, attr, value):
    """Verify that extras_require mapping is valid"""
    try:
        for k, v in value.items():
            if ':' in k:
                k, m = k.split(':', 1)
                if pkg_resources.invalid_marker(m):
                    raise DistutilsSetupError("Invalid environment marker: " + m)
            list(pkg_resources.parse_requirements(v))
    except (TypeError, ValueError, AttributeError):
        raise DistutilsSetupError(
            "'extras_require' must be a dictionary whose values are "
            "strings or lists of strings containing valid project/version "
            "requirement specifiers."
        )


def assert_bool(dist, attr, value):
    """Verify that value is True, False, 0, or 1"""
    if bool(value) != value:
        tmpl = "{attr!r} must be a boolean value (got {value!r})"
        raise DistutilsSetupError(tmpl.format(attr=attr, value=value))


def check_requirements(dist, attr, value):
    """Verify that install_requires is a valid requirements list"""
    try:
        list(pkg_resources.parse_requirements(value))
    except (TypeError, ValueError) as error:
        tmpl = (
            "{attr!r} must be a string or list of strings "
            "containing valid project/version requirement specifiers; {error}"
        )
        raise DistutilsSetupError(tmpl.format(attr=attr, error=error))


def check_specifier(dist, attr, value):
    """Verify that value is a valid version specifier"""
    try:
        packaging.specifiers.SpecifierSet(value)
    except packaging.specifiers.InvalidSpecifier as error:
        tmpl = (
            "{attr!r} must be a string or list of strings "
            "containing valid version specifiers; {error}"
        )
        raise DistutilsSetupError(tmpl.format(attr=attr, error=error))


def check_entry_points(dist, attr, value):
    """Verify that entry_points map is parseable"""
    try:
        pkg_resources.EntryPoint.parse_map(value)
    except ValueError as e:
        raise DistutilsSetupError(e)


def check_test_suite(dist, attr, value):
    if not isinstance(value, six.string_types):
        raise DistutilsSetupError("test_suite must be a string")


def check_package_data(dist, attr, value):
    """Verify that value is a dictionary of package names to glob lists"""
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                break
            try:
                iter(v)
            except TypeError:
                break
        else:
            return
    raise DistutilsSetupError(
        attr + " must be a dictionary mapping package names to lists of "
        "wildcard patterns"
    )


def check_packages(dist, attr, value):
    for pkgname in value:
        if not re.match(r'\w+(\.\w+)*', pkgname):
            distutils.log.warn(
                "WARNING: %r not a valid package name; please use only "
                ".-separated package names in setup.py", pkgname
            )


_Distribution = get_unpatched(distutils.core.Distribution)


class Distribution(Distribution_parse_config_files, _Distribution):
    """Distribution with support for features, tests, and package data

    This is an enhanced version of 'distutils.dist.Distribution' that
    effectively adds the following new optional keyword arguments to 'setup()':

     'install_requires' -- a string or sequence of strings specifying project
        versions that the distribution requires when installed, in the format
        used by 'pkg_resources.require()'.  They will be installed
        automatically when the package is installed.  If you wish to use
        packages that are not available in PyPI, or want to give your users an
        alternate download location, you can add a 'find_links' option to the
        '[easy_install]' section of your project's 'setup.cfg' file, and then
        setuptools will scan the listed web pages for links that satisfy the
        requirements.

     'extras_require' -- a dictionary mapping names of optional "extras" to the
        additional requirement(s) that using those extras incurs. For example,
        this::

            extras_require = dict(reST = ["docutils>=0.3", "reSTedit"])

        indicates that the distribution can optionally provide an extra
        capability called "reST", but it can only be used if docutils and
        reSTedit are installed.  If the user installs your package using
        EasyInstall and requests one of your extras, the corresponding
        additional requirements will be installed if needed.

     'features' **deprecated** -- a dictionary mapping option names to
        'setuptools.Feature'
        objects.  Features are a portion of the distribution that can be
        included or excluded based on user options, inter-feature dependencies,
        and availability on the current system.  Excluded features are omitted
        from all setup commands, including source and binary distributions, so
        you can create multiple distributions from the same source tree.
        Feature names should be valid Python identifiers, except that they may
        contain the '-' (minus) sign.  Features can be included or excluded
        via the command line options '--with-X' and '--without-X', where 'X' is
        the name of the feature.  Whether a feature is included by default, and
        whether you are allowed to control this from the command line, is
        determined by the Feature object.  See the 'Feature' class for more
        information.

     'test_suite' -- the name of a test suite to run for the 'test' command.
        If the user runs 'python setup.py test', the package will be installed,
        and the named test suite will be run.  The format is the same as
        would be used on a 'unittest.py' command line.  That is, it is the
        dotted name of an object to import and call to generate a test suite.

     'package_data' -- a dictionary mapping package names to lists of filenames
        or globs to use to find data files contained in the named packages.
        If the dictionary has filenames or globs listed under '""' (the empty
        string), those names will be searched for in every package, in addition
        to any names for the specific package.  Data files found using these
        names/globs will be installed along with the package, in the same
        location as the package.  Note that globs are allowed to reference
        the contents of non-package subdirectories, as long as you use '/' as
        a path separator.  (Globs are automatically converted to
        platform-specific paths at runtime.)

    In addition to these new keywords, this class also has several new methods
    for manipulating the distribution's contents.  For example, the 'include()'
    and 'exclude()' methods can be thought of as in-place add and subtract
    commands that add or remove packages, modules, extensions, and so on from
    the distribution.  They are used by the feature subsystem to configure the
    distribution for the included and excluded features.
    """

    _patched_dist = None

    def patch_missing_pkg_info(self, attrs):
        # Fake up a replacement for the data that would normally come from
        # PKG-INFO, but which might not yet be built if this is a fresh
        # checkout.
        #
        if not attrs or 'name' not in attrs or 'version' not in attrs:
            return
        key = pkg_resources.safe_name(str(attrs['name'])).lower()
        dist = pkg_resources.working_set.by_key.get(key)
        if dist is not None and not dist.has_metadata('PKG-INFO'):
            dist._version = pkg_resources.safe_version(str(attrs['version']))
            self._patched_dist = dist

    def __init__(self, attrs=None):
        have_package_data = hasattr(self, "package_data")
        if not have_package_data:
            self.package_data = {}
        _attrs_dict = attrs or {}
        if 'features' in _attrs_dict or 'require_features' in _attrs_dict:
            Feature.warn_deprecated()
        self.require_features = []
        self.features = {}
        self.dist_files = []
        self.src_root = attrs and attrs.pop("src_root", None)
        self.patch_missing_pkg_info(attrs)
        # Make sure we have any eggs needed to interpret 'attrs'
        if attrs is not None:
            self.dependency_links = attrs.pop('dependency_links', [])
            assert_string_list(self, 'dependency_links', self.dependency_links)
        if attrs and 'setup_requires' in attrs:
            self.fetch_build_eggs(attrs['setup_requires'])
        for ep in pkg_resources.iter_entry_points('distutils.setup_keywords'):
            vars(self).setdefault(ep.name, None)
        _Distribution.__init__(self, attrs)
        if isinstance(self.metadata.version, numbers.Number):
            # Some people apparently take "version number" too literally :)
            self.metadata.version = str(self.metadata.version)

        if self.metadata.version is not None:
            try:
                ver = packaging.version.Version(self.metadata.version)
                normalized_version = str(ver)
                if self.metadata.version != normalized_version:
                    warnings.warn(
                        "Normalizing '%s' to '%s'" % (
                            self.metadata.version,
                            normalized_version,
                        )
                    )
                    self.metadata.version = normalized_version
            except (packaging.version.InvalidVersion, TypeError):
                warnings.warn(
                    "The version specified (%r) is an invalid version, this "
                    "may not work as expected with newer versions of "
                    "setuptools, pip, and PyPI. Please see PEP 440 for more "
                    "details." % self.metadata.version
                )
        if getattr(self, 'python_requires', None):
            self.metadata.python_requires = self.python_requires

    def parse_config_files(self, filenames=None):
        """Parses configuration files from various levels
        and loads configuration.

        """
        _Distribution.parse_config_files(self, filenames=filenames)

        parse_configuration(self, self.command_options)

    def parse_command_line(self):
        """Process features after parsing command line options"""
        result = _Distribution.parse_command_line(self)
        if self.features:
            self._finalize_features()
        return result

    def _feature_attrname(self, name):
        """Convert feature name to corresponding option attribute name"""
        return 'with_' + name.replace('-', '_')

    def fetch_build_eggs(self, requires):
        """Resolve pre-setup requirements"""
        resolved_dists = pkg_resources.working_set.resolve(
            pkg_resources.parse_requirements(requires),
            installer=self.fetch_build_egg,
            replace_conflicting=True,
        )
        for dist in resolved_dists:
            pkg_resources.working_set.add(dist, replace=True)
        return resolved_dists

    def finalize_options(self):
        _Distribution.finalize_options(self)
        if self.features:
            self._set_global_opts_from_features()

        for ep in pkg_resources.iter_entry_points('distutils.setup_keywords'):
            value = getattr(self, ep.name, None)
            if value is not None:
                ep.require(installer=self.fetch_build_egg)
                ep.load()(self, ep.name, value)
        if getattr(self, 'convert_2to3_doctests', None):
            # XXX may convert to set here when we can rely on set being builtin
            self.convert_2to3_doctests = [os.path.abspath(p) for p in self.convert_2to3_doctests]
        else:
            self.convert_2to3_doctests = []

    def get_egg_cache_dir(self):
        egg_cache_dir = os.path.join(os.curdir, '.eggs')
        if not os.path.exists(egg_cache_dir):
            os.mkdir(egg_cache_dir)
            windows_support.hide_file(egg_cache_dir)
            readme_txt_filename = os.path.join(egg_cache_dir, 'README.txt')
            with open(readme_txt_filename, 'w') as f:
                f.write('This directory contains eggs that were downloaded '
                        'by setuptools to build, test, and run plug-ins.\n\n')
                f.write('This directory caches those eggs to prevent '
                        'repeated downloads.\n\n')
                f.write('However, it is safe to delete this directory.\n\n')

        return egg_cache_dir

    def fetch_build_egg(self, req):
        """Fetch an egg needed for building"""

        try:
            cmd = self._egg_fetcher
            cmd.package_index.to_scan = []
        except AttributeError:
            from setuptools.command.easy_install import easy_install
            dist = self.__class__({'script_args': ['easy_install']})
            dist.parse_config_files()
            opts = dist.get_option_dict('easy_install')
            keep = (
                'find_links', 'site_dirs', 'index_url', 'optimize',
                'site_dirs', 'allow_hosts'
            )
            for key in list(opts):
                if key not in keep:
                    del opts[key]  # don't use any other settings
            if self.dependency_links:
                links = self.dependency_links[:]
                if 'find_links' in opts:
                    links = opts['find_links'][1].split() + links
                opts['find_links'] = ('setup', links)
            install_dir = self.get_egg_cache_dir()
            cmd = easy_install(
                dist, args=["x"], install_dir=install_dir, exclude_scripts=True,
                always_copy=False, build_directory=None, editable=False,
                upgrade=False, multi_version=True, no_report=True, user=False
            )
            cmd.ensure_finalized()
            self._egg_fetcher = cmd
        return cmd.easy_install(req)

    def _set_global_opts_from_features(self):
        """Add --with-X/--without-X options based on optional features"""

        go = []
        no = self.negative_opt.copy()

        for name, feature in self.features.items():
            self._set_feature(name, None)
            feature.validate(self)

            if feature.optional:
                descr = feature.description
                incdef = ' (default)'
                excdef = ''
                if not feature.include_by_default():
                    excdef, incdef = incdef, excdef

                go.append(('with-' + name, None, 'include ' + descr + incdef))
                go.append(('without-' + name, None, 'exclude ' + descr + excdef))
                no['without-' + name] = 'with-' + name

        self.global_options = self.feature_options = go + self.global_options
        self.negative_opt = self.feature_negopt = no

    def _finalize_features(self):
        """Add/remove features and resolve dependencies between them"""

        # First, flag all the enabled items (and thus their dependencies)
        for name, feature in self.features.items():
            enabled = self.feature_is_included(name)
            if enabled or (enabled is None and feature.include_by_default()):
                feature.include_in(self)
                self._set_feature(name, 1)

        # Then disable the rest, so that off-by-default features don't
        # get flagged as errors when they're required by an enabled feature
        for name, feature in self.features.items():
            if not self.feature_is_included(name):
                feature.exclude_from(self)
                self._set_feature(name, 0)

    def get_command_class(self, command):
        """Pluggable version of get_command_class()"""
        if command in self.cmdclass:
            return self.cmdclass[command]

        for ep in pkg_resources.iter_entry_points('distutils.commands', command):
            ep.require(installer=self.fetch_build_egg)
            self.cmdclass[command] = cmdclass = ep.load()
            return cmdclass
        else:
            return _Distribution.get_command_class(self, command)

    def print_commands(self):
        for ep in pkg_resources.iter_entry_points('distutils.commands'):
            if ep.name not in self.cmdclass:
                # don't require extras as the commands won't be invoked
                cmdclass = ep.resolve()
                self.cmdclass[ep.name] = cmdclass
        return _Distribution.print_commands(self)

    def get_command_list(self):
        for ep in pkg_resources.iter_entry_points('distutils.commands'):
            if ep.name not in self.cmdclass:
                # don't require extras as the commands won't be invoked
                cmdclass = ep.resolve()
                self.cmdclass[ep.name] = cmdclass
        return _Distribution.get_command_list(self)

    def _set_feature(self, name, status):
        """Set feature's inclusion status"""
        setattr(self, self._feature_attrname(name), status)

    def feature_is_included(self, name):
        """Return 1 if feature is included, 0 if excluded, 'None' if unknown"""
        return getattr(self, self._feature_attrname(name))

    def include_feature(self, name):
        """Request inclusion of feature named 'name'"""

        if self.feature_is_included(name) == 0:
            descr = self.features[name].description
            raise DistutilsOptionError(
                descr + " is required, but was excluded or is not available"
            )
        self.features[name].include_in(self)
        self._set_feature(name, 1)

    def include(self, **attrs):
        """Add items to distribution that are named in keyword arguments

        For example, 'dist.exclude(py_modules=["x"])' would add 'x' to
        the distribution's 'py_modules' attribute, if it was not already
        there.

        Currently, this method only supports inclusion for attributes that are
        lists or tuples.  If you need to add support for adding to other
        attributes in this or a subclass, you can add an '_include_X' method,
        where 'X' is the name of the attribute.  The method will be called with
        the value passed to 'include()'.  So, 'dist.include(foo={"bar":"baz"})'
        will try to call 'dist._include_foo({"bar":"baz"})', which can then
        handle whatever special inclusion logic is needed.
        """
        for k, v in attrs.items():
            include = getattr(self, '_include_' + k, None)
            if include:
                include(v)
            else:
                self._include_misc(k, v)

    def exclude_package(self, package):
        """Remove packages, modules, and extensions in named package"""

        pfx = package + '.'
        if self.packages:
            self.packages = [
                p for p in self.packages
                if p != package and not p.startswith(pfx)
            ]

        if self.py_modules:
            self.py_modules = [
                p for p in self.py_modules
                if p != package and not p.startswith(pfx)
            ]

        if self.ext_modules:
            self.ext_modules = [
                p for p in self.ext_modules
                if p.name != package and not p.name.startswith(pfx)
            ]

    def has_contents_for(self, package):
        """Return true if 'exclude_package(package)' would do something"""

        pfx = package + '.'

        for p in self.iter_distribution_names():
            if p == package or p.startswith(pfx):
                return True

    def _exclude_misc(self, name, value):
        """Handle 'exclude()' for list/tuple attrs without a special handler"""
        if not isinstance(value, sequence):
            raise DistutilsSetupError(
                "%s: setting must be a list or tuple (%r)" % (name, value)
            )
        try:
            old = getattr(self, name)
        except AttributeError:
            raise DistutilsSetupError(
                "%s: No such distribution setting" % name
            )
        if old is not None and not isinstance(old, sequence):
            raise DistutilsSetupError(
                name + ": this setting cannot be changed via include/exclude"
            )
        elif old:
            setattr(self, name, [item for item in old if item not in value])

    def _include_misc(self, name, value):
        """Handle 'include()' for list/tuple attrs without a special handler"""

        if not isinstance(value, sequence):
            raise DistutilsSetupError(
                "%s: setting must be a list (%r)" % (name, value)
            )
        try:
            old = getattr(self, name)
        except AttributeError:
            raise DistutilsSetupError(
                "%s: No such distribution setting" % name
            )
        if old is None:
            setattr(self, name, value)
        elif not isinstance(old, sequence):
            raise DistutilsSetupError(
                name + ": this setting cannot be changed via include/exclude"
            )
        else:
            setattr(self, name, old + [item for item in value if item not in old])

    def exclude(self, **attrs):
        """Remove items from distribution that are named in keyword arguments

        For example, 'dist.exclude(py_modules=["x"])' would remove 'x' from
        the distribution's 'py_modules' attribute.  Excluding packages uses
        the 'exclude_package()' method, so all of the package's contained
        packages, modules, and extensions are also excluded.

        Currently, this method only supports exclusion from attributes that are
        lists or tuples.  If you need to add support for excluding from other
        attributes in this or a subclass, you can add an '_exclude_X' method,
        where 'X' is the name of the attribute.  The method will be called with
        the value passed to 'exclude()'.  So, 'dist.exclude(foo={"bar":"baz"})'
        will try to call 'dist._exclude_foo({"bar":"baz"})', which can then
        handle whatever special exclusion logic is needed.
        """
        for k, v in attrs.items():
            exclude = getattr(self, '_exclude_' + k, None)
            if exclude:
                exclude(v)
            else:
                self._exclude_misc(k, v)

    def _exclude_packages(self, packages):
        if not isinstance(packages, sequence):
            raise DistutilsSetupError(
                "packages: setting must be a list or tuple (%r)" % (packages,)
            )
        list(map(self.exclude_package, packages))

    def _parse_command_opts(self, parser, args):
        # Remove --with-X/--without-X options when processing command args
        self.global_options = self.__class__.global_options
        self.negative_opt = self.__class__.negative_opt

        # First, expand any aliases
        command = args[0]
        aliases = self.get_option_dict('aliases')
        while command in aliases:
            src, alias = aliases[command]
            del aliases[command]  # ensure each alias can expand only once!
            import shlex
            args[:1] = shlex.split(alias, True)
            command = args[0]

        nargs = _Distribution._parse_command_opts(self, parser, args)

        # Handle commands that want to consume all remaining arguments
        cmd_class = self.get_command_class(command)
        if getattr(cmd_class, 'command_consumes_arguments', None):
            self.get_option_dict(command)['args'] = ("command line", nargs)
            if nargs is not None:
                return []

        return nargs

    def get_cmdline_options(self):
        """Return a '{cmd: {opt:val}}' map of all command-line options

        Option names are all long, but do not include the leading '--', and
        contain dashes rather than underscores.  If the option doesn't take
        an argument (e.g. '--quiet'), the 'val' is 'None'.

        Note that options provided by config files are intentionally excluded.
        """

        d = {}

        for cmd, opts in self.command_options.items():

            for opt, (src, val) in opts.items():

                if src != "command line":
                    continue

                opt = opt.replace('_', '-')

                if val == 0:
                    cmdobj = self.get_command_obj(cmd)
                    neg_opt = self.negative_opt.copy()
                    neg_opt.update(getattr(cmdobj, 'negative_opt', {}))
                    for neg, pos in neg_opt.items():
                        if pos == opt:
                            opt = neg
                            val = None
                            break
                    else:
                        raise AssertionError("Shouldn't be able to get here")

                elif val == 1:
                    val = None

                d.setdefault(cmd, {})[opt] = val

        return d

    def iter_distribution_names(self):
        """Yield all packages, modules, and extension names in distribution"""

        for pkg in self.packages or ():
            yield pkg

        for module in self.py_modules or ():
            yield module

        for ext in self.ext_modules or ():
            if isinstance(ext, tuple):
                name, buildinfo = ext
            else:
                name = ext.name
            if name.endswith('module'):
                name = name[:-6]
            yield name

    def handle_display_options(self, option_order):
        """If there were any non-global "display-only" options
        (--help-commands or the metadata display options) on the command
        line, display the requested info and return true; else return
        false.
        """
        import sys

        if six.PY2 or self.help_commands:
            return _Distribution.handle_display_options(self, option_order)

        # Stdout may be StringIO (e.g. in tests)
        import io
        if not isinstance(sys.stdout, io.TextIOWrapper):
            return _Distribution.handle_display_options(self, option_order)

        # Don't wrap stdout if utf-8 is already the encoding. Provides
        #  workaround for #334.
        if sys.stdout.encoding.lower() in ('utf-8', 'utf8'):
            return _Distribution.handle_display_options(self, option_order)

        # Print metadata in UTF-8 no matter the platform
        encoding = sys.stdout.encoding
        errors = sys.stdout.errors
        newline = sys.platform != 'win32' and '\n' or None
        line_buffering = sys.stdout.line_buffering

        sys.stdout = io.TextIOWrapper(
            sys.stdout.detach(), 'utf-8', errors, newline, line_buffering)
        try:
            return _Distribution.handle_display_options(self, option_order)
        finally:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.detach(), encoding, errors, newline, line_buffering)


class Feature:
    """
    **deprecated** -- The `Feature` facility was never completely implemented
    or supported, `has reported issues
    <https://github.com/pypa/setuptools/issues/58>`_ and will be removed in
    a future version.

    A subset of the distribution that can be excluded if unneeded/wanted

    Features are created using these keyword arguments:

      'description' -- a short, human readable description of the feature, to
         be used in error messages, and option help messages.

      'standard' -- if true, the feature is included by default if it is
         available on the current system.  Otherwise, the feature is only
         included if requested via a command line '--with-X' option, or if
         another included feature requires it.  The default setting is 'False'.

      'available' -- if true, the feature is available for installation on the
         current system.  The default setting is 'True'.

      'optional' -- if true, the feature's inclusion can be controlled from the
         command line, using the '--with-X' or '--without-X' options.  If
         false, the feature's inclusion status is determined automatically,
         based on 'availabile', 'standard', and whether any other feature
         requires it.  The default setting is 'True'.

      'require_features' -- a string or sequence of strings naming features
         that should also be included if this feature is included.  Defaults to
         empty list.  May also contain 'Require' objects that should be
         added/removed from the distribution.

      'remove' -- a string or list of strings naming packages to be removed
         from the distribution if this feature is *not* included.  If the
         feature *is* included, this argument is ignored.  This argument exists
         to support removing features that "crosscut" a distribution, such as
         defining a 'tests' feature that removes all the 'tests' subpackages
         provided by other features.  The default for this argument is an empty
         list.  (Note: the named package(s) or modules must exist in the base
         distribution when the 'setup()' function is initially called.)

      other keywords -- any other keyword arguments are saved, and passed to
         the distribution's 'include()' and 'exclude()' methods when the
         feature is included or excluded, respectively.  So, for example, you
         could pass 'packages=["a","b"]' to cause packages 'a' and 'b' to be
         added or removed from the distribution as appropriate.

    A feature must include at least one 'requires', 'remove', or other
    keyword argument.  Otherwise, it can't affect the distribution in any way.
    Note also that you can subclass 'Feature' to create your own specialized
    feature types that modify the distribution in other ways when included or
    excluded.  See the docstrings for the various methods here for more detail.
    Aside from the methods, the only feature attributes that distributions look
    at are 'description' and 'optional'.
    """

    @staticmethod
    def warn_deprecated():
        warnings.warn(
            "Features are deprecated and will be removed in a future "
                "version. See https://github.com/pypa/setuptools/issues/65.",
            DeprecationWarning,
            stacklevel=3,
        )

    def __init__(self, description, standard=False, available=True,
            optional=True, require_features=(), remove=(), **extras):
        self.warn_deprecated()

        self.description = description
        self.standard = standard
        self.available = available
        self.optional = optional
        if isinstance(require_features, (str, Require)):
            require_features = require_features,

        self.require_features = [
            r for r in require_features if isinstance(r, str)
        ]
        er = [r for r in require_features if not isinstance(r, str)]
        if er:
            extras['require_features'] = er

        if isinstance(remove, str):
            remove = remove,
        self.remove = remove
        self.extras = extras

        if not remove and not require_features and not extras:
            raise DistutilsSetupError(
                "Feature %s: must define 'require_features', 'remove', or at least one"
                " of 'packages', 'py_modules', etc."
            )

    def include_by_default(self):
        """Should this feature be included by default?"""
        return self.available and self.standard

    def include_in(self, dist):
        """Ensure feature and its requirements are included in distribution

        You may override this in a subclass to perform additional operations on
        the distribution.  Note that this method may be called more than once
        per feature, and so should be idempotent.

        """

        if not self.available:
            raise DistutilsPlatformError(
                self.description + " is required, "
                "but is not available on this platform"
            )

        dist.include(**self.extras)

        for f in self.require_features:
            dist.include_feature(f)

    def exclude_from(self, dist):
        """Ensure feature is excluded from distribution

        You may override this in a subclass to perform additional operations on
        the distribution.  This method will be called at most once per
        feature, and only after all included features have been asked to
        include themselves.
        """

        dist.exclude(**self.extras)

        if self.remove:
            for item in self.remove:
                dist.exclude_package(item)

    def validate(self, dist):
        """Verify that feature makes sense in context of distribution

        This method is called by the distribution just before it parses its
        command line.  It checks to ensure that the 'remove' attribute, if any,
        contains only valid package/module names that are present in the base
        distribution when 'setup()' is called.  You may override it in a
        subclass to perform any other required validation of the feature
        against a target distribution.
        """

        for item in self.remove:
            if not dist.has_contents_for(item):
                raise DistutilsSetupError(
                    "%s wants to be able to remove %s, but the distribution"
                    " doesn't contain any packages or modules under %s"
                    % (self.description, item, item)
                )
