"""
Load setuptools configuration from ``setup.cfg`` files.

**API will be made private in the future**
"""
import os

import contextlib
import functools
import warnings
from collections import defaultdict
from functools import partial
from functools import wraps
from typing import (TYPE_CHECKING, Callable, Any, Dict, Generic, Iterable, List,
                    Optional, Tuple, TypeVar, Union)

from distutils.errors import DistutilsOptionError, DistutilsFileError
from setuptools.extern.packaging.requirements import Requirement, InvalidRequirement
from setuptools.extern.packaging.version import Version, InvalidVersion
from setuptools.extern.packaging.specifiers import SpecifierSet
from setuptools._deprecation_warning import SetuptoolsDeprecationWarning

from . import expand

if TYPE_CHECKING:
    from setuptools.dist import Distribution  # noqa
    from distutils.dist import DistributionMetadata  # noqa

_Path = Union[str, os.PathLike]
SingleCommandOptions = Dict["str", Tuple["str", Any]]
"""Dict that associate the name of the options of a particular command to a
tuple. The first element of the tuple indicates the origin of the option value
(e.g. the name of the configuration file where it was read from),
while the second element of the tuple is the option value itself
"""
AllCommandOptions = Dict["str", SingleCommandOptions]  # cmd name => its options
Target = TypeVar("Target", bound=Union["Distribution", "DistributionMetadata"])


def read_configuration(
    filepath: _Path,
    find_others=False,
    ignore_option_errors=False
) -> dict:
    """Read given configuration file and returns options from it as a dict.

    :param str|unicode filepath: Path to configuration file
        to get options from.

    :param bool find_others: Whether to search for other configuration files
        which could be on in various places.

    :param bool ignore_option_errors: Whether to silently ignore
        options, values of which could not be resolved (e.g. due to exceptions
        in directives such as file:, attr:, etc.).
        If False exceptions are propagated as expected.

    :rtype: dict
    """
    from setuptools.dist import Distribution

    dist = Distribution()
    filenames = dist.find_config_files() if find_others else []
    handlers = _apply(dist, filepath, filenames, ignore_option_errors)
    return configuration_to_dict(handlers)


def apply_configuration(dist: "Distribution", filepath: _Path) -> "Distribution":
    """Apply the configuration from a ``setup.cfg`` file into an existing
    distribution object.
    """
    _apply(dist, filepath)
    dist._finalize_requires()
    return dist


def _apply(
    dist: "Distribution", filepath: _Path,
    other_files: Iterable[_Path] = (),
    ignore_option_errors: bool = False,
) -> Tuple["ConfigHandler", ...]:
    """Read configuration from ``filepath`` and applies to the ``dist`` object."""
    from setuptools.dist import _Distribution

    filepath = os.path.abspath(filepath)

    if not os.path.isfile(filepath):
        raise DistutilsFileError('Configuration file %s does not exist.' % filepath)

    current_directory = os.getcwd()
    os.chdir(os.path.dirname(filepath))
    filenames = [*other_files, filepath]

    try:
        _Distribution.parse_config_files(dist, filenames=filenames)
        handlers = parse_configuration(
            dist, dist.command_options, ignore_option_errors=ignore_option_errors
        )
        dist._finalize_license_files()
    finally:
        os.chdir(current_directory)

    return handlers


def _get_option(target_obj: Target, key: str):
    """
    Given a target object and option key, get that option from
    the target object, either through a get_{key} method or
    from an attribute directly.
    """
    getter_name = 'get_{key}'.format(**locals())
    by_attribute = functools.partial(getattr, target_obj, key)
    getter = getattr(target_obj, getter_name, by_attribute)
    return getter()


def configuration_to_dict(handlers: Tuple["ConfigHandler", ...]) -> dict:
    """Returns configuration data gathered by given handlers as a dict.

    :param list[ConfigHandler] handlers: Handlers list,
        usually from parse_configuration()

    :rtype: dict
    """
    config_dict: dict = defaultdict(dict)

    for handler in handlers:
        for option in handler.set_options:
            value = _get_option(handler.target_obj, option)
            config_dict[handler.section_prefix][option] = value

    return config_dict


def parse_configuration(
    distribution: "Distribution",
    command_options: AllCommandOptions,
    ignore_option_errors=False
) -> Tuple["ConfigMetadataHandler", "ConfigOptionsHandler"]:
    """Performs additional parsing of configuration options
    for a distribution.

    Returns a list of used option handlers.

    :param Distribution distribution:
    :param dict command_options:
    :param bool ignore_option_errors: Whether to silently ignore
        options, values of which could not be resolved (e.g. due to exceptions
        in directives such as file:, attr:, etc.).
        If False exceptions are propagated as expected.
    :rtype: list
    """
    with expand.EnsurePackagesDiscovered(distribution) as ensure_discovered:
        options = ConfigOptionsHandler(
            distribution,
            command_options,
            ignore_option_errors,
            ensure_discovered,
        )

        options.parse()
        if not distribution.package_dir:
            distribution.package_dir = options.package_dir  # Filled by `find_packages`

        meta = ConfigMetadataHandler(
            distribution.metadata,
            command_options,
            ignore_option_errors,
            ensure_discovered,
            distribution.package_dir,
            distribution.src_root,
        )
        meta.parse()

    return meta, options


def _warn_accidental_env_marker_misconfig(label: str, orig_value: str, parsed: list):
    """Because users sometimes misinterpret this configuration:

    [options.extras_require]
    foo = bar;python_version<"4"

    It looks like one requirement with an environment marker
    but because there is no newline, it's parsed as two requirements
    with a semicolon as separator.

    Therefore, if:
        * input string does not contain a newline AND
        * parsed result contains two requirements AND
        * parsing of the two parts from the result ("<first>;<second>")
        leads in a valid Requirement with a valid marker
    a UserWarning is shown to inform the user about the possible problem.
    """
    if "\n" in orig_value or len(parsed) != 2:
        return

    with contextlib.suppress(InvalidRequirement):
        original_requirements_str = ";".join(parsed)
        req = Requirement(original_requirements_str)
        if req.marker is not None:
            msg = (
                f"One of the parsed requirements in `{label}` "
                f"looks like a valid environment marker: '{parsed[1]}'\n"
                "Make sure that the config is correct and check "
                "https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#opt-2"  # noqa: E501
            )
            warnings.warn(msg, UserWarning)


class ConfigHandler(Generic[Target]):
    """Handles metadata supplied in configuration files."""

    section_prefix: str
    """Prefix for config sections handled by this handler.
    Must be provided by class heirs.

    """

    aliases: Dict[str, str] = {}
    """Options aliases.
    For compatibility with various packages. E.g.: d2to1 and pbr.
    Note: `-` in keys is replaced with `_` by config parser.

    """

    def __init__(
        self,
        target_obj: Target,
        options: AllCommandOptions,
        ignore_option_errors,
        ensure_discovered: expand.EnsurePackagesDiscovered,
    ):
        sections: AllCommandOptions = {}

        section_prefix = self.section_prefix
        for section_name, section_options in options.items():
            if not section_name.startswith(section_prefix):
                continue

            section_name = section_name.replace(section_prefix, '').strip('.')
            sections[section_name] = section_options

        self.ignore_option_errors = ignore_option_errors
        self.target_obj = target_obj
        self.sections = sections
        self.set_options: List[str] = []
        self.ensure_discovered = ensure_discovered

    @property
    def parsers(self):
        """Metadata item name to parser function mapping."""
        raise NotImplementedError(
            '%s must provide .parsers property' % self.__class__.__name__
        )

    def __setitem__(self, option_name, value):
        unknown = tuple()
        target_obj = self.target_obj

        # Translate alias into real name.
        option_name = self.aliases.get(option_name, option_name)

        current_value = getattr(target_obj, option_name, unknown)

        if current_value is unknown:
            raise KeyError(option_name)

        if current_value:
            # Already inhabited. Skipping.
            return

        skip_option = False
        parser = self.parsers.get(option_name)
        if parser:
            try:
                value = parser(value)

            except Exception:
                skip_option = True
                if not self.ignore_option_errors:
                    raise

        if skip_option:
            return

        setter = getattr(target_obj, 'set_%s' % option_name, None)
        if setter is None:
            setattr(target_obj, option_name, value)
        else:
            setter(value)

        self.set_options.append(option_name)

    @classmethod
    def _parse_list(cls, value, separator=','):
        """Represents value as a list.

        Value is split either by separator (defaults to comma) or by lines.

        :param value:
        :param separator: List items separator character.
        :rtype: list
        """
        if isinstance(value, list):  # _get_parser_compound case
            return value

        if '\n' in value:
            value = value.splitlines()
        else:
            value = value.split(separator)

        return [chunk.strip() for chunk in value if chunk.strip()]

    @classmethod
    def _parse_dict(cls, value):
        """Represents value as a dict.

        :param value:
        :rtype: dict
        """
        separator = '='
        result = {}
        for line in cls._parse_list(value):
            key, sep, val = line.partition(separator)
            if sep != separator:
                raise DistutilsOptionError(
                    'Unable to parse option value to dict: %s' % value
                )
            result[key.strip()] = val.strip()

        return result

    @classmethod
    def _parse_bool(cls, value):
        """Represents value as boolean.

        :param value:
        :rtype: bool
        """
        value = value.lower()
        return value in ('1', 'true', 'yes')

    @classmethod
    def _exclude_files_parser(cls, key):
        """Returns a parser function to make sure field inputs
        are not files.

        Parses a value after getting the key so error messages are
        more informative.

        :param key:
        :rtype: callable
        """

        def parser(value):
            exclude_directive = 'file:'
            if value.startswith(exclude_directive):
                raise ValueError(
                    'Only strings are accepted for the {0} field, '
                    'files are not accepted'.format(key)
                )
            return value

        return parser

    @classmethod
    def _parse_file(cls, value, root_dir: _Path):
        """Represents value as a string, allowing including text
        from nearest files using `file:` directive.

        Directive is sandboxed and won't reach anything outside
        directory with setup.py.

        Examples:
            file: README.rst, CHANGELOG.md, src/file.txt

        :param str value:
        :rtype: str
        """
        include_directive = 'file:'

        if not isinstance(value, str):
            return value

        if not value.startswith(include_directive):
            return value

        spec = value[len(include_directive) :]
        filepaths = (path.strip() for path in spec.split(','))
        return expand.read_files(filepaths, root_dir)

    def _parse_attr(self, value, package_dir, root_dir: _Path):
        """Represents value as a module attribute.

        Examples:
            attr: package.attr
            attr: package.module.attr

        :param str value:
        :rtype: str
        """
        attr_directive = 'attr:'
        if not value.startswith(attr_directive):
            return value

        attr_desc = value.replace(attr_directive, '')

        # Make sure package_dir is populated correctly, so `attr:` directives can work
        package_dir.update(self.ensure_discovered.package_dir)
        return expand.read_attr(attr_desc, package_dir, root_dir)

    @classmethod
    def _get_parser_compound(cls, *parse_methods):
        """Returns parser function to represents value as a list.

        Parses a value applying given methods one after another.

        :param parse_methods:
        :rtype: callable
        """

        def parse(value):
            parsed = value

            for method in parse_methods:
                parsed = method(parsed)

            return parsed

        return parse

    @classmethod
    def _parse_section_to_dict_with_key(cls, section_options, values_parser):
        """Parses section options into a dictionary.

        Applies a given parser to each option in a section.

        :param dict section_options:
        :param callable values_parser: function with 2 args corresponding to key, value
        :rtype: dict
        """
        value = {}
        for key, (_, val) in section_options.items():
            value[key] = values_parser(key, val)
        return value

    @classmethod
    def _parse_section_to_dict(cls, section_options, values_parser=None):
        """Parses section options into a dictionary.

        Optionally applies a given parser to each value.

        :param dict section_options:
        :param callable values_parser: function with 1 arg corresponding to option value
        :rtype: dict
        """
        parser = (lambda _, v: values_parser(v)) if values_parser else (lambda _, v: v)
        return cls._parse_section_to_dict_with_key(section_options, parser)

    def parse_section(self, section_options):
        """Parses configuration file section.

        :param dict section_options:
        """
        for (name, (_, value)) in section_options.items():
            with contextlib.suppress(KeyError):
                # Keep silent for a new option may appear anytime.
                self[name] = value

    def parse(self):
        """Parses configuration file items from one
        or more related sections.

        """
        for section_name, section_options in self.sections.items():

            method_postfix = ''
            if section_name:  # [section.option] variant
                method_postfix = '_%s' % section_name

            section_parser_method: Optional[Callable] = getattr(
                self,
                # Dots in section names are translated into dunderscores.
                ('parse_section%s' % method_postfix).replace('.', '__'),
                None,
            )

            if section_parser_method is None:
                raise DistutilsOptionError(
                    'Unsupported distribution option section: [%s.%s]'
                    % (self.section_prefix, section_name)
                )

            section_parser_method(section_options)

    def _deprecated_config_handler(self, func, msg, warning_class):
        """this function will wrap around parameters that are deprecated

        :param msg: deprecation message
        :param warning_class: class of warning exception to be raised
        :param func: function to be wrapped around
        """

        @wraps(func)
        def config_handler(*args, **kwargs):
            warnings.warn(msg, warning_class)
            return func(*args, **kwargs)

        return config_handler


class ConfigMetadataHandler(ConfigHandler["DistributionMetadata"]):

    section_prefix = 'metadata'

    aliases = {
        'home_page': 'url',
        'summary': 'description',
        'classifier': 'classifiers',
        'platform': 'platforms',
    }

    strict_mode = False
    """We need to keep it loose, to be partially compatible with
    `pbr` and `d2to1` packages which also uses `metadata` section.

    """

    def __init__(
        self,
        target_obj: "DistributionMetadata",
        options: AllCommandOptions,
        ignore_option_errors: bool,
        ensure_discovered: expand.EnsurePackagesDiscovered,
        package_dir: Optional[dict] = None,
        root_dir: _Path = os.curdir
    ):
        super().__init__(target_obj, options, ignore_option_errors, ensure_discovered)
        self.package_dir = package_dir
        self.root_dir = root_dir

    @property
    def parsers(self):
        """Metadata item name to parser function mapping."""
        parse_list = self._parse_list
        parse_file = partial(self._parse_file, root_dir=self.root_dir)
        parse_dict = self._parse_dict
        exclude_files_parser = self._exclude_files_parser

        return {
            'platforms': parse_list,
            'keywords': parse_list,
            'provides': parse_list,
            'requires': self._deprecated_config_handler(
                parse_list,
                "The requires parameter is deprecated, please use "
                "install_requires for runtime dependencies.",
                SetuptoolsDeprecationWarning,
            ),
            'obsoletes': parse_list,
            'classifiers': self._get_parser_compound(parse_file, parse_list),
            'license': exclude_files_parser('license'),
            'license_file': self._deprecated_config_handler(
                exclude_files_parser('license_file'),
                "The license_file parameter is deprecated, "
                "use license_files instead.",
                SetuptoolsDeprecationWarning,
            ),
            'license_files': parse_list,
            'description': parse_file,
            'long_description': parse_file,
            'version': self._parse_version,
            'project_urls': parse_dict,
        }

    def _parse_version(self, value):
        """Parses `version` option value.

        :param value:
        :rtype: str

        """
        version = self._parse_file(value, self.root_dir)

        if version != value:
            version = version.strip()
            # Be strict about versions loaded from file because it's easy to
            # accidentally include newlines and other unintended content
            try:
                Version(version)
            except InvalidVersion:
                tmpl = (
                    'Version loaded from {value} does not '
                    'comply with PEP 440: {version}'
                )
                raise DistutilsOptionError(tmpl.format(**locals()))

            return version

        return expand.version(self._parse_attr(value, self.package_dir, self.root_dir))


class ConfigOptionsHandler(ConfigHandler["Distribution"]):

    section_prefix = 'options'

    def __init__(
        self,
        target_obj: "Distribution",
        options: AllCommandOptions,
        ignore_option_errors: bool,
        ensure_discovered: expand.EnsurePackagesDiscovered,
    ):
        super().__init__(target_obj, options, ignore_option_errors, ensure_discovered)
        self.root_dir = target_obj.src_root
        self.package_dir: Dict[str, str] = {}  # To be filled by `find_packages`

    @classmethod
    def _parse_list_semicolon(cls, value):
        return cls._parse_list(value, separator=';')

    def _parse_file_in_root(self, value):
        return self._parse_file(value, root_dir=self.root_dir)

    def _parse_requirements_list(self, label: str, value: str):
        # Parse a requirements list, either by reading in a `file:`, or a list.
        parsed = self._parse_list_semicolon(self._parse_file_in_root(value))
        _warn_accidental_env_marker_misconfig(label, value, parsed)
        # Filter it to only include lines that are not comments. `parse_list`
        # will have stripped each line and filtered out empties.
        return [line for line in parsed if not line.startswith("#")]

    @property
    def parsers(self):
        """Metadata item name to parser function mapping."""
        parse_list = self._parse_list
        parse_bool = self._parse_bool
        parse_dict = self._parse_dict
        parse_cmdclass = self._parse_cmdclass

        return {
            'zip_safe': parse_bool,
            'include_package_data': parse_bool,
            'package_dir': parse_dict,
            'scripts': parse_list,
            'eager_resources': parse_list,
            'dependency_links': parse_list,
            'namespace_packages': self._deprecated_config_handler(
                parse_list,
                "The namespace_packages parameter is deprecated, "
                "consider using implicit namespaces instead (PEP 420).",
                SetuptoolsDeprecationWarning,
            ),
            'install_requires': partial(
                self._parse_requirements_list, "install_requires"
            ),
            'setup_requires': self._parse_list_semicolon,
            'tests_require': self._parse_list_semicolon,
            'packages': self._parse_packages,
            'entry_points': self._parse_file_in_root,
            'py_modules': parse_list,
            'python_requires': SpecifierSet,
            'cmdclass': parse_cmdclass,
        }

    def _parse_cmdclass(self, value):
        package_dir = self.ensure_discovered.package_dir
        return expand.cmdclass(self._parse_dict(value), package_dir, self.root_dir)

    def _parse_packages(self, value):
        """Parses `packages` option value.

        :param value:
        :rtype: list
        """
        find_directives = ['find:', 'find_namespace:']
        trimmed_value = value.strip()

        if trimmed_value not in find_directives:
            return self._parse_list(value)

        # Read function arguments from a dedicated section.
        find_kwargs = self.parse_section_packages__find(
            self.sections.get('packages.find', {})
        )

        find_kwargs.update(
            namespaces=(trimmed_value == find_directives[1]),
            root_dir=self.root_dir,
            fill_package_dir=self.package_dir,
        )

        return expand.find_packages(**find_kwargs)

    def parse_section_packages__find(self, section_options):
        """Parses `packages.find` configuration file section.

        To be used in conjunction with _parse_packages().

        :param dict section_options:
        """
        section_data = self._parse_section_to_dict(section_options, self._parse_list)

        valid_keys = ['where', 'include', 'exclude']

        find_kwargs = dict(
            [(k, v) for k, v in section_data.items() if k in valid_keys and v]
        )

        where = find_kwargs.get('where')
        if where is not None:
            find_kwargs['where'] = where[0]  # cast list to single val

        return find_kwargs

    def parse_section_entry_points(self, section_options):
        """Parses `entry_points` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict(section_options, self._parse_list)
        self['entry_points'] = parsed

    def _parse_package_data(self, section_options):
        package_data = self._parse_section_to_dict(section_options, self._parse_list)
        return expand.canonic_package_data(package_data)

    def parse_section_package_data(self, section_options):
        """Parses `package_data` configuration file section.

        :param dict section_options:
        """
        self['package_data'] = self._parse_package_data(section_options)

    def parse_section_exclude_package_data(self, section_options):
        """Parses `exclude_package_data` configuration file section.

        :param dict section_options:
        """
        self['exclude_package_data'] = self._parse_package_data(section_options)

    def parse_section_extras_require(self, section_options):
        """Parses `extras_require` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict_with_key(
            section_options,
            lambda k, v: self._parse_requirements_list(f"extras_require[{k}]", v)
        )

        self['extras_require'] = parsed

    def parse_section_data_files(self, section_options):
        """Parses `data_files` configuration file section.

        :param dict section_options:
        """
        parsed = self._parse_section_to_dict(section_options, self._parse_list)
        self['data_files'] = expand.canonic_data_files(parsed, self.root_dir)
