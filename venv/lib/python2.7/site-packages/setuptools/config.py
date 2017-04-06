from __future__ import absolute_import, unicode_literals
import io
import os
import sys
from collections import defaultdict
from functools import partial

from distutils.errors import DistutilsOptionError, DistutilsFileError
from setuptools.py26compat import import_module
from six import string_types


def read_configuration(
        filepath, find_others=False, ignore_option_errors=False):
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
    from setuptools.dist import Distribution, _Distribution

    filepath = os.path.abspath(filepath)

    if not os.path.isfile(filepath):
        raise DistutilsFileError(
            'Configuration file %s does not exist.' % filepath)

    current_directory = os.getcwd()
    os.chdir(os.path.dirname(filepath))

    try:
        dist = Distribution()

        filenames = dist.find_config_files() if find_others else []
        if filepath not in filenames:
            filenames.append(filepath)

        _Distribution.parse_config_files(dist, filenames=filenames)

        handlers = parse_configuration(
            dist, dist.command_options,
            ignore_option_errors=ignore_option_errors)

    finally:
        os.chdir(current_directory)

    return configuration_to_dict(handlers)


def configuration_to_dict(handlers):
    """Returns configuration data gathered by given handlers as a dict.

    :param list[ConfigHandler] handlers: Handlers list,
        usually from parse_configuration()

    :rtype: dict
    """
    config_dict = defaultdict(dict)

    for handler in handlers:

        obj_alias = handler.section_prefix
        target_obj = handler.target_obj

        for option in handler.set_options:
            getter = getattr(target_obj, 'get_%s' % option, None)

            if getter is None:
                value = getattr(target_obj, option)

            else:
                value = getter()

            config_dict[obj_alias][option] = value

    return config_dict


def parse_configuration(
        distribution, command_options, ignore_option_errors=False):
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
    meta = ConfigMetadataHandler(
        distribution.metadata, command_options, ignore_option_errors)
    meta.parse()

    options = ConfigOptionsHandler(
        distribution, command_options, ignore_option_errors)
    options.parse()

    return [meta, options]


class ConfigHandler(object):
    """Handles metadata supplied in configuration files."""

    section_prefix = None
    """Prefix for config sections handled by this handler.
    Must be provided by class heirs.

    """

    aliases = {}
    """Options aliases.
    For compatibility with various packages. E.g.: d2to1 and pbr.
    Note: `-` in keys is replaced with `_` by config parser.

    """

    def __init__(self, target_obj, options, ignore_option_errors=False):
        sections = {}

        section_prefix = self.section_prefix
        for section_name, section_options in options.items():
            if not section_name.startswith(section_prefix):
                continue

            section_name = section_name.replace(section_prefix, '').strip('.')
            sections[section_name] = section_options

        self.ignore_option_errors = ignore_option_errors
        self.target_obj = target_obj
        self.sections = sections
        self.set_options = []

    @property
    def parsers(self):
        """Metadata item name to parser function mapping."""
        raise NotImplementedError(
            '%s must provide .parsers property' % self.__class__.__name__)

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
                    'Unable to parse option value to dict: %s' % value)
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
    def _parse_file(cls, value):
        """Represents value as a string, allowing including text
        from nearest files using `file:` directive.

        Directive is sandboxed and won't reach anything outside
        directory with setup.py.

        Examples:
            include: LICENSE
            include: src/file.txt

        :param str value:
        :rtype: str
        """
        if not isinstance(value, string_types):
            return value

        include_directive = 'file:'
        if not value.startswith(include_directive):
            return value

        current_directory = os.getcwd()

        filepath = value.replace(include_directive, '').strip()
        filepath = os.path.abspath(filepath)

        if not filepath.startswith(current_directory):
            raise DistutilsOptionError(
                '`file:` directive can not access %s' % filepath)

        if os.path.isfile(filepath):
            with io.open(filepath, encoding='utf-8') as f:
                value = f.read()

        return value

    @classmethod
    def _parse_attr(cls, value):
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

        attrs_path = value.replace(attr_directive, '').strip().split('.')
        attr_name = attrs_path.pop()

        module_name = '.'.join(attrs_path)
        module_name = module_name or '__init__'

        sys.path.insert(0, os.getcwd())
        try:
            module = import_module(module_name)
            value = getattr(module, attr_name)

        finally:
            sys.path = sys.path[1:]

        return value

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
    def _parse_section_to_dict(cls, section_options, values_parser=None):
        """Parses section options into a dictionary.

        Optionally applies a given parser to values.

        :param dict section_options:
        :param callable values_parser:
        :rtype: dict
        """
        value = {}
        values_parser = values_parser or (lambda val: val)
        for key, (_, val) in section_options.items():
            value[key] = values_parser(val)
        return value

    def parse_section(self, section_options):
        """Parses configuration file section.

        :param dict section_options:
        """
        for (name, (_, value)) in section_options.items():
            try:
                self[name] = value

            except KeyError:
                pass  # Keep silent for a new option may appear anytime.

    def parse(self):
        """Parses configuration file items from one
        or more related sections.

        """
        for section_name, section_options in self.sections.items():

            method_postfix = ''
            if section_name:  # [section.option] variant
                method_postfix = '_%s' % section_name

            section_parser_method = getattr(
                self,
                # Dots in section names are tranlsated into dunderscores.
                ('parse_section%s' % method_postfix).replace('.', '__'),
                None)

            if section_parser_method is None:
                raise DistutilsOptionError(
                    'Unsupported distribution option section: [%s.%s]' % (
                        self.section_prefix, section_name))

            section_parser_method(section_options)


class ConfigMetadataHandler(ConfigHandler):

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

    @property
    def parsers(self):
        """Metadata item name to parser function mapping."""
        parse_list = self._parse_list
        parse_file = self._parse_file

        return {
            'platforms': parse_list,
            'keywords': parse_list,
            'provides': parse_list,
            'requires': parse_list,
            'obsoletes': parse_list,
            'classifiers': self._get_parser_compound(parse_file, parse_list),
            'license': parse_file,
            'description': parse_file,
            'long_description': parse_file,
            'version': self._parse_version,
        }

    def _parse_version(self, value):
        """Parses `version` option value.

        :param value:
        :rtype: str

        """
        version = self._parse_attr(value)

        if callable(version):
            version = version()

        if not isinstance(version, string_types):
            if hasattr(version, '__iter__'):
                version = '.'.join(map(str, version))
            else:
                version = '%s' % version

        return version


class ConfigOptionsHandler(ConfigHandler):

    section_prefix = 'options'

    @property
    def parsers(self):
        """Metadata item name to parser function mapping."""
        parse_list = self._parse_list
        parse_list_semicolon = partial(self._parse_list, separator=';')
        parse_bool = self._parse_bool
        parse_dict = self._parse_dict

        return {
            'zip_safe': parse_bool,
            'use_2to3': parse_bool,
            'include_package_data': parse_bool,
            'package_dir': parse_dict,
            'use_2to3_fixers': parse_list,
            'use_2to3_exclude_fixers': parse_list,
            'convert_2to3_doctests': parse_list,
            'scripts': parse_list,
            'eager_resources': parse_list,
            'dependency_links': parse_list,
            'namespace_packages': parse_list,
            'install_requires': parse_list_semicolon,
            'setup_requires': parse_list_semicolon,
            'tests_require': parse_list_semicolon,
            'packages': self._parse_packages,
            'entry_points': self._parse_file,
        }

    def _parse_packages(self, value):
        """Parses `packages` option value.

        :param value:
        :rtype: list
        """
        find_directive = 'find:'

        if not value.startswith(find_directive):
            return self._parse_list(value)

        # Read function arguments from a dedicated section.
        find_kwargs = self.parse_section_packages__find(
            self.sections.get('packages.find', {}))

        from setuptools import find_packages

        return find_packages(**find_kwargs)

    def parse_section_packages__find(self, section_options):
        """Parses `packages.find` configuration file section.

        To be used in conjunction with _parse_packages().

        :param dict section_options:
        """
        section_data = self._parse_section_to_dict(
            section_options, self._parse_list)

        valid_keys = ['where', 'include', 'exclude']

        find_kwargs = dict(
            [(k, v) for k, v in section_data.items() if k in valid_keys and v])

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
        parsed = self._parse_section_to_dict(section_options, self._parse_list)

        root = parsed.get('*')
        if root:
            parsed[''] = root
            del parsed['*']

        return parsed

    def parse_section_package_data(self, section_options):
        """Parses `package_data` configuration file section.

        :param dict section_options:
        """
        self['package_data'] = self._parse_package_data(section_options)

    def parse_section_exclude_package_data(self, section_options):
        """Parses `exclude_package_data` configuration file section.

        :param dict section_options:
        """
        self['exclude_package_data'] = self._parse_package_data(
            section_options)

    def parse_section_extras_require(self, section_options):
        """Parses `extras_require` configuration file section.

        :param dict section_options:
        """
        parse_list = partial(self._parse_list, separator=';')
        self['extras_require'] = self._parse_section_to_dict(
            section_options, parse_list)
