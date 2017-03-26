"""
Tools for converting old- to new-style metadata.
"""

from collections import namedtuple
from .pkginfo import read_pkg_info
from .util import OrderedDefaultDict
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

import re
import os.path
import textwrap
import pkg_resources
import email.parser
import wheel

METADATA_VERSION = "2.0"

PLURAL_FIELDS = { "classifier" : "classifiers",
                  "provides_dist" : "provides",
                  "provides_extra" : "extras" }

SKIP_FIELDS = set()

CONTACT_FIELDS = (({"email":"author_email", "name": "author"},
                    "author"),
                  ({"email":"maintainer_email", "name": "maintainer"},
                    "maintainer"))

# commonly filled out as "UNKNOWN" by distutils:
UNKNOWN_FIELDS = set(("author", "author_email", "platform", "home_page",
                      "license"))

# Wheel itself is probably the only program that uses non-extras markers
# in METADATA/PKG-INFO. Support its syntax with the extra at the end only.
EXTRA_RE = re.compile("""^(?P<package>.*?)(;\s*(?P<condition>.*?)(extra == '(?P<extra>.*?)')?)$""")
KEYWORDS_RE = re.compile("[\0-,]+")

MayRequiresKey = namedtuple('MayRequiresKey', ('condition', 'extra'))

def unique(iterable):
    """
    Yield unique values in iterable, preserving order.
    """
    seen = set()
    for value in iterable:
        if not value in seen:
            seen.add(value)
            yield value


def handle_requires(metadata, pkg_info, key):
    """
    Place the runtime requirements from pkg_info into metadata.
    """
    may_requires = OrderedDefaultDict(list)
    for value in sorted(pkg_info.get_all(key)):
        extra_match = EXTRA_RE.search(value)
        if extra_match:
            groupdict = extra_match.groupdict()
            condition = groupdict['condition']
            extra = groupdict['extra']
            package = groupdict['package']
            if condition.endswith(' and '):
                condition = condition[:-5]
        else:
            condition, extra = None, None
            package = value
        key = MayRequiresKey(condition, extra)
        may_requires[key].append(package)

    if may_requires:
        metadata['run_requires'] = []
        def sort_key(item):
            # Both condition and extra could be None, which can't be compared
            # against strings in Python 3.
            key, value = item
            if key.condition is None:
                return ''
            return key.condition
        for key, value in sorted(may_requires.items(), key=sort_key):
            may_requirement = OrderedDict((('requires', value),))
            if key.extra:
                may_requirement['extra'] = key.extra
            if key.condition:
                may_requirement['environment'] = key.condition
            metadata['run_requires'].append(may_requirement)

        if not 'extras' in metadata:
            metadata['extras'] = []
        metadata['extras'].extend([key.extra for key in may_requires.keys() if key.extra])


def pkginfo_to_dict(path, distribution=None):
    """
    Convert PKG-INFO to a prototype Metadata 2.0 (PEP 426) dict.

    The description is included under the key ['description'] rather than
    being written to a separate file.

    path: path to PKG-INFO file
    distribution: optional distutils Distribution()
    """

    metadata = OrderedDefaultDict(lambda: OrderedDefaultDict(lambda: OrderedDefaultDict(OrderedDict)))
    metadata["generator"] = "bdist_wheel (" + wheel.__version__ + ")"
    try:
        unicode
        pkg_info = read_pkg_info(path)
    except NameError:
        pkg_info = email.parser.Parser().parsestr(open(path, 'rb').read().decode('utf-8'))
    description = None

    if pkg_info['Summary']:
        metadata['summary'] = pkginfo_unicode(pkg_info, 'Summary')
        del pkg_info['Summary']

    if pkg_info['Description']:
        description = dedent_description(pkg_info)
        del pkg_info['Description']
    else:
        payload = pkg_info.get_payload()
        if isinstance(payload, bytes):
            # Avoid a Python 2 Unicode error.
            # We still suffer ? glyphs on Python 3.
            payload = payload.decode('utf-8')
        if payload:
            description = payload

    if description:
        pkg_info['description'] = description

    for key in sorted(unique(k.lower() for k in pkg_info.keys())):
        low_key = key.replace('-', '_')

        if low_key in SKIP_FIELDS:
            continue

        if low_key in UNKNOWN_FIELDS and pkg_info.get(key) == 'UNKNOWN':
            continue

        if low_key in sorted(PLURAL_FIELDS):
            metadata[PLURAL_FIELDS[low_key]] = pkg_info.get_all(key)

        elif low_key == "requires_dist":
            handle_requires(metadata, pkg_info, key)

        elif low_key == 'provides_extra':
            if not 'extras' in metadata:
                metadata['extras'] = []
            metadata['extras'].extend(pkg_info.get_all(key))

        elif low_key == 'home_page':
            metadata['extensions']['python.details']['project_urls'] = {'Home':pkg_info[key]}

        elif low_key == 'keywords':
            metadata['keywords'] = KEYWORDS_RE.split(pkg_info[key])

        else:
            metadata[low_key] = pkg_info[key]

    metadata['metadata_version'] = METADATA_VERSION

    if 'extras' in metadata:
        metadata['extras'] = sorted(set(metadata['extras']))

    # include more information if distribution is available
    if distribution:
        for requires, attr in (('test_requires', 'tests_require'),):
            try:
                requirements = getattr(distribution, attr)
                if isinstance(requirements, list):
                    new_requirements = sorted(convert_requirements(requirements))
                    metadata[requires] = [{'requires':new_requirements}]
            except AttributeError:
                pass

    # handle contacts
    contacts = []
    for contact_type, role in CONTACT_FIELDS:
        contact = OrderedDict()
        for key in sorted(contact_type):
            if contact_type[key] in metadata:
                contact[key] = metadata.pop(contact_type[key])
        if contact:
            contact['role'] = role
            contacts.append(contact)
    if contacts:
        metadata['extensions']['python.details']['contacts'] = contacts

    # convert entry points to exports
    try:
        with open(os.path.join(os.path.dirname(path), "entry_points.txt"), "r") as ep_file:
            ep_map = pkg_resources.EntryPoint.parse_map(ep_file.read())
        exports = OrderedDict()
        for group, items in sorted(ep_map.items()):
            exports[group] = OrderedDict()
            for item in sorted(map(str, items.values())):
                name, export = item.split(' = ', 1)
                exports[group][name] = export
        if exports:
            metadata['extensions']['python.exports'] = exports
    except IOError:
        pass

    # copy console_scripts entry points to commands
    if 'python.exports' in metadata['extensions']:
        for (ep_script, wrap_script) in (('console_scripts', 'wrap_console'),
                                         ('gui_scripts', 'wrap_gui')):
            if ep_script in metadata['extensions']['python.exports']:
                metadata['extensions']['python.commands'][wrap_script] = \
                    metadata['extensions']['python.exports'][ep_script]

    return metadata

def requires_to_requires_dist(requirement):
    """Compose the version predicates for requirement in PEP 345 fashion."""
    requires_dist = []
    for op, ver in requirement.specs:
        requires_dist.append(op + ver)
    if not requires_dist:
        return ''
    return " (%s)" % ','.join(requires_dist)

def convert_requirements(requirements):
    """Yield Requires-Dist: strings for parsed requirements strings."""
    for req in requirements:
        parsed_requirement = pkg_resources.Requirement.parse(req)
        spec = requires_to_requires_dist(parsed_requirement)
        extras = ",".join(parsed_requirement.extras)
        if extras:
            extras = "[%s]" % extras
        yield (parsed_requirement.project_name + extras + spec)

def pkginfo_to_metadata(egg_info_path, pkginfo_path):
    """
    Convert .egg-info directory with PKG-INFO to the Metadata 1.3 aka
    old-draft Metadata 2.0 format.
    """
    pkg_info = read_pkg_info(pkginfo_path)
    pkg_info.replace_header('Metadata-Version', '2.0')
    requires_path = os.path.join(egg_info_path, 'requires.txt')
    if os.path.exists(requires_path):
        requires = open(requires_path).read()
        for extra, reqs in sorted(pkg_resources.split_sections(requires),
                                  key=lambda x: x[0] or ''):
            condition = ''
            if extra and ':' in extra: # setuptools extra:condition syntax
                extra, condition = extra.split(':', 1)
            if extra:
                pkg_info['Provides-Extra'] = extra
                if condition:
                    condition += " and "
                condition += 'extra == %s' % repr(extra)
            if condition:
                condition = '; ' + condition
            for new_req in sorted(convert_requirements(reqs)):
                pkg_info['Requires-Dist'] = new_req + condition

    description = pkg_info['Description']
    if description:
        pkg_info.set_payload(dedent_description(pkg_info))
        del pkg_info['Description']

    return pkg_info


def pkginfo_unicode(pkg_info, field):
    """Hack to coax Unicode out of an email Message() - Python 3.3+"""
    text = pkg_info[field]
    field = field.lower()
    if not isinstance(text, str):
        if not hasattr(pkg_info, 'raw_items'):  # Python 3.2
            return str(text)
        for item in pkg_info.raw_items():
            if item[0].lower() == field:
                text = item[1].encode('ascii', 'surrogateescape')\
                                      .decode('utf-8')
                break

    return text


def dedent_description(pkg_info):
    """
    Dedent and convert pkg_info['Description'] to Unicode.
    """
    description = pkg_info['Description']

    # Python 3 Unicode handling, sorta.
    surrogates = False
    if not isinstance(description, str):
        surrogates = True
        description = pkginfo_unicode(pkg_info, 'Description')

    description_lines = description.splitlines()
    description_dedent = '\n'.join(
            # if the first line of long_description is blank,
            # the first line here will be indented.
            (description_lines[0].lstrip(),
             textwrap.dedent('\n'.join(description_lines[1:])),
             '\n'))

    if surrogates:
        description_dedent = description_dedent\
                .encode("utf8")\
                .decode("ascii", "surrogateescape")

    return description_dedent


if __name__ == "__main__":
    import sys, pprint
    pprint.pprint(pkginfo_to_dict(sys.argv[1]))
