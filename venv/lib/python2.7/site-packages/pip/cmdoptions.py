"""
shared options and groups

The principle here is to define options once, but *not* instantiate them
globally. One reason being that options with action='append' can carry state
between parses. pip parses general options twice internally, and shouldn't
pass on state. To be consistent, all options will follow this design.

"""
from __future__ import absolute_import

from functools import partial
from optparse import OptionGroup, SUPPRESS_HELP, Option
import warnings

from pip.index import (
    FormatControl, fmt_ctl_handle_mutual_exclude, fmt_ctl_no_binary,
    fmt_ctl_no_use_wheel)
from pip.models import PyPI
from pip.locations import USER_CACHE_DIR, src_prefix
from pip.utils.hashes import STRONG_HASHES


def make_option_group(group, parser):
    """
    Return an OptionGroup object
    group  -- assumed to be dict with 'name' and 'options' keys
    parser -- an optparse Parser
    """
    option_group = OptionGroup(parser, group['name'])
    for option in group['options']:
        option_group.add_option(option())
    return option_group


def resolve_wheel_no_use_binary(options):
    if not options.use_wheel:
        control = options.format_control
        fmt_ctl_no_use_wheel(control)


def check_install_build_global(options, check_options=None):
    """Disable wheels if per-setup.py call options are set.

    :param options: The OptionParser options to update.
    :param check_options: The options to check, if not supplied defaults to
        options.
    """
    if check_options is None:
        check_options = options

    def getname(n):
        return getattr(check_options, n, None)
    names = ["build_options", "global_options", "install_options"]
    if any(map(getname, names)):
        control = options.format_control
        fmt_ctl_no_binary(control)
        warnings.warn(
            'Disabling all use of wheels due to the use of --build-options '
            '/ --global-options / --install-options.', stacklevel=2)


###########
# options #
###########

help_ = partial(
    Option,
    '-h', '--help',
    dest='help',
    action='help',
    help='Show help.')

isolated_mode = partial(
    Option,
    "--isolated",
    dest="isolated_mode",
    action="store_true",
    default=False,
    help=(
        "Run pip in an isolated mode, ignoring environment variables and user "
        "configuration."
    ),
)

require_virtualenv = partial(
    Option,
    # Run only if inside a virtualenv, bail if not.
    '--require-virtualenv', '--require-venv',
    dest='require_venv',
    action='store_true',
    default=False,
    help=SUPPRESS_HELP)

verbose = partial(
    Option,
    '-v', '--verbose',
    dest='verbose',
    action='count',
    default=0,
    help='Give more output. Option is additive, and can be used up to 3 times.'
)

version = partial(
    Option,
    '-V', '--version',
    dest='version',
    action='store_true',
    help='Show version and exit.')

quiet = partial(
    Option,
    '-q', '--quiet',
    dest='quiet',
    action='count',
    default=0,
    help=('Give less output. Option is additive, and can be used up to 3'
          ' times (corresponding to WARNING, ERROR, and CRITICAL logging'
          ' levels).')
)

log = partial(
    Option,
    "--log", "--log-file", "--local-log",
    dest="log",
    metavar="path",
    help="Path to a verbose appending log."
)

no_input = partial(
    Option,
    # Don't ask for input
    '--no-input',
    dest='no_input',
    action='store_true',
    default=False,
    help=SUPPRESS_HELP)

proxy = partial(
    Option,
    '--proxy',
    dest='proxy',
    type='str',
    default='',
    help="Specify a proxy in the form [user:passwd@]proxy.server:port.")

retries = partial(
    Option,
    '--retries',
    dest='retries',
    type='int',
    default=5,
    help="Maximum number of retries each connection should attempt "
         "(default %default times).")

timeout = partial(
    Option,
    '--timeout', '--default-timeout',
    metavar='sec',
    dest='timeout',
    type='float',
    default=15,
    help='Set the socket timeout (default %default seconds).')

default_vcs = partial(
    Option,
    # The default version control system for editables, e.g. 'svn'
    '--default-vcs',
    dest='default_vcs',
    type='str',
    default='',
    help=SUPPRESS_HELP)

skip_requirements_regex = partial(
    Option,
    # A regex to be used to skip requirements
    '--skip-requirements-regex',
    dest='skip_requirements_regex',
    type='str',
    default='',
    help=SUPPRESS_HELP)


def exists_action():
    return Option(
        # Option when path already exist
        '--exists-action',
        dest='exists_action',
        type='choice',
        choices=['s', 'i', 'w', 'b', 'a'],
        default=[],
        action='append',
        metavar='action',
        help="Default action when a path already exists: "
        "(s)witch, (i)gnore, (w)ipe, (b)ackup, (a)bort.")


cert = partial(
    Option,
    '--cert',
    dest='cert',
    type='str',
    metavar='path',
    help="Path to alternate CA bundle.")

client_cert = partial(
    Option,
    '--client-cert',
    dest='client_cert',
    type='str',
    default=None,
    metavar='path',
    help="Path to SSL client certificate, a single file containing the "
         "private key and the certificate in PEM format.")

index_url = partial(
    Option,
    '-i', '--index-url', '--pypi-url',
    dest='index_url',
    metavar='URL',
    default=PyPI.simple_url,
    help="Base URL of Python Package Index (default %default). "
         "This should point to a repository compliant with PEP 503 "
         "(the simple repository API) or a local directory laid out "
         "in the same format.")


def extra_index_url():
    return Option(
        '--extra-index-url',
        dest='extra_index_urls',
        metavar='URL',
        action='append',
        default=[],
        help="Extra URLs of package indexes to use in addition to "
             "--index-url. Should follow the same rules as "
             "--index-url."
    )


no_index = partial(
    Option,
    '--no-index',
    dest='no_index',
    action='store_true',
    default=False,
    help='Ignore package index (only looking at --find-links URLs instead).')


def find_links():
    return Option(
        '-f', '--find-links',
        dest='find_links',
        action='append',
        default=[],
        metavar='url',
        help="If a url or path to an html file, then parse for links to "
             "archives. If a local path or file:// url that's a directory, "
             "then look for archives in the directory listing.")


def allow_external():
    return Option(
        "--allow-external",
        dest="allow_external",
        action="append",
        default=[],
        metavar="PACKAGE",
        help=SUPPRESS_HELP,
    )


allow_all_external = partial(
    Option,
    "--allow-all-external",
    dest="allow_all_external",
    action="store_true",
    default=False,
    help=SUPPRESS_HELP,
)


def trusted_host():
    return Option(
        "--trusted-host",
        dest="trusted_hosts",
        action="append",
        metavar="HOSTNAME",
        default=[],
        help="Mark this host as trusted, even though it does not have valid "
             "or any HTTPS.",
    )


# Remove after 7.0
no_allow_external = partial(
    Option,
    "--no-allow-external",
    dest="allow_all_external",
    action="store_false",
    default=False,
    help=SUPPRESS_HELP,
)


# Remove --allow-insecure after 7.0
def allow_unsafe():
    return Option(
        "--allow-unverified", "--allow-insecure",
        dest="allow_unverified",
        action="append",
        default=[],
        metavar="PACKAGE",
        help=SUPPRESS_HELP,
    )

# Remove after 7.0
no_allow_unsafe = partial(
    Option,
    "--no-allow-insecure",
    dest="allow_all_insecure",
    action="store_false",
    default=False,
    help=SUPPRESS_HELP
)

# Remove after 1.5
process_dependency_links = partial(
    Option,
    "--process-dependency-links",
    dest="process_dependency_links",
    action="store_true",
    default=False,
    help="Enable the processing of dependency links.",
)


def constraints():
    return Option(
        '-c', '--constraint',
        dest='constraints',
        action='append',
        default=[],
        metavar='file',
        help='Constrain versions using the given constraints file. '
        'This option can be used multiple times.')


def requirements():
    return Option(
        '-r', '--requirement',
        dest='requirements',
        action='append',
        default=[],
        metavar='file',
        help='Install from the given requirements file. '
        'This option can be used multiple times.')


def editable():
    return Option(
        '-e', '--editable',
        dest='editables',
        action='append',
        default=[],
        metavar='path/url',
        help=('Install a project in editable mode (i.e. setuptools '
              '"develop mode") from a local project path or a VCS url.'),
    )

src = partial(
    Option,
    '--src', '--source', '--source-dir', '--source-directory',
    dest='src_dir',
    metavar='dir',
    default=src_prefix,
    help='Directory to check out editable projects into. '
    'The default in a virtualenv is "<venv path>/src". '
    'The default for global installs is "<current dir>/src".'
)

# XXX: deprecated, remove in 9.0
use_wheel = partial(
    Option,
    '--use-wheel',
    dest='use_wheel',
    action='store_true',
    default=True,
    help=SUPPRESS_HELP,
)

# XXX: deprecated, remove in 9.0
no_use_wheel = partial(
    Option,
    '--no-use-wheel',
    dest='use_wheel',
    action='store_false',
    default=True,
    help=('Do not Find and prefer wheel archives when searching indexes and '
          'find-links locations. DEPRECATED in favour of --no-binary.'),
)


def _get_format_control(values, option):
    """Get a format_control object."""
    return getattr(values, option.dest)


def _handle_no_binary(option, opt_str, value, parser):
    existing = getattr(parser.values, option.dest)
    fmt_ctl_handle_mutual_exclude(
        value, existing.no_binary, existing.only_binary)


def _handle_only_binary(option, opt_str, value, parser):
    existing = getattr(parser.values, option.dest)
    fmt_ctl_handle_mutual_exclude(
        value, existing.only_binary, existing.no_binary)


def no_binary():
    return Option(
        "--no-binary", dest="format_control", action="callback",
        callback=_handle_no_binary, type="str",
        default=FormatControl(set(), set()),
        help="Do not use binary packages. Can be supplied multiple times, and "
             "each time adds to the existing value. Accepts either :all: to "
             "disable all binary packages, :none: to empty the set, or one or "
             "more package names with commas between them. Note that some "
             "packages are tricky to compile and may fail to install when "
             "this option is used on them.")


def only_binary():
    return Option(
        "--only-binary", dest="format_control", action="callback",
        callback=_handle_only_binary, type="str",
        default=FormatControl(set(), set()),
        help="Do not use source packages. Can be supplied multiple times, and "
             "each time adds to the existing value. Accepts either :all: to "
             "disable all source packages, :none: to empty the set, or one or "
             "more package names with commas between them. Packages without "
             "binary distributions will fail to install when this option is "
             "used on them.")


cache_dir = partial(
    Option,
    "--cache-dir",
    dest="cache_dir",
    default=USER_CACHE_DIR,
    metavar="dir",
    help="Store the cache data in <dir>."
)

no_cache = partial(
    Option,
    "--no-cache-dir",
    dest="cache_dir",
    action="store_false",
    help="Disable the cache.",
)

no_deps = partial(
    Option,
    '--no-deps', '--no-dependencies',
    dest='ignore_dependencies',
    action='store_true',
    default=False,
    help="Don't install package dependencies.")

build_dir = partial(
    Option,
    '-b', '--build', '--build-dir', '--build-directory',
    dest='build_dir',
    metavar='dir',
    help='Directory to unpack packages into and build in.'
)

ignore_requires_python = partial(
    Option,
    '--ignore-requires-python',
    dest='ignore_requires_python',
    action='store_true',
    help='Ignore the Requires-Python information.')

install_options = partial(
    Option,
    '--install-option',
    dest='install_options',
    action='append',
    metavar='options',
    help="Extra arguments to be supplied to the setup.py install "
         "command (use like --install-option=\"--install-scripts=/usr/local/"
         "bin\"). Use multiple --install-option options to pass multiple "
         "options to setup.py install. If you are using an option with a "
         "directory path, be sure to use absolute path.")

global_options = partial(
    Option,
    '--global-option',
    dest='global_options',
    action='append',
    metavar='options',
    help="Extra global options to be supplied to the setup.py "
         "call before the install command.")

no_clean = partial(
    Option,
    '--no-clean',
    action='store_true',
    default=False,
    help="Don't clean up build directories.")

pre = partial(
    Option,
    '--pre',
    action='store_true',
    default=False,
    help="Include pre-release and development versions. By default, "
         "pip only finds stable versions.")

disable_pip_version_check = partial(
    Option,
    "--disable-pip-version-check",
    dest="disable_pip_version_check",
    action="store_true",
    default=False,
    help="Don't periodically check PyPI to determine whether a new version "
         "of pip is available for download. Implied with --no-index.")

# Deprecated, Remove later
always_unzip = partial(
    Option,
    '-Z', '--always-unzip',
    dest='always_unzip',
    action='store_true',
    help=SUPPRESS_HELP,
)


def _merge_hash(option, opt_str, value, parser):
    """Given a value spelled "algo:digest", append the digest to a list
    pointed to in a dict by the algo name."""
    if not parser.values.hashes:
        parser.values.hashes = {}
    try:
        algo, digest = value.split(':', 1)
    except ValueError:
        parser.error('Arguments to %s must be a hash name '
                     'followed by a value, like --hash=sha256:abcde...' %
                     opt_str)
    if algo not in STRONG_HASHES:
        parser.error('Allowed hash algorithms for %s are %s.' %
                     (opt_str, ', '.join(STRONG_HASHES)))
    parser.values.hashes.setdefault(algo, []).append(digest)


hash = partial(
    Option,
    '--hash',
    # Hash values eventually end up in InstallRequirement.hashes due to
    # __dict__ copying in process_line().
    dest='hashes',
    action='callback',
    callback=_merge_hash,
    type='string',
    help="Verify that the package's archive matches this "
         'hash before installing. Example: --hash=sha256:abcdef...')


require_hashes = partial(
    Option,
    '--require-hashes',
    dest='require_hashes',
    action='store_true',
    default=False,
    help='Require a hash to check each requirement against, for '
         'repeatable installs. This option is implied when any package in a '
         'requirements file has a --hash option.')


##########
# groups #
##########

general_group = {
    'name': 'General Options',
    'options': [
        help_,
        isolated_mode,
        require_virtualenv,
        verbose,
        version,
        quiet,
        log,
        no_input,
        proxy,
        retries,
        timeout,
        default_vcs,
        skip_requirements_regex,
        exists_action,
        trusted_host,
        cert,
        client_cert,
        cache_dir,
        no_cache,
        disable_pip_version_check,
    ]
}

non_deprecated_index_group = {
    'name': 'Package Index Options',
    'options': [
        index_url,
        extra_index_url,
        no_index,
        find_links,
        process_dependency_links,
    ]
}

index_group = {
    'name': 'Package Index Options (including deprecated options)',
    'options': non_deprecated_index_group['options'] + [
        allow_external,
        allow_all_external,
        no_allow_external,
        allow_unsafe,
        no_allow_unsafe,
    ]
}
