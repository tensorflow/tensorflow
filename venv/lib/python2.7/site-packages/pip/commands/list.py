from __future__ import absolute_import

import json
import logging
import warnings
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

from pip._vendor import six

from pip.basecommand import Command
from pip.exceptions import CommandError
from pip.index import PackageFinder
from pip.utils import (
    get_installed_distributions, dist_is_editable)
from pip.utils.deprecation import RemovedInPip10Warning
from pip.cmdoptions import make_option_group, index_group

logger = logging.getLogger(__name__)


class ListCommand(Command):
    """
    List installed packages, including editables.

    Packages are listed in a case-insensitive sorted order.
    """
    name = 'list'
    usage = """
      %prog [options]"""
    summary = 'List installed packages.'

    def __init__(self, *args, **kw):
        super(ListCommand, self).__init__(*args, **kw)

        cmd_opts = self.cmd_opts

        cmd_opts.add_option(
            '-o', '--outdated',
            action='store_true',
            default=False,
            help='List outdated packages')
        cmd_opts.add_option(
            '-u', '--uptodate',
            action='store_true',
            default=False,
            help='List uptodate packages')
        cmd_opts.add_option(
            '-e', '--editable',
            action='store_true',
            default=False,
            help='List editable projects.')
        cmd_opts.add_option(
            '-l', '--local',
            action='store_true',
            default=False,
            help=('If in a virtualenv that has global access, do not list '
                  'globally-installed packages.'),
        )
        self.cmd_opts.add_option(
            '--user',
            dest='user',
            action='store_true',
            default=False,
            help='Only output packages installed in user-site.')

        cmd_opts.add_option(
            '--pre',
            action='store_true',
            default=False,
            help=("Include pre-release and development versions. By default, "
                  "pip only finds stable versions."),
        )

        cmd_opts.add_option(
            '--format',
            action='store',
            dest='list_format',
            choices=('legacy', 'columns', 'freeze', 'json'),
            help="Select the output format among: legacy (default), columns, "
                 "freeze or json.",
        )

        cmd_opts.add_option(
            '--not-required',
            action='store_true',
            dest='not_required',
            help="List packages that are not dependencies of "
                 "installed packages.",
        )

        index_opts = make_option_group(index_group, self.parser)

        self.parser.insert_option_group(0, index_opts)
        self.parser.insert_option_group(0, cmd_opts)

    def _build_package_finder(self, options, index_urls, session):
        """
        Create a package finder appropriate to this list command.
        """
        return PackageFinder(
            find_links=options.find_links,
            index_urls=index_urls,
            allow_all_prereleases=options.pre,
            trusted_hosts=options.trusted_hosts,
            process_dependency_links=options.process_dependency_links,
            session=session,
        )

    def run(self, options, args):
        if options.allow_external:
            warnings.warn(
                "--allow-external has been deprecated and will be removed in "
                "the future. Due to changes in the repository protocol, it no "
                "longer has any effect.",
                RemovedInPip10Warning,
            )

        if options.allow_all_external:
            warnings.warn(
                "--allow-all-external has been deprecated and will be removed "
                "in the future. Due to changes in the repository protocol, it "
                "no longer has any effect.",
                RemovedInPip10Warning,
            )

        if options.allow_unverified:
            warnings.warn(
                "--allow-unverified has been deprecated and will be removed "
                "in the future. Due to changes in the repository protocol, it "
                "no longer has any effect.",
                RemovedInPip10Warning,
            )

        if options.list_format is None:
            warnings.warn(
                "The default format will switch to columns in the future. "
                "You can use --format=(legacy|columns) (or define a "
                "format=(legacy|columns) in your pip.conf under the [list] "
                "section) to disable this warning.",
                RemovedInPip10Warning,
            )

        if options.outdated and options.uptodate:
            raise CommandError(
                "Options --outdated and --uptodate cannot be combined.")

        packages = get_installed_distributions(
            local_only=options.local,
            user_only=options.user,
            editables_only=options.editable,
        )

        if options.outdated:
            packages = self.get_outdated(packages, options)
        elif options.uptodate:
            packages = self.get_uptodate(packages, options)

        if options.not_required:
            packages = self.get_not_required(packages, options)

        self.output_package_listing(packages, options)

    def get_outdated(self, packages, options):
        return [
            dist for dist in self.iter_packages_latest_infos(packages, options)
            if dist.latest_version > dist.parsed_version
        ]

    def get_uptodate(self, packages, options):
        return [
            dist for dist in self.iter_packages_latest_infos(packages, options)
            if dist.latest_version == dist.parsed_version
        ]

    def get_not_required(self, packages, options):
        dep_keys = set()
        for dist in packages:
            dep_keys.update(requirement.key for requirement in dist.requires())
        return set(pkg for pkg in packages if pkg.key not in dep_keys)

    def iter_packages_latest_infos(self, packages, options):
        index_urls = [options.index_url] + options.extra_index_urls
        if options.no_index:
            logger.debug('Ignoring indexes: %s', ','.join(index_urls))
            index_urls = []

        dependency_links = []
        for dist in packages:
            if dist.has_metadata('dependency_links.txt'):
                dependency_links.extend(
                    dist.get_metadata_lines('dependency_links.txt'),
                )

        with self._build_session(options) as session:
            finder = self._build_package_finder(options, index_urls, session)
            finder.add_dependency_links(dependency_links)

            for dist in packages:
                typ = 'unknown'
                all_candidates = finder.find_all_candidates(dist.key)
                if not options.pre:
                    # Remove prereleases
                    all_candidates = [candidate for candidate in all_candidates
                                      if not candidate.version.is_prerelease]

                if not all_candidates:
                    continue
                best_candidate = max(all_candidates,
                                     key=finder._candidate_sort_key)
                remote_version = best_candidate.version
                if best_candidate.location.is_wheel:
                    typ = 'wheel'
                else:
                    typ = 'sdist'
                # This is dirty but makes the rest of the code much cleaner
                dist.latest_version = remote_version
                dist.latest_filetype = typ
                yield dist

    def output_legacy(self, dist):
        if dist_is_editable(dist):
            return '%s (%s, %s)' % (
                dist.project_name,
                dist.version,
                dist.location,
            )
        else:
            return '%s (%s)' % (dist.project_name, dist.version)

    def output_legacy_latest(self, dist):
        return '%s - Latest: %s [%s]' % (
            self.output_legacy(dist),
            dist.latest_version,
            dist.latest_filetype,
        )

    def output_package_listing(self, packages, options):
        packages = sorted(
            packages,
            key=lambda dist: dist.project_name.lower(),
        )
        if options.list_format == 'columns' and packages:
            data, header = format_for_columns(packages, options)
            self.output_package_listing_columns(data, header)
        elif options.list_format == 'freeze':
            for dist in packages:
                logger.info("%s==%s", dist.project_name, dist.version)
        elif options.list_format == 'json':
            logger.info(format_for_json(packages, options))
        else:  # legacy
            for dist in packages:
                if options.outdated:
                    logger.info(self.output_legacy_latest(dist))
                else:
                    logger.info(self.output_legacy(dist))

    def output_package_listing_columns(self, data, header):
        # insert the header first: we need to know the size of column names
        if len(data) > 0:
            data.insert(0, header)

        pkg_strings, sizes = tabulate(data)

        # Create and add a separator.
        if len(data) > 0:
            pkg_strings.insert(1, " ".join(map(lambda x: '-' * x, sizes)))

        for val in pkg_strings:
            logger.info(val)


def tabulate(vals):
    # From pfmoore on GitHub:
    # https://github.com/pypa/pip/issues/3651#issuecomment-216932564
    assert len(vals) > 0

    sizes = [0] * max(len(x) for x in vals)
    for row in vals:
        sizes = [max(s, len(str(c))) for s, c in zip_longest(sizes, row)]

    result = []
    for row in vals:
        display = " ".join([str(c).ljust(s) if c is not None else ''
                            for s, c in zip_longest(sizes, row)])
        result.append(display)

    return result, sizes


def format_for_columns(pkgs, options):
    """
    Convert the package data into something usable
    by output_package_listing_columns.
    """
    running_outdated = options.outdated
    # Adjust the header for the `pip list --outdated` case.
    if running_outdated:
        header = ["Package", "Version", "Latest", "Type"]
    else:
        header = ["Package", "Version"]

    data = []
    if any(dist_is_editable(x) for x in pkgs):
        header.append("Location")

    for proj in pkgs:
        # if we're working on the 'outdated' list, separate out the
        # latest_version and type
        row = [proj.project_name, proj.version]

        if running_outdated:
            row.append(proj.latest_version)
            row.append(proj.latest_filetype)

        if dist_is_editable(proj):
            row.append(proj.location)

        data.append(row)

    return data, header


def format_for_json(packages, options):
    data = []
    for dist in packages:
        info = {
            'name': dist.project_name,
            'version': six.text_type(dist.version),
        }
        if options.outdated:
            info['latest_version'] = six.text_type(dist.latest_version)
            info['latest_filetype'] = dist.latest_filetype
        data.append(info)
    return json.dumps(data)
