from __future__ import absolute_import

import logging
import sys
import textwrap

from pip.basecommand import Command, SUCCESS
from pip.compat import OrderedDict
from pip.download import PipXmlrpcTransport
from pip.models import PyPI
from pip.utils import get_terminal_size
from pip.utils.logging import indent_log
from pip.exceptions import CommandError
from pip.status_codes import NO_MATCHES_FOUND
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor import pkg_resources
from pip._vendor.six.moves import xmlrpc_client


logger = logging.getLogger(__name__)


class SearchCommand(Command):
    """Search for PyPI packages whose name or summary contains <query>."""
    name = 'search'
    usage = """
      %prog [options] <query>"""
    summary = 'Search PyPI for packages.'

    def __init__(self, *args, **kw):
        super(SearchCommand, self).__init__(*args, **kw)
        self.cmd_opts.add_option(
            '-i', '--index',
            dest='index',
            metavar='URL',
            default=PyPI.pypi_url,
            help='Base URL of Python Package Index (default %default)')

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options, args):
        if not args:
            raise CommandError('Missing required argument (search query).')
        query = args
        pypi_hits = self.search(query, options)
        hits = transform_hits(pypi_hits)

        terminal_width = None
        if sys.stdout.isatty():
            terminal_width = get_terminal_size()[0]

        print_results(hits, terminal_width=terminal_width)
        if pypi_hits:
            return SUCCESS
        return NO_MATCHES_FOUND

    def search(self, query, options):
        index_url = options.index
        with self._build_session(options) as session:
            transport = PipXmlrpcTransport(index_url, session)
            pypi = xmlrpc_client.ServerProxy(index_url, transport)
            hits = pypi.search({'name': query, 'summary': query}, 'or')
            return hits


def transform_hits(hits):
    """
    The list from pypi is really a list of versions. We want a list of
    packages with the list of versions stored inline. This converts the
    list from pypi into one we can use.
    """
    packages = OrderedDict()
    for hit in hits:
        name = hit['name']
        summary = hit['summary']
        version = hit['version']

        if name not in packages.keys():
            packages[name] = {
                'name': name,
                'summary': summary,
                'versions': [version],
            }
        else:
            packages[name]['versions'].append(version)

            # if this is the highest version, replace summary and score
            if version == highest_version(packages[name]['versions']):
                packages[name]['summary'] = summary

    return list(packages.values())


def print_results(hits, name_column_width=None, terminal_width=None):
    if not hits:
        return
    if name_column_width is None:
        name_column_width = max([
            len(hit['name']) + len(hit.get('versions', ['-'])[-1])
            for hit in hits
        ]) + 4

    installed_packages = [p.project_name for p in pkg_resources.working_set]
    for hit in hits:
        name = hit['name']
        summary = hit['summary'] or ''
        version = hit.get('versions', ['-'])[-1]
        if terminal_width is not None:
            target_width = terminal_width - name_column_width - 5
            if target_width > 10:
                # wrap and indent summary to fit terminal
                summary = textwrap.wrap(summary, target_width)
                summary = ('\n' + ' ' * (name_column_width + 3)).join(summary)

        line = '%-*s - %s' % (name_column_width,
                              '%s (%s)' % (name, version), summary)
        try:
            logger.info(line)
            if name in installed_packages:
                dist = pkg_resources.get_distribution(name)
                with indent_log():
                    latest = highest_version(hit['versions'])
                    if dist.version == latest:
                        logger.info('INSTALLED: %s (latest)', dist.version)
                    else:
                        logger.info('INSTALLED: %s', dist.version)
                        logger.info('LATEST:    %s', latest)
        except UnicodeEncodeError:
            pass


def highest_version(versions):
    return max(versions, key=parse_version)
