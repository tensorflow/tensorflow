from __future__ import absolute_import

import logging
import os
import re

from pip._vendor.six.moves.urllib import parse as urllib_parse

from pip.index import Link
from pip.utils import rmtree, display_path
from pip.utils.logging import indent_log
from pip.vcs import vcs, VersionControl

_svn_xml_url_re = re.compile('url="([^"]+)"')
_svn_rev_re = re.compile('committed-rev="(\d+)"')
_svn_url_re = re.compile(r'URL: (.+)')
_svn_revision_re = re.compile(r'Revision: (.+)')
_svn_info_xml_rev_re = re.compile(r'\s*revision="(\d+)"')
_svn_info_xml_url_re = re.compile(r'<url>(.*)</url>')


logger = logging.getLogger(__name__)


class Subversion(VersionControl):
    name = 'svn'
    dirname = '.svn'
    repo_name = 'checkout'
    schemes = ('svn', 'svn+ssh', 'svn+http', 'svn+https', 'svn+svn')

    def get_info(self, location):
        """Returns (url, revision), where both are strings"""
        assert not location.rstrip('/').endswith(self.dirname), \
            'Bad directory: %s' % location
        output = self.run_command(
            ['info', location],
            show_stdout=False,
            extra_environ={'LANG': 'C'},
        )
        match = _svn_url_re.search(output)
        if not match:
            logger.warning(
                'Cannot determine URL of svn checkout %s',
                display_path(location),
            )
            logger.debug('Output that cannot be parsed: \n%s', output)
            return None, None
        url = match.group(1).strip()
        match = _svn_revision_re.search(output)
        if not match:
            logger.warning(
                'Cannot determine revision of svn checkout %s',
                display_path(location),
            )
            logger.debug('Output that cannot be parsed: \n%s', output)
            return url, None
        return url, match.group(1)

    def export(self, location):
        """Export the svn repository at the url to the destination location"""
        url, rev = self.get_url_rev()
        rev_options = get_rev_options(url, rev)
        url = self.remove_auth_from_url(url)
        logger.info('Exporting svn repository %s to %s', url, location)
        with indent_log():
            if os.path.exists(location):
                # Subversion doesn't like to check out over an existing
                # directory --force fixes this, but was only added in svn 1.5
                rmtree(location)
            self.run_command(
                ['export'] + rev_options + [url, location],
                show_stdout=False)

    def switch(self, dest, url, rev_options):
        self.run_command(['switch'] + rev_options + [url, dest])

    def update(self, dest, rev_options):
        self.run_command(['update'] + rev_options + [dest])

    def obtain(self, dest):
        url, rev = self.get_url_rev()
        rev_options = get_rev_options(url, rev)
        url = self.remove_auth_from_url(url)
        if rev:
            rev_display = ' (to revision %s)' % rev
        else:
            rev_display = ''
        if self.check_destination(dest, url, rev_options, rev_display):
            logger.info(
                'Checking out %s%s to %s',
                url,
                rev_display,
                display_path(dest),
            )
            self.run_command(['checkout', '-q'] + rev_options + [url, dest])

    def get_location(self, dist, dependency_links):
        for url in dependency_links:
            egg_fragment = Link(url).egg_fragment
            if not egg_fragment:
                continue
            if '-' in egg_fragment:
                # FIXME: will this work when a package has - in the name?
                key = '-'.join(egg_fragment.split('-')[:-1]).lower()
            else:
                key = egg_fragment
            if key == dist.key:
                return url.split('#', 1)[0]
        return None

    def get_revision(self, location):
        """
        Return the maximum revision for all files under a given location
        """
        # Note: taken from setuptools.command.egg_info
        revision = 0

        for base, dirs, files in os.walk(location):
            if self.dirname not in dirs:
                dirs[:] = []
                continue    # no sense walking uncontrolled subdirs
            dirs.remove(self.dirname)
            entries_fn = os.path.join(base, self.dirname, 'entries')
            if not os.path.exists(entries_fn):
                # FIXME: should we warn?
                continue

            dirurl, localrev = self._get_svn_url_rev(base)

            if base == location:
                base_url = dirurl + '/'   # save the root url
            elif not dirurl or not dirurl.startswith(base_url):
                dirs[:] = []
                continue    # not part of the same svn tree, skip it
            revision = max(revision, localrev)
        return revision

    def get_url_rev(self):
        # hotfix the URL scheme after removing svn+ from svn+ssh:// readd it
        url, rev = super(Subversion, self).get_url_rev()
        if url.startswith('ssh://'):
            url = 'svn+' + url
        return url, rev

    def get_url(self, location):
        # In cases where the source is in a subdirectory, not alongside
        # setup.py we have to look up in the location until we find a real
        # setup.py
        orig_location = location
        while not os.path.exists(os.path.join(location, 'setup.py')):
            last_location = location
            location = os.path.dirname(location)
            if location == last_location:
                # We've traversed up to the root of the filesystem without
                # finding setup.py
                logger.warning(
                    "Could not find setup.py for directory %s (tried all "
                    "parent directories)",
                    orig_location,
                )
                return None

        return self._get_svn_url_rev(location)[0]

    def _get_svn_url_rev(self, location):
        from pip.exceptions import InstallationError

        entries_path = os.path.join(location, self.dirname, 'entries')
        if os.path.exists(entries_path):
            with open(entries_path) as f:
                data = f.read()
        else:  # subversion >= 1.7 does not have the 'entries' file
            data = ''

        if (data.startswith('8') or
                data.startswith('9') or
                data.startswith('10')):
            data = list(map(str.splitlines, data.split('\n\x0c\n')))
            del data[0][0]  # get rid of the '8'
            url = data[0][3]
            revs = [int(d[9]) for d in data if len(d) > 9 and d[9]] + [0]
        elif data.startswith('<?xml'):
            match = _svn_xml_url_re.search(data)
            if not match:
                raise ValueError('Badly formatted data: %r' % data)
            url = match.group(1)    # get repository URL
            revs = [int(m.group(1)) for m in _svn_rev_re.finditer(data)] + [0]
        else:
            try:
                # subversion >= 1.7
                xml = self.run_command(
                    ['info', '--xml', location],
                    show_stdout=False,
                )
                url = _svn_info_xml_url_re.search(xml).group(1)
                revs = [
                    int(m.group(1)) for m in _svn_info_xml_rev_re.finditer(xml)
                ]
            except InstallationError:
                url, revs = None, []

        if revs:
            rev = max(revs)
        else:
            rev = 0

        return url, rev

    def get_src_requirement(self, dist, location):
        repo = self.get_url(location)
        if repo is None:
            return None
        # FIXME: why not project name?
        egg_project_name = dist.egg_name().split('-', 1)[0]
        rev = self.get_revision(location)
        return 'svn+%s@%s#egg=%s' % (repo, rev, egg_project_name)

    def check_version(self, dest, rev_options):
        """Always assume the versions don't match"""
        return False

    @staticmethod
    def remove_auth_from_url(url):
        # Return a copy of url with 'username:password@' removed.
        # username/pass params are passed to subversion through flags
        # and are not recognized in the url.

        # parsed url
        purl = urllib_parse.urlsplit(url)
        stripped_netloc = \
            purl.netloc.split('@')[-1]

        # stripped url
        url_pieces = (
            purl.scheme, stripped_netloc, purl.path, purl.query, purl.fragment
        )
        surl = urllib_parse.urlunsplit(url_pieces)
        return surl


def get_rev_options(url, rev):
    if rev:
        rev_options = ['-r', rev]
    else:
        rev_options = []

    r = urllib_parse.urlsplit(url)
    if hasattr(r, 'username'):
        # >= Python-2.5
        username, password = r.username, r.password
    else:
        netloc = r[1]
        if '@' in netloc:
            auth = netloc.split('@')[0]
            if ':' in auth:
                username, password = auth.split(':', 1)
            else:
                username, password = auth, None
        else:
            username, password = None, None

    if username:
        rev_options += ['--username', username]
    if password:
        rev_options += ['--password', password]
    return rev_options


vcs.register(Subversion)
