from __future__ import absolute_import

import logging
import os
import tempfile

# TODO: Get this into six.moves.urllib.parse
try:
    from urllib import parse as urllib_parse
except ImportError:
    import urlparse as urllib_parse

from pip.utils import rmtree, display_path
from pip.vcs import vcs, VersionControl
from pip.download import path_to_url


logger = logging.getLogger(__name__)


class Bazaar(VersionControl):
    name = 'bzr'
    dirname = '.bzr'
    repo_name = 'branch'
    schemes = (
        'bzr', 'bzr+http', 'bzr+https', 'bzr+ssh', 'bzr+sftp', 'bzr+ftp',
        'bzr+lp',
    )

    def __init__(self, url=None, *args, **kwargs):
        super(Bazaar, self).__init__(url, *args, **kwargs)
        # Python >= 2.7.4, 3.3 doesn't have uses_fragment or non_hierarchical
        # Register lp but do not expose as a scheme to support bzr+lp.
        if getattr(urllib_parse, 'uses_fragment', None):
            urllib_parse.uses_fragment.extend(['lp'])
            urllib_parse.non_hierarchical.extend(['lp'])

    def export(self, location):
        """
        Export the Bazaar repository at the url to the destination location
        """
        temp_dir = tempfile.mkdtemp('-export', 'pip-')
        self.unpack(temp_dir)
        if os.path.exists(location):
            # Remove the location to make sure Bazaar can export it correctly
            rmtree(location)
        try:
            self.run_command(['export', location], cwd=temp_dir,
                             show_stdout=False)
        finally:
            rmtree(temp_dir)

    def switch(self, dest, url, rev_options):
        self.run_command(['switch', url], cwd=dest)

    def update(self, dest, rev_options):
        self.run_command(['pull', '-q'] + rev_options, cwd=dest)

    def obtain(self, dest):
        url, rev = self.get_url_rev()
        if rev:
            rev_options = ['-r', rev]
            rev_display = ' (to revision %s)' % rev
        else:
            rev_options = []
            rev_display = ''
        if self.check_destination(dest, url, rev_options, rev_display):
            logger.info(
                'Checking out %s%s to %s',
                url,
                rev_display,
                display_path(dest),
            )
            self.run_command(['branch', '-q'] + rev_options + [url, dest])

    def get_url_rev(self):
        # hotfix the URL scheme after removing bzr+ from bzr+ssh:// readd it
        url, rev = super(Bazaar, self).get_url_rev()
        if url.startswith('ssh://'):
            url = 'bzr+' + url
        return url, rev

    def get_url(self, location):
        urls = self.run_command(['info'], show_stdout=False, cwd=location)
        for line in urls.splitlines():
            line = line.strip()
            for x in ('checkout of branch: ',
                      'parent branch: '):
                if line.startswith(x):
                    repo = line.split(x)[1]
                    if self._is_local_repository(repo):
                        return path_to_url(repo)
                    return repo
        return None

    def get_revision(self, location):
        revision = self.run_command(
            ['revno'], show_stdout=False, cwd=location)
        return revision.splitlines()[-1]

    def get_src_requirement(self, dist, location):
        repo = self.get_url(location)
        if not repo:
            return None
        if not repo.lower().startswith('bzr:'):
            repo = 'bzr+' + repo
        egg_project_name = dist.egg_name().split('-', 1)[0]
        current_rev = self.get_revision(location)
        return '%s@%s#egg=%s' % (repo, current_rev, egg_project_name)

    def check_version(self, dest, rev_options):
        """Always assume the versions don't match"""
        return False


vcs.register(Bazaar)
