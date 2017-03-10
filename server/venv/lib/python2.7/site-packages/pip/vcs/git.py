from __future__ import absolute_import

import logging
import tempfile
import os.path

from pip.compat import samefile
from pip.exceptions import BadCommand
from pip._vendor.six.moves.urllib import parse as urllib_parse
from pip._vendor.six.moves.urllib import request as urllib_request
from pip._vendor.packaging.version import parse as parse_version

from pip.utils import display_path, rmtree
from pip.vcs import vcs, VersionControl


urlsplit = urllib_parse.urlsplit
urlunsplit = urllib_parse.urlunsplit


logger = logging.getLogger(__name__)


class Git(VersionControl):
    name = 'git'
    dirname = '.git'
    repo_name = 'clone'
    schemes = (
        'git', 'git+http', 'git+https', 'git+ssh', 'git+git', 'git+file',
    )

    def __init__(self, url=None, *args, **kwargs):

        # Works around an apparent Git bug
        # (see http://article.gmane.org/gmane.comp.version-control.git/146500)
        if url:
            scheme, netloc, path, query, fragment = urlsplit(url)
            if scheme.endswith('file'):
                initial_slashes = path[:-len(path.lstrip('/'))]
                newpath = (
                    initial_slashes +
                    urllib_request.url2pathname(path)
                    .replace('\\', '/').lstrip('/')
                )
                url = urlunsplit((scheme, netloc, newpath, query, fragment))
                after_plus = scheme.find('+') + 1
                url = scheme[:after_plus] + urlunsplit(
                    (scheme[after_plus:], netloc, newpath, query, fragment),
                )

        super(Git, self).__init__(url, *args, **kwargs)

    def get_git_version(self):
        VERSION_PFX = 'git version '
        version = self.run_command(['version'], show_stdout=False)
        if version.startswith(VERSION_PFX):
            version = version[len(VERSION_PFX):]
        else:
            version = ''
        # get first 3 positions of the git version becasue
        # on windows it is x.y.z.windows.t, and this parses as
        # LegacyVersion which always smaller than a Version.
        version = '.'.join(version.split('.')[:3])
        return parse_version(version)

    def export(self, location):
        """Export the Git repository at the url to the destination location"""
        temp_dir = tempfile.mkdtemp('-export', 'pip-')
        self.unpack(temp_dir)
        try:
            if not location.endswith('/'):
                location = location + '/'
            self.run_command(
                ['checkout-index', '-a', '-f', '--prefix', location],
                show_stdout=False, cwd=temp_dir)
        finally:
            rmtree(temp_dir)

    def check_rev_options(self, rev, dest, rev_options):
        """Check the revision options before checkout to compensate that tags
        and branches may need origin/ as a prefix.
        Returns the SHA1 of the branch or tag if found.
        """
        revisions = self.get_short_refs(dest)

        origin_rev = 'origin/%s' % rev
        if origin_rev in revisions:
            # remote branch
            return [revisions[origin_rev]]
        elif rev in revisions:
            # a local tag or branch name
            return [revisions[rev]]
        else:
            logger.warning(
                "Could not find a tag or branch '%s', assuming commit.", rev,
            )
            return rev_options

    def check_version(self, dest, rev_options):
        """
        Compare the current sha to the ref. ref may be a branch or tag name,
        but current rev will always point to a sha. This means that a branch
        or tag will never compare as True. So this ultimately only matches
        against exact shas.
        """
        return self.get_revision(dest).startswith(rev_options[0])

    def switch(self, dest, url, rev_options):
        self.run_command(['config', 'remote.origin.url', url], cwd=dest)
        self.run_command(['checkout', '-q'] + rev_options, cwd=dest)

        self.update_submodules(dest)

    def update(self, dest, rev_options):
        # First fetch changes from the default remote
        if self.get_git_version() >= parse_version('1.9.0'):
            # fetch tags in addition to everything else
            self.run_command(['fetch', '-q', '--tags'], cwd=dest)
        else:
            self.run_command(['fetch', '-q'], cwd=dest)
        # Then reset to wanted revision (maybe even origin/master)
        if rev_options:
            rev_options = self.check_rev_options(
                rev_options[0], dest, rev_options,
            )
        self.run_command(['reset', '--hard', '-q'] + rev_options, cwd=dest)
        #: update submodules
        self.update_submodules(dest)

    def obtain(self, dest):
        url, rev = self.get_url_rev()
        if rev:
            rev_options = [rev]
            rev_display = ' (to %s)' % rev
        else:
            rev_options = ['origin/master']
            rev_display = ''
        if self.check_destination(dest, url, rev_options, rev_display):
            logger.info(
                'Cloning %s%s to %s', url, rev_display, display_path(dest),
            )
            self.run_command(['clone', '-q', url, dest])

            if rev:
                rev_options = self.check_rev_options(rev, dest, rev_options)
                # Only do a checkout if rev_options differs from HEAD
                if not self.check_version(dest, rev_options):
                    self.run_command(
                        ['checkout', '-q'] + rev_options,
                        cwd=dest,
                    )
            #: repo may contain submodules
            self.update_submodules(dest)

    def get_url(self, location):
        """Return URL of the first remote encountered."""
        remotes = self.run_command(
            ['config', '--get-regexp', 'remote\..*\.url'],
            show_stdout=False, cwd=location)
        remotes = remotes.splitlines()
        found_remote = remotes[0]
        for remote in remotes:
            if remote.startswith('remote.origin.url '):
                found_remote = remote
                break
        url = found_remote.split(' ')[1]
        return url.strip()

    def get_revision(self, location):
        current_rev = self.run_command(
            ['rev-parse', 'HEAD'], show_stdout=False, cwd=location)
        return current_rev.strip()

    def get_full_refs(self, location):
        """Yields tuples of (commit, ref) for branches and tags"""
        output = self.run_command(['show-ref'],
                                  show_stdout=False, cwd=location)
        for line in output.strip().splitlines():
            commit, ref = line.split(' ', 1)
            yield commit.strip(), ref.strip()

    def is_ref_remote(self, ref):
        return ref.startswith('refs/remotes/')

    def is_ref_branch(self, ref):
        return ref.startswith('refs/heads/')

    def is_ref_tag(self, ref):
        return ref.startswith('refs/tags/')

    def is_ref_commit(self, ref):
        """A ref is a commit sha if it is not anything else"""
        return not any((
            self.is_ref_remote(ref),
            self.is_ref_branch(ref),
            self.is_ref_tag(ref),
        ))

    # Should deprecate `get_refs` since it's ambiguous
    def get_refs(self, location):
        return self.get_short_refs(location)

    def get_short_refs(self, location):
        """Return map of named refs (branches or tags) to commit hashes."""
        rv = {}
        for commit, ref in self.get_full_refs(location):
            ref_name = None
            if self.is_ref_remote(ref):
                ref_name = ref[len('refs/remotes/'):]
            elif self.is_ref_branch(ref):
                ref_name = ref[len('refs/heads/'):]
            elif self.is_ref_tag(ref):
                ref_name = ref[len('refs/tags/'):]
            if ref_name is not None:
                rv[ref_name] = commit
        return rv

    def _get_subdirectory(self, location):
        """Return the relative path of setup.py to the git repo root."""
        # find the repo root
        git_dir = self.run_command(['rev-parse', '--git-dir'],
                                   show_stdout=False, cwd=location).strip()
        if not os.path.isabs(git_dir):
            git_dir = os.path.join(location, git_dir)
        root_dir = os.path.join(git_dir, '..')
        # find setup.py
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
        # relative path of setup.py to repo root
        if samefile(root_dir, location):
            return None
        return os.path.relpath(location, root_dir)

    def get_src_requirement(self, dist, location):
        repo = self.get_url(location)
        if not repo.lower().startswith('git:'):
            repo = 'git+' + repo
        egg_project_name = dist.egg_name().split('-', 1)[0]
        if not repo:
            return None
        current_rev = self.get_revision(location)
        req = '%s@%s#egg=%s' % (repo, current_rev, egg_project_name)
        subdirectory = self._get_subdirectory(location)
        if subdirectory:
            req += '&subdirectory=' + subdirectory
        return req

    def get_url_rev(self):
        """
        Prefixes stub URLs like 'user@hostname:user/repo.git' with 'ssh://'.
        That's required because although they use SSH they sometimes doesn't
        work with a ssh:// scheme (e.g. Github). But we need a scheme for
        parsing. Hence we remove it again afterwards and return it as a stub.
        """
        if '://' not in self.url:
            assert 'file:' not in self.url
            self.url = self.url.replace('git+', 'git+ssh://')
            url, rev = super(Git, self).get_url_rev()
            url = url.replace('ssh://', '')
        else:
            url, rev = super(Git, self).get_url_rev()

        return url, rev

    def update_submodules(self, location):
        if not os.path.exists(os.path.join(location, '.gitmodules')):
            return
        self.run_command(
            ['submodule', 'update', '--init', '--recursive', '-q'],
            cwd=location,
        )

    @classmethod
    def controls_location(cls, location):
        if super(Git, cls).controls_location(location):
            return True
        try:
            r = cls().run_command(['rev-parse'],
                                  cwd=location,
                                  show_stdout=False,
                                  on_returncode='ignore')
            return not r
        except BadCommand:
            logger.debug("could not determine if %s is under git control "
                         "because git is not available", location)
            return False


vcs.register(Git)
