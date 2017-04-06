from __future__ import absolute_import

import logging
import os
import tempfile

from pip.compat import uses_pycache, WINDOWS, cache_from_source
from pip.exceptions import UninstallationError
from pip.utils import rmtree, ask, is_local, renames, normalize_path
from pip.utils.logging import indent_log


logger = logging.getLogger(__name__)


class UninstallPathSet(object):
    """A set of file paths to be removed in the uninstallation of a
    requirement."""
    def __init__(self, dist):
        self.paths = set()
        self._refuse = set()
        self.pth = {}
        self.dist = dist
        self.save_dir = None
        self._moved_paths = []

    def _permitted(self, path):
        """
        Return True if the given path is one we are permitted to
        remove/modify, False otherwise.

        """
        return is_local(path)

    def add(self, path):
        head, tail = os.path.split(path)

        # we normalize the head to resolve parent directory symlinks, but not
        # the tail, since we only want to uninstall symlinks, not their targets
        path = os.path.join(normalize_path(head), os.path.normcase(tail))

        if not os.path.exists(path):
            return
        if self._permitted(path):
            self.paths.add(path)
        else:
            self._refuse.add(path)

        # __pycache__ files can show up after 'installed-files.txt' is created,
        # due to imports
        if os.path.splitext(path)[1] == '.py' and uses_pycache:
            self.add(cache_from_source(path))

    def add_pth(self, pth_file, entry):
        pth_file = normalize_path(pth_file)
        if self._permitted(pth_file):
            if pth_file not in self.pth:
                self.pth[pth_file] = UninstallPthEntries(pth_file)
            self.pth[pth_file].add(entry)
        else:
            self._refuse.add(pth_file)

    def compact(self, paths):
        """Compact a path set to contain the minimal number of paths
        necessary to contain all paths in the set. If /a/path/ and
        /a/path/to/a/file.txt are both in the set, leave only the
        shorter path."""
        short_paths = set()
        for path in sorted(paths, key=len):
            if not any([
                    (path.startswith(shortpath) and
                     path[len(shortpath.rstrip(os.path.sep))] == os.path.sep)
                    for shortpath in short_paths]):
                short_paths.add(path)
        return short_paths

    def _stash(self, path):
        return os.path.join(
            self.save_dir, os.path.splitdrive(path)[1].lstrip(os.path.sep))

    def remove(self, auto_confirm=False):
        """Remove paths in ``self.paths`` with confirmation (unless
        ``auto_confirm`` is True)."""
        if not self.paths:
            logger.info(
                "Can't uninstall '%s'. No files were found to uninstall.",
                self.dist.project_name,
            )
            return
        logger.info(
            'Uninstalling %s-%s:',
            self.dist.project_name, self.dist.version
        )

        with indent_log():
            paths = sorted(self.compact(self.paths))

            if auto_confirm:
                response = 'y'
            else:
                for path in paths:
                    logger.info(path)
                response = ask('Proceed (y/n)? ', ('y', 'n'))
            if self._refuse:
                logger.info('Not removing or modifying (outside of prefix):')
                for path in self.compact(self._refuse):
                    logger.info(path)
            if response == 'y':
                self.save_dir = tempfile.mkdtemp(suffix='-uninstall',
                                                 prefix='pip-')
                for path in paths:
                    new_path = self._stash(path)
                    logger.debug('Removing file or directory %s', path)
                    self._moved_paths.append(path)
                    renames(path, new_path)
                for pth in self.pth.values():
                    pth.remove()
                logger.info(
                    'Successfully uninstalled %s-%s',
                    self.dist.project_name, self.dist.version
                )

    def rollback(self):
        """Rollback the changes previously made by remove()."""
        if self.save_dir is None:
            logger.error(
                "Can't roll back %s; was not uninstalled",
                self.dist.project_name,
            )
            return False
        logger.info('Rolling back uninstall of %s', self.dist.project_name)
        for path in self._moved_paths:
            tmp_path = self._stash(path)
            logger.debug('Replacing %s', path)
            renames(tmp_path, path)
        for pth in self.pth.values():
            pth.rollback()

    def commit(self):
        """Remove temporary save dir: rollback will no longer be possible."""
        if self.save_dir is not None:
            rmtree(self.save_dir)
            self.save_dir = None
            self._moved_paths = []


class UninstallPthEntries(object):
    def __init__(self, pth_file):
        if not os.path.isfile(pth_file):
            raise UninstallationError(
                "Cannot remove entries from nonexistent file %s" % pth_file
            )
        self.file = pth_file
        self.entries = set()
        self._saved_lines = None

    def add(self, entry):
        entry = os.path.normcase(entry)
        # On Windows, os.path.normcase converts the entry to use
        # backslashes.  This is correct for entries that describe absolute
        # paths outside of site-packages, but all the others use forward
        # slashes.
        if WINDOWS and not os.path.splitdrive(entry)[0]:
            entry = entry.replace('\\', '/')
        self.entries.add(entry)

    def remove(self):
        logger.debug('Removing pth entries from %s:', self.file)
        with open(self.file, 'rb') as fh:
            # windows uses '\r\n' with py3k, but uses '\n' with py2.x
            lines = fh.readlines()
            self._saved_lines = lines
        if any(b'\r\n' in line for line in lines):
            endline = '\r\n'
        else:
            endline = '\n'
        for entry in self.entries:
            try:
                logger.debug('Removing entry: %s', entry)
                lines.remove((entry + endline).encode("utf-8"))
            except ValueError:
                pass
        with open(self.file, 'wb') as fh:
            fh.writelines(lines)

    def rollback(self):
        if self._saved_lines is None:
            logger.error(
                'Cannot roll back changes to %s, none were made', self.file
            )
            return False
        logger.debug('Rolling %s back to previous state', self.file)
        with open(self.file, 'wb') as fh:
            fh.writelines(self._saved_lines)
        return True
