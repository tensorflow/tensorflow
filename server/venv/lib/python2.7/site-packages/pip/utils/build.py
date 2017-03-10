from __future__ import absolute_import

import os.path
import tempfile

from pip.utils import rmtree


class BuildDirectory(object):

    def __init__(self, name=None, delete=None):
        # If we were not given an explicit directory, and we were not given an
        # explicit delete option, then we'll default to deleting.
        if name is None and delete is None:
            delete = True

        if name is None:
            # We realpath here because some systems have their default tmpdir
            # symlinked to another directory.  This tends to confuse build
            # scripts, so we canonicalize the path by traversing potential
            # symlinks here.
            name = os.path.realpath(tempfile.mkdtemp(prefix="pip-build-"))
            # If we were not given an explicit directory, and we were not given
            # an explicit delete option, then we'll default to deleting.
            if delete is None:
                delete = True

        self.name = name
        self.delete = delete

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def cleanup(self):
        if self.delete:
            rmtree(self.name)
