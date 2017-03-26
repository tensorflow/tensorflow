from __future__ import absolute_import

import time
import os

from . import (LockBase, LockFailed, NotLocked, NotMyLock, LockTimeout,
               AlreadyLocked)


class LinkLockFile(LockBase):
    """Lock access to a file using atomic property of link(2).

    >>> lock = LinkLockFile('somefile')
    >>> lock = LinkLockFile('somefile', threaded=False)
    """

    def acquire(self, timeout=None):
        try:
            open(self.unique_name, "wb").close()
        except IOError:
            raise LockFailed("failed to create %s" % self.unique_name)

        timeout = timeout if timeout is not None else self.timeout
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout

        while True:
            # Try and create a hard link to it.
            try:
                os.link(self.unique_name, self.lock_file)
            except OSError:
                # Link creation failed.  Maybe we've double-locked?
                nlinks = os.stat(self.unique_name).st_nlink
                if nlinks == 2:
                    # The original link plus the one I created == 2.  We're
                    # good to go.
                    return
                else:
                    # Otherwise the lock creation failed.
                    if timeout is not None and time.time() > end_time:
                        os.unlink(self.unique_name)
                        if timeout > 0:
                            raise LockTimeout("Timeout waiting to acquire"
                                              " lock for %s" %
                                              self.path)
                        else:
                            raise AlreadyLocked("%s is already locked" %
                                                self.path)
                    time.sleep(timeout is not None and timeout / 10 or 0.1)
            else:
                # Link creation succeeded.  We're good to go.
                return

    def release(self):
        if not self.is_locked():
            raise NotLocked("%s is not locked" % self.path)
        elif not os.path.exists(self.unique_name):
            raise NotMyLock("%s is locked, but not by me" % self.path)
        os.unlink(self.unique_name)
        os.unlink(self.lock_file)

    def is_locked(self):
        return os.path.exists(self.lock_file)

    def i_am_locking(self):
        return (self.is_locked() and
                os.path.exists(self.unique_name) and
                os.stat(self.unique_name).st_nlink == 2)

    def break_lock(self):
        if os.path.exists(self.lock_file):
            os.unlink(self.lock_file)
