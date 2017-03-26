from __future__ import absolute_import

import os
import time

from . import (LockBase, NotLocked, NotMyLock, LockTimeout,
               AlreadyLocked)


class SymlinkLockFile(LockBase):
    """Lock access to a file using symlink(2)."""

    def __init__(self, path, threaded=True, timeout=None):
        # super(SymlinkLockFile).__init(...)
        LockBase.__init__(self, path, threaded, timeout)
        # split it back!
        self.unique_name = os.path.split(self.unique_name)[1]

    def acquire(self, timeout=None):
        # Hopefully unnecessary for symlink.
        # try:
        #     open(self.unique_name, "wb").close()
        # except IOError:
        #     raise LockFailed("failed to create %s" % self.unique_name)
        timeout = timeout if timeout is not None else self.timeout
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout

        while True:
            # Try and create a symbolic link to it.
            try:
                os.symlink(self.unique_name, self.lock_file)
            except OSError:
                # Link creation failed.  Maybe we've double-locked?
                if self.i_am_locking():
                    # Linked to out unique name. Proceed.
                    return
                else:
                    # Otherwise the lock creation failed.
                    if timeout is not None and time.time() > end_time:
                        if timeout > 0:
                            raise LockTimeout("Timeout waiting to acquire"
                                              " lock for %s" %
                                              self.path)
                        else:
                            raise AlreadyLocked("%s is already locked" %
                                                self.path)
                    time.sleep(timeout / 10 if timeout is not None else 0.1)
            else:
                # Link creation succeeded.  We're good to go.
                return

    def release(self):
        if not self.is_locked():
            raise NotLocked("%s is not locked" % self.path)
        elif not self.i_am_locking():
            raise NotMyLock("%s is locked, but not by me" % self.path)
        os.unlink(self.lock_file)

    def is_locked(self):
        return os.path.islink(self.lock_file)

    def i_am_locking(self):
        return (os.path.islink(self.lock_file)
                and os.readlink(self.lock_file) == self.unique_name)

    def break_lock(self):
        if os.path.islink(self.lock_file):  # exists && link
            os.unlink(self.lock_file)
