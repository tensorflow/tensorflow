from __future__ import absolute_import, division

import time
import os
import sys
import errno

from . import (LockBase, LockFailed, NotLocked, NotMyLock, LockTimeout,
               AlreadyLocked)


class MkdirLockFile(LockBase):
    """Lock file by creating a directory."""
    def __init__(self, path, threaded=True, timeout=None):
        """
        >>> lock = MkdirLockFile('somefile')
        >>> lock = MkdirLockFile('somefile', threaded=False)
        """
        LockBase.__init__(self, path, threaded, timeout)
        # Lock file itself is a directory.  Place the unique file name into
        # it.
        self.unique_name = os.path.join(self.lock_file,
                                        "%s.%s%s" % (self.hostname,
                                                     self.tname,
                                                     self.pid))

    def acquire(self, timeout=None):
        timeout = timeout if timeout is not None else self.timeout
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout

        if timeout is None:
            wait = 0.1
        else:
            wait = max(0, timeout / 10)

        while True:
            try:
                os.mkdir(self.lock_file)
            except OSError:
                err = sys.exc_info()[1]
                if err.errno == errno.EEXIST:
                    # Already locked.
                    if os.path.exists(self.unique_name):
                        # Already locked by me.
                        return
                    if timeout is not None and time.time() > end_time:
                        if timeout > 0:
                            raise LockTimeout("Timeout waiting to acquire"
                                              " lock for %s" %
                                              self.path)
                        else:
                            # Someone else has the lock.
                            raise AlreadyLocked("%s is already locked" %
                                                self.path)
                    time.sleep(wait)
                else:
                    # Couldn't create the lock for some other reason
                    raise LockFailed("failed to create %s" % self.lock_file)
            else:
                open(self.unique_name, "wb").close()
                return

    def release(self):
        if not self.is_locked():
            raise NotLocked("%s is not locked" % self.path)
        elif not os.path.exists(self.unique_name):
            raise NotMyLock("%s is locked, but not by me" % self.path)
        os.unlink(self.unique_name)
        os.rmdir(self.lock_file)

    def is_locked(self):
        return os.path.exists(self.lock_file)

    def i_am_locking(self):
        return (self.is_locked() and
                os.path.exists(self.unique_name))

    def break_lock(self):
        if os.path.exists(self.lock_file):
            for name in os.listdir(self.lock_file):
                os.unlink(os.path.join(self.lock_file, name))
            os.rmdir(self.lock_file)
