"""Spawn a command with pipes to its stdin, stdout, and optionally stderr.

The normal os.popen(cmd, mode) call spawns a shell command and provides a
file interface to just the input or output of the process depending on
whether mode is 'r' or 'w'.  This module provides the functions popen2(cmd)
and popen3(cmd) which return two or three pipes to the spawned command.
"""

import os
import subprocess
import sys

__all__ = ["popen2", "popen3", "popen4"]

MAXFD = subprocess.MAXFD
_active = subprocess._active
_cleanup = subprocess._cleanup

class Popen3:
    """Class representing a child process.  Normally instances are created
    by the factory functions popen2() and popen3()."""

    sts = -1                    # Child not completed yet

    def __init__(self, cmd, capturestderr=False, bufsize=-1):
        """The parameter 'cmd' is the shell command to execute in a
        sub-process.  On UNIX, 'cmd' may be a sequence, in which case arguments
        will be passed directly to the program without shell intervention (as
        with os.spawnv()).  If 'cmd' is a string it will be passed to the shell
        (as with os.system()).   The 'capturestderr' flag, if true, specifies
        that the object should capture standard error output of the child
        process.  The default is false.  If the 'bufsize' parameter is
        specified, it specifies the size of the I/O buffers to/from the child
        process."""
        stderr = subprocess.PIPE if capturestderr else None
        PIPE = subprocess.PIPE
        self._popen = subprocess.Popen(cmd, bufsize=bufsize,
                                       shell=isinstance(cmd, basestring),
                                       stdin=PIPE, stdout=PIPE, stderr=stderr)
        self._setup(cmd)

    def _setup(self, cmd):
        """Setup the Popen attributes."""
        self.cmd = cmd
        self.pid = self._popen.pid
        self.tochild = self._popen.stdin
        self.fromchild = self._popen.stdout
        self.childerr = self._popen.stderr

    def __del__(self):
        # XXX: Should let _popen __del__ on its own, but it's a new
        # style class: http://bugs.jython.org/issue1057
        if hasattr(self, '_popen'):
            self._popen.__del__()

    def poll(self, _deadstate=None):
        """Return the exit status of the child process if it has finished,
        or -1 if it hasn't finished yet."""
        if self.sts < 0:
            result = self._popen.poll(_deadstate)
            if result is not None:
                self.sts = result
        return self.sts

    def wait(self):
        """Wait for and return the exit status of the child process."""
        if self.sts < 0:
            self.sts = self._popen.wait()
        return self.sts


class Popen4(Popen3):
    childerr = None

    def __init__(self, cmd, bufsize=-1):
        PIPE = subprocess.PIPE
        self._popen = subprocess.Popen(cmd, bufsize=bufsize,
                                       shell=isinstance(cmd, basestring),
                                       stdin=PIPE, stdout=PIPE,
                                       stderr=subprocess.STDOUT)
        self._setup(cmd)


if sys.platform[:3] == "win" or sys.platform == "os2emx":
    # Some things don't make sense on non-Unix platforms.
    del Popen3, Popen4

    def popen2(cmd, bufsize=-1, mode='t'):
        """Execute the shell command 'cmd' in a sub-process. On UNIX, 'cmd' may
        be a sequence, in which case arguments will be passed directly to the
        program without shell intervention (as with os.spawnv()). If 'cmd' is a
        string it will be passed to the shell (as with os.system()). If
        'bufsize' is specified, it sets the buffer size for the I/O pipes. The
        file objects (child_stdout, child_stdin) are returned."""
        w, r = os.popen2(cmd, mode, bufsize)
        return r, w

    def popen3(cmd, bufsize=-1, mode='t'):
        """Execute the shell command 'cmd' in a sub-process. On UNIX, 'cmd' may
        be a sequence, in which case arguments will be passed directly to the
        program without shell intervention (as with os.spawnv()). If 'cmd' is a
        string it will be passed to the shell (as with os.system()). If
        'bufsize' is specified, it sets the buffer size for the I/O pipes. The
        file objects (child_stdout, child_stdin, child_stderr) are returned."""
        w, r, e = os.popen3(cmd, mode, bufsize)
        return r, w, e

    def popen4(cmd, bufsize=-1, mode='t'):
        """Execute the shell command 'cmd' in a sub-process. On UNIX, 'cmd' may
        be a sequence, in which case arguments will be passed directly to the
        program without shell intervention (as with os.spawnv()). If 'cmd' is a
        string it will be passed to the shell (as with os.system()). If
        'bufsize' is specified, it sets the buffer size for the I/O pipes. The
        file objects (child_stdout_stderr, child_stdin) are returned."""
        w, r = os.popen4(cmd, mode, bufsize)
        return r, w
else:
    def popen2(cmd, bufsize=-1, mode='t'):
        """Execute the shell command 'cmd' in a sub-process. On UNIX, 'cmd' may
        be a sequence, in which case arguments will be passed directly to the
        program without shell intervention (as with os.spawnv()). If 'cmd' is a
        string it will be passed to the shell (as with os.system()). If
        'bufsize' is specified, it sets the buffer size for the I/O pipes. The
        file objects (child_stdout, child_stdin) are returned."""
        inst = Popen3(cmd, False, bufsize)
        return inst.fromchild, inst.tochild

    def popen3(cmd, bufsize=-1, mode='t'):
        """Execute the shell command 'cmd' in a sub-process. On UNIX, 'cmd' may
        be a sequence, in which case arguments will be passed directly to the
        program without shell intervention (as with os.spawnv()). If 'cmd' is a
        string it will be passed to the shell (as with os.system()). If
        'bufsize' is specified, it sets the buffer size for the I/O pipes. The
        file objects (child_stdout, child_stdin, child_stderr) are returned."""
        inst = Popen3(cmd, True, bufsize)
        return inst.fromchild, inst.tochild, inst.childerr

    def popen4(cmd, bufsize=-1, mode='t'):
        """Execute the shell command 'cmd' in a sub-process. On UNIX, 'cmd' may
        be a sequence, in which case arguments will be passed directly to the
        program without shell intervention (as with os.spawnv()). If 'cmd' is a
        string it will be passed to the shell (as with os.system()). If
        'bufsize' is specified, it sets the buffer size for the I/O pipes. The
        file objects (child_stdout_stderr, child_stdin) are returned."""
        inst = Popen4(cmd, bufsize)
        return inst.fromchild, inst.tochild

    __all__.extend(["Popen3", "Popen4"])

def _test():
    # When the test runs, there shouldn't be any open pipes
    _cleanup()
    assert not _active, "Active pipes when test starts " + repr([c.cmd for c in _active])
    cmd  = "cat"
    teststr = "ab cd\n"
    if os.name in ("nt", "java"):
        cmd = "more"
    # "more" doesn't act the same way across Windows flavors,
    # sometimes adding an extra newline at the start or the
    # end.  So we strip whitespace off both ends for comparison.
    expected = teststr.strip()
    print "testing popen2..."
    r, w = popen2(cmd)
    w.write(teststr)
    w.close()
    got = r.read()
    if got.strip() != expected:
        raise ValueError("wrote %r read %r" % (teststr, got))
    print "testing popen3..."
    try:
        r, w, e = popen3([cmd])
    except:
        r, w, e = popen3(cmd)
    w.write(teststr)
    w.close()
    got = r.read()
    if got.strip() != expected:
        raise ValueError("wrote %r read %r" % (teststr, got))
    got = e.read()
    if got:
        raise ValueError("unexpected %r on stderr" % (got,))
    for inst in _active[:]:
        inst.wait()
    _cleanup()
    if _active:
        raise ValueError("_active not empty")
    print "All OK"

if __name__ == '__main__':
    _test()
