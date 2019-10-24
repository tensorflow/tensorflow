r"""OS routines for Java, with some attempts to support NT, and Posix
functionality.

This exports:
  - all functions from posix, nt, dos, os2, mac, or ce, e.g. unlink, stat, etc.
  - os.path is one of the modules posixpath, ntpath, macpath, or dospath
  - os.name is 'posix', 'nt', 'dos', 'os2', 'mac', 'ce' or 'riscos'
  - os.curdir is a string representing the current directory ('.' or ':')
  - os.pardir is a string representing the parent directory ('..' or '::')
  - os.sep is the (or a most common) pathname separator ('/' or ':' or '\\')
  - os.altsep is the alternate pathname separator (None or '/')
  - os.pathsep is the component separator used in $PATH etc
  - os.linesep is the line separator in text files ('\r' or '\n' or '\r\n')
  - os.defpath is the default search path for executables

Programs that import and use 'os' stand a better chance of being
portable between different platforms.  Of course, they must then
only use functions that are defined by all platforms (e.g., unlink
and opendir), and leave all pathname manipulation to os.path
(e.g., split and join).
"""

# CPython os.py __all__
__all__ = ["altsep", "curdir", "pardir", "sep", "pathsep", "linesep",
           "defpath", "name", "path",
           "SEEK_SET", "SEEK_CUR", "SEEK_END"]

# Would come from the posix/nt/etc. modules on CPython
__all__.extend(['EX_OK', 'F_OK', 'O_APPEND', 'O_CREAT', 'O_EXCL', 'O_RDONLY',
                'O_RDWR', 'O_SYNC', 'O_TRUNC', 'O_WRONLY', 'R_OK', 'SEEK_CUR',
                'SEEK_END', 'SEEK_SET', 'W_OK', 'X_OK', '_exit', 'access',
                'altsep', 'chdir', 'chmod', 'close', 'curdir', 'defpath',
                'environ', 'error', 'fdopen', 'fsync', 'getcwd', 'getcwdu',
                'getenv', 'getpid', 'isatty', 'linesep', 'listdir', 'lseek',
                'lstat', 'makedirs', 'mkdir', 'name', 'open', 'pardir', 'path',
                'pathsep', 'popen', 'popen2', 'popen3', 'popen4', 'putenv',
                'read', 'remove', 'removedirs', 'rename', 'renames', 'rmdir',
                'sep', 'stat', 'stat_result', 'strerror', 'system', 'unlink',
                'unsetenv', 'utime', 'walk', 'write'])

import errno
import jarray
import java.lang.System
import time
import stat as _stat
import sys
from java.io import File
from org.python.core.io import FileDescriptors, FileIO, IOBase
from org.python.core.Py import newString as asPyString

try:
    from org.python.constantine.platform import Errno
except ImportError:
    from com.kenai.constantine.platform import Errno

# Mapping of: os._name: [name list, shell command list]
_os_map = dict(nt=[
        ['Windows'],
        [['cmd.exe', '/c'], ['command.com', '/c']]
        ],
               posix=[
        [], # posix is a fallback, instead of matching names
        [['/bin/sh', '-c']]
        ]
               )

def get_os_type():
    """Return the name of the type of the underlying OS.

    Returns a value suitable for the os.name variable (though not
    necessarily intended to be for os.name Jython).  This value may be
    overwritten in the Jython registry.
    """
    os_name = sys.registry.getProperty('python.os')
    if os_name:
        return asPyString(os_name)

    os_name = asPyString(java.lang.System.getProperty('os.name'))
    os_type = None
    for type, (patterns, shell_commands) in _os_map.iteritems():
        for pattern in patterns:
            if os_name.startswith(pattern):
                # determine the shell_command later, when it's needed:
                # it requires os.path (which isn't setup yet)
                return type
    return 'posix'

name = 'java'
# WARNING: _name is private: for Jython internal usage only! user code
# should *NOT* use it
_name = get_os_type()

try:
    from org.python.posix import JavaPOSIX, POSIXHandler, POSIXFactory
except ImportError:
    from org.jruby.ext.posix import JavaPOSIX, POSIXHandler, POSIXFactory

class PythonPOSIXHandler(POSIXHandler):
    def error(self, error, msg):
        err = getattr(errno, error.name(), None)
        if err is None:
            raise OSError('%s: %s' % (error, asPyString(msg)))
        raise OSError(err, strerror(err), asPyString(msg))
    def unimplementedError(self, method_name):
        raise NotImplementedError(method_name)
    def warn(self, warning_id, msg, rest):
        pass # XXX implement
    def isVerbose(self):
        return False
    def getCurrentWorkingDirectory(self):
        return File(getcwdu())
    def getEnv(self):
        return ['%s=%s' % (key, val) for key, val in environ.iteritems()]
    def getInputStream(self):
        return getattr(java.lang.System, 'in') # XXX handle resetting
    def getOutputStream(self):
        return java.lang.System.out # XXX handle resetting
    def getPID(self):
        return 0
    def getErrorStream(self):
        return java.lang.System.err # XXX handle resetting

_posix = POSIXFactory.getPOSIX(PythonPOSIXHandler(), True)
_native_posix = not isinstance(_posix, JavaPOSIX)

if _name == 'nt':
    import ntpath as path
else:
    import posixpath as path

sys.modules['os.path'] = _path = path
from os.path import curdir, pardir, sep, pathsep, defpath, extsep, altsep, devnull
linesep = java.lang.System.getProperty('line.separator')

# open for reading only
O_RDONLY = 0x0
# open for writing only
O_WRONLY = 0x1
# open for reading and writing
O_RDWR = 0x2

# set append mode
O_APPEND = 0x8
# synchronous writes
O_SYNC = 0x80

# create if nonexistant
O_CREAT = 0x200
# truncate to zero length
O_TRUNC = 0x400
# error if already exists
O_EXCL = 0x800

# seek variables
SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2

# test for existence of file
F_OK = 0
# test for execute or search permission
X_OK = 1<<0
# test for write permission
W_OK = 1<<1
# test for read permission
R_OK = 1<<2

# successful termination
EX_OK = 0

# Java class representing the size of a time_t. internal use, lazily set
_time_t = None

class stat_result:

    _stat_members = (
      ('st_mode', _stat.ST_MODE),
      ('st_ino', _stat.ST_INO),
      ('st_dev', _stat.ST_DEV),
      ('st_nlink', _stat.ST_NLINK),
      ('st_uid', _stat.ST_UID),
      ('st_gid', _stat.ST_GID),
      ('st_size', _stat.ST_SIZE),
      ('st_atime', _stat.ST_ATIME),
      ('st_mtime', _stat.ST_MTIME),
      ('st_ctime', _stat.ST_CTIME),
    )

    def __init__(self, results):
        if len(results) != 10:
            raise TypeError("stat_result() takes an a 10-sequence")
        for (name, index) in stat_result._stat_members:
            self.__dict__[name] = results[index]

    @classmethod
    def from_jnastat(cls, s):
        results = []
        for meth in (s.mode, s.ino, s.dev, s.nlink, s.uid, s.gid, s.st_size,
                     s.atime, s.mtime, s.ctime):
            try:
                results.append(meth())
            except NotImplementedError:
                results.append(0)
        return cls(results)

    def __getitem__(self, i):
        if i < 0 or i > 9:
            raise IndexError(i)
        return getattr(self, stat_result._stat_members[i][0])

    def __setitem__(self, x, value):
        raise TypeError("object doesn't support item assignment")

    def __setattr__(self, name, value):
        if name in [x[0] for x in stat_result._stat_members]:
            raise TypeError(name)
        raise AttributeError("readonly attribute")

    def __len__(self):
        return 10

    def __cmp__(self, other):
        if not isinstance(other, stat_result):
            return 1
        return cmp(self.__dict__, other.__dict__)

    def __repr__(self):
        return repr(tuple(self.__dict__[member[0]] for member
                          in stat_result._stat_members))

error = OSError

def _exit(n=0):
    """_exit(status)

    Exit to the system with specified status, without normal exit
    processing.
    """
    java.lang.System.exit(n)

def getcwd():
    """getcwd() -> path

    Return a string representing the current working directory.
    """
    return asPyString(sys.getCurrentWorkingDir())

def getcwdu():
    """getcwd() -> path

    Return a unicode string representing the current working directory.
    """
    return sys.getCurrentWorkingDir()

def chdir(path):
    """chdir(path)

    Change the current working directory to the specified path.
    """
    realpath = _path.realpath(path)
    if not _path.exists(realpath):
        raise OSError(errno.ENOENT, strerror(errno.ENOENT), path)
    if not _path.isdir(realpath):
        raise OSError(errno.ENOTDIR, strerror(errno.ENOTDIR), path)
    sys.setCurrentWorkingDir(realpath)

def listdir(path):
    """listdir(path) -> list_of_strings

    Return a list containing the names of the entries in the directory.

        path: path of directory to list

    The list is in arbitrary order.  It does not include the special
    entries '.' and '..' even if they are present in the directory.
    """
    l = File(sys.getPath(path)).list()
    if l is None:
        raise OSError(0, 'No such directory', path)
    return [asPyString(entry) for entry in l]

def chmod(path, mode):
    """chmod(path, mode)

    Change the access permissions of a file.
    """
    # XXX no error handling for chmod in jna-posix
    # catch not found errors explicitly here, for now
    abs_path = sys.getPath(path)
    if not File(abs_path).exists():
        raise OSError(errno.ENOENT, strerror(errno.ENOENT), path)
    _posix.chmod(abs_path, mode)

def mkdir(path, mode='ignored'):
    """mkdir(path [, mode=0777])

    Create a directory.

    The optional parameter is currently ignored.
    """
    # XXX: use _posix.mkdir when we can get the real errno upon failure
    fp = File(sys.getPath(path))
    if not fp.mkdir():
        if fp.isDirectory() or fp.isFile():
            err = errno.EEXIST
        else:
            err = 0
        msg = strerror(err) if err else "couldn't make directory"
        raise OSError(err, msg, path)

def makedirs(path, mode='ignored'):
    """makedirs(path [, mode=0777])

    Super-mkdir; create a leaf directory and all intermediate ones.

    Works like mkdir, except that any intermediate path segment (not
    just the rightmost) will be created if it does not exist.
    The optional parameter is currently ignored.
    """
    sys_path = sys.getPath(path)
    if File(sys_path).mkdirs():
        return

    # if making a /x/y/z/., java.io.File#mkdirs inexplicably fails. So we need
    # to force it

    # need to use _path instead of path, because param is hiding
    # os.path module in namespace!
    head, tail = _path.split(sys_path)
    if tail == curdir:
        if File(_path.join(head)).mkdirs():
            return

    raise OSError(0, "couldn't make directories", path)

def remove(path):
    """remove(path)

    Remove a file (same as unlink(path)).
    """
    if not File(sys.getPath(path)).delete():
        raise OSError(0, "couldn't delete file", path)

unlink = remove

def rename(path, newpath):
    """rename(old, new)

    Rename a file or directory.
    """
    if not File(sys.getPath(path)).renameTo(File(sys.getPath(newpath))):
        raise OSError(0, "couldn't rename file", path)

#XXX: copied from CPython 2.5.1
def renames(old, new):
    """renames(old, new)

    Super-rename; create directories as necessary and delete any left
    empty.  Works like rename, except creation of any intermediate
    directories needed to make the new pathname good is attempted
    first.  After the rename, directories corresponding to rightmost
    path segments of the old name will be pruned way until either the
    whole path is consumed or a nonempty directory is found.

    Note: this function can fail with the new directory structure made
    if you lack permissions needed to unlink the leaf directory or
    file.

    """
    head, tail = path.split(new)
    if head and tail and not path.exists(head):
        makedirs(head)
    rename(old, new)
    head, tail = path.split(old)
    if head and tail:
        try:
            removedirs(head)
        except error:
            pass

def rmdir(path):
    """rmdir(path)

    Remove a directory."""
    f = File(sys.getPath(path))
    if not f.exists():
        raise OSError(errno.ENOENT, strerror(errno.ENOENT), path)
    elif not f.isDirectory():
        raise OSError(errno.ENOTDIR, strerror(errno.ENOTDIR), path)
    elif not f.delete():
        raise OSError(0, "couldn't delete directory", path)

#XXX: copied from CPython 2.5.1
def removedirs(name):
    """removedirs(path)

    Super-rmdir; remove a leaf directory and empty all intermediate
    ones.  Works like rmdir except that, if the leaf directory is
    successfully removed, directories corresponding to rightmost path
    segments will be pruned away until either the whole path is
    consumed or an error occurs.  Errors during this latter phase are
    ignored -- they generally mean that a directory was not empty.

    """
    rmdir(name)
    head, tail = path.split(name)
    if not tail:
        head, tail = path.split(head)
    while head and tail:
        try:
            rmdir(head)
        except error:
            break
        head, tail = path.split(head)

__all__.extend(['makedirs', 'renames', 'removedirs'])

def strerror(code):
    """strerror(code) -> string

    Translate an error code to a message string.
    """
    if not isinstance(code, (int, long)):
        raise TypeError('an integer is required')
    constant = Errno.valueOf(code)
    if constant is Errno.__UNKNOWN_CONSTANT__:
        return 'Unknown error: %d' % code
    if constant.name() == constant.description():
        # XXX: have constantine handle this fallback
        # Fake constant or just lacks a description, fallback to Linux's
        try:
            from org.python.constantine.platform.linux import Errno as LinuxErrno
        except ImportError:
            from com.kenai.constantine.platform.linux import Errno as LinuxErrno
        constant = getattr(LinuxErrno, constant.name(), None)
        if not constant:
            return 'Unknown error: %d' % code
    return asPyString(constant.toString())

def access(path, mode):
    """access(path, mode) -> True if granted, False otherwise

    Use the real uid/gid to test for access to a path.  Note that most
    operations will use the effective uid/gid, therefore this routine can
    be used in a suid/sgid environment to test if the invoking user has the
    specified access to the path.  The mode argument can be F_OK to test
    existence, or the inclusive-OR of R_OK, W_OK, and X_OK.
    """
    if not isinstance(mode, (int, long)):
        raise TypeError('an integer is required')

    f = File(sys.getPath(path))
    result = True
    if not f.exists():
        result = False
    if mode & R_OK and not f.canRead():
        result = False
    if mode & W_OK and not f.canWrite():
        result = False
    if mode & X_OK:
        # NOTE: always False without jna-posix stat
        try:
            result = (stat(path).st_mode & _stat.S_IEXEC) != 0
        except OSError:
            result = False
    return result

def stat(path):
    """stat(path) -> stat result

    Perform a stat system call on the given path.

    The Java stat implementation only returns a small subset of
    the standard fields: size, modification time and change time.
    """
    abs_path = sys.getPath(path)
    try:
        return stat_result.from_jnastat(_posix.stat(abs_path))
    except NotImplementedError:
        pass
    except:
        raise
    f = File(abs_path)
    if not f.exists():
        raise OSError(errno.ENOENT, strerror(errno.ENOENT), path)
    size = f.length()
    mtime = f.lastModified() / 1000.0
    mode = 0
    if f.isDirectory():
        mode = _stat.S_IFDIR
    elif f.isFile():
        mode = _stat.S_IFREG
    if f.canRead():
        mode = mode | _stat.S_IREAD
    if f.canWrite():
        mode = mode | _stat.S_IWRITE
    return stat_result((mode, 0, 0, 0, 0, 0, size, mtime, mtime, 0))

def lstat(path):
    """lstat(path) -> stat result

    Like stat(path), but do not follow symbolic links.
    """
    abs_path = sys.getPath(path)
    try:
        return stat_result.from_jnastat(_posix.lstat(abs_path))
    except NotImplementedError:
        pass
    except:
        raise
    f = File(sys.getPath(path))
    # XXX: jna-posix implements similar link detection in
    # JavaFileStat.calculateSymlink, fallback to that instead when not
    # native
    abs_parent = f.getAbsoluteFile().getParentFile()
    if not abs_parent:
        # root isn't a link
        return stat(path)
    can_parent = abs_parent.getCanonicalFile()

    if can_parent.getAbsolutePath() == abs_parent.getAbsolutePath():
        # The parent directory's absolute path is canonical..
        if f.getAbsolutePath() != f.getCanonicalPath():
            # but the file's absolute and canonical paths differ (a
            # link)
            return stat_result((_stat.S_IFLNK, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    # The parent directory's path is not canonical (one of the parent
    # directories is a symlink). Build a new path with the parent's
    # canonical path and compare the files
    f = File(_path.join(can_parent.getAbsolutePath(), f.getName()))
    if f.getAbsolutePath() != f.getCanonicalPath():
        return stat_result((_stat.S_IFLNK, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    # Not a link, only now can we determine if it exists (because
    # File.exists() returns False for dead links)
    if not f.exists():
        raise OSError(errno.ENOENT, strerror(errno.ENOENT), path)
    return stat(path)

def utime(path, times):
    """utime(path, (atime, mtime))
    utime(path, None)

    Set the access and modification time of the file to the given values.
    If the second form is used, set the access and modification times to the
    current time.

    Due to Java limitations, on some platforms only the modification time
    may be changed.
    """
    if path is None:
        raise TypeError('path must be specified, not None')

    if times is None:
        atimeval = mtimeval = None
    elif isinstance(times, tuple) and len(times) == 2:
        atimeval = _to_timeval(times[0])
        mtimeval = _to_timeval(times[1])
    else:
        raise TypeError('utime() arg 2 must be a tuple (atime, mtime)')

    _posix.utimes(path, atimeval, mtimeval)

def _to_timeval(seconds):
    """Convert seconds (with a fraction) from epoch to a 2 item tuple of
    seconds, microseconds from epoch as longs
    """
    global _time_t
    if _time_t is None:
        from java.lang import Integer, Long
        try:
            from org.python.posix.util import Platform
        except ImportError:
            from org.jruby.ext.posix.util import Platform
        _time_t = Integer if Platform.IS_32_BIT else Long

    try:
        floor = long(seconds)
    except TypeError:
        raise TypeError('an integer is required')
    if not _time_t.MIN_VALUE <= floor <= _time_t.MAX_VALUE:
        raise OverflowError('long int too large to convert to int')

    # usec can't exceed 1000000
    usec = long((seconds - floor) * 1e6)
    if usec < 0:
        # If rounding gave us a negative number, truncate
        usec = 0
    return floor, usec

def close(fd):
    """close(fd)

    Close a file descriptor (for low level IO).
    """
    rawio = FileDescriptors.get(fd)
    _handle_oserror(rawio.close)

def fdopen(fd, mode='r', bufsize=-1):
    """fdopen(fd [, mode='r' [, bufsize]]) -> file_object

    Return an open file object connected to a file descriptor.
    """
    rawio = FileDescriptors.get(fd)
    if (len(mode) and mode[0] or '') not in 'rwa':
        raise ValueError("invalid file mode '%s'" % mode)
    if rawio.closed():
        raise OSError(errno.EBADF, strerror(errno.EBADF))

    try:
        fp = FileDescriptors.wrap(rawio, mode, bufsize)
    except IOError:
        raise OSError(errno.EINVAL, strerror(errno.EINVAL))
    return fp

def ftruncate(fd, length):
    """ftruncate(fd, length)

    Truncate a file to a specified length.
    """
    rawio = FileDescriptors.get(fd)
    try:
        rawio.truncate(length)
    except Exception, e:
        raise IOError(errno.EBADF, strerror(errno.EBADF))

def lseek(fd, pos, how):
    """lseek(fd, pos, how) -> newpos

    Set the current position of a file descriptor.
    """
    rawio = FileDescriptors.get(fd)
    return _handle_oserror(rawio.seek, pos, how)

def open(filename, flag, mode=0777):
    """open(filename, flag [, mode=0777]) -> fd

    Open a file (for low level IO).
    """
    reading = flag & O_RDONLY
    writing = flag & O_WRONLY
    updating = flag & O_RDWR
    creating = flag & O_CREAT

    truncating = flag & O_TRUNC
    exclusive = flag & O_EXCL
    sync = flag & O_SYNC
    appending = flag & O_APPEND

    if updating and writing:
        raise OSError(errno.EINVAL, strerror(errno.EINVAL), filename)

    if not creating and not path.exists(filename):
        raise OSError(errno.ENOENT, strerror(errno.ENOENT), filename)

    if not writing:
        if updating:
            writing = True
        else:
            reading = True

    if truncating and not writing:
        # Explicitly truncate, writing will truncate anyway
        FileIO(filename, 'w').close()

    if exclusive and creating:
        try:
            if not File(sys.getPath(filename)).createNewFile():
                raise OSError(errno.EEXIST, strerror(errno.EEXIST),
                              filename)
        except java.io.IOException, ioe:
            raise OSError(ioe)

    mode = '%s%s%s%s' % (reading and 'r' or '',
                         (not appending and writing) and 'w' or '',
                         (appending and (writing or updating)) and 'a' or '',
                         updating and '+' or '')

    if sync and (writing or updating):
        from java.io import FileNotFoundException, RandomAccessFile
        try:
            fchannel = RandomAccessFile(sys.getPath(filename), 'rws').getChannel()
        except FileNotFoundException, fnfe:
            if path.isdir(filename):
                raise OSError(errno.EISDIR, strerror(errno.EISDIR))
            raise OSError(errno.ENOENT, strerror(errno.ENOENT), filename)
        return FileIO(fchannel, mode)

    return FileIO(filename, mode)

def read(fd, buffersize):
    """read(fd, buffersize) -> string

    Read a file descriptor.
    """
    from org.python.core.util import StringUtil
    rawio = FileDescriptors.get(fd)
    buf = _handle_oserror(rawio.read, buffersize)
    return asPyString(StringUtil.fromBytes(buf))

def write(fd, string):
    """write(fd, string) -> byteswritten

    Write a string to a file descriptor.
    """
    from java.nio import ByteBuffer
    from org.python.core.util import StringUtil
    rawio = FileDescriptors.get(fd)
    return _handle_oserror(rawio.write,
                           ByteBuffer.wrap(StringUtil.toBytes(string)))

def _handle_oserror(func, *args, **kwargs):
    """Translate exceptions into OSErrors"""
    try:
        return func(*args, **kwargs)
    except:
        raise OSError(errno.EBADF, strerror(errno.EBADF))

def system(command):
    """system(command) -> exit_status

    Execute the command (a string) in a subshell.
    """
    import subprocess
    return subprocess.call(command, shell=True)

def popen(command, mode='r', bufsize=-1):
    """popen(command [, mode='r' [, bufsize]]) -> pipe

    Open a pipe to/from a command returning a file object.
    """
    import subprocess
    if mode == 'r':
        proc = subprocess.Popen(command, bufsize=bufsize, shell=True,
                                stdout=subprocess.PIPE)
        return _wrap_close(proc.stdout, proc)
    elif mode == 'w':
        proc = subprocess.Popen(command, bufsize=bufsize, shell=True,
                                stdin=subprocess.PIPE)
        return _wrap_close(proc.stdin, proc)
    else:
        raise OSError(errno.EINVAL, strerror(errno.EINVAL))

# Helper for popen() -- a proxy for a file whose close waits for the process
class _wrap_close(object):
    def __init__(self, stream, proc):
        self._stream = stream
        self._proc = proc
    def close(self):
        self._stream.close()
        returncode = self._proc.wait()
        if returncode == 0:
            return None
        if _name == 'nt':
            return returncode
        else:
            return returncode
    def __getattr__(self, name):
        return getattr(self._stream, name)
    def __iter__(self):
        return iter(self._stream)

# os module versions of the popen# methods have different return value
# order than popen2 functions

def popen2(cmd, mode="t", bufsize=-1):
    """Execute the shell command cmd in a sub-process.

    On UNIX, 'cmd' may be a sequence, in which case arguments will be
    passed directly to the program without shell intervention (as with
    os.spawnv()).  If 'cmd' is a string it will be passed to the shell
    (as with os.system()).  If 'bufsize' is specified, it sets the
    buffer size for the I/O pipes.  The file objects (child_stdin,
    child_stdout) are returned.
    """
    import popen2
    stdout, stdin = popen2.popen2(cmd, bufsize)
    return stdin, stdout

def popen3(cmd, mode="t", bufsize=-1):
    """Execute the shell command 'cmd' in a sub-process.

    On UNIX, 'cmd' may be a sequence, in which case arguments will be
    passed directly to the program without shell intervention
    (as with os.spawnv()).  If 'cmd' is a string it will be passed
    to the shell (as with os.system()).  If 'bufsize' is specified,
    it sets the buffer size for the I/O pipes.  The file objects
    (child_stdin, child_stdout, child_stderr) are returned.
    """
    import popen2
    stdout, stdin, stderr = popen2.popen3(cmd, bufsize)
    return stdin, stdout, stderr

def popen4(cmd, mode="t", bufsize=-1):
    """Execute the shell command 'cmd' in a sub-process.

    On UNIX, 'cmd' may be a sequence, in which case arguments will be
    passed directly to the program without shell intervention
    (as with os.spawnv()).  If 'cmd' is a string it will be passed
    to the shell (as with os.system()).  If 'bufsize' is specified,
    it sets the buffer size for the I/O pipes.  The file objects
    (child_stdin, child_stdout_stderr) are returned.
    """
    import popen2
    stdout, stdin = popen2.popen4(cmd, bufsize)
    return stdin, stdout

def getlogin():
    """getlogin() -> string

    Return the actual login name.
    """
    return java.lang.System.getProperty("user.name")

#XXX: copied from CPython's release23-maint branch revision 56502
def walk(top, topdown=True, onerror=None):
    """Directory tree generator.

    For each directory in the directory tree rooted at top (including top
    itself, but excluding '.' and '..'), yields a 3-tuple

        dirpath, dirnames, filenames

    dirpath is a string, the path to the directory.  dirnames is a list of
    the names of the subdirectories in dirpath (excluding '.' and '..').
    filenames is a list of the names of the non-directory files in dirpath.
    Note that the names in the lists are just names, with no path components.
    To get a full path (which begins with top) to a file or directory in
    dirpath, do os.path.join(dirpath, name).

    If optional arg 'topdown' is true or not specified, the triple for a
    directory is generated before the triples for any of its subdirectories
    (directories are generated top down).  If topdown is false, the triple
    for a directory is generated after the triples for all of its
    subdirectories (directories are generated bottom up).

    When topdown is true, the caller can modify the dirnames list in-place
    (e.g., via del or slice assignment), and walk will only recurse into the
    subdirectories whose names remain in dirnames; this can be used to prune
    the search, or to impose a specific order of visiting.  Modifying
    dirnames when topdown is false is ineffective, since the directories in
    dirnames have already been generated by the time dirnames itself is
    generated.

    By default errors from the os.listdir() call are ignored.  If
    optional arg 'onerror' is specified, it should be a function; it
    will be called with one argument, an os.error instance.  It can
    report the error to continue with the walk, or raise the exception
    to abort the walk.  Note that the filename is available as the
    filename attribute of the exception object.

    Caution:  if you pass a relative pathname for top, don't change the
    current working directory between resumptions of walk.  walk never
    changes the current directory, and assumes that the client doesn't
    either.

    Example:

    from os.path import join, getsize
    for root, dirs, files in walk('python/Lib/email'):
        print root, "consumes",
        print sum([getsize(join(root, name)) for name in files]),
        print "bytes in", len(files), "non-directory files"
        if 'CVS' in dirs:
            dirs.remove('CVS')  # don't visit CVS directories
    """

    from os.path import join, isdir, islink

    # We may not have read permission for top, in which case we can't
    # get a list of the files the directory contains.  os.path.walk
    # always suppressed the exception then, rather than blow up for a
    # minor reason when (say) a thousand readable directories are still
    # left to visit.  That logic is copied here.
    try:
        # Note that listdir and error are globals in this module due
        # to earlier import-*.
        names = listdir(top)
    except error, err:
        if onerror is not None:
            onerror(err)
        return

    dirs, nondirs = [], []
    for name in names:
        if isdir(join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    if topdown:
        yield top, dirs, nondirs
    for name in dirs:
        path = join(top, name)
        if not islink(path):
            for x in walk(path, topdown, onerror):
                yield x
    if not topdown:
        yield top, dirs, nondirs

__all__.append("walk")

environ = sys.getEnviron()

if _name in ('os2', 'nt'):  # Where Env Var Names Must Be UPPERCASE
    import UserDict

    # But we store them as upper case
    class _Environ(UserDict.IterableUserDict):
        def __init__(self, environ):
            UserDict.UserDict.__init__(self)
            data = self.data
            for k, v in environ.items():
                data[k.upper()] = v
        def __setitem__(self, key, item):
            self.data[key.upper()] = item
        def __getitem__(self, key):
            return self.data[key.upper()]
        def __delitem__(self, key):
            del self.data[key.upper()]
        def has_key(self, key):
            return key.upper() in self.data
        def __contains__(self, key):
            return key.upper() in self.data
        def get(self, key, failobj=None):
            return self.data.get(key.upper(), failobj)
        def update(self, dict=None, **kwargs):
            if dict:
                try:
                    keys = dict.keys()
                except AttributeError:
                    # List of (key, value)
                    for k, v in dict:
                        self[k] = v
                else:
                    # got keys
                    # cannot use items(), since mappings
                    # may not have them.
                    for k in keys:
                        self[k] = dict[k]
            if kwargs:
                self.update(kwargs)
        def copy(self):
            return dict(self)

    environ = _Environ(environ)

def putenv(key, value):
    """putenv(key, value)

    Change or add an environment variable.
    """
    environ[key] = value

def unsetenv(key):
    """unsetenv(key)

    Delete an environment variable.
    """
    if key in environ:
        del environ[key]

def getenv(key, default=None):
    """Get an environment variable, return None if it doesn't exist.
    The optional second argument can specify an alternate default."""
    return environ.get(key, default)

if _name == 'posix':
    def link(src, dst):
        """link(src, dst)

        Create a hard link to a file.
        """
        _posix.link(sys.getPath(src), sys.getPath(dst))

    def symlink(src, dst):
        """symlink(src, dst)

        Create a symbolic link pointing to src named dst.
        """
        _posix.symlink(src, sys.getPath(dst))

    def readlink(path):
        """readlink(path) -> path

        Return a string representing the path to which the symbolic link
        points.
        """
        return _posix.readlink(sys.getPath(path))

    def getegid():
        """getegid() -> egid

        Return the current process's effective group id."""
        return _posix.getegid()

    def geteuid():
        """geteuid() -> euid

        Return the current process's effective user id."""
        return _posix.geteuid()

    def getgid():
        """getgid() -> gid

        Return the current process's group id."""
        return _posix.getgid()

    def getlogin():
        """getlogin() -> string

        Return the actual login name."""
        return _posix.getlogin()

    def getpgrp():
        """getpgrp() -> pgrp

        Return the current process group id."""
        return _posix.getpgrp()

    def getppid():
        """getppid() -> ppid

        Return the parent's process id."""
        return _posix.getppid()

    def getuid():
        """getuid() -> uid

        Return the current process's user id."""
        return _posix.getuid()

    def setpgrp():
        """setpgrp()

        Make this process a session leader."""
        return _posix.setpgrp()

    def setsid():
        """setsid()

        Call the system call setsid()."""
        return _posix.setsid()

    # This implementation of fork partially works on
    # Jython. Diagnosing what works, what doesn't, and fixing it is
    # left for another day. In any event, this would only be
    # marginally useful.

    # def fork():
    #     """fork() -> pid
    #
    #     Fork a child process.
    #     Return 0 to child process and PID of child to parent process."""
    #     return _posix.fork()

    def kill(pid, sig):
        """kill(pid, sig)

        Kill a process with a signal."""
        return _posix.kill(pid, sig)

    def wait():
        """wait() -> (pid, status)

        Wait for completion of a child process."""

        status = jarray.zeros(1, 'i')
        res_pid = _posix.wait(status)
        if res_pid == -1:
            raise OSError(status[0], strerror(status[0]))
        return res_pid, status[0]

    def waitpid(pid, options):
        """waitpid(pid, options) -> (pid, status)

        Wait for completion of a given child process."""
        status = jarray.zeros(1, 'i')
        res_pid = _posix.waitpid(pid, status, options)
        if res_pid == -1:
            raise OSError(status[0], strerror(status[0]))
        return res_pid, status[0]

    def fdatasync(fd):
        """fdatasync(fildes)

        force write of file with filedescriptor to disk.
        does not force update of metadata.
        """
        _fsync(fd, False)

    __all__.extend(['link', 'symlink', 'readlink', 'getegid', 'geteuid',
                    'getgid', 'getlogin', 'getpgrp', 'getppid', 'getuid',
                    'setpgrp', 'setsid', 'kill', 'wait', 'waitpid',
                    'fdatasync'])

def fsync(fd):
    """fsync(fildes)

    force write of file with filedescriptor to disk.
    """
    _fsync(fd, True)

def _fsync(fd, metadata):
    """Internal fsync impl"""
    rawio = FileDescriptors.get(fd)
    rawio.checkClosed()

    from java.nio.channels import FileChannel
    channel = rawio.getChannel()
    if not isinstance(channel, FileChannel):
        raise OSError(errno.EINVAL, strerror(errno.EINVAL))

    try:
        channel.force(metadata)
    except java.io.IOException, ioe:
        raise OSError(ioe)

def getpid():
    """getpid() -> pid

    Return the current process id."""
    return _posix.getpid()

def isatty(fileno):
    """isatty(fd) -> bool

    Return True if the file descriptor 'fd' is an open file descriptor
    connected to the slave end of a terminal."""
    from java.io import FileDescriptor

    if isinstance(fileno, int):
        if fileno == 0:
            fd = getattr(FileDescriptor, 'in')
        elif fileno == 1:
            fd = FileDescriptor.out
        elif fileno == 2:
            fd = FileDescriptor.err
        else:
            raise NotImplemented('Integer file descriptor compatibility only '
                                 'available for stdin, stdout and stderr (0-2)')

        return _posix.isatty(fd)

    if isinstance(fileno, FileDescriptor):
        return _posix.isatty(fileno)

    if not isinstance(fileno, IOBase):
        raise TypeError('a file descriptor is required')

    return fileno.isatty()

def umask(new_mask):
    """umask(new_mask) -> old_mask

    Set the current numeric umask and return the previous umask."""
    return _posix.umask(int(new_mask))


from java.security import SecureRandom
urandom_source = None

def urandom(n):
    global urandom_source
    if urandom_source is None:
        urandom_source = SecureRandom()
    buffer = jarray.zeros(n, 'b')
    urandom_source.nextBytes(buffer)
    return buffer.tostring()
