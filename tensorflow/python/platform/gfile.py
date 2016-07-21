# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""File processing utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import errno
import functools
import glob as _glob
import os
import shutil
import threading

import six


class _GFileBase(six.Iterator):
  """Base I/O wrapper class.  Similar semantics to Python's file object."""

  # pylint: disable=protected-access
  def _synchronized(fn):
    """Synchronizes file I/O for methods in GFileBase."""
    @functools.wraps(fn)
    def sync(self, *args, **kwargs):
      # Sometimes a GFileBase method is called before the instance
      # has been properly initialized.  Check that _locker is available.
      if hasattr(self, '_locker'): self._locker.lock()
      try:
        return fn(self, *args, **kwargs)
      finally:
        if hasattr(self, '_locker'): self._locker.unlock()
    return sync
  # pylint: enable=protected-access

  def __init__(self, name, mode, locker):
    """Create the GFileBase object with the given filename, mode, and locker.

    Args:
      name: string, the filename.
      mode: string, the mode to open the file with (e.g. "r", "w", "a+").
      locker: the thread locking object (e.g. _PythonLocker) for controlling
        thread access to the I/O methods of this class.
    """
    self._name = name
    self._mode = mode
    self._locker = locker
    self._fp = open(name, mode)

  def __enter__(self):
    """Make GFileBase usable with "with" statement."""
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    """Make GFileBase usable with "with" statement."""
    self.close()

  @_synchronized
  def __del__(self):
    # __del__ is sometimes called before initialization, in which
    # case the object is not fully constructed.  Check for this here
    # before trying to close the file handle.
    if hasattr(self, '_fp'): self._fp.close()

  @_synchronized
  def flush(self):
    """Flush the underlying file handle."""
    return self._fp.flush()

  @property
  @_synchronized
  def closed(self):
    """Returns "True" if the file handle is closed.  Otherwise False."""
    return self._fp.closed

  @_synchronized
  def write(self, data):
    """Write data to the underlying file handle.

    Args:
      data: The string to write to the file handle.
    """
    self._fp.write(data)

  @_synchronized
  def writelines(self, seq):
    """Write a sequence of strings to the underlying file handle."""
    self._fp.writelines(seq)

  @_synchronized
  def tell(self):
    """Return the location from the underlying file handle.

    Returns:
      An integer location (which can be used in e.g., seek).
    """
    return self._fp.tell()

  @_synchronized
  def seek(self, offset, whence=0):
    """Seek to offset (conditioned on whence) in the underlying file handle.

    Args:
      offset: int, the offset within the file to seek to.
      whence: 0, 1, or 2.  See python's seek() documentation for details.
    """
    self._fp.seek(offset, whence)

  @_synchronized
  def truncate(self, new_size=None):
    """Truncate the underlying file handle to new_size.

    Args:
      new_size: Size after truncation.  If None, the file handle is truncated
      to 0 bytes.
    """
    self._fp.truncate(new_size)

  @_synchronized
  def readline(self, max_length=-1):
    """Read a single line (up to max_length) from the underlying file handle.

    Args:
      max_length: The maximum number of chsaracters to read.

    Returns:
      A string, including any newline at the end, or empty string if at EOF.
    """
    return self._fp.readline(max_length)

  @_synchronized
  def readlines(self, sizehint=None):
    """Read lines from the underlying file handle.

    Args:
      sizehint: See the python file.readlines() documentation.

    Returns:
      A list of strings from the underlying file handle.
    """
    if sizehint is not None:
      return self._fp.readlines(sizehint)
    else:
      return self._fp.readlines()

  def __iter__(self):
    """Enable line iteration on the underlying handle (not synchronized)."""
    return self

  # Not synchronized
  def __next__(self):
    """Enable line iteration on the underlying handle (not synchronized).

    Returns:
      An line iterator from the underlying handle.

    Example:
      # read a file's lines by consuming the iterator with a list
      with open("filename", "r") as fp: lines = list(fp)
    """
    return next(self._fp)

  @_synchronized
  def Size(self):   # pylint: disable=invalid-name
    """Get byte size of the file from the underlying file handle."""
    cur = self.tell()
    try:
      self.seek(0, 2)
      size = self.tell()
    finally:
      self.seek(cur)
    return size

  @_synchronized
  def read(self, n=-1):
    """Read n bytes from the underlying file handle.

    Args:
      n: Number of bytes to read (if negative, read to end of file handle.)

    Returns:
      A string of the bytes read, up to the end of file.
    """
    return self._fp.read(n)

  @_synchronized
  def close(self):
    """Close the underlying file handle."""
    self._fp.close()

  # Declare wrappers as staticmethods at the end so that we can
  # use them as decorators.
  _synchronized = staticmethod(_synchronized)


class GFile(_GFileBase):
  """File I/O wrappers with thread locking."""

  def __init__(self, name, mode='r'):
    super(GFile, self).__init__(name, mode, _Pythonlocker())


class FastGFile(_GFileBase):
  """File I/O wrappers without thread locking."""

  def __init__(self, name, mode='r'):
    super(FastGFile, self).__init__(name, mode, _Nulllocker())


# locker classes.  Note that locks must be reentrant, so that multiple
# lock() calls by the owning thread will not block.
class _Pythonlocker(object):
  """A locking strategy that uses standard locks from the thread module."""

  def __init__(self):
    self._lock = threading.RLock()

  def lock(self):
    self._lock.acquire()

  def unlock(self):
    self._lock.release()


class _Nulllocker(object):
  """A locking strategy where lock() and unlock() methods are no-ops."""

  def lock(self):
    pass

  def unlock(self):
    pass


def Exists(path):   # pylint: disable=invalid-name
  """Returns True iff "path" exists (as a dir, file, non-broken symlink)."""
  return os.path.exists(path)


def IsDirectory(path):   # pylint: disable=invalid-name
  """Return True iff "path" exists and is a directory."""
  return os.path.isdir(path)


def Glob(glob):   # pylint: disable=invalid-name
  """Return a list of filenames matching the glob "glob"."""
  return _glob.glob(glob)


def MkDir(path, mode=0o755):  # pylint: disable=invalid-name
  """Create the directory "path" with the given mode.

  Args:
    path: The directory path
    mode: The file mode for the directory

  Returns:
    None

  Raises:
    OSError: if the path already exists
  """
  os.mkdir(path, mode)


def MakeDirs(path, mode=0o755):  # pylint: disable=invalid-name
  """Recursively create the directory "path" with the given mode.

  Args:
    path: The directory path.
    mode: The file mode for the created directories

  Raises:
    OSError: if the path already exists
  """
  # NOTE(mrry): MakeDirs("") should be a no-op to match other
  # implementations of tf.gfile.
  if path:
    try:
      os.makedirs(path, mode)
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise


def RmDir(directory):   # pylint: disable=invalid-name
  """Removes the directory "directory" iff the directory is empty.

  Args:
    directory: The directory to remove.

  Raises:
    OSError: If the directory does not exist or is not empty.
  """
  os.rmdir(directory)


def Remove(path):   # pylint: disable=invalid-name
  """Delete the (non-directory) file "path".

  Args:
    path: The file to remove.

  Raises:
    OSError: If "path" does not exist, is a directory, or cannot be deleted.
  """
  os.remove(path)


def Rename(oldpath, newpath, overwrite=False):
  """Rename or move a file, or a local directory.

  Args:
    oldpath: string; a pathname of a file.
    newpath: string; a pathname to which the file will be moved.
    overwrite: boolean; if false, it is an error for newpath to be
      occupied by an existing file.

  Raises:
    OSError: If "newpath" is occupied by an existing file and overwrite=False.
  """
  if not overwrite and Exists(newpath) and not IsDirectory(newpath):
    raise OSError(errno.EEXIST, os.strerror(errno.EEXIST), newpath)
  os.rename(oldpath, newpath)


def DeleteRecursively(path):   # pylint: disable=invalid-name
  """Delete the file or directory "path" recursively.

  Args:
    path: The path to remove (may be a non-empty directory).

  Raises:
    OSError: If the path does not exist or cannot be deleted.
  """
  if IsDirectory(path):
    shutil.rmtree(path)
  else:
    Remove(path)


def ListDirectory(directory, return_dotfiles=False):  # pylint: disable=invalid-name
  """Returns a list of files in dir.

  As with the standard os.listdir(), the filenames in the returned list will be
  the basenames of the files in dir (not absolute paths).  To get a list of
  absolute paths of files in a directory, a client could do:
    file_list = gfile.ListDir(my_dir)
    file_list = [os.path.join(my_dir, f) for f in file_list]
  (assuming that my_dir itself specified an absolute path to a directory).

  Args:
    directory: the directory to list
    return_dotfiles: if True, dotfiles will be returned as well.  Even if
      this arg is True, '.' and '..' will not be returned.

  Returns:
    ['list', 'of', 'files']. The entries '.' and '..' are never returned.
    Other entries starting with a dot will only be returned if return_dotfiles
    is True.
  Raises:
    OSError: if there is an error retrieving the directory listing.
  """
  files = os.listdir(directory)
  if not return_dotfiles:
    files = [f for f in files if not f.startswith('.')]
  return files


def Walk(top, topdown=1, onerror=None):
  """Recursive directory tree generator.

  Args:
    top: string, a pathname.
    topdown: bool, should traversal be pre-order (True) or post-order (False)
    onerror: function, optional callback for errors.

  By default, errors that occur when listing a directory are ignored.
  (This is the same semantics as Python's os.walk() generator.)  If the
  optional argument "onerror" is specified, it should be a function.  It
  will be called with one argument, an os.error instance.  It can return
  to continue with the walk, or reraise the exception to abort the walk.
  By default, the walk follows symlinks that resolve into directories.

  Yields:
    # Each yield is a 3-tuple:  the pathname of a directory, followed
    # by lists of all its subdirectories and leaf files.
    (dirname, [subdirname, subdirname, ...], [filename, filename, ...])
  """
  return os.walk(top, topdown=topdown, onerror=onerror, followlinks=True)


def Stat(path):   # pylint: disable=invalid-name
  """Gets the status of a file.

  Args:
    path: The file to call Stat() on.

  Does the equivalent of Stat() on the specified "path" and return file
  properties.

  Returns:
    An object whose attributes give information on the file.

  Raises:
    OSError: If "path" does not exist.
  """
  statinfo = os.stat(path)
  filestat = collections.namedtuple('FileStat', ['mtime'])
  filestat.mtime = statinfo.st_mtime
  return filestat


def Copy(oldpath, newpath, overwrite=False):
  """Copy a file.

  Args:
    oldpath: string; a pathname of a file.
    newpath: string; a pathname to which the file will be copied.
    overwrite: boolean; if false, it is an error for newpath to be
      occupied by an existing file.

  Raises:
    OSError: If "newpath" is occupied by an existing file and overwrite=False,
             or any error thrown by shutil.copy.
  """
  if not overwrite and Exists(newpath):
    raise OSError(errno.EEXIST, os.strerror(errno.EEXIST), newpath)
  shutil.copy(oldpath, newpath)


def Open(name, mode='r'):
  """Exact API match to the standard open.

  Args:
    name:  a file name, either local or a gfile compatible.
    mode:  for example "w" to open the file for writing.

  Returns:
    A threadsafe gfile.GFile object.
  """
  return GFile(name, mode=mode)
