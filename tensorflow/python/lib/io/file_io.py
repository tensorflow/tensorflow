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
"""File IO methods that wrap the C++ FileSystem API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import binascii
import os
import uuid

import six

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import _pywrap_file_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

# A good default block size depends on the system in question.
# A somewhat conservative default chosen here.
_DEFAULT_BLOCK_SIZE = 16 * 1024 * 1024


class FileIO(object):
  """FileIO class that exposes methods to read / write to / from files.

  The constructor takes the following arguments:
  name: [path-like object](https://docs.python.org/3/glossary.html#term-path-like-object)
    giving the pathname of the file to be opened.
  mode: one of `r`, `w`, `a`, `r+`, `w+`, `a+`. Append `b` for bytes mode.

  Can be used as an iterator to iterate over lines in the file.

  The default buffer size used for the BufferedInputStream used for reading
  the file line by line is 1024 * 512 bytes.
  """

  def __init__(self, name, mode):
    self.__name = name
    self.__mode = mode
    self._read_buf = None
    self._writable_file = None
    self._binary_mode = "b" in mode
    mode = mode.replace("b", "")
    if mode not in ("r", "w", "a", "r+", "w+", "a+"):
      raise errors.InvalidArgumentError(
          None, None, "mode is not 'r' or 'w' or 'a' or 'r+' or 'w+' or 'a+'")
    self._read_check_passed = mode in ("r", "r+", "a+", "w+")
    self._write_check_passed = mode in ("a", "w", "r+", "a+", "w+")

  @property
  def name(self):
    """Returns the file name."""
    return self.__name

  @property
  def mode(self):
    """Returns the mode in which the file was opened."""
    return self.__mode

  def _preread_check(self):
    if not self._read_buf:
      if not self._read_check_passed:
        raise errors.PermissionDeniedError(None, None,
                                           "File isn't open for reading")
      self._read_buf = _pywrap_file_io.BufferedInputStream(
          compat.path_to_str(self.__name), 1024 * 512)

  def _prewrite_check(self):
    if not self._writable_file:
      if not self._write_check_passed:
        raise errors.PermissionDeniedError(None, None,
                                           "File isn't open for writing")
      self._writable_file = _pywrap_file_io.WritableFile(
          compat.path_to_bytes(self.__name), compat.as_bytes(self.__mode))

  def _prepare_value(self, val):
    if self._binary_mode:
      return compat.as_bytes(val)
    else:
      return compat.as_str_any(val)

  def size(self):
    """Returns the size of the file."""
    return stat(self.__name).length

  def write(self, file_content):
    """Writes file_content to the file. Appends to the end of the file."""
    self._prewrite_check()
    self._writable_file.append(compat.as_bytes(file_content))

  def read(self, n=-1):
    """Returns the contents of a file as a string.

    Starts reading from current position in file.

    Args:
      n: Read `n` bytes if `n != -1`. If `n = -1`, reads to end of file.

    Returns:
      `n` bytes of the file (or whole file) in bytes mode or `n` bytes of the
      string if in string (regular) mode.
    """
    self._preread_check()
    if n == -1:
      length = self.size() - self.tell()
    else:
      length = n
    return self._prepare_value(self._read_buf.read(length))

  @deprecation.deprecated_args(
      None, "position is deprecated in favor of the offset argument.",
      "position")
  def seek(self, offset=None, whence=0, position=None):
    # TODO(jhseu): Delete later. Used to omit `position` from docs.
    # pylint: disable=g-doc-args
    """Seeks to the offset in the file.

    Args:
      offset: The byte count relative to the whence argument.
      whence: Valid values for whence are:
        0: start of the file (default)
        1: relative to the current position of the file
        2: relative to the end of file. `offset` is usually negative.
    """
    # pylint: enable=g-doc-args
    self._preread_check()
    # We needed to make offset a keyword argument for backwards-compatibility.
    # This check exists so that we can convert back to having offset be a
    # positional argument.
    # TODO(jhseu): Make `offset` a positional argument after `position` is
    # deleted.
    if offset is None and position is None:
      raise TypeError("seek(): offset argument required")
    if offset is not None and position is not None:
      raise TypeError("seek(): offset and position may not be set "
                      "simultaneously.")

    if position is not None:
      offset = position

    if whence == 0:
      pass
    elif whence == 1:
      offset += self.tell()
    elif whence == 2:
      offset += self.size()
    else:
      raise errors.InvalidArgumentError(
          None, None,
          "Invalid whence argument: {}. Valid values are 0, 1, or 2.".format(
              whence))
    self._read_buf.seek(offset)

  def readline(self):
    r"""Reads the next line, keeping \n. At EOF, returns ''."""
    self._preread_check()
    return self._prepare_value(self._read_buf.readline())

  def readlines(self):
    """Returns all lines from the file in a list."""
    self._preread_check()
    lines = []
    while True:
      s = self.readline()
      if not s:
        break
      lines.append(s)
    return lines

  def tell(self):
    """Returns the current position in the file."""
    if self._read_check_passed:
      self._preread_check()
      return self._read_buf.tell()
    else:
      self._prewrite_check()

      return self._writable_file.tell()

  def __enter__(self):
    """Make usable with "with" statement."""
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    """Make usable with "with" statement."""
    self.close()

  def __iter__(self):
    return self

  def __next__(self):
    retval = self.readline()
    if not retval:
      raise StopIteration()
    return retval

  def next(self):
    return self.__next__()

  def flush(self):
    """Flushes the Writable file.

    This only ensures that the data has made its way out of the process without
    any guarantees on whether it's written to disk. This means that the
    data would survive an application crash but not necessarily an OS crash.
    """
    if self._writable_file:
      self._writable_file.flush()

  def close(self):
    r"""Closes the file.

    Should be called for the WritableFile to be flushed.

    In general, if you use the context manager pattern, you don't need to call
    this directly.

    >>> with tf.io.gfile.GFile("/tmp/x", "w") as f:
    ...   f.write("asdf\n")
    ...   f.write("qwer\n")
    >>> # implicit f.close() at the end of the block

    For cloud filesystems, forgetting to call `close()` might result in data
    loss as last write might not have been replicated.
    """
    self._read_buf = None
    if self._writable_file:
      self._writable_file.close()
      self._writable_file = None

  def seekable(self):
    """Returns True as FileIO supports random access ops of seek()/tell()"""
    return True


@tf_export("io.gfile.exists")
def file_exists_v2(path):
  """Determines whether a path exists or not.

  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.exists("/tmp/x")
  True

  You can also specify the URI scheme for selecting a different filesystem:

  >>> # for a GCS filesystem path:
  >>> # tf.io.gfile.exists("gs://bucket/file")
  >>> # for a local filesystem:
  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.exists("file:///tmp/x")
  True

  This currently returns `True` for existing directories but don't rely on this
  behavior, especially if you are using cloud filesystems (e.g., GCS, S3,
  Hadoop):

  >>> tf.io.gfile.exists("/tmp")
  True

  Args:
    path: string, a path

  Returns:
    True if the path exists, whether it's a file or a directory.
    False if the path does not exist and there are no filesystem errors.

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.
  """
  try:
    _pywrap_file_io.FileExists(compat.path_to_bytes(path))
  except errors.NotFoundError:
    return False
  return True


@tf_export(v1=["gfile.Exists"])
def file_exists(filename):
  return file_exists_v2(filename)


file_exists.__doc__ = file_exists_v2.__doc__


@tf_export(v1=["gfile.Remove"])
def delete_file(filename):
  """Deletes the file located at 'filename'.

  Args:
    filename: string, a filename

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
    `NotFoundError` if the file does not exist.
  """
  delete_file_v2(filename)


@tf_export("io.gfile.remove")
def delete_file_v2(path):
  """Deletes the path located at 'path'.

  Args:
    path: string, a path

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
    `NotFoundError` if the path does not exist.
  """
  _pywrap_file_io.DeleteFile(compat.path_to_bytes(path))


def read_file_to_string(filename, binary_mode=False):
  """Reads the entire contents of a file to a string.

  Args:
    filename: string, path to a file
    binary_mode: whether to open the file in binary mode or not. This changes
      the type of the object returned.

  Returns:
    contents of the file as a string or bytes.

  Raises:
    errors.OpError: Raises variety of errors that are subtypes e.g.
    `NotFoundError` etc.
  """
  if binary_mode:
    f = FileIO(filename, mode="rb")
  else:
    f = FileIO(filename, mode="r")
  return f.read()


def write_string_to_file(filename, file_content):
  """Writes a string to a given file.

  Args:
    filename: string, path to a file
    file_content: string, contents that need to be written to the file

  Raises:
    errors.OpError: If there are errors during the operation.
  """
  with FileIO(filename, mode="w") as f:
    f.write(file_content)


@tf_export(v1=["gfile.Glob"])
def get_matching_files(filename):
  """Returns a list of files that match the given pattern(s).

  Args:
    filename: string or iterable of strings. The glob pattern(s).

  Returns:
    A list of strings containing filenames that match the given pattern(s).

  Raises:
  *  errors.OpError: If there are filesystem / directory listing errors.
  *  errors.NotFoundError: If pattern to be matched is an invalid directory.
  """
  return get_matching_files_v2(filename)


@tf_export("io.gfile.glob")
def get_matching_files_v2(pattern):
  r"""Returns a list of files that match the given pattern(s).

  The patterns are defined as strings. Supported patterns are defined
  here. Note that the pattern can be a Python iteratable of string patterns.

  The format definition of the pattern is:

  **pattern**: `{ term }`

  **term**:
    * `'*'`: matches any sequence of non-'/' characters
    * `'?'`: matches a single non-'/' character
    * `'[' [ '^' ] { match-list } ']'`: matches any single
      character (not) on the list
    * `c`: matches character `c`  where `c != '*', '?', '\\', '['`
    * `'\\' c`: matches character `c`

  **character range**:
    * `c`: matches character `c` while `c != '\\', '-', ']'`
    * `'\\' c`: matches character `c`
    * `lo '-' hi`: matches character `c` for `lo <= c <= hi`

  Examples:

  >>> tf.io.gfile.glob("*.py")
  ... # For example, ['__init__.py']

  >>> tf.io.gfile.glob("__init__.??")
  ... # As above

  >>> files = {"*.py"}
  >>> the_iterator = iter(files)
  >>> tf.io.gfile.glob(the_iterator)
  ... # As above

  See the C++ function `GetMatchingPaths` in
  [`core/platform/file_system.h`]
  (../../../core/platform/file_system.h)
  for implementation details.

  Args:
    pattern: string or iterable of strings. The glob pattern(s).

  Returns:
    A list of strings containing filenames that match the given pattern(s).

  Raises:
    errors.OpError: If there are filesystem / directory listing errors.
    errors.NotFoundError: If pattern to be matched is an invalid directory.
  """
  if isinstance(pattern, six.string_types):
    return [
        # Convert the filenames to string from bytes.
        compat.as_str_any(matching_filename)
        for matching_filename in _pywrap_file_io.GetMatchingFiles(
            compat.as_bytes(pattern))
    ]
  else:
    return [
        # Convert the filenames to string from bytes.
        compat.as_str_any(matching_filename)  # pylint: disable=g-complex-comprehension
        for single_filename in pattern
        for matching_filename in _pywrap_file_io.GetMatchingFiles(
            compat.as_bytes(single_filename))
    ]


@tf_export(v1=["gfile.MkDir"])
def create_dir(dirname):
  """Creates a directory with the name `dirname`.

  Args:
    dirname: string, name of the directory to be created

  Notes: The parent directories need to exist. Use `tf.io.gfile.makedirs`
    instead if there is the possibility that the parent dirs don't exist.

  Raises:
    errors.OpError: If the operation fails.
  """
  create_dir_v2(dirname)


@tf_export("io.gfile.mkdir")
def create_dir_v2(path):
  """Creates a directory with the name given by `path`.

  Args:
    path: string, name of the directory to be created

  Notes: The parent directories need to exist. Use `tf.io.gfile.makedirs`
    instead if there is the possibility that the parent dirs don't exist.

  Raises:
    errors.OpError: If the operation fails.
  """
  _pywrap_file_io.CreateDir(compat.path_to_bytes(path))


@tf_export(v1=["gfile.MakeDirs"])
def recursive_create_dir(dirname):
  """Creates a directory and all parent/intermediate directories.

  It succeeds if dirname already exists and is writable.

  Args:
    dirname: string, name of the directory to be created

  Raises:
    errors.OpError: If the operation fails.
  """
  recursive_create_dir_v2(dirname)


@tf_export("io.gfile.makedirs")
def recursive_create_dir_v2(path):
  """Creates a directory and all parent/intermediate directories.

  It succeeds if path already exists and is writable.

  Args:
    path: string, name of the directory to be created

  Raises:
    errors.OpError: If the operation fails.
  """
  _pywrap_file_io.RecursivelyCreateDir(compat.path_to_bytes(path))


@tf_export("io.gfile.copy")
def copy_v2(src, dst, overwrite=False):
  """Copies data from `src` to `dst`.

  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.exists("/tmp/x")
  True
  >>> tf.io.gfile.copy("/tmp/x", "/tmp/y")
  >>> tf.io.gfile.exists("/tmp/y")
  True
  >>> tf.io.gfile.remove("/tmp/y")

  You can also specify the URI scheme for selecting a different filesystem:

  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.copy("/tmp/x", "file:///tmp/y")
  >>> tf.io.gfile.exists("/tmp/y")
  True
  >>> tf.io.gfile.remove("/tmp/y")

  Note that you need to always specify a file name, even if moving into a new
  directory. This is because some cloud filesystems don't have the concept of a
  directory.

  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.mkdir("/tmp/new_dir")
  >>> tf.io.gfile.copy("/tmp/x", "/tmp/new_dir/y")
  >>> tf.io.gfile.exists("/tmp/new_dir/y")
  True
  >>> tf.io.gfile.rmtree("/tmp/new_dir")

  If you want to prevent errors if the path already exists, you can use
  `overwrite` argument:

  >>> with open("/tmp/x", "w") as f:
  ...   f.write("asdf")
  ...
  4
  >>> tf.io.gfile.copy("/tmp/x", "file:///tmp/y")
  >>> tf.io.gfile.copy("/tmp/x", "file:///tmp/y", overwrite=True)
  >>> tf.io.gfile.remove("/tmp/y")

  Note that the above will still result in an error if you try to overwrite a
  directory with a file.

  Note that you cannot copy a directory, only file arguments are supported.

  Args:
    src: string, name of the file whose contents need to be copied
    dst: string, name of the file to which to copy to
    overwrite: boolean, if false it's an error for `dst` to be occupied by an
      existing file.

  Raises:
    errors.OpError: If the operation fails.
  """
  _pywrap_file_io.CopyFile(
      compat.path_to_bytes(src), compat.path_to_bytes(dst), overwrite)


@tf_export(v1=["gfile.Copy"])
def copy(oldpath, newpath, overwrite=False):
  copy_v2(oldpath, newpath, overwrite)


copy.__doc__ = copy_v2.__doc__


@tf_export(v1=["gfile.Rename"])
def rename(oldname, newname, overwrite=False):
  """Rename or move a file / directory.

  Args:
    oldname: string, pathname for a file
    newname: string, pathname to which the file needs to be moved
    overwrite: boolean, if false it's an error for `newname` to be occupied by
      an existing file.

  Raises:
    errors.OpError: If the operation fails.
  """
  rename_v2(oldname, newname, overwrite)


@tf_export("io.gfile.rename")
def rename_v2(src, dst, overwrite=False):
  """Rename or move a file / directory.

  Args:
    src: string, pathname for a file
    dst: string, pathname to which the file needs to be moved
    overwrite: boolean, if false it's an error for `dst` to be occupied by an
      existing file.

  Raises:
    errors.OpError: If the operation fails.
  """
  _pywrap_file_io.RenameFile(
      compat.path_to_bytes(src), compat.path_to_bytes(dst), overwrite)


def atomic_write_string_to_file(filename, contents, overwrite=True):
  """Writes to `filename` atomically.

  This means that when `filename` appears in the filesystem, it will contain
  all of `contents`. With write_string_to_file, it is possible for the file
  to appear in the filesystem with `contents` only partially written.

  Accomplished by writing to a temp file and then renaming it.

  Args:
    filename: string, pathname for a file
    contents: string, contents that need to be written to the file
    overwrite: boolean, if false it's an error for `filename` to be occupied by
      an existing file.
  """
  if not has_atomic_move(filename):
    write_string_to_file(filename, contents)
  else:
    temp_pathname = filename + ".tmp" + uuid.uuid4().hex
    write_string_to_file(temp_pathname, contents)
    try:
      rename(temp_pathname, filename, overwrite)
    except errors.OpError:
      delete_file(temp_pathname)
      raise


@tf_export(v1=["gfile.DeleteRecursively"])
def delete_recursively(dirname):
  """Deletes everything under dirname recursively.

  Args:
    dirname: string, a path to a directory

  Raises:
    errors.OpError: If the operation fails.
  """
  delete_recursively_v2(dirname)


@tf_export("io.gfile.rmtree")
def delete_recursively_v2(path):
  """Deletes everything under path recursively.

  Args:
    path: string, a path

  Raises:
    errors.OpError: If the operation fails.
  """
  _pywrap_file_io.DeleteRecursively(compat.path_to_bytes(path))


@tf_export(v1=["gfile.IsDirectory"])
def is_directory(dirname):
  """Returns whether the path is a directory or not.

  Args:
    dirname: string, path to a potential directory

  Returns:
    True, if the path is a directory; False otherwise
  """
  return is_directory_v2(dirname)


@tf_export("io.gfile.isdir")
def is_directory_v2(path):
  """Returns whether the path is a directory or not.

  Args:
    path: string, path to a potential directory

  Returns:
    True, if the path is a directory; False otherwise
  """
  try:
    return _pywrap_file_io.IsDirectory(compat.path_to_bytes(path))
  except errors.OpError:
    return False


def has_atomic_move(path):
  """Checks whether the file system supports atomic moves.

  Returns whether or not the file system of the given path supports the atomic
  move operation for a file or folder.  If atomic move is supported, it is
  recommended to use a temp location for writing and then move to the final
  location.

  Args:
    path: string, path to a file

  Returns:
    True, if the path is on a file system that supports atomic move
    False, if the file system does not support atomic move. In such cases
           we need to be careful about using moves. In some cases it is safer
           not to use temporary locations in this case.
  """
  try:
    return _pywrap_file_io.HasAtomicMove(compat.path_to_bytes(path))
  except errors.OpError:
    # defaults to True
    return True


@tf_export(v1=["gfile.ListDirectory"])
def list_directory(dirname):
  """Returns a list of entries contained within a directory.

  The list is in arbitrary order. It does not contain the special entries "."
  and "..".

  Args:
    dirname: string, path to a directory

  Returns:
    [filename1, filename2, ... filenameN] as strings

  Raises:
    errors.NotFoundError if directory doesn't exist
  """
  return list_directory_v2(dirname)


@tf_export("io.gfile.listdir")
def list_directory_v2(path):
  """Returns a list of entries contained within a directory.

  The list is in arbitrary order. It does not contain the special entries "."
  and "..".

  Args:
    path: string, path to a directory

  Returns:
    [filename1, filename2, ... filenameN] as strings

  Raises:
    errors.NotFoundError if directory doesn't exist
  """
  if not is_directory(path):
    raise errors.NotFoundError(
        node_def=None,
        op=None,
        message="Could not find directory {}".format(path))

  # Convert each element to string, since the return values of the
  # vector of string should be interpreted as strings, not bytes.
  return [
      compat.as_str_any(filename)
      for filename in _pywrap_file_io.GetChildren(compat.path_to_bytes(path))
  ]


@tf_export(v1=["gfile.Walk"])
def walk(top, in_order=True):
  """Recursive directory tree generator for directories.

  Args:
    top: string, a Directory name
    in_order: bool, Traverse in order if True, post order if False.  Errors that
      happen while listing directories are ignored.

  Yields:
    Each yield is a 3-tuple:  the pathname of a directory, followed by lists of
    all its subdirectories and leaf files. That is, each yield looks like:
    `(dirname, [subdirname, subdirname, ...], [filename, filename, ...])`.
    Each item is a string.
  """
  return walk_v2(top, in_order)


@tf_export("io.gfile.walk")
def walk_v2(top, topdown=True, onerror=None):
  """Recursive directory tree generator for directories.

  Args:
    top: string, a Directory name
    topdown: bool, Traverse pre order if True, post order if False.
    onerror: optional handler for errors. Should be a function, it will be
      called with the error as argument. Rethrowing the error aborts the walk.
      Errors that happen while listing directories are ignored.

  Yields:
    Each yield is a 3-tuple:  the pathname of a directory, followed by lists of
    all its subdirectories and leaf files. That is, each yield looks like:
    `(dirname, [subdirname, subdirname, ...], [filename, filename, ...])`.
    Each item is a string.
  """

  def _make_full_path(parent, item):
    # Since `os.path.join` discards paths before one that starts with the path
    # separator (https://docs.python.org/3/library/os.path.html#os.path.join),
    # we have to manually handle that case as `/` is a valid character on GCS.
    if item[0] == os.sep:
      return "".join([os.path.join(parent, ""), item])
    return os.path.join(parent, item)

  top = compat.as_str_any(compat.path_to_str(top))
  try:
    listing = list_directory(top)
  except errors.NotFoundError as err:
    if onerror:
      onerror(err)
    else:
      return

  files = []
  subdirs = []
  for item in listing:
    full_path = _make_full_path(top, item)
    if is_directory(full_path):
      subdirs.append(item)
    else:
      files.append(item)

  here = (top, subdirs, files)

  if topdown:
    yield here

  for subdir in subdirs:
    for subitem in walk_v2(
        _make_full_path(top, subdir), topdown, onerror=onerror):
      yield subitem

  if not topdown:
    yield here


@tf_export(v1=["gfile.Stat"])
def stat(filename):
  """Returns file statistics for a given path.

  Args:
    filename: string, path to a file

  Returns:
    FileStatistics struct that contains information about the path

  Raises:
    errors.OpError: If the operation fails.
  """
  return stat_v2(filename)


@tf_export("io.gfile.stat")
def stat_v2(path):
  """Returns file statistics for a given path.

  Args:
    path: string, path to a file

  Returns:
    FileStatistics struct that contains information about the path

  Raises:
    errors.OpError: If the operation fails.
  """
  return _pywrap_file_io.Stat(compat.path_to_str(path))


def filecmp(filename_a, filename_b):
  """Compare two files, returning True if they are the same, False otherwise.

  We check size first and return False quickly if the files are different sizes.
  If they are the same size, we continue to generating a crc for the whole file.

  You might wonder: why not use Python's `filecmp.cmp()` instead? The answer is
  that the builtin library is not robust to the many different filesystems
  TensorFlow runs on, and so we here perform a similar comparison with
  the more robust FileIO.

  Args:
    filename_a: string path to the first file.
    filename_b: string path to the second file.

  Returns:
    True if the files are the same, False otherwise.
  """
  size_a = FileIO(filename_a, "rb").size()
  size_b = FileIO(filename_b, "rb").size()
  if size_a != size_b:
    return False

  # Size is the same. Do a full check.
  crc_a = file_crc32(filename_a)
  crc_b = file_crc32(filename_b)
  return crc_a == crc_b


def file_crc32(filename, block_size=_DEFAULT_BLOCK_SIZE):
  """Get the crc32 of the passed file.

  The crc32 of a file can be used for error checking; two files with the same
  crc32 are considered equivalent. Note that the entire file must be read
  to produce the crc32.

  Args:
    filename: string, path to a file
    block_size: Integer, process the files by reading blocks of `block_size`
      bytes. Use -1 to read the file as once.

  Returns:
    hexadecimal as string, the crc32 of the passed file.
  """
  crc = 0
  with FileIO(filename, mode="rb") as f:
    chunk = f.read(n=block_size)
    while chunk:
      crc = binascii.crc32(chunk, crc)
      chunk = f.read(n=block_size)
  return hex(crc & 0xFFFFFFFF)
