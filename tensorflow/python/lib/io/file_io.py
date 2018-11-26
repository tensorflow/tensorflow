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
"""File IO methods that wrap the C++ FileSystem API.

The C++ FileSystem API is SWIG wrapped in file_io.i. These functions call those
to accomplish basic File IO operations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import binascii
import os
import uuid

import six

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

# A good default block size depends on the system in question.
# A somewhat conservative default chosen here.
_DEFAULT_BLOCK_SIZE = 16 * 1024 * 1024


class FileIO(object):
  """FileIO class that exposes methods to read / write to / from files.

  The constructor takes the following arguments:
  name: name of the file
  mode: one of 'r', 'w', 'a', 'r+', 'w+', 'a+'. Append 'b' for bytes mode.

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
      with errors.raise_exception_on_not_ok_status() as status:
        self._read_buf = pywrap_tensorflow.CreateBufferedInputStream(
            compat.as_bytes(self.__name), 1024 * 512, status)

  def _prewrite_check(self):
    if not self._writable_file:
      if not self._write_check_passed:
        raise errors.PermissionDeniedError(None, None,
                                           "File isn't open for writing")
      with errors.raise_exception_on_not_ok_status() as status:
        self._writable_file = pywrap_tensorflow.CreateWritableFile(
            compat.as_bytes(self.__name), compat.as_bytes(self.__mode), status)

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
    with errors.raise_exception_on_not_ok_status() as status:
      pywrap_tensorflow.AppendToFile(
          compat.as_bytes(file_content), self._writable_file, status)

  def read(self, n=-1):
    """Returns the contents of a file as a string.

    Starts reading from current position in file.

    Args:
      n: Read 'n' bytes if n != -1. If n = -1, reads to end of file.

    Returns:
      'n' bytes of the file (or whole file) in bytes mode or 'n' bytes of the
      string if in string (regular) mode.
    """
    self._preread_check()
    with errors.raise_exception_on_not_ok_status() as status:
      if n == -1:
        length = self.size() - self.tell()
      else:
        length = n
      return self._prepare_value(
          pywrap_tensorflow.ReadFromStream(self._read_buf, length, status))

  @deprecation.deprecated_args(
      None,
      "position is deprecated in favor of the offset argument.",
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
        2: relative to the end of file. offset is usually negative.
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

    with errors.raise_exception_on_not_ok_status() as status:
      if whence == 0:
        pass
      elif whence == 1:
        offset += self.tell()
      elif whence == 2:
        offset += self.size()
      else:
        raise errors.InvalidArgumentError(
            None, None,
            "Invalid whence argument: {}. Valid values are 0, 1, or 2."
            .format(whence))
      ret_status = self._read_buf.Seek(offset)
      pywrap_tensorflow.Set_TF_Status_from_Status(status, ret_status)

  def readline(self):
    r"""Reads the next line from the file. Leaves the '\n' at the end."""
    self._preread_check()
    return self._prepare_value(self._read_buf.ReadLineAsString())

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
    self._preread_check()
    return self._read_buf.Tell()

  def __enter__(self):
    """Make usable with "with" statement."""
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    """Make usable with "with" statement."""
    self.close()

  def __iter__(self):
    return self

  def next(self):
    retval = self.readline()
    if not retval:
      raise StopIteration()
    return retval

  def __next__(self):
    return self.next()

  def flush(self):
    """Flushes the Writable file.

    This only ensures that the data has made its way out of the process without
    any guarantees on whether it's written to disk. This means that the
    data would survive an application crash but not necessarily an OS crash.
    """
    if self._writable_file:
      with errors.raise_exception_on_not_ok_status() as status:
        ret_status = self._writable_file.Flush()
        pywrap_tensorflow.Set_TF_Status_from_Status(status, ret_status)

  def close(self):
    """Closes FileIO. Should be called for the WritableFile to be flushed."""
    self._read_buf = None
    if self._writable_file:
      with errors.raise_exception_on_not_ok_status() as status:
        ret_status = self._writable_file.Close()
        pywrap_tensorflow.Set_TF_Status_from_Status(status, ret_status)
    self._writable_file = None


@tf_export(v1=["gfile.Exists"])
def file_exists(filename):
  """Determines whether a path exists or not.

  Args:
    filename: string, a path

  Returns:
    True if the path exists, whether its a file or a directory.
    False if the path does not exist and there are no filesystem errors.

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.
  """
  return file_exists_v2(filename)


@tf_export("io.gfile.exists")
def file_exists_v2(path):
  """Determines whether a path exists or not.

  Args:
    path: string, a path

  Returns:
    True if the path exists, whether its a file or a directory.
    False if the path does not exist and there are no filesystem errors.

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.
  """
  try:
    with errors.raise_exception_on_not_ok_status() as status:
      pywrap_tensorflow.FileExists(compat.as_bytes(path), status)
  except errors.NotFoundError:
    return False
  return True


@tf_export(v1=["gfile.Remove"])
def delete_file(filename):
  """Deletes the file located at 'filename'.

  Args:
    filename: string, a filename

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
    NotFoundError if the file does not exist.
  """
  delete_file_v2(filename)


@tf_export("io.gfile.remove")
def delete_file_v2(path):
  """Deletes the path located at 'path'.

  Args:
    path: string, a path

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
    NotFoundError if the path does not exist.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.DeleteFile(compat.as_bytes(path), status)


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
    NotFoundError etc.
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
    errors.OpError: If there are filesystem / directory listing errors.
  """
  return get_matching_files_v2(filename)


@tf_export("io.gfile.glob")
def get_matching_files_v2(pattern):
  """Returns a list of files that match the given pattern(s).

  Args:
    pattern: string or iterable of strings. The glob pattern(s).

  Returns:
    A list of strings containing filenames that match the given pattern(s).

  Raises:
    errors.OpError: If there are filesystem / directory listing errors.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    if isinstance(pattern, six.string_types):
      return [
          # Convert the filenames to string from bytes.
          compat.as_str_any(matching_filename)
          for matching_filename in pywrap_tensorflow.GetMatchingFiles(
              compat.as_bytes(pattern), status)
      ]
    else:
      return [
          # Convert the filenames to string from bytes.
          compat.as_str_any(matching_filename)
          for single_filename in pattern
          for matching_filename in pywrap_tensorflow.GetMatchingFiles(
              compat.as_bytes(single_filename), status)
      ]


@tf_export(v1=["gfile.MkDir"])
def create_dir(dirname):
  """Creates a directory with the name 'dirname'.

  Args:
    dirname: string, name of the directory to be created

  Notes:
    The parent directories need to exist. Use recursive_create_dir instead if
    there is the possibility that the parent dirs don't exist.

  Raises:
    errors.OpError: If the operation fails.
  """
  create_dir_v2(dirname)


@tf_export("io.gfile.mkdir")
def create_dir_v2(path):
  """Creates a directory with the name given by 'path'.

  Args:
    path: string, name of the directory to be created

  Notes:
    The parent directories need to exist. Use recursive_create_dir instead if
    there is the possibility that the parent dirs don't exist.

  Raises:
    errors.OpError: If the operation fails.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.CreateDir(compat.as_bytes(path), status)


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
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.RecursivelyCreateDir(compat.as_bytes(path), status)


@tf_export(v1=["gfile.Copy"])
def copy(oldpath, newpath, overwrite=False):
  """Copies data from oldpath to newpath.

  Args:
    oldpath: string, name of the file who's contents need to be copied
    newpath: string, name of the file to which to copy to
    overwrite: boolean, if false its an error for newpath to be occupied by an
        existing file.

  Raises:
    errors.OpError: If the operation fails.
  """
  copy_v2(oldpath, newpath, overwrite)


@tf_export("io.gfile.copy")
def copy_v2(src, dst, overwrite=False):
  """Copies data from src to dst.

  Args:
    src: string, name of the file whose contents need to be copied
    dst: string, name of the file to which to copy to
    overwrite: boolean, if false its an error for newpath to be occupied by an
        existing file.

  Raises:
    errors.OpError: If the operation fails.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.CopyFile(
        compat.as_bytes(src), compat.as_bytes(dst), overwrite, status)


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
def rename_v2(src, dst, overwrite):
  """Rename or move a file / directory.

  Args:
    src: string, pathname for a file
    dst: string, pathname to which the file needs to be moved
    overwrite: boolean, if false it's an error for `dst` to be occupied by
        an existing file.

  Raises:
    errors.OpError: If the operation fails.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.RenameFile(
        compat.as_bytes(src), compat.as_bytes(dst), overwrite, status)


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
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.DeleteRecursively(compat.as_bytes(path), status)


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
  status = c_api_util.ScopedTFStatus()
  return pywrap_tensorflow.IsDirectory(compat.as_bytes(path), status)


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
    raise errors.NotFoundError(None, None, "Could not find directory")
  with errors.raise_exception_on_not_ok_status() as status:
    # Convert each element to string, since the return values of the
    # vector of string should be interpreted as strings, not bytes.
    return [
        compat.as_str_any(filename)
        for filename in pywrap_tensorflow.GetChildren(
            compat.as_bytes(path), status)
    ]


@tf_export(v1=["gfile.Walk"])
def walk(top, in_order=True):
  """Recursive directory tree generator for directories.

  Args:
    top: string, a Directory name
    in_order: bool, Traverse in order if True, post order if False.

  Errors that happen while listing directories are ignored.

  Yields:
    Each yield is a 3-tuple:  the pathname of a directory, followed by lists of
    all its subdirectories and leaf files.
    (dirname, [subdirname, subdirname, ...], [filename, filename, ...])
    as strings
  """
  return walk_v2(top, in_order)


@tf_export("io.gfile.walk")
def walk_v2(top, topdown, onerror=None):
  """Recursive directory tree generator for directories.

  Args:
    top: string, a Directory name
    topdown: bool, Traverse pre order if True, post order if False.
    onerror: optional handler for errors. Should be a function, it will be
      called with the error as argument. Rethrowing the error aborts the walk.

  Errors that happen while listing directories are ignored.

  Yields:
    Each yield is a 3-tuple:  the pathname of a directory, followed by lists of
    all its subdirectories and leaf files.
    (dirname, [subdirname, subdirname, ...], [filename, filename, ...])
    as strings
  """
  top = compat.as_str_any(top)
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
    full_path = os.path.join(top, item)
    if is_directory(full_path):
      subdirs.append(item)
    else:
      files.append(item)

  here = (top, subdirs, files)

  if topdown:
    yield here

  for subdir in subdirs:
    for subitem in walk_v2(os.path.join(top, subdir), topdown, onerror=onerror):
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
  file_statistics = pywrap_tensorflow.FileStatistics()
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.Stat(compat.as_bytes(path), file_statistics, status)
    return file_statistics


def filecmp(filename_a, filename_b):
  """Compare two files, returning True if they are the same, False otherwise.

  We check size first and return False quickly if the files are different sizes.
  If they are the same size, we continue to generating a crc for the whole file.

  You might wonder: why not use Python's filecmp.cmp() instead? The answer is
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
