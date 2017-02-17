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

import os
import uuid

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors
from tensorflow.python.util import compat


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

  def seek(self, position):
    """Seeks to the position in the file."""
    self._preread_check()
    with errors.raise_exception_on_not_ok_status() as status:
      ret_status = self._read_buf.Seek(position)
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
  try:
    with errors.raise_exception_on_not_ok_status() as status:
      pywrap_tensorflow.FileExists(compat.as_bytes(filename), status)
  except errors.NotFoundError:
    return False
  return True


def delete_file(filename):
  """Deletes the file located at 'filename'.

  Args:
    filename: string, a filename

  Raises:
    errors.OpError: Propagates any errors reported by the FileSystem API.  E.g.,
    NotFoundError if the file does not exist.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.DeleteFile(compat.as_bytes(filename), status)


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


def get_matching_files(filename):
  """Returns a list of files that match the given pattern.

  Args:
    filename: string, the pattern

  Returns:
    Returns a list of strings containing filenames that match the given pattern.

  Raises:
    errors.OpError: If there are filesystem / directory listing errors.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    # Convert each element to string, since the return values of the
    # vector of string should be interpreted as strings, not bytes.
    return [compat.as_str_any(matching_filename)
            for matching_filename in pywrap_tensorflow.GetMatchingFiles(
                compat.as_bytes(filename), status)]


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
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.CreateDir(compat.as_bytes(dirname), status)


def recursive_create_dir(dirname):
  """Creates a directory and all parent/intermediate directories.

  It succeeds if dirname already exists and is writable.

  Args:
    dirname: string, name of the directory to be created

  Raises:
    errors.OpError: If the operation fails.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.RecursivelyCreateDir(compat.as_bytes(dirname), status)


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
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.CopyFile(
        compat.as_bytes(oldpath), compat.as_bytes(newpath), overwrite, status)


def rename(oldname, newname, overwrite=False):
  """Rename or move a file / directory.

  Args:
    oldname: string, pathname for a file
    newname: string, pathname to which the file needs to be moved
    overwrite: boolean, if false its an error for newpath to be occupied by an
        existing file.

  Raises:
    errors.OpError: If the operation fails.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.RenameFile(
        compat.as_bytes(oldname), compat.as_bytes(newname), overwrite, status)


def atomic_write_string_to_file(filename, contents):
  """Writes to `filename` atomically.

  This means that when `filename` appears in the filesystem, it will contain
  all of `contents`. With write_string_to_file, it is possible for the file
  to appear in the filesystem with `contents` only partially written.

  Accomplished by writing to a temp file and then renaming it.

  Args:
    filename: string, pathname for a file
    contents: string, contents that need to be written to the file
  """
  temp_pathname = filename + ".tmp" + uuid.uuid4().hex
  write_string_to_file(temp_pathname, contents)
  rename(temp_pathname, filename, overwrite=True)


def delete_recursively(dirname):
  """Deletes everything under dirname recursively.

  Args:
    dirname: string, a path to a directory

  Raises:
    errors.OpError: If the operation fails.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.DeleteRecursively(compat.as_bytes(dirname), status)


def is_directory(dirname):
  """Returns whether the path is a directory or not.

  Args:
    dirname: string, path to a potential directory

  Returns:
    True, if the path is a directory; False otherwise
  """
  try:
    status = pywrap_tensorflow.TF_NewStatus()
    return pywrap_tensorflow.IsDirectory(compat.as_bytes(dirname), status)
  finally:
    pywrap_tensorflow.TF_DeleteStatus(status)


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
  if not is_directory(dirname):
    raise errors.NotFoundError(None, None, "Could not find directory")
  with errors.raise_exception_on_not_ok_status() as status:
    # Convert each element to string, since the return values of the
    # vector of string should be interpreted as strings, not bytes.
    return [
        compat.as_str_any(filename)
        for filename in pywrap_tensorflow.GetChildren(
            compat.as_bytes(dirname), status)
    ]


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
  top = compat.as_str_any(top)
  try:
    listing = list_directory(top)
  except errors.NotFoundError:
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

  if in_order:
    yield here

  for subdir in subdirs:
    for subitem in walk(os.path.join(top, subdir), in_order):
      yield subitem

  if not in_order:
    yield here


def stat(filename):
  """Returns file statistics for a given path.

  Args:
    filename: string, path to a file

  Returns:
    FileStatistics struct that contains information about the path

  Raises:
    errors.OpError: If the operation fails.
  """
  file_statistics = pywrap_tensorflow.FileStatistics()
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.Stat(compat.as_bytes(filename), file_statistics, status)
    return file_statistics
