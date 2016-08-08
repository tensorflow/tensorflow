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

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors
from tensorflow.python.util import compat


def file_exists(filename):
  """Determines whether a path exists or not.

  Args:
    filename: string, a path

  Returns:
    True if the path exists, whether its a file or a directory.
  """
  return pywrap_tensorflow.FileExists(compat.as_bytes(filename))


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


def read_file_to_string(filename):
  """Reads the entire contents of a file to a string.

  Args:
    filename: string, path to a file

  Returns:
    contents of the file as a string

  Raises:
    errors.OpError: Raises variety of errors that are subtypes e.g.
    NotFoundError etc.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    return pywrap_tensorflow.ReadFileToString(compat.as_bytes(filename), status)


def write_string_to_file(filename, file_content):
  """Writes a string to a given file.

  Args:
    filename: string, path to a file
    file_content: string, contents that need to be written to the file

  Raises:
    errors.OpError: If there are errors during the operation.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.WriteStringToFile(
        compat.as_bytes(filename), compat.as_bytes(file_content), status)


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
    return pywrap_tensorflow.GetMatchingFiles(compat.as_bytes(filename), status)


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
  """Create a directory and all parent/intermediate directories.

  Args:
    dirname: string, name of the directory to be created

  Raises:
    errors.OpError: If the operation fails.
  """
  with errors.raise_exception_on_not_ok_status() as status:
    dirs = dirname.split('/')
    for i in range(len(dirs)):
      partial_dir = '/'.join(dirs[0:i + 1])
      if partial_dir and not file_exists(partial_dir):
        pywrap_tensorflow.CreateDir(compat.as_bytes(partial_dir), status)


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

  Raises:
    errors.OpError: If the path doesn't exist or other errors
  """
  with errors.raise_exception_on_not_ok_status() as status:
    return pywrap_tensorflow.IsDirectory(compat.as_bytes(dirname), status)


def list_directory(dirname):
  """Returns a list of entries contained within a directory.

  The list is in arbitrary order. It does not contain the special entries "."
  and "..".

  Args:
    dirname: string, path to a directory

  Returns:
    [filename1, filename2, ... filenameN]

  Raises:
    errors.NotFoundError if directory doesn't exist
  """
  if not is_directory(dirname):
    raise errors.NotFoundError(None, None, 'Could not find directory')
  file_list = get_matching_files(os.path.join(compat.as_str_any(dirname), '*'))
  return [compat.as_bytes(pywrap_tensorflow.Basename(compat.as_bytes(filename)))
          for filename in file_list]


def walk(top, in_order=True):
  """Recursive directory tree generator for directories.

  Args:
    top: string, a Directory name
    in_order: bool, Traverse in order if True, post order if False.

  Errors that happen while listing directories are ignored.

  Yields:
    # Each yield is a 3-tuple:  the pathname of a directory, followed
    # by lists of all its subdirectories and leaf files.
    (dirname, [subdirname, subdirname, ...], [filename, filename, ...])
  """
  top = compat.as_bytes(top)
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
