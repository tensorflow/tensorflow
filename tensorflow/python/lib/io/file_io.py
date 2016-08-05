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
  return pywrap_tensorflow.FileExists(compat.as_bytes(filename))


def delete_file(filename):
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.DeleteFile(compat.as_bytes(filename), status)


def read_file_to_string(filename):
  with errors.raise_exception_on_not_ok_status() as status:
    return pywrap_tensorflow.ReadFileToString(compat.as_bytes(filename), status)


def write_string_to_file(filename, file_content):
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.WriteStringToFile(
        compat.as_bytes(filename), compat.as_bytes(file_content), status)


def get_matching_files(filename):
  with errors.raise_exception_on_not_ok_status() as status:
    return pywrap_tensorflow.GetMatchingFiles(compat.as_bytes(filename), status)


def create_dir(dirname):
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.CreateDir(compat.as_bytes(dirname), status)


def recursive_create_dir(dirname):
  with errors.raise_exception_on_not_ok_status() as status:
    dirs = dirname.split('/')
    for i in range(len(dirs)):
      partial_dir = '/'.join(dirs[0:i + 1])
      if partial_dir and not file_exists(partial_dir):
        pywrap_tensorflow.CreateDir(compat.as_bytes(partial_dir), status)


def copy(oldpath, newpath, overwrite=False):
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.CopyFile(
        compat.as_bytes(oldpath), compat.as_bytes(newpath), overwrite, status)


def rename(oldname, newname, overwrite=False):
  with errors.raise_exception_on_not_ok_status() as status:
    return pywrap_tensorflow.RenameFile(
        compat.as_bytes(oldname), compat.as_bytes(newname), overwrite, status)


def delete_recursively(dirname):
  with errors.raise_exception_on_not_ok_status() as status:
    return pywrap_tensorflow.DeleteRecursively(compat.as_bytes(dirname), status)


def is_directory(dirname):
  with errors.raise_exception_on_not_ok_status() as status:
    return pywrap_tensorflow.IsDirectory(compat.as_bytes(dirname), status)


def list_directory(dirname):
  """Returns a list of entries contained within a directory.

  The list is in arbitrary order. It does not contain the special entries "."
  and "..".

  Args:
    dirname: string, path to a directory

  Raises:
    NotFoundError if directory doesn't exist

  Returns:
    [filename1, filename2, ... filenameN]
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
  file_statistics = pywrap_tensorflow.FileStatistics()
  with errors.raise_exception_on_not_ok_status() as status:
    pywrap_tensorflow.Stat(compat.as_bytes(filename), file_statistics, status)
    return file_statistics
