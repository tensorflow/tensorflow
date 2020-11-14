# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-import-not-at-top
"""Utilities related to disk I/O."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import six


if sys.version_info >= (3, 6):

  def _path_to_string(path):
    if isinstance(path, os.PathLike):
      return os.fspath(path)
    return path
elif sys.version_info >= (3, 4):

  def _path_to_string(path):
    import pathlib
    if isinstance(path, pathlib.Path):
      return str(path)
    return path
else:

  def _path_to_string(path):
    return path


def path_to_string(path):
  """Convert `PathLike` objects to their string representation.

  If given a non-string typed path object, converts it to its string
  representation. Depending on the python version used, this function
  can handle the following arguments:
  python >= 3.6: Everything supporting the fs path protocol
    https://www.python.org/dev/peps/pep-0519
  python >= 3.4: Only `pathlib.Path` objects

  If the object passed to `path` is not among the above, then it is
  returned unchanged. This allows e.g. passthrough of file objects
  through this function.

  Args:
    path: `PathLike` object that represents a path

  Returns:
    A string representation of the path argument, if Python support exists.
  """
  return _path_to_string(path)


def ask_to_proceed_with_overwrite(filepath):
  """Produces a prompt asking about overwriting a file.

  Arguments:
      filepath: the path to the file to be overwritten.

  Returns:
      True if we can proceed with overwrite, False otherwise.
  """
  overwrite = six.moves.input('[WARNING] %s already exists - overwrite? '
                              '[y/n]' % (filepath)).strip().lower()
  while overwrite not in ('y', 'n'):
    overwrite = six.moves.input('Enter "y" (overwrite) or "n" '
                                '(cancel).').strip().lower()
  if overwrite == 'n':
    return False
  print('[TIP] Next time specify overwrite=True!')
  return True
