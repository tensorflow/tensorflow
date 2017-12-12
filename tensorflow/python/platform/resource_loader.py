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
"""Resource management library.

@@get_abs_data_path
@@get_data_files_path
@@get_grandparent
@@get_path_to_datafile
@@get_root_dir_with_all_resources
@@load_resource
@@readahead_file_path
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os as _os
import sys as _sys

from tensorflow.python.util import tf_inspect as _inspect
from tensorflow.python.util.all_util import remove_undocumented


def get_grandparent(path, degree):
  """Get a files grandparent of the given degree.

  Args:
    path: a string resource path to the file.
    degree: a integer indicating the grandparents degree.

  Returns:
    The absolute path for the grandparent.
  """
  for _ in range(degree):
    path = _os.path.join(path, _os.pardir)

  path = _os.path.abspath(path)
  return path


def get_abs_data_path(path, depth, frame=0):
  """Get the absolute path relative to a file upwards the callstack.

  Args:
    path: a string resource path relative to the parent of tensorflow/.
    depth: a integer indicating the depth of the file.
    frame: a integer indicating the position in the callstack.

  Returns:
    The absolute path for the given relative path.
  """
  root = get_data_files_path(1 + frame)
  root = get_grandparent(root, depth)
  path = _os.path.join(root, path)
  path = _os.path.abspath(path)
  return path


def load_resource(path):
  """Load the resource at given path, where path is relative to tensorflow/.

  Args:
    path: a string resource path relative to tensorflow/.

  Returns:
    The contents of that resource.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  root = get_grandparent(os_.path.dirname(__file__), 2)
  path = _os.path.join(root, path)

  with open(path, 'rb') as f:
    return f.read()


# pylint: disable=protected-access
def get_data_files_path(frame=0):
  """Get a direct path to the data files colocated with the script.

  Returns:
    The directory where files specified in data attribute of py_test
    and py_binary are stored.
  """
  return _os.path.dirname(_inspect.getfile(_sys._getframe(1 + frame)))


def get_root_dir_with_all_resources():
  """Get a root directory containing all the data attributes in the build rule.

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary. Falls back to returning the same as get_data_files_path if it
    fails to detect a bazel runfiles directory.
  """
  script_dir = get_data_files_path()

  # Create a history of the paths, because the data files are located relative
  # to the repository root directory, which is directly under runfiles
  # directory.
  directories = [script_dir]
  data_files_dir = ''

  while True:
    candidate_dir = directories[-1]
    current_directory = _os.path.basename(candidate_dir)
    if '.runfiles' in current_directory:
      # Our file should never be directly under runfiles.
      # If the history has only one item, it means we are directly inside the
      # runfiles directory, something is wrong, fall back to the default return
      # value, script directory.
      if len(directories) > 1:
        data_files_dir = directories[-2]

      break
    else:
      new_candidate_dir = _os.path.dirname(candidate_dir)
      # If we are at the root directory these two will be the same.
      if new_candidate_dir == candidate_dir:
        break
      else:
        directories.append(new_candidate_dir)

  return data_files_dir or script_dir


def get_path_to_datafile(path):
  """Get the path to the specified file in the data dependencies.

  The path is relative to tensorflow/

  Args:
    path: a string resource path relative to tensorflow/

  Returns:
    The path to the specified file present in the data attribute of py_test
    or py_binary.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  root = get_data_files_path(1)
  return _os.path.join(root, path)


def readahead_file_path(path, readahead='128M'):  # pylint: disable=unused-argument
  """Readahead files not implemented; simply returns given path."""
  return path


_allowed_symbols = []
remove_undocumented(__name__, _allowed_symbols)
