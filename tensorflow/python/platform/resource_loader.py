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
"""Resource management library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os as _os
import sys as _sys

from tensorflow.python.util import tf_inspect as _inspect
from tensorflow.python.util.tf_export import tf_export


@tf_export('resource_loader.load_resource')
def load_resource(path):
  """Load the resource at given path, where path is relative to tensorflow/.

  Args:
    path: a string resource path relative to tensorflow/.

  Returns:
    The contents of that resource.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  tensorflow_root = (_os.path.join(
      _os.path.dirname(__file__), _os.pardir, _os.pardir))
  path = _os.path.join(tensorflow_root, path)
  path = _os.path.abspath(path)
  with open(path, 'rb') as f:
    return f.read()


# pylint: disable=protected-access
@tf_export('resource_loader.get_data_files_path')
def get_data_files_path():
  """Get a direct path to the data files colocated with the script.

  Returns:
    The directory where files specified in data attribute of py_test
    and py_binary are stored.
  """
  return _os.path.dirname(_inspect.getfile(_sys._getframe(1)))


@tf_export('resource_loader.get_root_dir_with_all_resources')
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


@tf_export('resource_loader.get_path_to_datafile')
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
  data_files_path = _os.path.dirname(_inspect.getfile(_sys._getframe(1)))
  return _os.path.join(data_files_path, path)


@tf_export('resource_loader.readahead_file_path')
def readahead_file_path(path, readahead='128M'):  # pylint: disable=unused-argument
  """Readahead files not implemented; simply returns given path."""
  return path
