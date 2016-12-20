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

"""## Resource management.

@@get_data_files_path
@@get_path_to_datafile
@@load_resource
@@readahead_file_path
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect as _inspect
import os as _os
import sys as _sys

from tensorflow.python.util.all_util import remove_undocumented


def load_resource(path):
  """Load the resource at given path, where path is relative to tensorflow/.

  Args:
    path: a string resource path relative to tensorflow/.

  Returns:
    The contents of that resource.

  Raises:
    IOError: If the path is not found, or the resource can't be opened.
  """
  tensorflow_root = (
      _os.path.join(
          _os.path.dirname(__file__), _os.pardir, _os.pardir))
  path = _os.path.join(tensorflow_root, path)
  path = _os.path.abspath(path)
  with open(path, 'rb') as f:
    return f.read()


# pylint: disable=protected-access
def get_data_files_path():
  """Get the directory where files specified in data attribute are stored.

  Returns:
    The directory where files specified in data attribute of py_test
    and py_binary are stored.
  """
  return _os.path.dirname(_inspect.getfile(_sys._getframe(1)))


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


def readahead_file_path(path, unused_readahead=None):
  """Readahead files not implemented; simply returns given path."""
  return path

_allowed_symbols = []
remove_undocumented(__name__, _allowed_symbols)
