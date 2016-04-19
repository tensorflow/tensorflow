# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Read a file and return its contents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os.path
import sys

from tensorflow.python.platform import logging


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
      os.path.join(
          os.path.dirname(__file__), os.pardir, os.pardir))
  path = os.path.join(tensorflow_root, path)
  path = os.path.abspath(path)
  try:
    with open(path, 'rb') as f:
      return f.read()
  except IOError as e:
    logging.warning('IOError %s on path %s', e, path)
    raise e


# pylint: disable=protected-access
def get_data_files_path():
  """Get the directory where files specified in data attribute are stored.

  Returns:
    The directory where files specified in data attribute of py_test
    and py_binary are stored.
  """
  return os.path.dirname(inspect.getfile(sys._getframe(1)))


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
  data_files_path = os.path.dirname(inspect.getfile(sys._getframe(1)))
  return os.path.join(data_files_path, path)

def readahead_file_path(path, unused_readahead=None):
  """Readahead files not implemented; simply returns given path."""
  return path
