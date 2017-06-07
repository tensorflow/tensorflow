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
"""IO helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


def IsGCSPath(path):
  return path.startswith("gs://")


def ListDirectoryAbsolute(directory):
  """Yields all files in the given directory. The paths are absolute."""
  return (os.path.join(directory, path)
          for path in tf.gfile.ListDirectory(directory))


def ListRecursively(top):
  """Walks a directory tree, yielding (dir_path, file_paths) tuples.

  For each of `top` and its subdirectories, yields a tuple containing the path
  to the directory and the path to each of the contained files.  Note that
  unlike os.Walk()/tf.gfile.Walk(), this does not list subdirectories and the
  file paths are all absolute.

  If the directory does not exist, this yields nothing.

  Args:
    top: A path to a directory..
  Yields:
    A list of (dir_path, file_paths) tuples.
  """
  for dir_path, _, filenames in tf.gfile.Walk(top):
    yield (dir_path, (os.path.join(dir_path, filename)
                      for filename in filenames))
