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
"""Functions that wrap both gfile and gcs.

This module is *not* intended to be a general-purpose IO wrapper library; it
only implements the operations that are necessary for loading event files. The
functions either dispatch to the gcs library or to gfile, depending on whether
the path is a GCS 'pseudo-path' (i.e., it satisfies gcs.IsGCSPath) or not.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.platform import gfile
from tensorflow.python.summary.impl import event_file_loader
from tensorflow.python.summary.impl import gcs
from tensorflow.python.summary.impl import gcs_file_loader


def CreateFileLoader(path):
  """Creates a file loader for the given path.

  Args:
    path: A string representing either a normal path or a GCS
  Returns:
    An object with a Load() method that yields event_pb2.Event protos.
  """
  if gcs.IsGCSPath(path):
    return gcs_file_loader.GCSFileLoader(path)
  else:
    return event_file_loader.EventFileLoader(path)


def ListDirectoryAbsolute(directory):
  """Yields all files in the given directory. The paths are absolute."""
  if gcs.IsGCSPath(directory):
    return gcs.ListDirectory(directory)
  else:
    return (os.path.join(directory, path)
            for path in gfile.ListDirectory(directory))


def ListRecursively(top):
  """Walks a directory tree, yielding (dir_path, file_paths) tuples.

  For each of `top` and its subdirectories, yields a tuple containing the path
  to the directory and the path to each of the contained files.  Note that
  unlike os.Walk()/gfile.Walk(), this does not list subdirectories and the file
  paths are all absolute.

  If the directory does not exist, this yields nothing.

  Args:
    top: A path to a directory..
  Yields:
    A list of (dir_path, file_paths) tuples.
  """
  if gcs.IsGCSPath(top):
    for x in gcs.ListRecursively(top):
      yield x
  else:
    for dir_path, _, filenames in gfile.Walk(top):
      yield (dir_path, (os.path.join(dir_path, filename)
                        for filename in filenames))


def IsDirectory(path):
  """Returns true if path exists and is a directory."""
  if gcs.IsGCSPath(path):
    return gcs.IsDirectory(path)
  else:
    return gfile.IsDirectory(path)


def Exists(path):
  if gcs.IsGCSPath(path):
    return gcs.Exists(path)
  else:
    return gfile.Exists(path)


def Size(path):
  """Returns the number of bytes in the given file. Doesn't work on GCS."""
  if gcs.IsGCSPath(path):
    raise NotImplementedError("io_wrapper.Size doesn't support GCS paths")
  else:
    return gfile.Open(path).Size()
