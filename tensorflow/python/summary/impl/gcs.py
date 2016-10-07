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
"""Functions for communicating with Google Cloud Storage."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from tensorflow.python.platform import tf_logging as logging

# All GCS paths should start with this.
PATH_PREFIX = 'gs://'

# TODO(phurst): We should use the GCS Python API.


def CopyContents(gcs_path, byte_offset, local_file):
  """Copies the contents of gcs_path from byte_offset onwards to local_file.

  Args:
    gcs_path: The path to the GCS object.
    byte_offset: The byte offset to start appending from.
    local_file: The file object to write into.

  Raises:
    ValueError: If offset is negative or gcs_path is not a valid GCS path.
    CalledProcessError: If the gsutil command failed.
  """
  if byte_offset < 0:
    raise ValueError('byte_offset must not be negative')
  command = ['gsutil', 'cat', '-r', '%d-' % byte_offset, gcs_path]
  subprocess.check_call(command, stdout=local_file)
  local_file.flush()


def ListDirectory(directory):
  """Lists all files in the given directory."""
  command = ['gsutil', 'ls', directory]
  return subprocess.check_output(command).splitlines()


def ListRecursively(top):
  """Walks a directory tree, yielding (dir_path, file_paths) tuples.

  For each top |top| and its subdirectories, yields a tuple containing the path
  to the directory and the path to each of the contained files.  Note that
  unlike os.Walk()/gfile.Walk(), this does not list subdirectories and the file
  paths are all absolute.

  Args:
    top: A path to a GCS directory.
  Returns:
    A list of (dir_path, file_paths) tuples.

  """
  if top.endswith('/'):
    wildcard = top + '**'
  else:
    wildcard = top + '/**'
  tuples = []
  try:
    file_paths = ListDirectory(wildcard)
  except subprocess.CalledProcessError as e:
    logging.info('%s, assuming it means no files were found', e)
    return []
  for file_path in file_paths:
    dir_path = os.path.dirname(file_path)
    if tuples and tuples[-1][0] == dir_path:
      tuples[-1][1].append(file_path)
    else:
      tuples.append((dir_path, [file_path]))
  return tuples


def IsDirectory(path):
  """Returns true if path exists and is a directory."""
  path = path.rstrip('/')
  try:
    ls = ListDirectory(path)
  except subprocess.CalledProcessError:
    # Doesn't exist.
    return False
  if len(ls) == 1:
    # Either it's a file (which ls-es as itself) or it's a dir with one file.
    return ls[0] != path
  else:
    return True


def Exists(path):
  """Returns true if path exists."""
  try:
    ListDirectory(path)
    return True
  except subprocess.CalledProcessError:
    return False


def IsGCSPath(path):
  return path.startswith(PATH_PREFIX)


def CheckIsSupported():
  """Raises an OSError if the system isn't set up for Google Cloud Storage.

  Raises:
    OSError: If the system hasn't been set up so that TensorBoard can access
      Google Cloud Storage.   The error's message contains installation
      instructions.
  """
  try:
    subprocess.check_output(['gsutil', 'version'])
  except OSError as e:
    logging.error('Error while checking for gsutil: %s', e)
    raise OSError(
        'Unable to execute the gsutil binary, which is required for Google '
        'Cloud Storage support. You can find installation instructions at '
        'https://goo.gl/sST520')
