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

"""Import router for file_io."""
# pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.lib.io import file_io


class GFile(file_io.FileIO):
  """File I/O wrappers with thread locking."""

  def __init__(self, name, mode='r'):
    mode = mode.replace('b', '')
    super(GFile, self).__init__(name=name, mode=mode)


class FastGFile(file_io.FileIO):
  """File I/O wrappers without thread locking."""

  def __init__(self, name, mode='r'):
    mode = mode.replace('b', '')
    super(FastGFile, self).__init__(name=name, mode=mode)


# This should be kept consistent with the OSS implementation
# of the gfile interface.

# Does not alias to Open so that we use our version of GFile to strip
# 'b' mode.
Open = GFile

# pylint: disable=invalid-name
Exists = file_io.file_exists
IsDirectory = file_io.is_directory
Glob = file_io.get_matching_files
MkDir = file_io.create_dir
MakeDirs = file_io.recursive_create_dir
Remove = file_io.delete_file
DeleteRecursively = file_io.delete_recursively
ListDirectory = file_io.list_directory
Walk = file_io.walk
Stat = file_io.stat
Rename = file_io.rename
Copy = file_io.copy
