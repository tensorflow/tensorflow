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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from tensorflow.python.lib.io.file_io import copy as Copy
from tensorflow.python.lib.io.file_io import create_dir as MkDir
from tensorflow.python.lib.io.file_io import delete_file as Remove
from tensorflow.python.lib.io.file_io import delete_recursively as DeleteRecursively
from tensorflow.python.lib.io.file_io import file_exists as Exists
from tensorflow.python.lib.io.file_io import FileIO as _FileIO
from tensorflow.python.lib.io.file_io import get_matching_files as Glob
from tensorflow.python.lib.io.file_io import is_directory as IsDirectory
from tensorflow.python.lib.io.file_io import list_directory as ListDirectory
from tensorflow.python.lib.io.file_io import recursive_create_dir as MakeDirs
from tensorflow.python.lib.io.file_io import rename as Rename
from tensorflow.python.lib.io.file_io import stat as Stat
from tensorflow.python.lib.io.file_io import walk as Walk
# pylint: enable=unused-import
from tensorflow.python.util.all_util import remove_undocumented


class GFile(_FileIO):
  """File I/O wrappers without thread locking."""

  def __init__(self, name, mode='r'):
    mode = mode.replace('b', '')
    super(GFile, self).__init__(name=name, mode=mode)


class FastGFile(_FileIO):
  """File I/O wrappers without thread locking."""

  def __init__(self, name, mode='r'):
    mode = mode.replace('b', '')
    super(FastGFile, self).__init__(name=name, mode=mode)


# Does not alias to Open so that we use our version of GFile to strip
# 'b' mode.
Open = GFile

# TODO(drpng): Find the right place to document these.
_allowed_symbols = [
    'Copy',
    'DeleteRecursively',
    'Exists',
    'FastGFile',
    'GFile',
    'Glob',
    'IsDirectory',
    'ListDirectory',
    'Open',
    'MakeDirs',
    'MkDir',
    'Remove',
    'Rename',
    'Stat',
    'Walk',
]

remove_undocumented(__name__, _allowed_symbols)
