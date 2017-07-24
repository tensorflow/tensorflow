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

"""System configuration library.

@@get_include
@@get_lib
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as _os_path

from tensorflow.python.util.all_util import remove_undocumented


# pylint: disable=g-import-not-at-top
def get_include():
  """Get the directory containing the TensorFlow C++ header files.

  Returns:
    The directory as string.
  """
  # Import inside the function.
  # sysconfig is imported from the tensorflow module, so having this
  # import at the top would cause a circular import, resulting in
  # the tensorflow module missing symbols that come after sysconfig.
  import tensorflow as tf
  return _os_path.join(_os_path.dirname(tf.__file__), 'include')


def get_lib():
  """Get the directory containing the TensorFlow framework library.

  Returns:
    The directory as string.
  """
  import tensorflow as tf
  return _os_path.join(_os_path.dirname(tf.__file__), 'core')

_allowed_symbols = []
remove_undocumented(__name__, _allowed_symbols)
