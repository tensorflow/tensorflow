# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for loading op libraries.

@@load_op_library
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


def load_op_library(path):
  """Loads a contrib op library from the given path.

  NOTE(mrry): On Windows, we currently assume that some contrib op
  libraries are statically linked into the main TensorFlow Python
  extension DLL - use dynamically linked ops if the .so is present.

  Args:
    path: An absolute path to a shared object file.

  Returns:
    A Python module containing the Python wrappers for Ops defined in the
    plugin.
  """
  if os.name == 'nt':
    # To avoid making every user_ops aware of windows, re-write
    # the file extension from .so to .dll if .so file doesn't exist.
    if not os.path.exists(path):
      path = re.sub(r'\.so$', '.dll', path)

    # Currently we have only some user_ops as dlls on windows - don't try
    # to load them if the dll is not found.
    # TODO(mrry): Once we have all of them this check should be removed.
    if not os.path.exists(path):
      return None
  path = resource_loader.get_path_to_datafile(path)
  ret = load_library.load_op_library(path)
  assert ret, 'Could not load %s' % path
  return ret
