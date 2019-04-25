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

"""System configuration library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as _os_path
import platform as _platform

from tensorflow.python.framework.versions import CXX11_ABI_FLAG as _CXX11_ABI_FLAG
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as _MONOLITHIC_BUILD
from tensorflow.python.framework.versions import VERSION as _VERSION
from tensorflow.python.util.tf_export import tf_export


# pylint: disable=g-import-not-at-top
@tf_export('sysconfig.get_include')
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


@tf_export('sysconfig.get_lib')
def get_lib():
  """Get the directory containing the TensorFlow framework library.

  Returns:
    The directory as string.
  """
  import tensorflow as tf
  return _os_path.join(_os_path.dirname(tf.__file__))


@tf_export('sysconfig.get_compile_flags')
def get_compile_flags():
  """Get the compilation flags for custom operators.

  Returns:
    The compilation flags.
  """
  flags = []
  flags.append('-I%s' % get_include())
  flags.append('-D_GLIBCXX_USE_CXX11_ABI=%d' % _CXX11_ABI_FLAG)
  return flags


@tf_export('sysconfig.get_link_flags')
def get_link_flags():
  """Get the link flags for custom operators.

  Returns:
    The link flags.
  """
  is_mac = _platform.system() == 'Darwin'
  ver = _VERSION.split('.')[0]
  flags = []
  if not _MONOLITHIC_BUILD:
    flags.append('-L%s' % get_lib())
    if is_mac:
      flags.append('-l:libtensorflow_framework.%s.dylib' % ver)
    else:
      flags.append('-l:libtensorflow_framework.so.%s' % ver)
  return flags
