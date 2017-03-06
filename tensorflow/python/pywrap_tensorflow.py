# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""pywrap_tensorflow wrapper that exports all symbols with RTLD_GLOBAL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import sys
import traceback

# pylint: disable=wildcard-import,g-import-not-at-top,unused-import,line-too-long

# On UNIX-based platforms, pywrap_tensorflow is a SWIG-generated
# python library that dynamically loads _pywrap_tensorflow.so. The
# default mode for loading keeps all the symbol private and not
# visible to other libraries that may be loaded. Setting the mode to
# RTLD_GLOBAL to make the symbols visible, so that custom op libraries
# imported using `tf.load_op_library()` can access symbols defined in
# _pywrap_tensorflow.so.
try:
  # TODO(keveman,mrry): Support dynamic op loading on platforms that do not
  # use `dlopen()` for dynamic loading.
  _use_rtld_global = hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags')
  if _use_rtld_global:
    _default_dlopen_flags = sys.getdlopenflags()
    sys.setdlopenflags(_default_dlopen_flags | ctypes.RTLD_GLOBAL)
  from tensorflow.python.pywrap_tensorflow_internal import *
  from tensorflow.python.pywrap_tensorflow_internal import __version__
  from tensorflow.python.pywrap_tensorflow_internal import __git_version__
  from tensorflow.python.pywrap_tensorflow_internal import __compiler_version__
  if _use_rtld_global:
    sys.setdlopenflags(_default_dlopen_flags)
except ImportError:
  msg = """%s\n\nFailed to load the native TensorFlow runtime.\n
See https://www.tensorflow.org/install/install_sources#common_installation_problems\n
for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.""" % traceback.format_exc()
  raise ImportError(msg)

# pylint: enable=wildcard-import,g-import-not-at-top,unused-import,line-too-long
