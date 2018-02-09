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
"""A wrapper for TensorFlow SWIG-generated bindings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import sys
import traceback

from tensorflow.python.platform import self_check


# Perform pre-load sanity checks in order to produce a more actionable error
# than we get from an error during SWIG import.
self_check.preload_check()

# pylint: disable=wildcard-import,g-import-not-at-top,unused-import,line-too-long

try:
  # This import is expected to fail if there is an explicit shared object
  # dependency (with_framework_lib=true), since we do not need RTLD_GLOBAL.
  from tensorflow.python import pywrap_dlopen_global_flags
  _use_dlopen_global_flags = True
except ImportError:
  _use_dlopen_global_flags = False

# On UNIX-based platforms, pywrap_tensorflow is a SWIG-generated
# python library that dynamically loads _pywrap_tensorflow.so.
_can_set_rtld_local = (hasattr(sys, 'getdlopenflags')
                       and hasattr(sys, 'setdlopenflags'))
if _can_set_rtld_local:
  _default_dlopen_flags = sys.getdlopenflags()

try:
  if _use_dlopen_global_flags:
    pywrap_dlopen_global_flags.set_dlopen_flags()
  elif _can_set_rtld_local:
    # Ensure RTLD_LOCAL behavior for platforms where it isn't the default
    # (macOS). On Linux RTLD_LOCAL is 0, so this does nothing (and would not
    # override an RTLD_GLOBAL in _default_dlopen_flags).
    sys.setdlopenflags(_default_dlopen_flags | ctypes.RTLD_LOCAL)

  from tensorflow.python.pywrap_tensorflow_internal import *
  from tensorflow.python.pywrap_tensorflow_internal import __version__
  from tensorflow.python.pywrap_tensorflow_internal import __git_version__
  from tensorflow.python.pywrap_tensorflow_internal import __compiler_version__
  from tensorflow.python.pywrap_tensorflow_internal import __cxx11_abi_flag__
  from tensorflow.python.pywrap_tensorflow_internal import __monolithic_build__

  if _use_dlopen_global_flags:
    pywrap_dlopen_global_flags.reset_dlopen_flags()
  elif _can_set_rtld_local:
    sys.setdlopenflags(_default_dlopen_flags)
except ImportError:
  msg = """%s\n\nFailed to load the native TensorFlow runtime.\n
See https://www.tensorflow.org/install/install_sources#common_installation_problems\n
for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.""" % traceback.format_exc()
  raise ImportError(msg)

# pylint: enable=wildcard-import,g-import-not-at-top,unused-import,line-too-long
