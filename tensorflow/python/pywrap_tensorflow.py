# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""A Python wrapper that loads _pywrap_tensorflow_internal.so."""

import ctypes
import sys
import traceback

# TODO(vam): Override boringssl static linking by explicitly importing openssl
# Please refer to b/304394983 for more information.
try:
  import ssl
except ImportError:
  print(f'{traceback.format_exc()}')
  print('\nWarning: Failed to load ssl module. Continuing without ssl support.')

from tensorflow.python.platform import self_check

# TODO(mdan): Cleanup antipattern: import for side effects.

# Perform pre-load sanity checks in order to produce a more actionable error.
self_check.preload_check()

# pylint: disable=wildcard-import,g-import-not-at-top,unused-import,line-too-long

try:
  # This import is expected to fail if there is an explicit shared object
  # dependency (with_framework_lib=true), since we do not need RTLD_GLOBAL.
  from tensorflow.python import pywrap_dlopen_global_flags
  _use_dlopen_global_flags = True
except ImportError:
  _use_dlopen_global_flags = False

# On UNIX-based platforms, pywrap_tensorflow is a python library that
# dynamically loads _pywrap_tensorflow.so.
_can_set_rtld_local = (
    hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'))
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

  # Python2.7 does not have a ModuleNotFoundError.
  try:
    ModuleNotFoundError
  except NameError:
    ModuleNotFoundError = ImportError  # pylint: disable=redefined-builtin

  # pylint: disable=wildcard-import,g-import-not-at-top,line-too-long,undefined-variable
  try:
    from tensorflow.python._pywrap_tensorflow_internal import *
  # This try catch logic is because there is no bazel equivalent for py_extension.
  # Externally in opensource we must enable exceptions to load the shared object
  # by exposing the PyInit symbols with pybind. This error will only be
  # caught internally or if someone changes the name of the target _pywrap_tensorflow_internal.

  # This logic is used in other internal projects using py_extension.
  except ModuleNotFoundError:
    pass

  if _use_dlopen_global_flags:
    pywrap_dlopen_global_flags.reset_dlopen_flags()
  elif _can_set_rtld_local:
    sys.setdlopenflags(_default_dlopen_flags)
except ImportError:
  raise ImportError(
      f'{traceback.format_exc()}'
      f'\n\nFailed to load the native TensorFlow runtime.\n'
      f'See https://www.tensorflow.org/install/errors '
      f'for some common causes and solutions.\n'
      f'If you need help, create an issue '
      f'at https://github.com/tensorflow/tensorflow/issues '
      f'and include the entire stack trace above this error message.')

# pylint: enable=wildcard-import,g-import-not-at-top,unused-import,line-too-long
