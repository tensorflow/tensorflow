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
"""If possible, exports all symbols with RTLD_GLOBAL.

This module is packaged only for monolithic builds (framework_shared_object
unset/false, e.g. --config=monolithic). Shared-object builds (default pip
wheels) omit it on purpose so pywrap_tensorflow.py hits ImportError and loads
with RTLD_LOCAL instead.

RTLD_GLOBAL is required for monolithic Python loads so custom op shared objects
can resolve TF symbols from the global table. It must NOT be used for shared
wheels: TF links LLVM/MLIR, and leaking those symbols globally causes ABI
clashes (SIGSEGV) when a later import (e.g. triton) brings its own LLVM. See
tensorflow/tensorflow#124205.
"""

import ctypes
import sys

# On UNIX-based platforms, pywrap_tensorflow is a SWIG-generated python library
# that dynamically loads _pywrap_tensorflow.so. The default mode for loading
# keeps all the symbol private and not visible to other libraries that may be
# loaded. Setting the mode to RTLD_GLOBAL to make the symbols visible, so that
# custom op libraries imported using `tf.load_op_library()` can access symbols
# defined in _pywrap_tensorflow.so.
_use_rtld_global = (hasattr(sys, 'getdlopenflags')
                    and hasattr(sys, 'setdlopenflags'))
if _use_rtld_global:
  _default_dlopen_flags = sys.getdlopenflags()


def set_dlopen_flags():
  if _use_rtld_global:
    sys.setdlopenflags(_default_dlopen_flags | ctypes.RTLD_GLOBAL)


def reset_dlopen_flags():
  if _use_rtld_global:
    sys.setdlopenflags(_default_dlopen_flags)
