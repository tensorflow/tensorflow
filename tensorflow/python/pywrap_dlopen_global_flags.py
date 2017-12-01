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

Note that this file is only imported by pywrap_tensorflow.py if this is a static
build (meaning there is no explicit framework cc_binary shared object dependency
of _pywrap_tensorflow_internal.so). For regular (non-static) builds, RTLD_GLOBAL
is not necessary, since the dynamic dependencies of custom/contrib ops are
explicit.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import sys

# On UNIX-based platforms, pywrap_tensorflow is a SWIG-generated
# python library that dynamically loads _pywrap_tensorflow.so. The
# default mode for loading keeps all the symbol private and not
# visible to other libraries that may be loaded. Setting the mode to
# RTLD_GLOBAL to make the symbols visible, so that custom op libraries
# imported using `tf.load_op_library()` can access symbols defined in
# _pywrap_tensorflow.so.
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
