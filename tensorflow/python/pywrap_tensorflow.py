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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Python2.7 does not have a ModuleNotFoundError.
try:
  ModuleNotFoundError
except NameError:
  ModuleNotFoundError = ImportError

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
