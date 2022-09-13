# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Exposes the python wrapper for TensorRT graph transforms."""

from tensorflow.compiler.tf2tensorrt.ops import gen_trt_ops
from tensorflow.python.util.lazy_loader import LazyLoader

# Lazily load the op, since it's not available in cpu-only builds. Importing
# this at top will cause tests that imports TF-TRT fail when they're built
# and run without CUDA/GPU.
_pywrap_py_utils = LazyLoader(
    "_pywrap_py_utils", globals(),
    "tensorflow.compiler.tf2tensorrt._pywrap_py_utils")

_pywrap_trt_convert = LazyLoader(
    "_pywrap_trt_convert", globals(),
    "tensorflow.python.compiler.tensorrt._pywrap_trt_convert")

# Register TRT ops in python, so that when users import this module they can
# execute a TRT-converted graph without calling any of the methods in this
# module.
#
# This will call register_op_list() in
# tensorflow/python/framework/op_def_registry.py, but it doesn't register
# the op or the op kernel in C++ runtime.
try:
  gen_trt_ops.trt_engine_op  # pylint: disable=pointless-statement
except AttributeError:
  pass

from tensorflow.python.compiler.tensorrt import trt_convert as trt
