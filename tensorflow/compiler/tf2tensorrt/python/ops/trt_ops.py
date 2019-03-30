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
"""Exposes the Python wrapper of TRTEngineOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

import platform
from tensorflow.compiler.tf2tensorrt.wrap_py_utils import is_tensorrt_enabled
from tensorflow.python.framework import errors

_tf_trt_so = None
_module_lock = threading.Lock()


def load_trt_ops():
  """Load TF-TRT op libraries so if it hasn't been loaded already."""
  global _tf_trt_so

  if not is_tensorrt_enabled():
    return

  if platform.system() == "Windows":
    raise RuntimeError("Windows platforms are not supported")

  with _module_lock:
    if _tf_trt_so:
      return

    try:
      # pylint: disable=g-import-not-at-top,unused-variable
      # This will call register_op_list() in
      # tensorflow/python/framework/op_def_registry.py, but it doesn't register
      # the op or the op kernel in C++ runtime.
      from tensorflow.compiler.tf2tensorrt.ops.gen_trt_ops import trt_engine_op
      # pylint: enable=g-import-not-at-top,unused-variable
    except ImportError as e:
      print("**** Failed to import TF-TRT ops. This is because the binary was "
            "not built with CUDA or TensorRT enabled. ****")
      raise e

    try:
      # pylint: disable=g-import-not-at-top
      from tensorflow.python.framework import load_library
      from tensorflow.python.platform import resource_loader
      # pylint: enable=g-import-not-at-top

      # Loading the shared object will cause registration of the op and the op
      # kernel if we link TF-TRT dynamically.
      _tf_trt_so = load_library.load_op_library(
          resource_loader.get_path_to_datafile("libtftrt.so"))
    except errors.NotFoundError as e:
      no_trt_message = (
          "**** Failed to initialize TensorRT. This is either because the "
          "TensorRT installation path is not in LD_LIBRARY_PATH, or because "
          "you do not have it installed. If not installed, please go to "
          "https://developer.nvidia.com/tensorrt to download and install "
          "TensorRT ****")
      print(no_trt_message)
      raise e
