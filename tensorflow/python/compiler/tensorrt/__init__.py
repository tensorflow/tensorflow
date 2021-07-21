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

# pylint: disable=unused-import,line-too-long
import os
import sys
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.platform import tf_logging as logging
# pylint: enable=unused-import,line-too-long


def _raise_trt_version_not_supported(version_type, trt_version):
  assert version_type in ["linked", "loaded"], (
    f"Incorrect value received for version_type: {version_type}. " \
    f"Accepted: ['linked', 'loaded']"
  )

  logging.error(
      f"The support for {version_type} version of TensorRT: "\
      f"`{trt_utils.version_tuple_to_string(trt_version)}` has now been " \
      f"removed. Please upgrade to TensorRT 7 or more recent.")

  raise RuntimeError(f"Incompatible {version_type} TensorRT versions")


if not _pywrap_py_utils.is_tensorrt_enabled():
  logging.error(
      "Tensorflow needs to be built with TensorRT support enabled to allow "
      "TF-TRT to operate.")

  raise RuntimeError("Tensorflow has not been built with TensorRT support.")


# Do not execute the following on a machine that is not linux-based and during
# bazel build execution.
if "bazel" not in os.environ["_"] and sys.platform == "linux":
  linked_version = _pywrap_py_utils.get_linked_tensorrt_version()
  loaded_version = _pywrap_py_utils.get_loaded_tensorrt_version()

  logging.info("Linked TensorRT version: %s", str(linked_version))
  logging.info("Loaded TensorRT version: %s", str(loaded_version))

  if not trt_utils.is_linked_tensorrt_version_greater_equal(7, 0, 0):
    _raise_trt_version_not_supported("linked", linked_version)

  if not trt_utils.is_loaded_tensorrt_version_greater_equal(7, 0, 0):
    _raise_trt_version_not_supported("loaded", loaded_version)

  if (loaded_version[0] != linked_version[0] or
      not trt_utils.is_loaded_tensorrt_version_greater_equal(*linked_version)):
    loaded_version_str = trt_utils.version_tuple_to_string(loaded_version)
    linked_version_str = trt_utils.version_tuple_to_string(linked_version)
    logging.error(
        f"Loaded TensorRT `{loaded_version_str}` but linked against " \
        f"TensorRT: `{linked_version_str}`. Not all requirements are met:\n" \
        f"\t- It is required to use the same major version of TensorRT " \
        f"during compilation and runtime.\n" \
        f"\t- TensorRT does not support forward compatibility. The loaded " \
        f"version has to be not later than the linked version."
    )
    raise RuntimeError("Incompatible TensorRT major version")

  elif loaded_version != linked_version:
    logging.info(
        "Loaded TensorRT %s and linked TensorFlow against TensorRT %s. This is "
        "supported because TensorRT minor/patch upgrades are backward "
        "compatible.", trt_utils.version_tuple_to_string(loaded_version),
        trt_utils.version_tuple_to_string(linked_version))
