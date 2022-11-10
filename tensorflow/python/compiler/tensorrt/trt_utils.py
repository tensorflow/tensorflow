# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Exposes the Python wrapper conversion to trt_graph."""

import os
import platform

from enum import Enum

from tensorflow.python.compiler.tensorrt import _pywrap_py_utils
from tensorflow.python.compiler.tensorrt import gen_trt_ops
from tensorflow.python.compiler.tensorrt.constants import TrtVersionEnv
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging


def cast_to_bytes(s):
  """Encode s if it is a sequence of chars."""
  if isinstance(s, Enum):
    s = s.value
  if isinstance(s, str):
    return s.encode("utf-8", errors="surrogateescape")
  return s


def convert_to_tensor(inp):
  try:
    if isinstance(inp, dict):
      args = []
      kwargs = {k: ops.convert_to_tensor(v) for k, v in inp.items()}
    else:
      kwargs = {}
      if isinstance(inp, (list, tuple)):
        args = map(ops.convert_to_tensor, inp)
      else:
        args = [ops.convert_to_tensor(inp)]
  except:
    error_msg = "Failed to convert input to tensor."
    logging.error(f"{error_msg}\ninp = `{inp}`\n")
    raise RuntimeError(error_msg)

  return args, kwargs


# Remove all scope prefixes in the node name. In TF 2.0, the same concrete
# function can be initialized multiple times with different prefixes, and
# this will result in the same TRTEngineOp being initialized multiple times
# with different cache and duplicate TRT engines.
# TODO(laigd): this may be caused by the fact that TRTEngineOp is not
# stateful, need to investigate.
# TODO(laigd): we rely on the fact that all functions are fully inlined
# before TF-TRT optimizer is called, as otherwise it may generate the same
# name when optimizing a different function graph. Fix this.
def get_canonical_engine_name(name):
  return name.split("/")[-1]


def is_experimental_feature_activated(feature_name):
  """Determines if a TF-TRT experimental feature is enabled.

  This helper function checks if an experimental feature was enabled using
  the environment variable `TF_TRT_EXPERIMENTAL_FEATURES=feature_1,feature_2`.

  Args:
    feature_name: Name of the feature being tested for activation.
  """

  return (feature_name
          in os.environ.get("TF_TRT_EXPERIMENTAL_FEATURES",
                            default="").split(","))


def print_row(fields, positions, print_fn):
  """Prints a row."""
  line = ""
  for i, field in enumerate(fields):
    field = str(field)
    end_line_pos = positions[i]
    if i > 0:
      line = line + " "
    line = "{0:{min_length}}".format(line + field, min_length=end_line_pos)

    if len(line) > end_line_pos:
      line = line[:(end_line_pos - 4)] + " ..."

  print_fn(line)


def save_calibration_table(node):
  try:
    calibration_table = gen_trt_ops.get_calibration_data_op(
        get_canonical_engine_name(node.name))
    node.attr["calibration_data"].s = calibration_table.numpy()
  except (errors.UnknownError, errors.NotFoundError):
    logging.warning(f"Warning calibration error for {node.name}", )


def validate_environment():
  """Check compatibility of TensorRT version.

  Raises:
    RuntimeError: if the TensorRT library version is incompatible.
  """

  if not _pywrap_py_utils.is_tensorrt_enabled():
    logging.error(
        "Tensorflow needs to be built with TensorRT support enabled to allow "
        "TF-TRT to operate.")

    raise RuntimeError("Tensorflow has not been built with TensorRT support.")

  if platform.system() == "Windows":
    logging.warn(
        "Windows support is provided experimentally. No guarantee is made "
        "regarding functionality or engineering support. Use at your own risk.")

  # Force loading Lazy Objects
  TrtVersionEnv.load()
