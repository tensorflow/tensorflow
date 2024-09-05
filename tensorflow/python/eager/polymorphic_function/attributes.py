# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""This file lists FunctionDef attributes and corresponding allowlists."""

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.util import compat

# IMPORTANT: The usage of all the attributes below should be considered tech
# debt and new additions to this list are discouraged.
#
# Historically, attributes have been used as means to pipe extra information
# down to runtime that is not related to the actual function definition itself.
#
# This information is better layered independently and future work is encouraged
# to pursue that direction instead.

API_IMPLEMENTS = "api_implements"
API_PREFERRED_DEVICE = "api_preferred_device"
BACKWARD_FUNCTION = "backward_function_name"
DISABLE_ACD = "_disable_acd"
DISABLE_CALL_SHAPE_INFERENCE = "_disable_call_shape_inference"
DISABLE_SUMMARIES_AT_RUNTIME = "disable_summaries_at_runtime"
EAGER_RUNTIME_CONSTRUCTION_CONTEXT = "_construction_context"
FORWARD_FUNCTION = "forward_function_name"
GO_BACKWARDS = "go_backwards"
IMPLEMENTS = "_implements"
INPUT_SHAPES = "_input_shapes"
INTS_ON_DEVICE = "experimental_ints_on_device"
NO_INLINE = "_noinline"
ORIGINAL_FUNCTION_NAME = "_original_func_name"
OUTPUTS_ON_OP_DEVICE = "_OutputsOnOpDevice"
QUANTIZED_COMPOSITE_FUNCTION = "tf_quant.composite_function"
QUANTIZED_OPS = "tf_quant.quantized_ops"
RUNTIME_CONSTANT_OPTIMIZATION = "runtime_constant_optimization"
SHARED_RENDEZVOUS = "shared_rendezvous"
TF_DATA_FUNCTION = "_tf_data_function"
TFTRT_ALLOW_BUILD_AT_RUNTIME = "_tftrt_allow_build_at_runtime"
TFTRT_CONVERT_FUNCTION = "_tftrt_convert_function"
TFTRT_IS_DYN_OP = "_tftrt_is_dyn_op"
TFTRT_LOGGER = "_tftrt_trt_logger_name"
TFTRT_MAX_BATCH_SIZE = "_tftrt_max_batch_size"
TFTRT_MAX_CACHED_ENGINES = "_tftrt_max_cached_engines"
TFTRT_MAX_WORKSPACE_SIZE = "_tftrt_max_workspace_size_bytes"
TFTRT_MIN_SEGMENT_SIZE = "_tftrt_minimum_segment_size"
TFTRT_PRECISION_MODE = "_tftrt_precision_mode"
TFTRT_PROFILE_STRATEGY = "_tftrt_profile_strategy"
TFTRT_USE_CALIBRATION = "_tftrt_use_calibration"
TFTRT_USE_IMPLICIT_BATCH = "_tftrt_use_implicit_batch"
TIME_MAJOR = "time_major"
XLA_COMPILE = "_XlaMustCompile"
XLA_COMPILE_OPTIONAL = "_XlaCompile"
XLA_SCOPE = "_XlaScope"
XLA_SEPERATE_COMPILED_GRADIENTS = "_XlaSeparateCompiledGradients"

POLYMORPHIC_FUNCTION_ALLOWLIST = frozenset({
    API_IMPLEMENTS,
    API_PREFERRED_DEVICE,
    DISABLE_ACD,
    DISABLE_SUMMARIES_AT_RUNTIME,
    GO_BACKWARDS,
    IMPLEMENTS,
    INTS_ON_DEVICE,
    NO_INLINE,
    RUNTIME_CONSTANT_OPTIMIZATION,
    TF_DATA_FUNCTION,
    TIME_MAJOR,
    OUTPUTS_ON_OP_DEVICE,
})

TRACING_COMPILATION_ALLOWLIST = frozenset().union(
    POLYMORPHIC_FUNCTION_ALLOWLIST,
    {
        SHARED_RENDEZVOUS,
        XLA_COMPILE,
    },
)

MONOMORPHIC_FUNCTION_ALLOWLIST = frozenset().union(
    TRACING_COMPILATION_ALLOWLIST,
    {
        BACKWARD_FUNCTION,
        DISABLE_CALL_SHAPE_INFERENCE,
        EAGER_RUNTIME_CONSTRUCTION_CONTEXT,
        FORWARD_FUNCTION,
        INPUT_SHAPES,
        ORIGINAL_FUNCTION_NAME,
        QUANTIZED_COMPOSITE_FUNCTION,
        QUANTIZED_OPS,
        TFTRT_ALLOW_BUILD_AT_RUNTIME,
        TFTRT_CONVERT_FUNCTION,
        TFTRT_IS_DYN_OP,
        TFTRT_LOGGER,
        TFTRT_MAX_BATCH_SIZE,
        TFTRT_MAX_CACHED_ENGINES,
        TFTRT_MAX_WORKSPACE_SIZE,
        TFTRT_MIN_SEGMENT_SIZE,
        TFTRT_PRECISION_MODE,
        TFTRT_PROFILE_STRATEGY,
        TFTRT_USE_CALIBRATION,
        TFTRT_USE_IMPLICIT_BATCH,
        XLA_COMPILE_OPTIONAL,
        XLA_SCOPE,
        XLA_SEPERATE_COMPILED_GRADIENTS,
    },
)


def _parse_func_attr_value(key, value):
  """Converts a python object to an attr_value_pb2.AttrValue object."""
  if isinstance(value, attr_value_pb2.AttrValue):
    return value
  # bool type check has to happen before int since bool is a subclass of int.
  elif isinstance(value, bool):
    return attr_value_pb2.AttrValue(b=value)
  elif isinstance(value, int):
    return attr_value_pb2.AttrValue(i=value)
  elif isinstance(value, float):
    return attr_value_pb2.AttrValue(f=value)
  elif isinstance(value, (str, bytes)):
    return attr_value_pb2.AttrValue(s=compat.as_bytes(value))
  elif isinstance(value, list):
    list_value = attr_value_pb2.AttrValue.ListValue()
    for v in value:
      if isinstance(v, bool):
        list_value.b.append(v)
      elif isinstance(v, int):
        list_value.i.append(v)
      elif isinstance(v, float):
        list_value.f.append(v)
      elif isinstance(v, (str, bytes)):
        list_value.s.append(compat.as_bytes(v))
      else:
        raise ValueError(
            f"Attributes for {key} must be bool, int, float, or string. "
            f"Got {type(v)}."
        )
    return attr_value_pb2.AttrValue(list=list_value)
  else:
    raise ValueError(
        f"Attribute {key} must be bool, int, float, string, list, or "
        f"AttrValue. Got {type(value)}."
    )


def parse_func_attrs(attributes, allowlist=None):
  """Convert the keyword arguments into function_def attributes.

  Currently only support primitive types: bool, int, float and string.

  Args:
    attributes: the dictionary of attributes.
    allowlist: set of attribute names allowed.
  Returns:
    A dict of attributes where the key is the name of attribute and the value
      is the AttrValue proto.
  Raises:
    ValueError: If the kwargs contains unallowlisted name or unsupported value
      types.
  """
  if not allowlist:
    allowlist = MONOMORPHIC_FUNCTION_ALLOWLIST

  attrs = {}
  for key, value in attributes.items():
    if key not in allowlist:
      raise ValueError(
          f"Allowlist does not support `{key}` as an attribute.")
    attrs[key] = _parse_func_attr_value(key, value)
  return attrs
