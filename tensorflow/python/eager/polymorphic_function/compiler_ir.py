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
"""Implmentation for defining get_compiler_ir."""
from tensorflow.core.function import trace_type
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest


def maybe_get_device_name(device_name):
  # TODO(cheshire): This is a hack to get the current "preferred" device,
  # there is no current API to get it otherwise.
  if device_name is None:
    device_name = random_ops.random_normal([]).device
  return device_name


def make_handledata_tensor_specs(resource_vars):
  """Convert tf.Variable list to its corresponding TensorSpec list."""
  inner_context = trace_type.InternalTracingContext()
  trace_type_inputs = trace_type.from_value(
      tuple(resource_vars), inner_context
  ).components
  handledata_mapping = inner_context.get_handledata_mapping()

  def to_spec(traced_input):
    try:
      my_id = id(traced_input)
      handled_data = handledata_mapping[my_id]
      shape_and_type = handled_data.shape_and_type[0]
      spec = tensor_spec.TensorSpec(
          shape=shape_and_type.shape, dtype=shape_and_type.dtype
      )
      return spec
    except Exception as e:
      raise ValueError(
          "Fail to convert tf.Variable list to TensorSpec list. The error"
          " is: %s" % e
      ) from e

  return [to_spec(trace_type) for trace_type in trace_type_inputs]


def from_concrete_function(concrete_fn):
  """Generate the Compiler Ir from tf concrete function.

  Args:
    concrete_fn: returned by using get_concrete_function.

  Returns:
    Function callable that generate the HLO text.

  Raises:
      ValueError: if concrete_fn is not "compilable" without concrete
      inputs.
  """
  context.ensure_initialized()
  # TODO(b/265073174) support users input tf.TensorSpec list here.
  if not all(
      [s.shape.is_fully_defined() for s in nest.flatten(concrete_fn.inputs)]
  ):
    raise ValueError(
        f"Only support static input shape but got inputs = {concrete_fn.inputs}"
    )
  fn_name = concrete_fn.name

  def compiler_ir_generator(stage="hlo", device_name=None):
    device_name = maybe_get_device_name(device_name)
    res_bytes = context.context().get_compiler_ir(
        device_name=device_name,
        stage=stage,
        function_name=fn_name,
        # args list is empty for using_tensor_spec case
        args=[],
    )
    if stage in (
        "hlo_serialized",
        "optimized_hlo_serialized",
        "optimized_hlo_proto_serialized",
    ):
      return res_bytes
    else:
      return res_bytes.decode("utf-8")

  return compiler_ir_generator
