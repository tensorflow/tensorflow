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
# ==============================================================================
"""Graph-only versions of a few op functions, for internal use only."""

# Must be separate from array_ops to avoid a cyclic dependency.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


def graph_placeholder(dtype, shape, name=None):
  """Graph-only version of tf.compat.v1.placeholder(), for internal use only."""
  dtype = dtype.base_dtype
  dtype_value = attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)
  if isinstance(shape, (list, tuple)):
    shape = tensor_shape.TensorShape(shape)
  shape = attr_value_pb2.AttrValue(shape=shape.as_proto())
  g = ops.get_default_graph()
  op = g._create_op_internal(  # pylint: disable=protected-access
      "Placeholder", [], [dtype], input_types=[],
      attrs={"dtype": dtype_value, "shape": shape}, name=name)
  result, = op.outputs
  return result
