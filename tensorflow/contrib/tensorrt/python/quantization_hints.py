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
"""Exposes the scope for NVIDIA optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op as cop
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import attr_value_pb2


def numerical_hint(op, max_value, min_value):
  """Add a const numerical hint node"""
  graph = ops.get_default_graph()
  if isinstance(op, ops.Tensor):
    top = op.op
  else:
    top = op
  const_name = graph.unique_name("%s_numerical_hint" % top.name, False)
  const_op = cop.constant(
      [min_value, max_value], dtype=dtypes.float32, name=const_name)
  # pylint: disable=protected-access
  top._add_control_input(const_op.op)
  top._set_attr(
      "_NvidiaQuantizationHint",
      attr_value_pb2.AttrValue(s=const_name.encode()))
  # pylint: enable=protected-access
  print("const op=", const_op.op, "quantized_op=", top)
  return op
