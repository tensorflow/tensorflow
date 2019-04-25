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
"""Operations to execute a subgraph on a remote processor."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import, line-too-long
from tensorflow.contrib.remote_fused_graph.pylib.python.ops import gen_remote_fused_graph_ops
from tensorflow.core.framework import remote_fused_graph_execute_info_pb2 as info_pb2
# pylint: enable=unused-import,wildcard-import,line-too-long

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

# RemoteFusedGraphExecute is not differenciable op.
ops.NotDifferentiable("RemoteFusedGraphExecute")


def remote_fused_graph_execute(inputs,
                               output_types,
                               graph_def,
                               graph_input_node_names,
                               graph_output_node_names,
                               executor_name,
                               serialized_executor_parameters,
                               default_graph_input_tensor_type_shapes=None,
                               default_graph_output_tensor_type_shapes=None):
  """A wrapper for remote_fused_graph_execute."""
  info_proto = info_pb2.RemoteFusedGraphExecuteInfo()
  info_proto.remote_graph.CopyFrom(graph_def)
  info_proto.graph_input_node_name.extend(graph_input_node_names)
  info_proto.graph_output_node_name.extend(graph_output_node_names)
  info_proto.executor_name = executor_name
  info_proto.serialized_executor_parameters = serialized_executor_parameters
  if default_graph_input_tensor_type_shapes:
    for type_shape in default_graph_input_tensor_type_shapes:
      type_shape_proto = info_proto.default_graph_input_tensor_shape.add()
      type_shape_proto.dtype = dtypes.as_dtype(type_shape[0]).as_datatype_enum
      for dim in type_shape[1]:
        type_shape_proto.shape.dim.add().size = dim
  if default_graph_output_tensor_type_shapes:
    for type_shape in default_graph_output_tensor_type_shapes:
      type_shape_proto = info_proto.default_graph_output_tensor_shape.add()
      type_shape_proto.dtype = dtypes.as_dtype(type_shape[0]).as_datatype_enum
      for dim in type_shape[1]:
        type_shape_proto.shape.dim.add().size = dim

  serialized_info = info_proto.SerializeToString()

  return gen_remote_fused_graph_ops.remote_fused_graph_execute(
      inputs, output_types, serialized_info)
