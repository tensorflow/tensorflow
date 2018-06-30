# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""SavedModel utility functions implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.util.tf_export import tf_export


# TensorInfo helpers.


@tf_export("saved_model.utils.build_tensor_info")
def build_tensor_info(tensor):
  """Utility function to build TensorInfo proto.

  Args:
    tensor: Tensor or SparseTensor whose name, dtype and shape are used to
        build the TensorInfo. For SparseTensors, the names of the three
        constitutent Tensors are used.

  Returns:
    A TensorInfo protocol buffer constructed based on the supplied argument.
  """
  tensor_info = meta_graph_pb2.TensorInfo(
      dtype=dtypes.as_dtype(tensor.dtype).as_datatype_enum,
      tensor_shape=tensor.get_shape().as_proto())
  if isinstance(tensor, sparse_tensor.SparseTensor):
    tensor_info.coo_sparse.values_tensor_name = tensor.values.name
    tensor_info.coo_sparse.indices_tensor_name = tensor.indices.name
    tensor_info.coo_sparse.dense_shape_tensor_name = tensor.dense_shape.name
  else:
    tensor_info.name = tensor.name
  return tensor_info


@tf_export("saved_model.utils.get_tensor_from_tensor_info")
def get_tensor_from_tensor_info(tensor_info, graph=None, import_scope=None):
  """Returns the Tensor or SparseTensor described by a TensorInfo proto.

  Args:
    tensor_info: A TensorInfo proto describing a Tensor or SparseTensor.
    graph: The tf.Graph in which tensors are looked up. If None, the
        current default graph is used.
    import_scope: If not None, names in `tensor_info` are prefixed with this
        string before lookup.

  Returns:
    The Tensor or SparseTensor in `graph` described by `tensor_info`.

  Raises:
    KeyError: If `tensor_info` does not correspond to a tensor in `graph`.
    ValueError: If `tensor_info` is malformed.
  """
  graph = graph if graph is not None else ops.get_default_graph()
  def _get_tensor(name):
    return graph.get_tensor_by_name(
        ops.prepend_name_scope(name, import_scope=import_scope))
  encoding = tensor_info.WhichOneof("encoding")
  if encoding == "name":
    return _get_tensor(tensor_info.name)
  elif encoding == "coo_sparse":
    return sparse_tensor.SparseTensor(
        _get_tensor(tensor_info.coo_sparse.indices_tensor_name),
        _get_tensor(tensor_info.coo_sparse.values_tensor_name),
        _get_tensor(tensor_info.coo_sparse.dense_shape_tensor_name))
  else:
    raise ValueError("Invalid TensorInfo.encoding: %s" % encoding)
