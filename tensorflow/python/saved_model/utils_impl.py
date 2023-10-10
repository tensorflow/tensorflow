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

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


# TensorInfo helpers.
_DEPRECATION_MSG = (
    "This API was designed for TensorFlow v1. See "
    "https://www.tensorflow.org/guide/migrate for instructions on how to "
    "migrate your code to TensorFlow v2.")


@tf_export(
    v1=["saved_model.build_tensor_info", "saved_model.utils.build_tensor_info"])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def build_tensor_info(tensor):
  """Utility function to build TensorInfo proto from a Tensor.

  Args:
    tensor: Tensor or SparseTensor whose name, dtype and shape are used to
        build the TensorInfo. For SparseTensors, the names of the three
        constituent Tensors are used.

  Returns:
    A TensorInfo protocol buffer constructed based on the supplied argument.

  Raises:
    RuntimeError: If eager execution is enabled.

  @compatibility(TF2)
  This API is not compatible with eager execution as `tensor` needs to be a
  graph tensor, and there is no replacement for it in TensorFlow 2.x. To start
  writing programs using TensorFlow 2.x, please refer to the [Effective
  TensorFlow 2](https://www.tensorflow.org/guide/effective_tf2) guide.
  @end_compatibility
  """
  if context.executing_eagerly():
    raise RuntimeError("`build_tensor_info` is not supported in eager "
                       "execution.")
  return build_tensor_info_internal(tensor)


def build_tensor_info_internal(tensor):
  """Utility function to build TensorInfo proto from a Tensor."""
  if (isinstance(tensor, composite_tensor.CompositeTensor) and
      not isinstance(tensor, sparse_tensor.SparseTensor) and
      not isinstance(tensor, resource_variable_ops.ResourceVariable)):
    return _build_composite_tensor_info_internal(tensor)

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


def _build_composite_tensor_info_internal(tensor):
  """Utility function to build TensorInfo proto from a CompositeTensor."""
  spec = tensor._type_spec  # pylint: disable=protected-access
  tensor_info = meta_graph_pb2.TensorInfo()
  spec_proto = nested_structure_coder.encode_structure(spec)
  tensor_info.composite_tensor.type_spec.CopyFrom(spec_proto.type_spec_value)
  for component in nest.flatten(tensor, expand_composites=True):
    tensor_info.composite_tensor.components.add().CopyFrom(
        build_tensor_info_internal(component))
  return tensor_info


def build_tensor_info_from_op(op):
  """Utility function to build TensorInfo proto from an Op.

  Note that this function should be used with caution. It is strictly restricted
  to TensorFlow internal use-cases only. Please make sure you do need it before
  using it.

  This utility function overloads the TensorInfo proto by setting the name to
  the Op's name, dtype to DT_INVALID and tensor_shape as None. One typical usage
  is for the Op of the call site for the defunned function:
  ```python
    @function.defun
    def some_variable_initialization_fn(value_a, value_b):
      a = value_a
      b = value_b

    value_a = constant_op.constant(1, name="a")
    value_b = constant_op.constant(2, name="b")
    op_info = utils.build_op_info(
        some_variable_initialization_fn(value_a, value_b))
  ```

  Args:
    op: An Op whose name is used to build the TensorInfo. The name that points
        to the Op could be fetched at run time in the Loader session.

  Returns:
    A TensorInfo protocol buffer constructed based on the supplied argument.

  Raises:
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError(
        "`build_tensor_info_from_op` is not supported in eager execution.")
  return meta_graph_pb2.TensorInfo(
      dtype=types_pb2.DT_INVALID,
      tensor_shape=tensor_shape.unknown_shape().as_proto(),
      name=op.name)


@tf_export(v1=["saved_model.get_tensor_from_tensor_info",
               "saved_model.utils.get_tensor_from_tensor_info"])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def get_tensor_from_tensor_info(tensor_info, graph=None, import_scope=None):
  """Returns the Tensor or CompositeTensor described by a TensorInfo proto.

  Args:
    tensor_info: A TensorInfo proto describing a Tensor or SparseTensor or
      CompositeTensor.
    graph: The tf.Graph in which tensors are looked up. If None, the
        current default graph is used.
    import_scope: If not None, names in `tensor_info` are prefixed with this
        string before lookup.

  Returns:
    The Tensor or SparseTensor or CompositeTensor in `graph` described by
    `tensor_info`.

  Raises:
    KeyError: If `tensor_info` does not correspond to a tensor in `graph`.
    ValueError: If `tensor_info` is malformed.
  """
  graph = graph or ops.get_default_graph()
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
  elif encoding == "composite_tensor":
    spec_proto = struct_pb2.StructuredValue(
        type_spec_value=tensor_info.composite_tensor.type_spec)
    spec = nested_structure_coder.decode_proto(spec_proto)
    components = [_get_tensor(component.name) for component in
                  tensor_info.composite_tensor.components]
    return nest.pack_sequence_as(spec, components, expand_composites=True)
  else:
    raise ValueError(f"Invalid TensorInfo.encoding: {encoding}. Expected `"
                     "coo_sparse`, `composite_tensor`, or `name` for a dense "
                     "tensor.")


def get_element_from_tensor_info(tensor_info, graph=None, import_scope=None):
  """Returns the element in the graph described by a TensorInfo proto.

  Args:
    tensor_info: A TensorInfo proto describing an Op or Tensor by name.
    graph: The tf.Graph in which tensors are looked up. If None, the current
      default graph is used.
    import_scope: If not None, names in `tensor_info` are prefixed with this
      string before lookup.

  Returns:
    Op or tensor in `graph` described by `tensor_info`.

  Raises:
    KeyError: If `tensor_info` does not correspond to an op or tensor in `graph`
  """
  graph = graph or ops.get_default_graph()
  return graph.as_graph_element(
      ops.prepend_name_scope(tensor_info.name, import_scope=import_scope))


def swap_function_tensor_content(meta_graph_def, from_endiness, to_endiness):
  bst.swap_tensor_content_in_graph_function(
      meta_graph_def, from_endiness, to_endiness
  )
