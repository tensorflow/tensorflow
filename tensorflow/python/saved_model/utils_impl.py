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

import os

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# TensorInfo helpers.


@tf_export(v1=["saved_model.build_tensor_info",
               "saved_model.utils.build_tensor_info"])
@deprecation.deprecated(
    None,
    "This function will only be available through the v1 compatibility "
    "library as tf.compat.v1.saved_model.utils.build_tensor_info or "
    "tf.compat.v1.saved_model.build_tensor_info.")
def build_tensor_info(tensor):
  """Utility function to build TensorInfo proto from a Tensor.

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
    def some_vairable_initialiation_fn(value_a, value_b):
      a = value_a
      b = value_b

    value_a = constant_op.constant(1, name="a")
    value_b = constant_op.constant(2, name="b")
    op_info = utils.build_op_info(
        some_vairable_initialiation_fn(value_a, value_b))
  ```

  Args:
    op: An Op whose name is used to build the TensorInfo. The name that points
        to the Op could be fetched at run time in the Loader session.

  Returns:
    A TensorInfo protocol buffer constructed based on the supplied argument.
  """
  return meta_graph_pb2.TensorInfo(
      dtype=types_pb2.DT_INVALID,
      tensor_shape=tensor_shape.unknown_shape().as_proto(),
      name=op.name)


@tf_export(v1=["saved_model.get_tensor_from_tensor_info",
               "saved_model.utils.get_tensor_from_tensor_info"])
@deprecation.deprecated(
    None,
    "This function will only be available through the v1 compatibility "
    "library as tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info or "
    "tf.compat.v1.saved_model.get_tensor_from_tensor_info.")
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
  else:
    raise ValueError("Invalid TensorInfo.encoding: %s" % encoding)


# Path helpers.


def get_or_create_variables_dir(export_dir):
  """Return variables sub-directory, or create one if it doesn't exist."""
  variables_dir = get_variables_dir(export_dir)
  if not file_io.file_exists(variables_dir):
    file_io.recursive_create_dir(variables_dir)
  return variables_dir


def get_variables_dir(export_dir):
  """Return variables sub-directory in the SavedModel."""
  return os.path.join(
      compat.as_text(export_dir),
      compat.as_text(constants.VARIABLES_DIRECTORY))


def get_variables_path(export_dir):
  """Return the variables path, used as the prefix for checkpoint files."""
  return os.path.join(
      compat.as_text(get_variables_dir(export_dir)),
      compat.as_text(constants.VARIABLES_FILENAME))


def get_or_create_assets_dir(export_dir):
  """Return assets sub-directory, or create one if it doesn't exist."""
  assets_destination_dir = get_assets_dir(export_dir)

  if not file_io.file_exists(assets_destination_dir):
    file_io.recursive_create_dir(assets_destination_dir)

  return assets_destination_dir


def get_assets_dir(export_dir):
  """Return path to asset directory in the SavedModel."""
  return os.path.join(
      compat.as_text(export_dir),
      compat.as_text(constants.ASSETS_DIRECTORY))
