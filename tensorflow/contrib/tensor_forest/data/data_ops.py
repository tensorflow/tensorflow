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
"""Ops for preprocessing data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import threading

from tensorflow.contrib.learn.python.learn.learn_io import graph_io
from tensorflow.contrib.tensor_forest.python import constants

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging

DATA_OPS_FILE = '_data_ops.so'

_data_ops = None
_ops_lock = threading.Lock()

ops.NotDifferentiable('SparseValuesToIndices')
ops.NotDifferentiable('StringToFloat')


ops.RegisterShape('SparseValuesToIndices')(common_shapes.call_cpp_shape_fn)
ops.RegisterShape('StringToFloat')(common_shapes.call_cpp_shape_fn)


# Workaround for the fact that importing tensorflow imports contrib
# (even if a user isn't using this or any other contrib op), but
# there's not yet any guarantee that the shared object exists.
# In which case, "import tensorflow" will always crash, even for users that
# never use contrib.
def Load():
  """Load the data ops library and return the loaded module."""
  with _ops_lock:
    global _data_ops
    if not _data_ops:
      ops_path = resource_loader.get_path_to_datafile(DATA_OPS_FILE)
      logging.info('data path: %s', ops_path)
      _data_ops = load_library.load_op_library(ops_path)

      assert _data_ops, 'Could not load _data_ops.so'
  return _data_ops


def _ParseSparse(data):
  """Concat sparse tensors together.

  A common use of sparse tensors is to treat strings as a sparse bit vector
  with a large number of features representing the presence of all possible
  values.  Here we convert these strings to integer indices in a sparse bit
  tensor.  In order to pack each incoming feature into a single sparse tensor,
  we add an offset to the converted indices to indicate that they came from
  different features in the source data.

  Args:
    data: A dict of name -> Tensor.

  Returns:
    A single sparse tensor with float values and a 1-D input spec Tensor.

  Raises:
    NotImplementedError:  Combining dense and sparse tensors is not yet
      supported.
    ValueError: If data contains non-string Tensors.
  """
  convert_ops = Load()

  # TODO(gilberth): Support mixed string/float sparse tensors.
  # We currently only support string (categorical) data if we're using sparse
  # tensors.
  for v in data.values():
    if v.dtype != dtypes.string:
      raise ValueError('Only sparse tensors of type string are supported.')

  # Sparse tensor indices have 63 bits to use for information. We use the
  # minimum number of these (MSBs) for the offset, and pack the rest with the
  # actual data.
  num_features = len(data)
  offset_bits = int(math.ceil(math.log(num_features, 2)))

  # We condense data to 26 bits, see sparse_values_to_indices.cc
  offset_increment = int(math.pow(2, 26 - offset_bits))
  offset = 0

  sparse_tensors = []
  keys = None
  for k in sorted(data.keys()):
    if k == graph_io.KEY_FEATURE_NAME:
      keys = data[k]
    elif isinstance(data[k], ops.SparseTensor):
      sparse_indices = data[k].indices
      sparse_values = data[k].values
      new_shape = array_ops.concat(
          0, [array_ops.slice(data[k].shape, [0], [1]), [offset_increment]])

      new_indices, new_values = convert_ops.sparse_values_to_indices(
          sparse_indices,
          sparse_values,
          offset, offset_bits=offset_bits)
    else:
      # Convert dense to sparse.
      raise NotImplementedError('Dense to sparse conversion not implemented.')

    sparse_tensors.append(ops.SparseTensor(indices=new_indices,
                                           values=new_values,
                                           shape=new_shape))

  return (sparse_ops.sparse_concat(1, sparse_tensors), keys,
          [constants.DATA_CATEGORICAL])


def _ParseDense(data):
  """Return a single flat tensor, keys, and a data spec list.

  Args:
    data: A dict mapping feature names to Tensors.

  Returns:
    A tuple of (single dense float Tensor, keys tensor (if exists), data spec).
  """
  convert_ops = Load()
  data_spec = [constants.DATA_CATEGORICAL if data[k].dtype == dtypes.string else
               constants.DATA_FLOAT for k in sorted(data.keys())]
  data_spec = [constants.DATA_FLOAT] + data_spec
  keys = None
  features = []
  for k in sorted(data.keys()):
    if k == graph_io.KEY_FEATURE_NAME:
      keys = data[k]
    else:
      features.append(
          convert_ops.string_to_float(data[k]) if data[k].dtype == dtypes.string
          else data[k])
  return array_ops.concat(1, features), keys, data_spec


def ParseDataTensorOrDict(data):
  """Return a tensor to use for input data.

  The incoming features can be a dict where keys are the string names of the
  columns, which we turn into a single 2-D tensor.

  Args:
    data: `Tensor` or `dict` of `Tensor` objects.

  Returns:
    A 2-D tensor for input to tensor_forest, a keys tensor for the
    tf.Examples if they exist, and a list of the type of each column
    (e.g. continuous float, categorical).
  """
  if isinstance(data, dict):
    # If there's at least one sparse tensor, everything has to be sparse.
    is_sparse = False
    for v in data.values():
      if isinstance(v, ops.SparseTensor):
        is_sparse = True
        break
    if is_sparse:
      return _ParseSparse(data)
    else:
      return _ParseDense(data)
  else:
    return data, None, [constants.DATA_FLOAT] * data.get_shape().as_list()[1]


def ParseLabelTensorOrDict(labels):
  """Return a tensor to use for input labels to tensor_forest.

  The incoming targets can be a dict where keys are the string names of the
  columns, which we turn into a single 1-D tensor for classification or
  2-D tensor for regression.

  Converts sparse tensors to dense ones.

  Args:
    labels: `Tensor` or `dict` of `Tensor` objects.

  Returns:
    A 2-D tensor for labels/outputs.
  """
  if isinstance(labels, dict):
    return math_ops.to_float(array_ops.concat(
        1, [sparse_ops.sparse_tensor_to_dense(labels[
            k], default_value=-1) if isinstance(labels, ops.SparseTensor) else
            labels[k] for k in sorted(labels.keys())]))
  else:
    if isinstance(labels, ops.SparseTensor):
      return math_ops.to_float(sparse_ops.sparse_tensor_to_dense(
          labels, default_value=-1))
    else:
      return math_ops.to_float(labels)
