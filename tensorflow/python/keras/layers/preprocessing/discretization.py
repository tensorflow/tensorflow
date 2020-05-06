# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras preprocessing layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor

INTEGER = "int"
BINARY = "binary"


class Discretization(Layer):
  """Buckets data into discrete ranges.

  This layer will place each element of its input data into one of several
  contiguous ranges and output either an integer index or a one-hot vector
  indicating which range each element was placed in.

  What happens in `adapt()`: The dataset is examined and sliced.

  Input shape:
    Any `tf.Tensor` or `tf.RaggedTensor` of dimension 2 or higher.

  Output shape:
    The same as the input shape if `output_mode` is 'int', or
      `[output_shape, num_buckets]` if `output_mode` is 'binary'.

  Attributes:
    bins: Optional boundary specification. Bins include the left boundary and
      exclude the right boundary, so `bins=[0., 1., 2.]` generates bins
      `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +inf)`.
    output_mode: One of 'int', 'binary'. Defaults to 'int'.
  """

  def __init__(self, bins, output_mode=INTEGER, **kwargs):
    super(Discretization, self).__init__(**kwargs)
    self.bins = bins
    self.output_mode = output_mode

  def get_config(self):
    config = {
        "bins": self.bins,
        "output_mode": self.output_mode,
    }
    base_config = super(Discretization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    if self.output_mode == INTEGER:
      return input_shape
    else:
      return tensor_shape.TensorShape([dim for dim in input_shape] +
                                      [len(self.bins)])

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = dtypes.int64
    if isinstance(input_spec, sparse_tensor.SparseTensorSpec):
      return sparse_tensor.SparseTensorSpec(
          shape=output_shape, dtype=output_dtype)
    return tensor_spec.TensorSpec(shape=output_shape, dtype=output_dtype)

  def call(self, inputs):
    if ragged_tensor.is_ragged(inputs):
      integer_buckets = ragged_functional_ops.map_flat_values(
          math_ops._bucketize, inputs, boundaries=self.bins)  # pylint: disable=protected-access
      # Ragged map_flat_values doesn't touch the non-values tensors in the
      # ragged composite tensor. If this op is the only op a Keras model,
      # this can cause errors in Graph mode, so wrap the tensor in an identity.
      integer_buckets = array_ops.identity(integer_buckets)
    elif isinstance(inputs, sparse_tensor.SparseTensor):
      integer_buckets = math_ops._bucketize(  # pylint: disable=protected-access
          inputs.values,
          boundaries=self.bins)
    else:
      integer_buckets = math_ops._bucketize(inputs, boundaries=self.bins)  # pylint: disable=protected-access

    if self.output_mode == INTEGER:
      if isinstance(inputs, sparse_tensor.SparseTensor):
        return sparse_tensor.SparseTensor(
            indices=array_ops.identity(inputs.indices),
            values=integer_buckets,
            dense_shape=array_ops.identity(inputs.dense_shape))
      return integer_buckets
    else:
      if isinstance(inputs, sparse_tensor.SparseTensor):
        raise ValueError("`output_mode=binary` is not supported for "
                         "sparse input")
      # The 'bins' array is the set of boundaries between the bins. We actually
      # have 'len(bins)+1' outputs.
      # TODO(momernick): This will change when we have the ability to adapt().
      return array_ops.one_hot(integer_buckets, depth=len(self.bins) + 1)
