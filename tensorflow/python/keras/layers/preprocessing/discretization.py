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
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.experimental.preprocessing.Discretization")
class Discretization(base_preprocessing_layer.PreprocessingLayer):
  """Buckets data into discrete ranges.

  This layer will place each element of its input data into one of several
  contiguous ranges and output an integer index indicating which range each
  element was placed in.

  Input shape:
    Any `tf.Tensor` or `tf.RaggedTensor` of dimension 2 or higher.

  Output shape:
    Same as input shape.

  Attributes:
    bins: Optional boundary specification. Bins include the left boundary and
      exclude the right boundary, so `bins=[0., 1., 2.]` generates bins
      `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +inf)`.

  Examples:

  Bucketize float values based on provided buckets.
  >>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
  >>> layer = tf.keras.layers.experimental.preprocessing.Discretization(
  ...          bins=[0., 1., 2.])
  >>> layer(input)
  <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
  array([[0, 2, 3, 1],
         [1, 3, 2, 1]], dtype=int32)>
  """

  def __init__(self, bins, **kwargs):
    super(Discretization, self).__init__(**kwargs)
    base_preprocessing_layer._kpl_gauge.get_cell("V2").set("Discretization")
    self.bins = bins

  def get_config(self):
    config = {
        "bins": self.bins,
    }
    base_config = super(Discretization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = dtypes.int64
    if isinstance(input_spec, sparse_tensor.SparseTensorSpec):
      return sparse_tensor.SparseTensorSpec(
          shape=output_shape, dtype=output_dtype)
    return tensor_spec.TensorSpec(shape=output_shape, dtype=output_dtype)

  def call(self, inputs):
    if tf_utils.is_ragged(inputs):
      integer_buckets = ragged_functional_ops.map_flat_values(
          gen_math_ops.Bucketize, input=inputs, boundaries=self.bins)
      # Ragged map_flat_values doesn't touch the non-values tensors in the
      # ragged composite tensor. If this op is the only op a Keras model,
      # this can cause errors in Graph mode, so wrap the tensor in an identity.
      return array_ops.identity(integer_buckets)
    elif isinstance(inputs, sparse_tensor.SparseTensor):
      integer_buckets = gen_math_ops.Bucketize(
          input=inputs.values, boundaries=self.bins)
      return sparse_tensor.SparseTensor(
          indices=array_ops.identity(inputs.indices),
          values=integer_buckets,
          dense_shape=array_ops.identity(inputs.dense_shape))
    else:
      return gen_math_ops.Bucketize(input=inputs, boundaries=self.bins)
