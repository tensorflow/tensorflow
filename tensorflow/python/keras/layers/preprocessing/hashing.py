# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras categorical preprocessing layers."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor


class Hashing(Layer):
  """Implements categorical feature hashing, also known as "hashing trick".

  This layer transforms categorical inputs to hashed output. It converts a
  sequence of int or string to a sequence of int. The stable hash function uses
  tensorflow::ops::Fingerprint to produce universal output that is consistent
  across platforms.

  Usage:
  ```python
    layer = Hashing(num_bins=3)
    inp = np.asarray([['A', 'B'], ['C', 'A']])
    layer(inputs)
    [[0, 0], [1, 0]]
  ```

  Arguments:
    num_bins: Number of hash bins.
    name: Name to give to the layer.
    **kwargs: Keyword arguments to construct a layer.

  Input shape: A string, int32 or int64 tensor of shape
    `[batch_size, d1, ..., dm]`

  Output shape: An int64 tensor of shape `[batch_size, d1, ..., dm]`

  Example:
    If the input is a 5 by 1 string tensor '[['A'], ['B'], ['C'], ['D'], ['E']]'
    with `num_bins=2`, then output is 5 by 1 integer tensor
    [[hash('A')], [hash('B')], [hash('C')], [hash('D')], [hash('E')]].
  """

  def __init__(self, num_bins, name=None, **kwargs):
    # TODO(tanzheny): consider adding strong hash variant.
    super(Hashing, self).__init__(name=name, **kwargs)
    self._num_bins = num_bins
    self._supports_ragged_inputs = True

  def call(self, inputs):
    # TODO(tanzheny): Add int support.
    # string_to_hash_bucket_fast uses FarmHash as hash function.
    if ragged_tensor.is_ragged(inputs):
      return ragged_functional_ops.map_flat_values(
          string_ops.string_to_hash_bucket_fast,
          inputs,
          num_buckets=self._num_bins,
          name='hash')
    elif isinstance(inputs, sparse_tensor.SparseTensor):
      sparse_values = inputs.values
      sparse_hashed_values = string_ops.string_to_hash_bucket_fast(
          sparse_values, self._num_bins, name='hash')
      return sparse_tensor.SparseTensor(
          indices=inputs.indices,
          values=sparse_hashed_values,
          dense_shape=inputs.dense_shape)
    else:
      return string_ops.string_to_hash_bucket_fast(
          inputs, self._num_bins, name='hash')

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = dtypes.int64
    if isinstance(input_spec, sparse_tensor.SparseTensorSpec):
      return sparse_tensor.SparseTensorSpec(
          shape=output_shape, dtype=output_dtype)
    else:
      return tensor_spec.TensorSpec(shape=output_shape, dtype=output_dtype)

  def get_config(self):
    config = {'num_bins': self._num_bins}
    base_config = super(Hashing, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
