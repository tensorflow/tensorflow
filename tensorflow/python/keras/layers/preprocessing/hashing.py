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

import functools

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

  This layer uses [FarmHash64](https://github.com/google/farmhash) by default,
  which provides a consistent hashed output across different platforms and is
  stable across invocations, regardless of device and context, by mixing the
  input bits thoroughly.

  If you want to obfuscate the hashed output, you can also pass a random `salt`
  argument in the constructor. In that case, the layer will use the
  [SipHash64](https://github.com/google/highwayhash) hash function, with
  the `salt` value serving as additional input to the hash function.

  Example (FarmHash64):
  ```python
    layer = Hashing(num_bins=3)
    inp = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    layer(inputs)
    [[1], [0], [1], [1], [2]]
  ```

  Example (SipHash64):
  ```python
    layer = Hashing(num_bins=3, salt=[133, 137])
    inp = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    layer(inputs)
    [[1], [2], [1], [0], [2]]
  ```

  Arguments:
    num_bins: Number of hash bins.
    salt: A tuple/list of 2 unsigned integer numbers. If passed, the hash
      function used will be SipHash64, with these values used as an additional
      input (known as a "salt" in cryptography).
      These should be non-zero. Defaults to `None` (in that
      case, the FarmHash64 hash function is used).
    name: Name to give to the layer.
    **kwargs: Keyword arguments to construct a layer.

  Input shape: A string, int32 or int64 tensor of shape
    `[batch_size, d1, ..., dm]`

  Output shape: An int64 tensor of shape `[batch_size, d1, ..., dm]`

  """

  def __init__(self, num_bins, salt=None, name=None, **kwargs):
    if num_bins is None or num_bins <= 0:
      raise ValueError('`num_bins` cannot be `None` or non-positive values.')
    if salt is not None:
      if not isinstance(salt, (tuple, list)) or len(salt) != 2:
        raise ValueError('`salt` must be a tuple or list of 2 unsigned '
                         'integer numbers, got {}'.format(salt))
    super(Hashing, self).__init__(name=name, **kwargs)
    self.num_bins = num_bins
    self.salt = salt
    self._supports_ragged_inputs = True

  def call(self, inputs):
    # Converts integer inputs to string.
    if inputs.dtype.is_integer:
      if isinstance(inputs, sparse_tensor.SparseTensor):
        inputs = sparse_tensor.SparseTensor(
            indices=inputs.indices,
            values=string_ops.as_string(inputs.values),
            dense_shape=inputs.dense_shape)
      else:
        inputs = string_ops.as_string(inputs)
    str_to_hash_bucket = self._get_string_to_hash_bucket_fn()
    if ragged_tensor.is_ragged(inputs):
      return ragged_functional_ops.map_flat_values(
          str_to_hash_bucket, inputs, num_buckets=self.num_bins, name='hash')
    elif isinstance(inputs, sparse_tensor.SparseTensor):
      sparse_values = inputs.values
      sparse_hashed_values = str_to_hash_bucket(
          sparse_values, self.num_bins, name='hash')
      return sparse_tensor.SparseTensor(
          indices=inputs.indices,
          values=sparse_hashed_values,
          dense_shape=inputs.dense_shape)
    else:
      return str_to_hash_bucket(inputs, self.num_bins, name='hash')

  def _get_string_to_hash_bucket_fn(self):
    """Returns the string_to_hash_bucket op to use based on `hasher_key`."""
    # string_to_hash_bucket_fast uses FarmHash64 as hash function.
    if self.salt is None:
      return string_ops.string_to_hash_bucket_fast
    # string_to_hash_bucket_strong uses SipHash64 as hash function.
    else:
      return functools.partial(
          string_ops.string_to_hash_bucket_strong, key=self.salt)

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
    config = {'num_bins': self.num_bins, 'salt': self.salt}
    base_config = super(Hashing, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
