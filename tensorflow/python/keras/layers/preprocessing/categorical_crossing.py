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

import itertools

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor


class CategoryCrossing(Layer):
  """Category crossing layer.

  This layer transforms multiple categorical inputs to categorical outputs
  by Cartesian product, and hash the output if necessary. Without hashing
  (`num_bins=None`) the output dtype is string, with hashing the output dtype
  is int64.

  For each input, the hash function uses a specific fingerprint method, i.e.,
  [FarmHash64](https://github.com/google/farmhash) to compute the hashed output,
  that provides a consistent hashed output across different platforms.
  For multiple inputs, the final output is calculated by first computing the
  fingerprint of `hash_key`, and concatenate it with the fingerprints of
  each input. The user can also obfuscate the output with customized `hash_key`.

  If [SipHash64[(https://github.com/google/highwayhash) is desired instead, the
  user can set `num_bins=None` to get string outputs, and use Hashing layer to
  get hashed output with SipHash64.

  Usage:

  Use with string output.
  >>> inp_1 = tf.constant([['a'], ['b'], ['c']])
  >>> inp_2 = tf.constant([['d'], ['e'], ['f']])
  >>> layer = categorical_crossing.CategoryCrossing()
  >>> output = layer([inp_1, inp_2])

  Use with hashed output.
  >>> layer = categorical_crossing.CategoryCrossing(num_bins=2)
  >>> output = layer([inp_1, inp_2])

  Use with customized hashed output.
  >>> layer = categorical_crossing.CategoryCrossing(num_bins=2, hash_key=133)
  >>> output = layer([inp_1, inp_2])

  Arguments:
    depth: depth of input crossing. By default None, all inputs are crossed into
      one output. It can also be an int or tuple/list of ints. Passing an
      integer will create combinations of crossed outputs with depth up to that
      integer, i.e., [1, 2, ..., `depth`), and passing a tuple of integers will
      create crossed outputs with depth for the specified values in the tuple,
      i.e., `depth`=(N1, N2) will create all possible crossed outputs with depth
      equal to N1 or N2. Passing `None` means a single crossed output with all
      inputs. For example, with inputs `a`, `b` and `c`, `depth=2` means the
      output will be [a;b;c;cross(a, b);cross(bc);cross(ca)].
    num_bins: Number of hash bins. By default None, no hashing is performed.
    hash_key: Integer hash_key that will be used by the concatenate
      fingerprints. If not given, will use a default key from
      `tf.sparse.cross_hashed`. This is only valid when `num_bins` is not None.
    name: Name to give to the layer.
    **kwargs: Keyword arguments to construct a layer.

  Input shape: a list of string or int tensors or sparse tensors of shape
    `[batch_size, d1, ..., dm]`

  Output shape: a single string or int tensor or sparse tensor of shape
    `[batch_size, d1, ..., dm]`

  Below 'hash' stands for tf.fingerprint, and cat stands for 'FingerprintCat'.

  Example: (`depth`=None)
    If the layer receives three inputs:
    `a=[[1], [4]]`, `b=[[2], [5]]`, `c=[[3], [6]]`
    the output will be a string tensor if not hashed:
    `[[b'1_X_2_X_3'], [b'4_X_5_X_6']]`
    the output will be an int64 tensor if hashed:
    `[[cat(hash(3), cat(hash(2), cat(hash(1), hash(hash_key))))],
     [[cat(hash(6), cat(hash(5), cat(hash(4), hash(hash_key))))]`

  Example: (`depth` is an integer)
    With the same input above, and if `depth`=2,
    the output will be a list of 6 string tensors if not hashed:
    `[[b'1'], [b'4']]`
    `[[b'2'], [b'5']]`
    `[[b'3'], [b'6']]`
    `[[b'1_X_2'], [b'4_X_5']]`,
    `[[b'2_X_3'], [b'5_X_6']]`,
    `[[b'3_X_1'], [b'6_X_4']]`
    the output will be a list of 6 int64 tensors if hashed:
    `[[hash(b'1')], [hash(b'4')]]`
    `[[hash(b'2')], [hash(b'5')]]`
    `[[hash(b'3')], [hash(b'6')]]`
    `[[cat(hash(2), cat(hash(1), hash(hash_key)))],
      [cat(hash(5), cat(hash(4), hash(hash_key)))]`,
    `[[cat(hash(3), cat(hash(1), hash(hash_key)))],
      [cat(hash(6), cat(hash(4), hash(hash_key)))]`,
    `[[cat(hash(3), cat(hash(2), hash(hash_key)))],
      [cat(hash(6), cat(hash(5), hash(hash_key)))]`,

  Example: (`depth` is a tuple/list of integers)
    With the same input above, and if `depth`=(2, 3)
    the output will be a list of 4 string tensors if not hashed:
    `[[b'1_X_2'], [b'4_X_5']]`,
    `[[b'2_X_3'], [b'5_X_6']]`,
    `[[b'3_X_1'], [b'6_X_4']]`,
    `[[b'1_X_2_X_3'], [b'4_X_5_X_6']]`
    the output will be a list of 4 int64 tensors if hashed:
    `[
      [cat(hash(2), cat(hash(1), hash(hash_key)))],
      [cat(hash(5), cat(hash(4), hash(hash_key)))]
     ]`,
    `[
      [cat(hash(3), cat(hash(1), hash(hash_key)))],
      [cat(hash(6), cat(hash(4), hash(hash_key)))]
     ]`,
    `[
      [cat(hash(3), cat(hash(2), hash(hash_key)))],
      [cat(hash(6), cat(hash(5), hash(hash_key)))]
     ]`,
    `[
      [cat(hash(3), cat(hash(2), cat(hash(1), hash(hash_key))))],
      [cat(hash(6), cat(hash(5), cat(hash(4), hash(hash_key))))]
     ]`
  """

  def __init__(self,
               depth=None,
               num_bins=None,
               hash_key=None,
               name=None,
               **kwargs):
    # TODO(tanzheny): Consider making seperator configurable.
    if num_bins is None and hash_key is not None:
      raise ValueError('`hash_key` is only valid when `num_bins` is not None')
    super(CategoryCrossing, self).__init__(name=name, **kwargs)
    self.depth = depth
    self.num_bins = num_bins
    self.hash_key = hash_key
    if isinstance(depth, (tuple, list)):
      self._depth_tuple = depth
    elif depth is not None:
      self._depth_tuple = tuple([i for i in range(1, depth + 1)])

  def partial_crossing(self, partial_inputs, ragged_out, sparse_out):
    """Gets the crossed output from a partial list/tuple of inputs."""
    if self.num_bins is not None:
      partial_output = sparse_ops.sparse_cross_hashed(
          partial_inputs, num_buckets=self.num_bins, hash_key=self.hash_key)
    else:
      partial_output = sparse_ops.sparse_cross(partial_inputs)

    # If ragged_out=True, convert output from sparse to ragged.
    if ragged_out:
      return ragged_tensor.RaggedTensor.from_sparse(partial_output)
    elif sparse_out:
      return partial_output
    else:
      return sparse_ops.sparse_tensor_to_dense(partial_output)

  def call(self, inputs):
    depth_tuple = self._depth_tuple if self.depth else (len(inputs),)
    ragged_out = sparse_out = False
    if all([ragged_tensor.is_ragged(inp) for inp in inputs]):
      # (b/144500510) ragged.map_flat_values(sparse_cross_hashed, inputs) will
      # cause kernel failure. Investigate and find a more efficient
      # implementation
      inputs = [inp.to_sparse() for inp in inputs]
      ragged_out = True
    else:
      if any([ragged_tensor.is_ragged(inp) for inp in inputs]):
        raise ValueError(
            'Inputs must be either all `RaggedTensor`, or none of them should '
            'be `RaggedTensor`, got {}'.format(inputs))

      if any([isinstance(inp, sparse_tensor.SparseTensor) for inp in inputs]):
        sparse_out = True

    outputs = []
    for depth in depth_tuple:
      if len(inputs) < depth:
        raise ValueError(
            'Number of inputs cannot be less than depth, got {} input tensors, '
            'and depth {}'.format(len(inputs), depth))
      for partial_inps in itertools.combinations(inputs, depth):
        partial_out = self.partial_crossing(
            partial_inps, ragged_out, sparse_out)
        outputs.append(partial_out)
    if sparse_out:
      return sparse_ops.sparse_concat_v2(axis=1, sp_inputs=outputs)
    return array_ops.concat(outputs, axis=1)

  def compute_output_shape(self, input_shape):
    if not isinstance(input_shape, (tuple, list)):
      raise ValueError('A `CategoryCrossing` layer should be called '
                       'on a list of inputs.')
    input_shapes = input_shape
    batch_size = None
    for inp_shape in input_shapes:
      inp_tensor_shape = tensor_shape.TensorShape(inp_shape).as_list()
      if len(inp_tensor_shape) != 2:
        raise ValueError('Inputs must be rank 2, get {}'.format(input_shapes))
      if batch_size is None:
        batch_size = inp_tensor_shape[0]
    # The second dimension is dynamic based on inputs.
    output_shape = [batch_size, None]
    return tensor_shape.TensorShape(output_shape)

  def compute_output_signature(self, input_spec):
    input_shapes = [x.shape for x in input_spec]
    output_shape = self.compute_output_shape(input_shapes)
    output_dtype = dtypes.int64 if self.num_bins else dtypes.string
    return sparse_tensor.SparseTensorSpec(
        shape=output_shape, dtype=output_dtype)

  def get_config(self):
    config = {
        'depth': self.depth,
        'num_bins': self.num_bins,
        'hash_key': self.hash_key
    }
    base_config = super(CategoryCrossing, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
