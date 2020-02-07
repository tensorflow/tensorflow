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
"""Keras categorical preprocessing layers."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor


class CategoryCrossing(Layer):
  """Category crossing layer.

  This layer transforms multiple categorical inputs to categorical outputs
  by Cartesian product, and hash the output if necessary. Without hashing
  (`num_bins=None`) the output dtype is string, with hashing the output dtype
  is int64.

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
    name: Name to give to the layer.
    **kwargs: Keyword arguments to construct a layer.

  Input shape: a list of string or int tensors or sparse tensors of shape
    `[batch_size, d1, ..., dm]`

  Output shape: a single string or int tensor or sparse tensor of shape
    `[batch_size, d1, ..., dm]`

  Example: (`depth`=None)
    If the layer receives three inputs:
    `a=[[1], [4]]`, `b=[[2], [5]]`, `c=[[3], [6]]`
    the output will be a string tensor if not hashed:
    `[[b'1_X_2_X_3'], [b'4_X_5_X_6']]`
    the output will be an int64 tensor if hashed:
    `[[hash(b'1_X_2_X_3')], [hash(b'4_X_5_X_6')]]`

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
    `[[hash(b'1_X_2')], [hash(b'4_X_5')]]`,
    `[[hash(b'2_X_3')], [hash(b'5_X_6')]]`,
    `[[hash(b'3_X_1')], [hash(b'6_X_4')]]`

  Example: (`depth` is a tuple/list of integers)
    With the same input above, and if `depth`=(2, 3)
    the output will be a list of 4 string tensors if not hashed:
    `[[b'1_X_2'], [b'4_X_5']]`,
    `[[b'2_X_3'], [b'5_X_6']]`,
    `[[b'3_X_1'], [b'6_X_4']]`,
    `[[b'1_X_2_X_3'], [b'4_X_5_X_6']]`
    the output will be a list of 4 int64 tensors if hashed:
    `[[hash(b'1_X_2')], [hash(b'4_X_5')]]`,
    `[[hash(b'2_X_3')], [hash(b'5_X_6')]]`,
    `[[hash(b'3_X_1')], [hash(b'6_X_4')]]`,
    `[[hash(b'1_X_2_X_3')], [hash(b'4_X_5_X_6')]]`
  """

  def __init__(self, depth=None, num_bins=None, name=None, **kwargs):
    # TODO(tanzheny): Add support for depth.
    # TODO(tanzheny): Consider making seperator configurable.
    if depth is not None:
      raise NotImplementedError('`depth` is not supported yet.')
    super(CategoryCrossing, self).__init__(name=name, **kwargs)
    self.num_bins = num_bins
    self.depth = depth
    self._supports_ragged_inputs = True

  def call(self, inputs):
    # (b/144500510) ragged.map_flat_values(sparse_cross_hashed, inputs) will
    # cause kernel failure. Investigate and find a more efficient implementation
    if all([ragged_tensor.is_ragged(inp) for inp in inputs]):
      inputs = [inp.to_sparse() if ragged_tensor.is_ragged(inp) else inp
                for inp in inputs]
      if self.num_bins is not None:
        output = sparse_ops.sparse_cross_hashed(
            inputs, num_buckets=self.num_bins)
      else:
        output = sparse_ops.sparse_cross(inputs)
      return ragged_tensor.RaggedTensor.from_sparse(output)
    if any([ragged_tensor.is_ragged(inp) for inp in inputs]):
      raise ValueError('Inputs must be either all `RaggedTensor`, or none of '
                       'them should be `RaggedTensor`, got {}'.format(inputs))
    sparse_output = False
    if any([isinstance(inp, sparse_tensor.SparseTensor) for inp in inputs]):
      sparse_output = True
    if self.num_bins is not None:
      output = sparse_ops.sparse_cross_hashed(
          inputs, num_buckets=self.num_bins)
    else:
      output = sparse_ops.sparse_cross(inputs)
    if not sparse_output:
      output = sparse_ops.sparse_tensor_to_dense(output)
    return output

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
    config = {'depth': self.depth, 'num_bins': self.num_bins}
    base_config = super(CategoryCrossing, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


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
