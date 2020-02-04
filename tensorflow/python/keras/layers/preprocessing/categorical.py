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

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import sparse_ops


class CategoryLookup(Layer):
  """Category lookup layer.

  This layer looks up tokens (int or string) in a vocabulary table,
  and return their indices (int). It converts a sequence of int or string to a
  sequence of int.

  Attributes:
    vocabulary: The vocabulary to lookup the input. If it is a file, it
      represents the source vocab file; If it is a list/tuple, it represents the
      source vocab list. If it is None, the vocabulary can later be set.
    max_tokens: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary. This is used when `adapt`
      is called.
    num_oov_tokens: Non-negative integer. The number of out-of-vocab tokens. All
      out-of-vocab inputs will be assigned IDs in the range of [0,
      num_oov_tokens) based on a hash.
    name: Name to give to the layer.
    **kwargs: Keyword arguments to construct a layer.
  Input shape: A string or int tensor of shape `[batch_size, d1, ..., dm]`
  Output shape: An int tensor of shape `[batch_size, d1, .., dm]`
  Example: Consider a batch of a single input sample, `[["a", "c", "d", "a",
    "x"]]`. Let's say the vocabulary is `["a", "b", "c", "d"]` and a single OOV
    token is used (`num_oov_tokens=1`). Then the corresponding output is `[[1,
    3, 4, 1, 0]]`. 0 stands for an OOV token.
  """

  def __init__(self,
               max_tokens=None,
               num_oov_tokens=1,
               vocabulary=None,
               name=None,
               **kwargs):
    if max_tokens is not None:
      raise ValueError('`max_tokens` and `adapt` is not supported yet.')
    if vocabulary is None:
      raise ValueError('for now, you must pass a `vocabulary` argument')
    self.max_tokens = max_tokens
    self.num_oov_tokens = num_oov_tokens
    self.vocabulary = vocabulary
    super(CategoryLookup, self).__init__(name, **kwargs)

  def __call__(self, inputs, *args, **kwargs):
    if isinstance(inputs, (np.ndarray, float, int)):
      inputs = ops.convert_to_tensor(inputs)
    self._input_dtype = inputs.dtype
    return super(CategoryLookup, self).__call__(inputs, *args, **kwargs)

  def build(self, input_shape):
    # categorical with vocabulary list.
    if isinstance(self.vocabulary, (tuple, list, np.ndarray)):
      self.table = lookup_ops.index_table_from_tensor(
          vocabulary_list=self.vocabulary,
          num_oov_buckets=self.num_oov_tokens,
          dtype=self._input_dtype)
    # categorical with vocabulary file.
    elif self.vocabulary:
      self.table = lookup_ops.index_table_from_file(
          vocabulary_file=self.vocabulary,
          num_oov_buckets=self.num_oov_tokens,
          key_dtype=self._input_dtype)

  def call(self, inputs):
    return self.table.lookup(inputs)

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
    self.num_bins = num_bins
    self.depth = depth
    super(CategoryCrossing, self).__init__(name=name, **kwargs)

  def call(self, inputs):
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
