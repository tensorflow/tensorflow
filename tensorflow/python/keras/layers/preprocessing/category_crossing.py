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
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.experimental.preprocessing.CategoryCrossing')
class CategoryCrossing(base_preprocessing_layer.PreprocessingLayer):
  """Category crossing layer.

  This layer concatenates multiple categorical inputs into a single categorical
  output (similar to Cartesian product). The output dtype is string.

  Usage:
  >>> inp_1 = ['a', 'b', 'c']
  >>> inp_2 = ['d', 'e', 'f']
  >>> layer = tf.keras.layers.experimental.preprocessing.CategoryCrossing()
  >>> layer([inp_1, inp_2])
  <tf.Tensor: shape=(3, 1), dtype=string, numpy=
    array([[b'a_X_d'],
           [b'b_X_e'],
           [b'c_X_f']], dtype=object)>


  >>> inp_1 = ['a', 'b', 'c']
  >>> inp_2 = ['d', 'e', 'f']
  >>> layer = tf.keras.layers.experimental.preprocessing.CategoryCrossing(
  ...    separator='-')
  >>> layer([inp_1, inp_2])
  <tf.Tensor: shape=(3, 1), dtype=string, numpy=
    array([[b'a-d'],
           [b'b-e'],
           [b'c-f']], dtype=object)>

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
    separator: A string added between each input being joined. Defaults to
      '_X_'.
    name: Name to give to the layer.
    **kwargs: Keyword arguments to construct a layer.

  Input shape: a list of string or int tensors or sparse tensors of shape
    `[batch_size, d1, ..., dm]`

  Output shape: a single string or int tensor or sparse tensor of shape
    `[batch_size, d1, ..., dm]`

  Returns:
    If any input is `RaggedTensor`, the output is `RaggedTensor`.
    Else, if any input is `SparseTensor`, the output is `SparseTensor`.
    Otherwise, the output is `Tensor`.

  Example: (`depth`=None)
    If the layer receives three inputs:
    `a=[[1], [4]]`, `b=[[2], [5]]`, `c=[[3], [6]]`
    the output will be a string tensor:
    `[[b'1_X_2_X_3'], [b'4_X_5_X_6']]`

  Example: (`depth` is an integer)
    With the same input above, and if `depth`=2,
    the output will be a list of 6 string tensors:
    `[[b'1'], [b'4']]`
    `[[b'2'], [b'5']]`
    `[[b'3'], [b'6']]`
    `[[b'1_X_2'], [b'4_X_5']]`,
    `[[b'2_X_3'], [b'5_X_6']]`,
    `[[b'3_X_1'], [b'6_X_4']]`

  Example: (`depth` is a tuple/list of integers)
    With the same input above, and if `depth`=(2, 3)
    the output will be a list of 4 string tensors:
    `[[b'1_X_2'], [b'4_X_5']]`,
    `[[b'2_X_3'], [b'5_X_6']]`,
    `[[b'3_X_1'], [b'6_X_4']]`,
    `[[b'1_X_2_X_3'], [b'4_X_5_X_6']]`
  """

  def __init__(self, depth=None, name=None, separator=None, **kwargs):
    super(CategoryCrossing, self).__init__(name=name, **kwargs)
    base_preprocessing_layer._kpl_gauge.get_cell('V2').set('CategoryCrossing')
    self.depth = depth
    if separator is None:
      separator = '_X_'
    self.separator = separator
    if isinstance(depth, (tuple, list)):
      self._depth_tuple = depth
    elif depth is not None:
      self._depth_tuple = tuple([i for i in range(1, depth + 1)])

  def partial_crossing(self, partial_inputs, ragged_out, sparse_out):
    """Gets the crossed output from a partial list/tuple of inputs."""
    # If ragged_out=True, convert output from sparse to ragged.
    if ragged_out:
      # TODO(momernick): Support separator with ragged_cross.
      if self.separator != '_X_':
        raise ValueError('Non-default separator with ragged input is not '
                         'supported yet, given {}'.format(self.separator))
      return ragged_array_ops.cross(partial_inputs)
    elif sparse_out:
      return sparse_ops.sparse_cross(partial_inputs, separator=self.separator)
    else:
      return sparse_ops.sparse_tensor_to_dense(
          sparse_ops.sparse_cross(partial_inputs, separator=self.separator))

  def _preprocess_input(self, inp):
    if isinstance(inp, (list, tuple, np.ndarray)):
      inp = ops.convert_to_tensor_v2_with_dispatch(inp)
    if inp.shape.rank == 1:
      inp = array_ops.expand_dims(inp, axis=-1)
    return inp

  def call(self, inputs):
    inputs = [self._preprocess_input(inp) for inp in inputs]
    depth_tuple = self._depth_tuple if self.depth else (len(inputs),)
    ragged_out = sparse_out = False
    if any(tf_utils.is_ragged(inp) for inp in inputs):
      ragged_out = True
    elif any(isinstance(inp, sparse_tensor.SparseTensor) for inp in inputs):
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
    if any(
        isinstance(inp_spec, ragged_tensor.RaggedTensorSpec)
        for inp_spec in input_spec):
      return tensor_spec.TensorSpec(shape=output_shape, dtype=dtypes.string)
    elif any(
        isinstance(inp_spec, sparse_tensor.SparseTensorSpec)
        for inp_spec in input_spec):
      return sparse_tensor.SparseTensorSpec(
          shape=output_shape, dtype=dtypes.string)
    return tensor_spec.TensorSpec(shape=output_shape, dtype=dtypes.string)

  def get_config(self):
    config = {
        'depth': self.depth,
        'separator': self.separator,
    }
    base_config = super(CategoryCrossing, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
