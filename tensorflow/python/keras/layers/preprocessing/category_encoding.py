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
"""Keras CategoryEncoding preprocessing layer."""
# pylint: disable=g-classes-have-attributes

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

INT = "int"
BINARY = "binary"
COUNT = "count"


@keras_export("keras.layers.experimental.preprocessing.CategoryEncoding")
class CategoryEncoding(base_preprocessing_layer.PreprocessingLayer):
  """Category encoding layer.

  This layer provides options for condensing data into a categorical encoding
  when the total number of tokens are known in advance. It accepts integer
  values as inputs and outputs a dense representation (one sample = 1-index
  tensor of float values representing data about the sample's tokens) of those
  inputs. For integer inputs where the total number of tokens is not known, see
  `tf.keras.layers.experimental.preprocessing.IntegerLookup`.

  Examples:

  **Multi-hot encoding data**

  >>> layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
  ...           num_tokens=4, output_mode="binary")
  >>> layer([[0, 1], [0, 0], [1, 2], [3, 1]])
  <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
    array([[1., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 1., 1., 0.],
           [0., 1., 0., 1.]], dtype=float32)>

  **Using weighted inputs in `count` mode**

  >>> layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
  ...           num_tokens=4, output_mode="count")
  >>> count_weights = np.array([[.1, .2], [.1, .1], [.2, .3], [.4, .2]])
  >>> layer([[0, 1], [0, 0], [1, 2], [3, 1]], count_weights=count_weights)
  <tf.Tensor: shape=(4, 4), dtype=float64, numpy=
    array([[0.1, 0.2, 0. , 0. ],
           [0.2, 0. , 0. , 0. ],
           [0. , 0.2, 0.3, 0. ],
           [0. , 0.2, 0. , 0.4]])>

  Args:
    num_tokens: The total number of tokens the layer should support. All inputs
      to the layer must integers in the range 0 <= value < num_tokens or an
      error will be thrown.
    output_mode: Specification for the output of the layer.
      Defaults to "binary". Values can
      be "binary" or "count", configuring the layer as follows:
        "binary": Outputs a single int array per batch, of num_tokens size,
          containing 1s in all elements where the token mapped to that index
          exists at least once in the batch item.
        "count": As "binary", but the int array contains a count of the number
          of times the token at that index appeared in the batch item.
    sparse: Boolean. If true, returns a `SparseTensor` instead of a dense
      `Tensor`. Defaults to `False`.

  Call arguments:
    inputs: A 2D tensor `(samples, timesteps)`.
    count_weights: A 2D tensor in the same shape as `inputs` indicating the
      weight for each sample value when summing up in `count` mode. Not used in
      `binary` mode.
  """

  def __init__(self,
               num_tokens=None,
               output_mode=BINARY,
               sparse=False,
               **kwargs):
    # max_tokens is an old name for the num_tokens arg we continue to support
    # because of usage.
    if "max_tokens" in kwargs:
      logging.warning(
          "max_tokens is deprecated, please use num_tokens instead.")
      num_tokens = kwargs["max_tokens"]
      del kwargs["max_tokens"]

    super(CategoryEncoding, self).__init__(**kwargs)

    # 'output_mode' must be one of (COUNT, BINARY)
    layer_utils.validate_string_arg(
        output_mode,
        allowable_strings=(COUNT, BINARY),
        layer_name="CategoryEncoding",
        arg_name="output_mode")

    if num_tokens is None:
      raise ValueError("num_tokens must be set to use this layer. If the "
                       "number of tokens is not known beforehand, use the "
                       "IntegerLookup layer instead.")
    if num_tokens < 1:
      raise ValueError("num_tokens must be >= 1.")

    self.num_tokens = num_tokens
    self.output_mode = output_mode
    self.sparse = sparse

  def compute_output_shape(self, input_shape):
    return tensor_shape.TensorShape([input_shape[0], self.num_tokens])

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    if self.sparse:
      return sparse_tensor.SparseTensorSpec(
          shape=output_shape, dtype=dtypes.int64)
    else:
      return tensor_spec.TensorSpec(shape=output_shape, dtype=dtypes.int64)

  def get_config(self):
    config = {
        "num_tokens": self.num_tokens,
        "output_mode": self.output_mode,
        "sparse": self.sparse,
    }
    base_config = super(CategoryEncoding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, count_weights=None):
    if isinstance(inputs, (list, np.ndarray)):
      inputs = ops.convert_to_tensor_v2_with_dispatch(inputs)
    if inputs.shape.rank == 1:
      inputs = array_ops.expand_dims(inputs, 1)

    if count_weights is not None and self.output_mode != COUNT:
      raise ValueError("count_weights is not used in `output_mode='binary'`. "
                       "Please pass a single input.")

    out_depth = self.num_tokens
    binary_output = (self.output_mode == BINARY)
    if isinstance(inputs, sparse_tensor.SparseTensor):
      max_value = math_ops.reduce_max(inputs.values)
      min_value = math_ops.reduce_min(inputs.values)
    else:
      max_value = math_ops.reduce_max(inputs)
      min_value = math_ops.reduce_min(inputs)
    condition = math_ops.logical_and(
        math_ops.greater(
            math_ops.cast(out_depth, max_value.dtype), max_value),
        math_ops.greater_equal(
            min_value, math_ops.cast(0, min_value.dtype)))
    control_flow_ops.Assert(condition, [
        "Input values must be in the range 0 <= values < num_tokens"
        " with num_tokens={}".format(out_depth)
    ])
    if self.sparse:
      return sparse_bincount(inputs, out_depth, binary_output, count_weights)
    else:
      return dense_bincount(inputs, out_depth, binary_output, count_weights)


def sparse_bincount(inputs, out_depth, binary_output, count_weights=None):
  """Apply binary or count encoding to an input and return a sparse tensor."""
  result = bincount_ops.sparse_bincount(
      inputs,
      weights=count_weights,
      minlength=out_depth,
      maxlength=out_depth,
      axis=-1,
      binary_output=binary_output)
  result = math_ops.cast(result, backend.floatx())
  batch_size = array_ops.shape(result)[0]
  result = sparse_tensor.SparseTensor(
      indices=result.indices,
      values=result.values,
      dense_shape=[batch_size, out_depth])
  return result


def dense_bincount(inputs, out_depth, binary_output, count_weights=None):
  """Apply binary or count encoding to an input."""
  result = bincount_ops.bincount(
      inputs,
      weights=count_weights,
      minlength=out_depth,
      maxlength=out_depth,
      dtype=backend.floatx(),
      axis=-1,
      binary_output=binary_output)
  batch_size = inputs.shape.as_list()[0]
  result.set_shape(tensor_shape.TensorShape((batch_size, out_depth)))
  return result
