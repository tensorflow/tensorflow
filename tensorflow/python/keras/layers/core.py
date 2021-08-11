# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Core Keras layers."""

import copy
import functools
import operator
import sys
import textwrap
import types as python_types
import warnings

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name
from tensorflow.python.util.tf_export import keras_export


# pylint: disable=g-classes-have-attributes
@keras_export('keras.layers.Masking')
class Masking(Layer):
  """Masks a sequence by using a mask value to skip timesteps.

  For each timestep in the input tensor (dimension #1 in the tensor),
  if all values in the input tensor at that timestep
  are equal to `mask_value`, then the timestep will be masked (skipped)
  in all downstream layers (as long as they support masking).

  If any downstream layer does not support masking yet receives such
  an input mask, an exception will be raised.

  Example:

  Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
  to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you
  lack data for these timesteps. You can:

  - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
  - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

  ```python
  samples, timesteps, features = 32, 10, 8
  inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
  inputs[:, 3, :] = 0.
  inputs[:, 5, :] = 0.

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Masking(mask_value=0.,
                                    input_shape=(timesteps, features)))
  model.add(tf.keras.layers.LSTM(32))

  output = model(inputs)
  # The time step 3 and 5 will be skipped from LSTM calculation.
  ```

  See [the masking and padding guide](
    https://www.tensorflow.org/guide/keras/masking_and_padding)
  for more details.
  """

  def __init__(self, mask_value=0., **kwargs):
    super(Masking, self).__init__(**kwargs)
    self.supports_masking = True
    self.mask_value = mask_value
    self._compute_output_and_mask_jointly = True

  def compute_mask(self, inputs, mask=None):
    return K.any(math_ops.not_equal(inputs, self.mask_value), axis=-1)

  def call(self, inputs):
    boolean_mask = K.any(
        math_ops.not_equal(inputs, self.mask_value), axis=-1, keepdims=True)
    outputs = inputs * math_ops.cast(boolean_mask, inputs.dtype)
    # Compute the mask and outputs simultaneously.
    outputs._keras_mask = array_ops.squeeze(boolean_mask, axis=-1)  # pylint: disable=protected-access
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'mask_value': self.mask_value}
    base_config = super(Masking, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.Dropout')
class Dropout(Layer):
  """Applies Dropout to the input.

  The Dropout layer randomly sets input units to 0 with a frequency of `rate`
  at each step during training time, which helps prevent overfitting.
  Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
  all inputs is unchanged.

  Note that the Dropout layer only applies when `training` is set to True
  such that no values are dropped during inference. When using `model.fit`,
  `training` will be appropriately set to True automatically, and in other
  contexts, you can set the kwarg explicitly to True when calling the layer.

  (This is in contrast to setting `trainable=False` for a Dropout layer.
  `trainable` does not affect the layer's behavior, as Dropout does
  not have any variables/weights that can be frozen during training.)

  >>> tf.random.set_seed(0)
  >>> layer = tf.keras.layers.Dropout(.2, input_shape=(2,))
  >>> data = np.arange(10).reshape(5, 2).astype(np.float32)
  >>> print(data)
  [[0. 1.]
   [2. 3.]
   [4. 5.]
   [6. 7.]
   [8. 9.]]
  >>> outputs = layer(data, training=True)
  >>> print(outputs)
  tf.Tensor(
  [[ 0.    1.25]
   [ 2.5   3.75]
   [ 5.    6.25]
   [ 7.5   8.75]
   [10.    0.  ]], shape=(5, 2), dtype=float32)

  Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    noise_shape: 1D integer tensor representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)` and
      you want the dropout mask to be the same for all timesteps,
      you can use `noise_shape=(batch_size, 1, features)`.
    seed: A Python integer to use as random seed.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).
  """

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(Dropout, self).__init__(**kwargs)
    if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
      raise ValueError(f'Invalid value {rate} received for '
                       f'`rate`, expected a value between 0 and 1.')
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, inputs):
    # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
    # which will override `self.noise_shape`, and allows for custom noise
    # shapes with dynamically sized inputs.
    if self.noise_shape is None:
      return None

    concrete_inputs_shape = array_ops.shape(inputs)
    noise_shape = []
    for i, value in enumerate(self.noise_shape):
      noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def dropped_inputs():
      return nn.dropout(
          inputs,
          noise_shape=self._get_noise_shape(inputs),
          seed=self.seed,
          rate=self.rate)

    output = control_flow_util.smart_cond(training, dropped_inputs,
                                          lambda: array_ops.identity(inputs))
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed
    }
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.SpatialDropout1D')
class SpatialDropout1D(Dropout):
  """Spatial 1D version of Dropout.

  This version performs the same function as Dropout, however, it drops
  entire 1D feature maps instead of individual elements. If adjacent frames
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout1D will help promote independence
  between feature maps and should be used instead.

  Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.

  Call arguments:
    inputs: A 3D tensor.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    3D tensor with shape:
    `(samples, timesteps, channels)`

  Output shape:
    Same as input.

  References:
    - [Efficient Object Localization Using Convolutional
      Networks](https://arxiv.org/abs/1411.4280)
  """

  def __init__(self, rate, **kwargs):
    super(SpatialDropout1D, self).__init__(rate, **kwargs)
    self.input_spec = InputSpec(ndim=3)

  def _get_noise_shape(self, inputs):
    input_shape = array_ops.shape(inputs)
    noise_shape = (input_shape[0], 1, input_shape[2])
    return noise_shape


@keras_export('keras.layers.SpatialDropout2D')
class SpatialDropout2D(Dropout):
  """Spatial 2D version of Dropout.

  This version performs the same function as Dropout, however, it drops
  entire 2D feature maps instead of individual elements. If adjacent pixels
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout2D will help promote independence
  between feature maps and should be used instead.

  Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    data_format: 'channels_first' or 'channels_last'.
      In 'channels_first' mode, the channels dimension
      (the depth) is at index 1,
      in 'channels_last' mode is it at index 3.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Call arguments:
    inputs: A 4D tensor.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    4D tensor with shape:
    `(samples, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
    Same as input.

  References:
    - [Efficient Object Localization Using Convolutional
      Networks](https://arxiv.org/abs/1411.4280)
  """

  def __init__(self, rate, data_format=None, **kwargs):
    super(SpatialDropout2D, self).__init__(rate, **kwargs)
    if data_format is None:
      data_format = K.image_data_format()
    if data_format not in {'channels_last', 'channels_first'}:
      raise ValueError('data_format must be in '
                       '{"channels_last", "channels_first"}')
    self.data_format = data_format
    self.input_spec = InputSpec(ndim=4)

  def _get_noise_shape(self, inputs):
    input_shape = array_ops.shape(inputs)
    if self.data_format == 'channels_first':
      return (input_shape[0], input_shape[1], 1, 1)
    elif self.data_format == 'channels_last':
      return (input_shape[0], 1, 1, input_shape[3])


@keras_export('keras.layers.SpatialDropout3D')
class SpatialDropout3D(Dropout):
  """Spatial 3D version of Dropout.

  This version performs the same function as Dropout, however, it drops
  entire 3D feature maps instead of individual elements. If adjacent voxels
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout3D will help promote independence
  between feature maps and should be used instead.

  Args:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    data_format: 'channels_first' or 'channels_last'.
        In 'channels_first' mode, the channels dimension (the depth)
        is at index 1, in 'channels_last' mode is it at index 4.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

  Call arguments:
    inputs: A 5D tensor.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    5D tensor with shape:
    `(samples, channels, dim1, dim2, dim3)` if data_format='channels_first'
    or 5D tensor with shape:
    `(samples, dim1, dim2, dim3, channels)` if data_format='channels_last'.

  Output shape:
    Same as input.

  References:
    - [Efficient Object Localization Using Convolutional
      Networks](https://arxiv.org/abs/1411.4280)
  """

  def __init__(self, rate, data_format=None, **kwargs):
    super(SpatialDropout3D, self).__init__(rate, **kwargs)
    if data_format is None:
      data_format = K.image_data_format()
    if data_format not in {'channels_last', 'channels_first'}:
      raise ValueError('data_format must be in '
                       '{"channels_last", "channels_first"}')
    self.data_format = data_format
    self.input_spec = InputSpec(ndim=5)

  def _get_noise_shape(self, inputs):
    input_shape = array_ops.shape(inputs)
    if self.data_format == 'channels_first':
      return (input_shape[0], input_shape[1], 1, 1, 1)
    elif self.data_format == 'channels_last':
      return (input_shape[0], 1, 1, 1, input_shape[4])


@keras_export('keras.layers.Activation')
class Activation(Layer):
  """Applies an activation function to an output.

  Args:
    activation: Activation function, such as `tf.nn.relu`, or string name of
      built-in activation function, such as "relu".

  Usage:

  >>> layer = tf.keras.layers.Activation('relu')
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]
  >>> layer = tf.keras.layers.Activation(tf.nn.relu)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, activation, **kwargs):
    super(Activation, self).__init__(**kwargs)
    self.supports_masking = True
    self.activation = activations.get(activation)

  def call(self, inputs):
    return self.activation(inputs)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'activation': activations.serialize(self.activation)}
    base_config = super(Activation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.Reshape')
class Reshape(Layer):
  """Layer that reshapes inputs into the given shape.

  Input shape:
    Arbitrary, although all dimensions in the input shape must be known/fixed.
    Use the keyword argument `input_shape` (tuple of integers, does not include
    the samples/batch size axis) when using this layer as the first layer
    in a model.

  Output shape:
    `(batch_size,) + target_shape`

  Example:

  >>> # as first layer in a Sequential model
  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Reshape((3, 4), input_shape=(12,)))
  >>> # model.output_shape == (None, 3, 4), `None` is the batch size.
  >>> model.output_shape
  (None, 3, 4)

  >>> # as intermediate layer in a Sequential model
  >>> model.add(tf.keras.layers.Reshape((6, 2)))
  >>> model.output_shape
  (None, 6, 2)

  >>> # also supports shape inference using `-1` as dimension
  >>> model.add(tf.keras.layers.Reshape((-1, 2, 2)))
  >>> model.output_shape
  (None, 3, 2, 2)
  """

  def __init__(self, target_shape, **kwargs):
    """Creates a `tf.keras.layers.Reshape`  layer instance.

    Args:
      target_shape: Target shape. Tuple of integers, does not include the
        samples dimension (batch size).
      **kwargs: Any additional layer keyword arguments.
    """
    super(Reshape, self).__init__(**kwargs)
    self.target_shape = tuple(target_shape)

  def _fix_unknown_dimension(self, input_shape, output_shape):
    """Find and replace a missing dimension in an output shape.

    This is a near direct port of the internal Numpy function
    `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

    Args:
      input_shape: Shape of array being reshaped
      output_shape: Desired shape of the array with at most
        a single -1 which indicates a dimension that should be
        derived from the input shape.

    Returns:
      The new output shape with a -1 replaced with its computed value.

    Raises:
      ValueError: If the total array size of the output_shape is
      different than the input_shape, or more than one unknown dimension
      is specified.
    """
    output_shape = list(output_shape)
    msg = ('total size of new array must be unchanged, '
           'input_shape = {}, output_shape = {}'
           .format(input_shape, output_shape))

    known, unknown = 1, None
    for index, dim in enumerate(output_shape):
      if dim < 0:
        if unknown is None:
          unknown = index
        else:
          raise ValueError('Can only specify one unknown dimension.')
      else:
        known *= dim

    original = np.prod(input_shape, dtype=int)
    if unknown is not None:
      if known == 0 or original % known != 0:
        raise ValueError(msg)
      output_shape[unknown] = original // known
    elif original != known:
      raise ValueError(msg)
    return output_shape

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if None in input_shape[1:]:
      output_shape = [input_shape[0]]
      # input shape (partially) unknown? replace -1's with None's
      output_shape += tuple(s if s != -1 else None for s in self.target_shape)
    else:
      output_shape = [input_shape[0]]
      output_shape += self._fix_unknown_dimension(input_shape[1:],
                                                  self.target_shape)
    return tensor_shape.TensorShape(output_shape)

  def call(self, inputs):
    result = array_ops.reshape(
        inputs, (array_ops.shape(inputs)[0],) + self.target_shape)
    if not context.executing_eagerly():
      # Set the static shape for the result since it might lost during array_ops
      # reshape, eg, some `None` dim in the result could be inferred.
      result.set_shape(self.compute_output_shape(inputs.shape))
    return result

  def get_config(self):
    config = {'target_shape': self.target_shape}
    base_config = super(Reshape, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.Permute')
class Permute(Layer):
  """Permutes the dimensions of the input according to a given pattern.

  Useful e.g. connecting RNNs and convnets.

  Example:

  ```python
  model = Sequential()
  model.add(Permute((2, 1), input_shape=(10, 64)))
  # now: model.output_shape == (None, 64, 10)
  # note: `None` is the batch dimension
  ```

  Args:
    dims: Tuple of integers. Permutation pattern does not include the
      samples dimension. Indexing starts at 1.
      For instance, `(2, 1)` permutes the first and second dimensions
      of the input.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same as the input shape, but with the dimensions re-ordered according
    to the specified pattern.
  """

  def __init__(self, dims, **kwargs):
    super(Permute, self).__init__(**kwargs)
    self.dims = tuple(dims)
    if sorted(dims) != list(range(1, len(dims) + 1)):
      raise ValueError(
          'Invalid permutation `dims` for Permute Layer: %s. '
          'The set of indices in `dims` must be consecutive and start from 1.' %
          (dims,))
    self.input_spec = InputSpec(ndim=len(self.dims) + 1)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = copy.copy(input_shape)
    for i, dim in enumerate(self.dims):
      target_dim = input_shape[dim]
      output_shape[i + 1] = target_dim
    return tensor_shape.TensorShape(output_shape)

  def call(self, inputs):
    return array_ops.transpose(inputs, perm=(0,) + self.dims)

  def get_config(self):
    config = {'dims': self.dims}
    base_config = super(Permute, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.Flatten')
class Flatten(Layer):
  """Flattens the input. Does not affect the batch size.

  Note: If inputs are shaped `(batch,)` without a feature axis, then
  flattening adds an extra channel dimension and output shape is `(batch, 1)`.

  Args:
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Example:

  >>> model = tf.keras.Sequential()
  >>> model.add(tf.keras.layers.Conv2D(64, 3, 3, input_shape=(3, 32, 32)))
  >>> model.output_shape
  (None, 1, 10, 64)

  >>> model.add(Flatten())
  >>> model.output_shape
  (None, 640)

  """

  def __init__(self, data_format=None, **kwargs):
    super(Flatten, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(min_ndim=1)
    self._channels_first = self.data_format == 'channels_first'

  def call(self, inputs):
    if self._channels_first:
      rank = inputs.shape.rank
      if rank and rank > 1:
        # Switch to channels-last format.
        permutation = [0]
        permutation.extend(range(2, rank))
        permutation.append(1)
        inputs = array_ops.transpose(inputs, perm=permutation)

    if context.executing_eagerly():
      # Full static shape is guaranteed to be available.
      # Performance: Using `constant_op` is much faster than passing a list.
      flattened_shape = constant_op.constant([inputs.shape[0], -1])
      return array_ops.reshape(inputs, flattened_shape)
    else:
      input_shape = inputs.shape
      rank = input_shape.rank
      if rank == 1:
        return array_ops.expand_dims_v2(inputs, axis=1)
      else:
        batch_dim = tensor_shape.dimension_value(input_shape[0])
        non_batch_dims = input_shape[1:]
        # Reshape in a way that preserves as much shape info as possible.
        if non_batch_dims.is_fully_defined():
          last_dim = int(functools.reduce(operator.mul, non_batch_dims))
          flattened_shape = constant_op.constant([-1, last_dim])
        elif batch_dim is not None:
          flattened_shape = constant_op.constant([int(batch_dim), -1])
        else:
          flattened_shape = [array_ops.shape_v2(inputs)[0], -1]
        return array_ops.reshape(inputs, flattened_shape)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if not input_shape:
      output_shape = tensor_shape.TensorShape([1])
    else:
      output_shape = [input_shape[0]]
    if np.all(input_shape[1:]):
      output_shape += [np.prod(input_shape[1:], dtype=int)]
    else:
      output_shape += [None]
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(Flatten, self).get_config()
    config.update({'data_format': self.data_format})
    return config


@keras_export('keras.layers.RepeatVector')
class RepeatVector(Layer):
  """Repeats the input n times.

  Example:

  ```python
  model = Sequential()
  model.add(Dense(32, input_dim=32))
  # now: model.output_shape == (None, 32)
  # note: `None` is the batch dimension

  model.add(RepeatVector(3))
  # now: model.output_shape == (None, 3, 32)
  ```

  Args:
    n: Integer, repetition factor.

  Input shape:
    2D tensor of shape `(num_samples, features)`.

  Output shape:
    3D tensor of shape `(num_samples, n, features)`.
  """

  def __init__(self, n, **kwargs):
    super(RepeatVector, self).__init__(**kwargs)
    self.n = n
    if not isinstance(n, int):
      raise TypeError(f'Expected an integer value for `n`, got {type(n)}.')
    self.input_spec = InputSpec(ndim=2)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape([input_shape[0], self.n, input_shape[1]])

  def call(self, inputs):
    return K.repeat(inputs, self.n)

  def get_config(self):
    config = {'n': self.n}
    base_config = super(RepeatVector, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.Lambda')
class Lambda(Layer):
  """Wraps arbitrary expressions as a `Layer` object.

  The `Lambda` layer exists so that arbitrary expressions can be used
  as a `Layer` when constructing `Sequential`
  and Functional API models. `Lambda` layers are best suited for simple
  operations or quick experimentation. For more advanced use cases, follow
  [this guide](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
  for subclassing `tf.keras.layers.Layer`.

  WARNING: `tf.keras.layers.Lambda` layers have (de)serialization limitations!

  The main reason to subclass `tf.keras.layers.Layer` instead of using a
  `Lambda` layer is saving and inspecting a Model. `Lambda` layers
  are saved by serializing the Python bytecode, which is fundamentally
  non-portable. They should only be loaded in the same environment where
  they were saved. Subclassed layers can be saved in a more portable way
  by overriding their `get_config` method. Models that rely on
  subclassed Layers are also often easier to visualize and reason about.

  Examples:

  ```python
  # add a x -> x^2 layer
  model.add(Lambda(lambda x: x ** 2))
  ```
  ```python
  # add a layer that returns the concatenation
  # of the positive part of the input and
  # the opposite of the negative part

  def antirectifier(x):
      x -= K.mean(x, axis=1, keepdims=True)
      x = K.l2_normalize(x, axis=1)
      pos = K.relu(x)
      neg = K.relu(-x)
      return K.concatenate([pos, neg], axis=1)

  model.add(Lambda(antirectifier))
  ```

  Variables:
    While it is possible to use Variables with Lambda layers, this practice is
    discouraged as it can easily lead to bugs. For instance, consider the
    following layer:

    ```python
      scale = tf.Variable(1.)
      scale_layer = tf.keras.layers.Lambda(lambda x: x * scale)
    ```

    Because scale_layer does not directly track the `scale` variable, it will
    not appear in `scale_layer.trainable_weights` and will therefore not be
    trained if `scale_layer` is used in a Model.

    A better pattern is to write a subclassed Layer:

    ```python
      class ScaleLayer(tf.keras.layers.Layer):
        def __init__(self):
          super(ScaleLayer, self).__init__()
          self.scale = tf.Variable(1.)

        def call(self, inputs):
          return inputs * self.scale
    ```

    In general, Lambda layers can be convenient for simple stateless
    computation, but anything more complex should use a subclass Layer instead.

  Args:
    function: The function to be evaluated. Takes input tensor as first
      argument.
    output_shape: Expected output shape from function. This argument can be
      inferred if not explicitly provided. Can be a tuple or function. If a
      tuple, it only specifies the first dimension onward;
      sample dimension is assumed either the same as the input: `output_shape =
        (input_shape[0], ) + output_shape` or, the input is `None` and
      the sample dimension is also `None`: `output_shape = (None, ) +
        output_shape` If a function, it specifies the entire shape as a function
        of the
      input shape: `output_shape = f(input_shape)`
    mask: Either None (indicating no masking) or a callable with the same
      signature as the `compute_mask` layer method, or a tensor that will be
      returned as output mask regardless of what the input is.
    arguments: Optional dictionary of keyword arguments to be passed to the
      function.

  Input shape:
    Arbitrary. Use the keyword argument input_shape (tuple of
    integers, does not include the samples axis) when using this layer as the
    first layer in a model.

  Output shape:
    Specified by `output_shape` argument
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, function, output_shape=None, mask=None, arguments=None,
               **kwargs):
    super(Lambda, self).__init__(**kwargs)

    self.arguments = arguments or {}
    self.function = function

    if mask is not None:
      self.supports_masking = True
    self.mask = mask
    self._output_shape = output_shape

    # Warning on every invocation will be quite irksome in Eager mode.
    self._already_warned = False

    function_args = tf_inspect.getfullargspec(function).args
    self._fn_expects_training_arg = 'training' in function_args
    self._fn_expects_mask_arg = 'mask' in function_args

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self._output_shape is None:
      # Make use of existing autocomputation but provide Lambda-specific
      # error message. This is always safe to run even when the outer context
      # is Graph mode because Lambda layers don't have side effects such as
      # `add_loss`.
      with context.eager_mode():
        try:
          return super(Lambda, self).compute_output_shape(input_shape)
        except NotImplementedError:
          raise NotImplementedError(
              'We could not automatically infer the shape of the Lambda\'s '
              'output. Please specify `output_shape` for this Lambda.')

    if callable(self._output_shape):
      output_shapes = self._output_shape(input_shape)
      return tf_utils.convert_shapes(output_shapes, to_tuples=False)

    # Output shapes are passed directly and don't include batch dimension.
    input_tensor_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
    batch_size = nest.flatten(input_tensor_shape)[0][0] if input_shape else None

    def _add_batch(shape):
      return tensor_shape.TensorShape([batch_size] + shape.as_list())

    output_shapes = tf_utils.convert_shapes(self._output_shape, to_tuples=False)
    return nest.map_structure(_add_batch, output_shapes)

  def call(self, inputs, mask=None, training=None):
    # We must copy for thread safety, but it only needs to be a shallow copy.
    kwargs = {k: v for k, v in self.arguments.items()}
    if self._fn_expects_mask_arg:
      kwargs['mask'] = mask
    if self._fn_expects_training_arg:
      kwargs['training'] = training

    created_variables = []
    def _variable_creator(next_creator, **kwargs):
      var = next_creator(**kwargs)
      created_variables.append(var)
      return var

    with backprop.GradientTape(watch_accessed_variables=True) as tape,\
        variable_scope.variable_creator_scope(_variable_creator):
      result = self.function(inputs, **kwargs)
    self._check_variables(created_variables, tape.watched_variables())
    return result

  def _check_variables(self, created_variables, accessed_variables):
    if not created_variables and not accessed_variables:
      # In the common case that a Lambda layer does not touch a Variable, we
      # don't want to incur the runtime cost of assembling any state used for
      # checking only to immediately discard it.
      return

    tracked_weights = set(v.ref() for v in self.weights)
    untracked_new_vars = [
        v for v in created_variables if v.ref() not in tracked_weights
    ]
    if untracked_new_vars:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_new_vars)
      error_str = textwrap.dedent(
          '''
          The following Variables were created within a Lambda layer ({name})
          but are not tracked by said layer:
          {variable_str}
          The layer cannot safely ensure proper Variable reuse across multiple
          calls, and consquently this behavior is disallowed for safety. Lambda
          layers are not well suited to stateful computation; instead, writing a
          subclassed Layer is the recommend way to define layers with
          Variables.'''
      ).format(name=self.name, variable_str=variable_str)
      raise ValueError(error_str)

    untracked_used_vars = [
        v for v in accessed_variables if v.ref() not in tracked_weights
    ]
    if untracked_used_vars and not self._already_warned:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_used_vars)
      self._warn(textwrap.dedent(
          '''
          The following Variables were used a Lambda layer's call ({name}), but
          are not present in its tracked objects:
          {variable_str}
          It is possible that this is intended behavior, but it is more likely
          an omission. This is a strong indication that this layer should be
          formulated as a subclassed Layer rather than a Lambda layer.'''
      ).format(name=self.name, variable_str=variable_str))
      self._already_warned = True

  def _warn(self, msg):
    # This method will be overridden in a unit test to raise an error, because
    # self.assertWarns is not universally implemented.
    return tf_logging.warning(msg)

  def compute_mask(self, inputs, mask=None):
    if callable(self.mask):
      return self.mask(inputs, mask)
    return self.mask

  def get_config(self):
    function_config = self._serialize_function_to_config(self.function)
    output_shape_config = self._serialize_function_to_config(self._output_shape,
                                                             allow_raw=True)
    config = {
        'function': function_config[0],
        'function_type': function_config[1],
        'module': function_config[2],
        'output_shape': output_shape_config[0],
        'output_shape_type': output_shape_config[1],
        'output_shape_module': output_shape_config[2],
    }
    if self.mask is not None:
      mask_config = self._serialize_function_to_config(self.mask)
      config.update({
          'mask': mask_config[0],
          'mask_type': mask_config[1],
          'mask_module': mask_config[2]
      })
    config['arguments'] = self.arguments

    base_config = super(Lambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _serialize_function_to_config(self, inputs, allow_raw=False):
    if isinstance(inputs, python_types.LambdaType):
      output = generic_utils.func_dump(inputs)
      output_type = 'lambda'
      module = inputs.__module__
    elif callable(inputs):
      output = inputs.__name__
      output_type = 'function'
      module = inputs.__module__
    elif allow_raw:
      output = inputs
      output_type = 'raw'
      module = None
    else:
      raise ValueError(
          'Invalid input for serialization, type: %s ' % type(inputs))

    return output, output_type, module

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    function = cls._parse_function_from_config(
        config, custom_objects, 'function', 'module', 'function_type')

    output_shape = cls._parse_function_from_config(
        config, custom_objects, 'output_shape', 'output_shape_module',
        'output_shape_type')
    if 'mask' in config:
      mask = cls._parse_function_from_config(
          config, custom_objects, 'mask', 'mask_module', 'mask_type')
    else:
      mask = None

    config['function'] = function
    config['output_shape'] = output_shape
    config['mask'] = mask

    # If arguments were numpy array, they have been saved as
    # list. We need to recover the ndarray
    if 'arguments' in config:
      for key in config['arguments']:
        if isinstance(config['arguments'][key], dict):
          arg_dict = config['arguments'][key]
          if 'type' in arg_dict and arg_dict['type'] == 'ndarray':
            # Overwrite the argument with its numpy translation
            config['arguments'][key] = np.array(arg_dict['value'])

    return cls(**config)

  @classmethod
  def _parse_function_from_config(
      cls, config, custom_objects, func_attr_name, module_attr_name,
      func_type_attr_name):
    globs = globals().copy()
    module = config.pop(module_attr_name, None)
    if module in sys.modules:
      globs.update(sys.modules[module].__dict__)
    elif module is not None:
      # Note: we don't know the name of the function if it's a lambda.
      warnings.warn('{} is not loaded, but a Lambda layer uses it. '
                    'It may cause errors.'.format(module)
                    , UserWarning)
    if custom_objects:
      globs.update(custom_objects)
    function_type = config.pop(func_type_attr_name)
    if function_type == 'function':
      # Simple lookup in custom objects
      function = generic_utils.deserialize_keras_object(
          config[func_attr_name],
          custom_objects=custom_objects,
          printable_module_name='function in Lambda layer')
    elif function_type == 'lambda':
      # Unsafe deserialization from bytecode
      function = generic_utils.func_load(
          config[func_attr_name], globs=globs)
    elif function_type == 'raw':
      function = config[func_attr_name]
    else:
      raise TypeError('Unknown function type:', function_type)
    return function


@keras_export('keras.layers.Dense')
class Dense(Layer):
  """Just your regular densely-connected NN layer.

  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`). These are all attributes of
  `Dense`.

  Note: If the input to the layer has a rank greater than 2, then `Dense`
  computes the dot product between the `inputs` and the `kernel` along the
  last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
  For example, if input has dimensions `(batch_size, d0, d1)`,
  then we create a `kernel` with shape `(d1, units)`, and the `kernel` operates
  along axis 2 of the `input`, on every sub-tensor of shape `(1, 1, d1)`
  (there are `batch_size * d0` such sub-tensors).
  The output in this case will have shape `(batch_size, d0, units)`.

  Besides, layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).
  When a popular kwarg `input_shape` is passed, then keras will create
  an input layer to insert before the current layer. This can be treated
  equivalent to explicitly defining an `InputLayer`.

  Example:

  >>> # Create a `Sequential` model and add a Dense layer as the first layer.
  >>> model = tf.keras.models.Sequential()
  >>> model.add(tf.keras.Input(shape=(16,)))
  >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
  >>> # Now the model will take as input arrays of shape (None, 16)
  >>> # and output arrays of shape (None, 32).
  >>> # Note that after the first layer, you don't need to specify
  >>> # the size of the input anymore:
  >>> model.add(tf.keras.layers.Dense(32))
  >>> model.output_shape
  (None, 32)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Dense, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    if self.units < 0:
      raise ValueError(f'Received an invalid value for `units`, expected '
                       f'a positive integer, got {units}.')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
      inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

    rank = inputs.shape.rank
    if rank == 2 or rank is None:
      # We use embedding_lookup_sparse as a more efficient matmul operation for
      # large sparse input tensors. The op will result in a sparse gradient, as
      # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
      # gradients. This can lead to sigfinicant speedups, see b/171762937.
      if isinstance(inputs, sparse_tensor.SparseTensor):
        # We need to fill empty rows, as the op assumes at least one id per row.
        inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
        # We need to do some munging of our input to use the embedding lookup as
        # a matrix multiply. We split our input matrix into separate ids and
        # weights tensors. The values of the ids tensor should be the column
        # indices of our input matrix and the values of the weights tensor
        # can continue to the actual matrix weights.
        # The column arrangement of ids and weights
        # will be summed over and does not matter. See the documentation for
        # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
        # of the inputs to both ops.
        ids = sparse_tensor.SparseTensor(
            indices=inputs.indices,
            values=inputs.indices[:, 1],
            dense_shape=inputs.dense_shape)
        weights = inputs
        outputs = embedding_ops.embedding_lookup_sparse_v2(
            self.kernel, ids, weights, combiner='sum')
      else:
        outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel)
    # Broadcast kernel to inputs.
    else:
      outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.kernel.shape[-1]]
        outputs.set_shape(output_shape)

    if self.use_bias:
      outputs = nn_ops.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % (input_shape,))
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super(Dense, self).get_config()
    config.update({
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    })
    return config


@keras_export('keras.layers.ActivityRegularization')
class ActivityRegularization(Layer):
  """Layer that applies an update to the cost function based input activity.

  Args:
    l1: L1 regularization factor (positive float).
    l2: L2 regularization factor (positive float).

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, l1=0., l2=0., **kwargs):
    super(ActivityRegularization, self).__init__(
        activity_regularizer=regularizers.L1L2(l1=l1, l2=l2), **kwargs)
    self.supports_masking = True
    self.l1 = l1
    self.l2 = l2

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'l1': self.l1, 'l2': self.l2}
    base_config = super(ActivityRegularization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class TFOpLambda(Layer):
  """Wraps TF API symbols in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF symbol on KerasTensors.

  Like Lambda layers, this layer tries to raise warnings when it detects users
  explicitly use variables in the call. (To let them know
  that the layer will not capture the variables).

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = tf.Variable(...)
  out = x * tf_variable
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, function, **kwargs):
    self.function = function
    self.symbol = (
        get_canonical_name_for_symbol(self.function,
                                      add_prefix_to_v1_names=True) or
        get_canonical_name_for_symbol(self.function,
                                      api_name='keras',
                                      add_prefix_to_v1_names=True))
    if 'name' not in kwargs:
      # Generate a name.
      # TFOpLambda layers avoid already-observed names,
      # because users cannot easily control the generated names.
      # Without this avoidance, users would be more likely to run
      # into unavoidable duplicate layer name collisions.
      # (For standard layers users could just set `name` when creating the
      # layer to work around a collision, but they can't do that for
      # auto-generated layers)
      if self.symbol:
        name = 'tf.' + self.symbol
      else:
        name = self.function.__name__
      kwargs['name'] = K.unique_object_name(
          name, zero_based=True, avoid_observed_names=True)
    kwargs['autocast'] = False

    # Decorate the function to produce this layer's call method
    def _call_wrapper(*args, **kwargs):
      return self._call_wrapper(*args, **kwargs)
    self.call = tf_decorator.make_decorator(function, _call_wrapper)

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(TFOpLambda, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

    # Warning on every invocation will be quite irksome in Eager mode.
    self._already_warned = False

    self._expects_training_arg = False
    self._expects_mask_arg = False

  def _call_wrapper(self, *args, **kwargs):
    created_variables = []
    def _variable_creator(next_creator, **creator_kwargs):
      var = next_creator(**creator_kwargs)
      created_variables.append(var)
      return var

    with backprop.GradientTape(watch_accessed_variables=True) as tape, \
        variable_scope.variable_creator_scope(_variable_creator):
      # We explicitly drop `name` arguments here,
      # to guard against the case where an op explicitly has a
      # `name` passed (which is susceptible to producing
      # multiple ops w/ the same name when the layer is reused)
      kwargs.pop('name', None)
      result = self.function(*args, **kwargs)
    self._check_variables(created_variables, tape.watched_variables())
    return result

  def _check_variables(self, created_variables, accessed_variables):
    if not created_variables and not accessed_variables:
      # In the common case that a Lambda layer does not touch a Variable, we
      # don't want to incur the runtime cost of assembling any state used for
      # checking only to immediately discard it.
      return

    tracked_weights = set(v.ref() for v in self.weights)
    untracked_new_vars = [
        v for v in created_variables if v.ref() not in tracked_weights
    ]
    if untracked_new_vars:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_new_vars)
      error_str = textwrap.dedent(
          '''
          The following Variables were created within a Lambda layer ({name})
          but are not tracked by said layer:
          {variable_str}
          The layer cannot safely ensure proper Variable reuse across multiple
          calls, and consquently this behavior is disallowed for safety. Lambda
          layers are not well suited to stateful computation; instead, writing a
          subclassed Layer is the recommend way to define layers with
          Variables.'''
      ).format(name=self.name, variable_str=variable_str)
      raise ValueError(error_str)

    untracked_used_vars = [
        v for v in accessed_variables if v.ref() not in tracked_weights
    ]
    if untracked_used_vars and not self._already_warned:
      variable_str = '\n'.join('  {}'.format(i) for i in untracked_used_vars)
      self._warn(textwrap.dedent(
          '''
          The following Variables were used a Lambda layer's call ({name}), but
          are not present in its tracked objects:
          {variable_str}
          It is possible that this is intended behavior, but it is more likely
          an omission. This is a strong indication that this layer should be
          formulated as a subclassed Layer rather than a Lambda layer.'''
      ).format(name=self.name, variable_str=variable_str))
      self._already_warned = True

  def _warn(self, msg):
    # This method will be overridden in a unit test to raise an error, because
    # self.assertWarns is not universally implemented.
    return tf_logging.warning(msg)

  def get_config(self):
    if not self.symbol:
      raise ValueError('This Keras op layer was generated from %s, a method '
                       'that is not an exposed in the TensorFlow API. This '
                       'may have happened if the method was explicitly '
                       'decorated to add dispatching support, and it was used '
                       'during Functional model construction. '
                       'To ensure cross-version compatibility of Keras models '
                       'that use op layers, only op layers produced from '
                       'exported TF API symbols can be serialized.'
                       % self.function)
    config = {
        'function': self.symbol
    }

    base_config = super(TFOpLambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    symbol_name = config['function']
    function = get_symbol_from_name(symbol_name)
    if not function:
      raise ValueError(
          'TF symbol `tf.%s` could not be found.' % symbol_name)

    config['function'] = function

    return cls(**config)


class KerasOpDispatcher(dispatch.GlobalOpDispatcher):
  """A global dispatcher that allows building a functional model with TF Ops."""

  def handle(self, op, args, kwargs):
    """Handle the specified operation with the specified arguments."""
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in nest.flatten([args, kwargs])):
      return TFOpLambda(op)(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED

KerasOpDispatcher().register()


def _slice_to_dict(x):
  if isinstance(x, slice):
    return {'start': x.start, 'stop': x.stop, 'step': x.step}
  return x


def _dict_to_slice(x):
  if isinstance(x, dict):
    return slice(x['start'], x['stop'], x['step'])
  return x


class SlicingOpLambda(TFOpLambda):
  """Wraps TF API symbols in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF symbol on KerasTensors.

  Like Lambda layers, this layer tries to raise warnings when it detects users
  explicitly use variables in the call. (To let them know
  that the layer will not capture the variables).

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = tf.Variable(...)
  out = x * tf_variable
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, function, **kwargs):
    super(SlicingOpLambda, self).__init__(function, **kwargs)

    original_call = self.call
    # Decorate the function to produce this layer's call method
    def _call_wrapper(*args, **kwargs):
      # Turn any slice dicts in the args back into `slice` objects.
      # This conversion cannot use nest.flatten/map_structure,
      # because dicts are flattened by nest while slices aren't.
      # So, map_structure would only see the individual elements in the
      # dict.
      # This can't use map_structure_up_to either because the 'shallowness' of
      # the shallow tree would have to vary depending on if only one dim or
      # multiple are being sliced.
      new_args = []
      for arg in args:
        arg = _dict_to_slice(arg)
        if isinstance(arg, (list, tuple)):
          new_arg = []
          for sub_arg in arg:
            new_arg.append(_dict_to_slice(sub_arg))
          arg = new_arg
        new_args.append(arg)

      # Handle the kwargs too.
      new_kwargs = {}
      for key, value in kwargs.items():
        value = _dict_to_slice(value)
        if isinstance(value, (list, tuple)):
          new_value = []
          for v in value:
            new_value.append(_dict_to_slice(v))
          value = new_value
        new_kwargs[key] = value

      return original_call(*new_args, **new_kwargs)
    self.call = tf_decorator.make_decorator(original_call, _call_wrapper)


class TFSlicingOpDispatcher(dispatch.OpDispatcher):
  """A global dispatcher that allows building a functional model with TF Ops."""

  def __init__(self, op):
    self.op = op

  def handle(self, args, kwargs):
    """Handle the specified operation with the specified arguments."""
    args = nest.map_structure(_slice_to_dict, args)
    kwargs = nest.map_structure(_slice_to_dict, kwargs)
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in nest.flatten([args, kwargs])):
      return SlicingOpLambda(self.op)(*args, **kwargs)
    else:
      return self.NOT_SUPPORTED

for slicing_op in [
    array_ops._slice_helper,  # pylint: disable=protected-access
    array_ops.boolean_mask,
    array_ops.boolean_mask_v2,
    ragged_getitem.ragged_tensor_getitem
]:
  TFSlicingOpDispatcher(slicing_op).register(slicing_op)


class InstanceProperty(Layer):
  """Wraps an instance property access (e.g. `x.foo`) in a Keras Layer.

  This layer takes an attribute name `attr_name` in the constructor and,
  when called on input tensor `obj` returns `obj.attr_name`.

  KerasTensors specialized for specific extension types use it to
  represent instance property accesses on the represented object in the
  case where the property needs to be dynamically accessed as opposed to
  being statically computed from the typespec, e.g.

  x = keras.Input(..., ragged=True)
  out = x.flat_values
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, attr_name, **kwargs):
    self.attr_name = attr_name

    if 'name' not in kwargs:
      kwargs['name'] = K.unique_object_name(
          'input.' + self.attr_name, zero_based=True, avoid_observed_names=True)
    kwargs['autocast'] = False

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(InstanceProperty, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

  def call(self, obj):
    return getattr(obj, self.attr_name)

  def get_config(self):
    config = {
        'attr_name': self.attr_name
    }
    base_config = super(InstanceProperty, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


class InstanceMethod(InstanceProperty):
  """Wraps an instance method access (e.g. `x.foo(arg)` in a Keras Layer.

  This layer takes an attribute name `attr_name` in the constructor and,
  when called on input tensor `obj` with additional arguments `args` and
  `kwargs` returns `obj.attr_name(*args, **kwargs)`.

  KerasTensors specialized for specific extension types use it to
  represent dynamic instance method calls on the represented object, e.g.

  x = keras.Input(..., ragged=True)
  new_values = keras.Input(...)
  out = x.with_values(new_values)
  """

  def call(self, obj, args, kwargs):
    method = getattr(obj, self.attr_name)
    return method(*args, **kwargs)


def _delegate_property(keras_tensor_cls, property_name):  # pylint: disable=invalid-name
  """Register property on a KerasTensor class.

  Calling this multiple times with the same arguments should be a no-op.

  This method exposes a property on the KerasTensor class that will use an
  `InstanceProperty` layer to access the property on the represented
  intermediate values in the model.

  Args:
    keras_tensor_cls: The KerasTensor subclass that should expose the property.
    property_name: The name of the property to expose and delegate to the
      represented (Composite)Tensor.
  """
  # We use a lambda because we can't create a Keras layer at import time
  # due to dynamic layer class versioning.
  property_access = property(lambda self: InstanceProperty(property_name)(self))  # pylint: disable=unnecessary-lambda
  setattr(keras_tensor_cls, property_name, property_access)


def _delegate_method(keras_tensor_cls, method_name):  # pylint: disable=invalid-name
  """Register method on a KerasTensor class.

  Calling this function times with the same arguments should be a no-op.

  This method exposes an instance method on the KerasTensor class that will use
  an `InstanceMethod` layer to run the desired method on the represented
  intermediate values in the model.

  Args:
    keras_tensor_cls: The KerasTensor subclass that should expose the property.
    method_name: The name of the method to expose and delegate to the
      represented (Composite)Tensor.
  """
  def delegate(self, *args, **kwargs):
    return InstanceMethod(method_name)(self, args, kwargs)
  setattr(keras_tensor_cls, method_name, delegate)

# We do not support the `uniform_row_length` property because it
# returns either `None` or an int tensor, and code that relies on it tends
# to check `is None` directly. Delegating it here would always return a
# `KerasTensor`, regardless of what can be statically inferred. This would
# never equal `None`, breaking code that expects it to be partially-static
# in unpredictable ways.
for ragged_property in [
    'values',
    'flat_values',
    'row_splits',
    'nested_row_splits'
]:
  _delegate_property(keras_tensor.RaggedKerasTensor, ragged_property)

for ragged_method_name in [
    'value_rowids',
    'nested_value_rowids',
    'nrows',
    'row_starts',
    'row_limits',
    'row_lengths',
    'nested_row_lengths',
    'bounding_shape',
    'with_values',
    'with_flat_values',
    'with_row_splits_dtype',
    'merge_dims',
    'to_tensor',
    'to_sparse',
]:
  _delegate_method(keras_tensor.RaggedKerasTensor, ragged_method_name)

for sparse_property in [
    'indices',
    'values',
]:
  _delegate_property(keras_tensor.SparseKerasTensor, sparse_property)

for sparse_method in [
    'with_values',
]:
  _delegate_method(keras_tensor.SparseKerasTensor, sparse_method)


class ClassMethod(Layer):
  """Wraps a TF API Class's class method  in a `Layer` object.

  It is inserted by the Functional API construction whenever users call
  a supported TF Class's class method on KerasTensors.

  This is useful in the case where users do something like:
  x = keras.Input(...)
  y = keras.Input(...)
  out = tf.RaggedTensor.from_row_splits(x, y)
  """

  @trackable.no_automatic_dependency_tracking
  def __init__(self, cls_ref, method_name, **kwargs):
    self.cls_ref = cls_ref
    self.method_name = method_name
    self.cls_symbol = (
        get_canonical_name_for_symbol(self.cls_ref,
                                      add_prefix_to_v1_names=True) or
        get_canonical_name_for_symbol(self.cls_ref,
                                      api_name='keras',
                                      add_prefix_to_v1_names=True))
    if 'name' not in kwargs:
      kwargs['name'] = K.unique_object_name(
          'tf.' + self.cls_symbol + '.' + self.method_name, zero_based=True,
          avoid_observed_names=True)
    kwargs['autocast'] = False

    # Do not individually trace op layers in the SavedModel.
    self._must_restore_from_config = True

    super(ClassMethod, self).__init__(**kwargs)

    # Preserve all argument data structures when saving/loading a config
    # (e.g., don't unnest lists that contain one element)
    self._preserve_input_structure_in_config = True

    self._expects_training_arg = False
    self._expects_mask_arg = False

  def call(self, args, kwargs):
    return getattr(self.cls_ref, self.method_name)(*args, **kwargs)

  def get_config(self):
    if not self.cls_symbol:
      raise ValueError('This Keras class method conversion tried to convert '
                       'a method belonging to class %s, a class '
                       'that is not an exposed in the TensorFlow API. '
                       'To ensure cross-version compatibility of Keras models '
                       'that use op layers, only op layers produced from '
                       'exported TF API symbols can be serialized.'
                       % self.cls_symbol)
    config = {
        'cls_symbol': self.cls_symbol,
        'method_name': self.method_name
    }

    base_config = super(ClassMethod, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    symbol_name = config.pop('cls_symbol')
    cls_ref = get_symbol_from_name(symbol_name)
    if not cls_ref:
      raise ValueError(
          'TF symbol `tf.%s` could not be found.' % symbol_name)

    config['cls_ref'] = cls_ref

    return cls(**config)


class TFClassMethodDispatcher(dispatch.OpDispatcher):
  """A class method dispatcher that allows building a functional model with TF class methods."""

  def __init__(self, cls, method_name):
    self.cls = cls
    self.method_name = method_name

  def handle(self, args, kwargs):
    """Handle the specified operation with the specified arguments."""
    if any(
        isinstance(x, keras_tensor.KerasTensor)
        for x in nest.flatten([args, kwargs])):
      return ClassMethod(self.cls, self.method_name)(args[1:], kwargs)
    else:
      return self.NOT_SUPPORTED

for ragged_class_method in [
    'from_value_rowids',
    'from_row_splits',
    'from_row_lengths',
    'from_row_starts',
    'from_row_limits',
    'from_uniform_row_length',
    'from_nested_value_rowids',
    'from_nested_row_splits',
    'from_nested_row_lengths',
    'from_tensor',
    'from_sparse',
]:
  TFClassMethodDispatcher(
      ragged_tensor.RaggedTensor, ragged_class_method).register(
          getattr(ragged_tensor.RaggedTensor, ragged_class_method))
