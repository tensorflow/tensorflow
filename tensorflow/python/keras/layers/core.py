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
"""Core Keras layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys
import types as python_types
import warnings

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export


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
  to be fed to an LSTM layer.
  You want to mask timestep #3 and #5 because you lack data for
  these timesteps. You can:

  - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
  - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

  ```python
  model = Sequential()
  model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
  model.add(LSTM(32))
  ```
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

  Dropout consists in randomly setting
  a fraction `rate` of input units to 0 at each update during training time,
  which helps prevent overfitting.

  Arguments:
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
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, inputs):
    # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
    # which will override `self.noise_shape`, and allows for custom noise
    # shapes with dynamically sized inputs.
    if self.noise_shape is None:
      return self.noise_shape
    return nn_ops._get_noise_shape(inputs, self.noise_shape)  # pylint: disable=protected-access

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def dropped_inputs():
      return nn.dropout(
          inputs,
          noise_shape=self._get_noise_shape(inputs),
          seed=self.seed,
          rate=self.rate)

    output = tf_utils.smart_cond(training,
                                 dropped_inputs,
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

  This version performs the same function as Dropout, however it drops
  entire 1D feature maps instead of individual elements. If adjacent frames
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout1D will help promote independence
  between feature maps and should be used instead.

  Arguments:
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

  This version performs the same function as Dropout, however it drops
  entire 2D feature maps instead of individual elements. If adjacent pixels
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout2D will help promote independence
  between feature maps and should be used instead.

  Arguments:
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

  This version performs the same function as Dropout, however it drops
  entire 3D feature maps instead of individual elements. If adjacent voxels
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout3D will help promote independence
  between feature maps and should be used instead.

  Arguments:
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

  Arguments:
    activation: Activation function, such as `tf.nn.relu`, or string name of
      built-in activation function, such as "relu".

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
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
  """Reshapes an output to a certain shape.

  Arguments:
    target_shape: Target shape. Tuple of integers,
      does not include the samples dimension (batch size).

  Input shape:
    Arbitrary, although all dimensions in the input shaped must be fixed.
    Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    `(batch_size,) + target_shape`

  Example:

  ```python
  # as first layer in a Sequential model
  model = Sequential()
  model.add(Reshape((3, 4), input_shape=(12,)))
  # now: model.output_shape == (None, 3, 4)
  # note: `None` is the batch dimension

  # as intermediate layer in a Sequential model
  model.add(Reshape((6, 2)))
  # now: model.output_shape == (None, 6, 2)

  # also supports shape inference using `-1` as dimension
  model.add(Reshape((-1, 2, 2)))
  # now: model.output_shape == (None, 3, 2, 2)
  ```
  """

  def __init__(self, target_shape, **kwargs):
    super(Reshape, self).__init__(**kwargs)
    self.target_shape = tuple(target_shape)

  def _fix_unknown_dimension(self, input_shape, output_shape):
    """Find and replace a missing dimension in an output shape.

    This is a near direct port of the internal Numpy function
    `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

    Arguments:
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
    msg = 'total size of new array must be unchanged'

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
    return array_ops.reshape(inputs,
                             (array_ops.shape(inputs)[0],) + self.target_shape)

  def get_config(self):
    config = {'target_shape': self.target_shape}
    base_config = super(Reshape, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.Permute')
class Permute(Layer):
  """Permutes the dimensions of the input according to a given pattern.

  Useful for e.g. connecting RNNs and convnets together.

  Example:

  ```python
  model = Sequential()
  model.add(Permute((2, 1), input_shape=(10, 64)))
  # now: model.output_shape == (None, 64, 10)
  # note: `None` is the batch dimension
  ```

  Arguments:
    dims: Tuple of integers. Permutation pattern, does not include the
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

  If inputs are shaped `(batch,)` without a channel dimension, then flattening
  adds an extra channel dimension and output shapes are `(batch, 1)`.

  Arguments:
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

  ```python
  model = Sequential()
  model.add(Convolution2D(64, 3, 3,
                          border_mode='same',
                          input_shape=(3, 32, 32)))
  # now: model.output_shape == (None, 64, 32, 32)

  model.add(Flatten())
  # now: model.output_shape == (None, 65536)
  ```
  """

  def __init__(self, data_format=None, **kwargs):
    super(Flatten, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(min_ndim=1)

  def call(self, inputs):
    if (self.data_format == 'channels_first'
        and K.ndim(inputs) is not None and K.ndim(inputs) > 1):
      permutation = [0]
      permutation.extend([i for i in
                          range(2, K.ndim(inputs))])
      permutation.append(1)
      inputs = array_ops.transpose(inputs, perm=permutation)

    outputs = array_ops.reshape(
        inputs, (tensor_shape.dimension_value(inputs.shape[0]) or
                 array_ops.shape(inputs)[0], -1))
    if not context.executing_eagerly():
      outputs.set_shape(self.compute_output_shape(inputs.shape))
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if not input_shape:
      output_shape = tensor_shape.TensorShape([1])
    output_shape = [input_shape[0]]
    if all(input_shape[1:]):
      output_shape += [np.prod(input_shape[1:])]
    else:
      output_shape += [None]
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = {'data_format': self.data_format}
    base_config = super(Flatten, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


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

  Arguments:
    n: Integer, repetition factor.

  Input shape:
    2D tensor of shape `(num_samples, features)`.

  Output shape:
    3D tensor of shape `(num_samples, n, features)`.
  """

  def __init__(self, n, **kwargs):
    super(RepeatVector, self).__init__(**kwargs)
    self.n = n
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

  The `Lambda` layer exists so that arbitrary TensorFlow functions
  can be used when constructing `Sequential` and Functional API
  models. `Lambda` layers are best suited for simple operations or
  quick experimentation. For more advanced use cases, subclassing
  `keras.layers.Layer` is preferred. One reason for this is that
  when saving a Model, `Lambda` layers are saved by serializing the
  Python bytecode, whereas subclassed Layers are saved via overriding
  their `get_config` method and are thus more portable. Models that rely
  on subclassed Layers are also often easier to visualize and reason
  about.

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

  Variables can be created within a `Lambda` layer. Like with
  other layers, these variables will be created only once and reused
  if the `Lambda` layer is called on new inputs. If creating more
  than one variable in a given `Lambda` instance, be sure to use
  a different name for each variable. Note that calling sublayers
  from within a `Lambda` is not supported.

  Example of variable creation:

  ```python
  def linear_transform(x):
    v1 = tf.Variable(1., name='multiplier')
    v2 = tf.Variable(0., name='bias')
    return x*v1 + v2

  linear_layer = Lambda(linear_transform)
  model.add(linear_layer)
  model.add(keras.layers.Dense(10, activation='relu'))
  model.add(linear_layer)  # Reuses existing Variables
  ```

  Note that creating two instances of `Lambda` using the same function
  will *not* share Variables between the two instances. Each instance of
  `Lambda` will create and manage its own weights.

  Arguments:
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
      returned as output mask regardless what the input is.
    arguments: Optional dictionary of keyword arguments to be passed to the
      function.
  Input shape: Arbitrary. Use the keyword argument input_shape (tuple of
    integers, does not include the samples axis) when using this layer as the
    first layer in a model.
  Output shape: Specified by `output_shape` argument
  """

  def __init__(self, function, output_shape=None, mask=None, arguments=None,
               **kwargs):
    super(Lambda, self).__init__(**kwargs)
    self.function = function
    self.arguments = arguments if arguments else {}
    if mask is not None:
      self.supports_masking = True
    self.mask = mask
    self._output_shape = output_shape
    self._variable_dict = {}
    # These attributes are inherited from `Layer`.
    self._trainable_weights = []
    self._non_trainable_weights = []

    function_args = tf_inspect.getfullargspec(self.function).args
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
    arguments = self.arguments
    if self._fn_expects_mask_arg:
      arguments['mask'] = mask
    if self._fn_expects_training_arg:
      arguments['training'] = training
    with variable_scope.variable_creator_scope(self._variable_creator):
      return self.function(inputs, **arguments)

  def _variable_creator(self, next_creator, **kwargs):
    name = kwargs['name']
    if name in self._variable_dict:
      return self._variable_dict[name]
    var = next_creator(**kwargs)
    self._variable_dict[name] = var
    if var.trainable:
      self._trainable_weights.append(var)
    else:
      self._non_trainable_weights.append(var)
    K.track_variable(var)
    return var

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
    globs = globals()
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
  (only applicable if `use_bias` is `True`).

  Note: If the input to the layer has a rank greater than 2, then
  it is flattened prior to the initial dot product with `kernel`.

  Example:

  ```python
  # as first layer in a sequential model:
  model = Sequential()
  model.add(Dense(32, input_shape=(16,)))
  # now the model will take as input arrays of shape (*, 16)
  # and output arrays of shape (*, 32)

  # after the first layer, you don't need to specify
  # the size of the input anymore:
  model.add(Dense(32))
  ```

  Arguments:
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
      the output of the layer (its "activation")..
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
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(Dense, self).__init__(
        activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
    self.units = int(units)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = InputSpec(min_ndim=2)

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))
    input_shape = tensor_shape.TensorShape(input_shape)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})
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
    inputs = ops.convert_to_tensor(inputs)
    rank = common_shapes.rank(inputs)
    if rank > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.units]
        outputs.set_shape(output_shape)
    else:
      # Cast the inputs to self.dtype, which is the variable dtype. We do not
      # cast if `should_cast_variables` is True, as in that case the variable
      # will be automatically casted to inputs.dtype.
      if not self._mixed_precision_policy.should_cast_variables:
        inputs = math_ops.cast(inputs, self.dtype)
      outputs = gen_math_ops.mat_mul(inputs, self.kernel)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
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
    }
    base_config = super(Dense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.ActivityRegularization')
class ActivityRegularization(Layer):
  """Layer that applies an update to the cost function based input activity.

  Arguments:
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
