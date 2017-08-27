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
import types as python_types

import numpy as np

from tensorflow.contrib.keras.python.keras import activations
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import constraints
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.engine import Layer
from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.contrib.keras.python.keras.utils.generic_utils import func_dump
from tensorflow.contrib.keras.python.keras.utils.generic_utils import func_load
from tensorflow.contrib.keras.python.keras.utils.generic_utils import has_arg
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as tf_core_layers


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
  to be fed to a LSTM layer.
  You want to mask timestep #3 and #5 because you lack data for
  these timesteps. You can:

      - set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
      - insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

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

  def compute_mask(self, inputs, mask=None):
    return K.any(K.not_equal(inputs, self.mask_value), axis=-1)

  def call(self, inputs):
    boolean_mask = K.any(
        K.not_equal(inputs, self.mask_value), axis=-1, keepdims=True)
    return inputs * K.cast(boolean_mask, K.floatx())

  def get_config(self):
    config = {'mask_value': self.mask_value}
    base_config = super(Masking, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Dropout(tf_core_layers.Dropout, Layer):
  """Applies Dropout to the input.

  Dropout consists in randomly setting
  a fraction `rate` of input units to 0 at each update during training time,
  which helps prevent overfitting.

  Arguments:
      rate: float between 0 and 1. Fraction of the input units to drop.
      noise_shape: 1D integer tensor representing the shape of the
          binary dropout mask that will be multiplied with the input.
          For instance, if your inputs have shape
          `(batch_size, timesteps, features)` and
          you want the dropout mask to be the same for all timesteps,
          you can use `noise_shape=(batch_size, 1, features)`.
      seed: A Python integer to use as random seed.
  """

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    self.supports_masking = True
    # Inheritance call order:
    # 1) tf.layers.Dropout, 2) keras.layers.Layer, 3) tf.layers.Layer
    super(Dropout, self).__init__(rate=rate, noise_shape=noise_shape, seed=seed, **kwargs)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()
    output = super(Dropout, self).call(inputs, training=training)
    if training is K.learning_phase():
      output._uses_learning_phase = True  # pylint: disable=protected-access
    return output

  def get_config(self):
    config = {'rate': self.rate}
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


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
      rate: float between 0 and 1. Fraction of the input units to drop.

  Input shape:
      3D tensor with shape:
      `(samples, timesteps, channels)`

  Output shape:
      Same as input

  References:
      - [Efficient Object Localization Using Convolutional
        Networks](https://arxiv.org/abs/1411.4280)
  """

  def __init__(self, rate, **kwargs):
    super(SpatialDropout1D, self).__init__(rate, **kwargs)
    self.input_spec = InputSpec(ndim=3)

  def _get_noise_shape(self, inputs):
    input_shape = K.shape(inputs)
    noise_shape = (input_shape[0], 1, input_shape[2])
    return noise_shape


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
      rate: float between 0 and 1. Fraction of the input units to drop.
      data_format: 'channels_first' or 'channels_last'.
          In 'channels_first' mode, the channels dimension
          (the depth) is at index 1,
          in 'channels_last' mode is it at index 3.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
      Same as input

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
    input_shape = K.shape(inputs)
    if self.data_format == 'channels_first':
      return (input_shape[0], input_shape[1], 1, 1)
    elif self.data_format == 'channels_last':
      return (input_shape[0], 1, 1, input_shape[3])


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
      rate: float between 0 and 1. Fraction of the input units to drop.
      data_format: 'channels_first' or 'channels_last'.
          In 'channels_first' mode, the channels dimension (the depth)
          is at index 1, in 'channels_last' mode is it at index 4.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      5D tensor with shape:
      `(samples, channels, dim1, dim2, dim3)` if data_format='channels_first'
      or 5D tensor with shape:
      `(samples, dim1, dim2, dim3, channels)` if data_format='channels_last'.

  Output shape:
      Same as input

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
    input_shape = K.shape(inputs)
    if self.data_format == 'channels_first':
      return (input_shape[0], input_shape[1], 1, 1, 1)
    elif self.data_format == 'channels_last':
      return (input_shape[0], 1, 1, 1, input_shape[4])


class Activation(Layer):
  """Applies an activation function to an output.

  Arguments:
      activation: name of activation function to use
          or alternatively, a Theano or TensorFlow operation.

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

  def get_config(self):
    config = {'activation': activations.serialize(self.activation)}
    base_config = super(Activation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Reshape(Layer):
  """Reshapes an output to a certain shape.

  Arguments:
      target_shape: target shape. Tuple of integers,
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
        input_shape: shape of array being reshaped
        output_shape: desired shape of the array with at most
            a single -1 which indicates a dimension that should be
            derived from the input shape.

    Returns:
        The new output shape with a -1 replaced with its computed value.

        Raises a ValueError if the total array size of the output_shape is
        different then the input_shape, or more then one unknown dimension
        is specified.

    Raises:
        ValueError: in case of invalid values
            for `input_shape` or `input_shape`.
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

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = [input_shape[0]]
    output_shape += self._fix_unknown_dimension(input_shape[1:],
                                                self.target_shape)
    return tensor_shape.TensorShape(output_shape)

  def call(self, inputs):
    # In case the target shape is not fully defined,
    # we need access to the shape of x.
    target_shape = self.target_shape
    if -1 in target_shape:
      # target shape not fully defined
      target_shape = self._compute_output_shape(inputs.get_shape())
      target_shape = target_shape.as_list()[1:]
    return K.reshape(inputs, (-1,) + tuple(target_shape))

  def get_config(self):
    config = {'target_shape': self.target_shape}
    base_config = super(Reshape, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


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
          For instance, `(2, 1)` permutes the first and second dimension
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
    self.input_spec = InputSpec(ndim=len(self.dims) + 1)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = copy.copy(input_shape)
    for i, dim in enumerate(self.dims):
      target_dim = input_shape[dim]
      output_shape[i + 1] = target_dim
    return tensor_shape.TensorShape(output_shape)

  def call(self, inputs):
    return K.permute_dimensions(inputs, (0,) + self.dims)

  def get_config(self):
    config = {'dims': self.dims}
    base_config = super(Permute, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Flatten(Layer):
  """Flattens the input. Does not affect the batch size.

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

  def __init__(self, **kwargs):
    super(Flatten, self).__init__(**kwargs)
    self.input_spec = InputSpec(min_ndim=3)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if not all(input_shape[1:]):
      raise ValueError('The shape of the input to "Flatten" '
                       'is not fully defined '
                       '(got ' + str(input_shape[1:]) + '. '
                       'Make sure to pass a complete "input_shape" '
                       'or "batch_input_shape" argument to the first '
                       'layer in your model.')
    return tensor_shape.TensorShape([input_shape[0], np.prod(input_shape[1:])])

  def call(self, inputs):
    outputs = K.batch_flatten(inputs)
    outputs.set_shape(self._compute_output_shape(inputs.get_shape()))
    return outputs


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
      n: integer, repetition factor.

  Input shape:
      2D tensor of shape `(num_samples, features)`.

  Output shape:
      3D tensor of shape `(num_samples, n, features)`.
  """

  def __init__(self, n, **kwargs):
    super(RepeatVector, self).__init__(**kwargs)
    self.n = n
    self.input_spec = InputSpec(ndim=2)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape([input_shape[0], self.n, input_shape[1]])

  def call(self, inputs):
    return K.repeat(inputs, self.n)

  def get_config(self):
    config = {'n': self.n}
    base_config = super(RepeatVector, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Lambda(Layer):
  """Wraps arbitrary expression as a `Layer` object.

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

  Arguments:
      function: The function to be evaluated.
          Takes input tensor as first argument.
      arguments: optional dictionary of keyword arguments to be passed
          to the function.

  Input shape:
      Arbitrary. Use the keyword argument input_shape
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Specified by `output_shape` argument
      (or auto-inferred when using TensorFlow).
  """

  def __init__(self, function, mask=None, arguments=None, **kwargs):
    super(Lambda, self).__init__(**kwargs)
    self.function = function
    self.arguments = arguments if arguments else {}
    if mask is not None:
      self.supports_masking = True
    self.mask = mask

  def call(self, inputs, mask=None):
    arguments = self.arguments
    if has_arg(self.function, 'mask'):
      arguments['mask'] = mask
    return self.function(inputs, **arguments)

  def compute_mask(self, inputs, mask=None):
    if callable(self.mask):
      return self.mask(inputs, mask)
    return self.mask

  def get_config(self):
    if isinstance(self.function, python_types.LambdaType):
      function = func_dump(self.function)
      function_type = 'lambda'
    else:
      function = self.function.__name__
      function_type = 'function'

    config = {
        'function': function,
        'function_type': function_type,
        'arguments': self.arguments
    }
    base_config = super(Lambda, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    globs = globals()
    if custom_objects:
      globs = dict(list(globs.items()) + list(custom_objects.items()))
    function_type = config.pop('function_type')
    if function_type == 'function':
      # Simple lookup in custom objects
      function = deserialize_keras_object(
          config['function'],
          custom_objects=custom_objects,
          printable_module_name='function in Lambda layer')
    elif function_type == 'lambda':
      # Unsafe deserialization from bytecode
      function = func_load(config['function'], globs=globs)
    else:
      raise TypeError('Unknown function type:', function_type)

    # If arguments were numpy array, they have been saved as
    # list. We need to recover the ndarray
    if 'arguments' in config:
      for key in config['arguments']:
        if isinstance(config['arguments'][key], dict):
          arg_dict = config['arguments'][key]
          if 'type' in arg_dict and arg_dict['type'] == 'ndarray':
            # Overwrite the argument with its numpy translation
            config['arguments'][key] = np.array(arg_dict['value'])

    config['function'] = function
    return cls(**config)


class Dense(tf_core_layers.Dense, Layer):
  """Just your regular densely-connected NN layer.

  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`).

  Note: if the input to the layer has a rank greater than 2, then
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
      nD tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
      nD tensor with shape: `(batch_size, ..., units)`.
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

    # Inheritance call order:
    # 1) tf.layers.Dense, 2) keras.layers.Layer, 3) tf.layers.Layer
    super(Dense, self).__init__(
        units,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)
    self.supports_masking = True

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
    super(ActivityRegularization, self).__init__(**kwargs)
    self.supports_masking = True
    self.l1 = l1
    self.l2 = l2
    self.activity_regularizer = regularizers.L1L2(l1=l1, l2=l2)

  def get_config(self):
    config = {'l1': self.l1, 'l2': self.l2}
    base_config = super(ActivityRegularization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
