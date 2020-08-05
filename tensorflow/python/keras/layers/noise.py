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
"""Layers that operate regularization via the addition of noise."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers


@keras_export('keras.layers.GaussianNoise')
class GaussianNoise(Layer):
  """Apply additive zero-centered Gaussian noise.

  This is useful to mitigate overfitting
  (you could see it as a form of random data augmentation).
  Gaussian Noise (GS) is a natural choice as corruption process
  for real valued inputs.

  As it is a regularization layer, it is only active at training time.

  Arguments:
    stddev: Float, standard deviation of the noise distribution.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding noise) or in inference mode (doing nothing).

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, stddev, **kwargs):
    super(GaussianNoise, self).__init__(**kwargs)
    self.supports_masking = True
    self.stddev = stddev

  def call(self, inputs, training=None):

    def noised():
      return inputs + K.random_normal(
          shape=array_ops.shape(inputs),
          mean=0.,
          stddev=self.stddev,
          dtype=inputs.dtype)

    return K.in_train_phase(noised, inputs, training=training)

  def get_config(self):
    config = {'stddev': self.stddev}
    base_config = super(GaussianNoise, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape


@keras_export('keras.layers.GaussianDropout')
class GaussianDropout(Layer):
  """Apply multiplicative 1-centered Gaussian noise.

  As it is a regularization layer, it is only active at training time.

  Arguments:
    rate: Float, drop probability (as with `Dropout`).
      The multiplicative noise will have
      standard deviation `sqrt(rate / (1 - rate))`.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, rate, **kwargs):
    super(GaussianDropout, self).__init__(**kwargs)
    self.supports_masking = True
    self.rate = rate

  def call(self, inputs, training=None):
    if 0 < self.rate < 1:

      def noised():
        stddev = np.sqrt(self.rate / (1.0 - self.rate))
        return inputs * K.random_normal(
            shape=array_ops.shape(inputs),
            mean=1.0,
            stddev=stddev,
            dtype=inputs.dtype)

      return K.in_train_phase(noised, inputs, training=training)
    return inputs

  def get_config(self):
    config = {'rate': self.rate}
    base_config = super(GaussianDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape


@keras_export('keras.layers.AlphaDropout')
class AlphaDropout(Layer):
  """Applies Alpha Dropout to the input.

  Alpha Dropout is a `Dropout` that keeps mean and variance of inputs
  to their original values, in order to ensure the self-normalizing property
  even after this dropout.
  Alpha Dropout fits well to Scaled Exponential Linear Units
  by randomly setting activations to the negative saturation value.

  Arguments:
    rate: float, drop probability (as with `Dropout`).
      The multiplicative noise will have
      standard deviation `sqrt(rate / (1 - rate))`.
    seed: A Python integer to use as random seed.

  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(AlphaDropout, self).__init__(**kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, inputs):
    return self.noise_shape if self.noise_shape else array_ops.shape(inputs)

  def call(self, inputs, training=None):
    if 0. < self.rate < 1.:
      noise_shape = self._get_noise_shape(inputs)

      def dropped_inputs(inputs=inputs, rate=self.rate, seed=self.seed):  # pylint: disable=missing-docstring
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale

        kept_idx = math_ops.greater_equal(
            K.random_uniform(noise_shape, seed=seed), rate)
        kept_idx = math_ops.cast(kept_idx, inputs.dtype)

        # Get affine transformation params
        a = ((1 - rate) * (1 + rate * alpha_p**2))**-0.5
        b = -a * alpha_p * rate

        # Apply mask
        x = inputs * kept_idx + alpha_p * (1 - kept_idx)

        # Do affine transformation
        return a * x + b

      return K.in_train_phase(dropped_inputs, inputs, training=training)
    return inputs

  def get_config(self):
    config = {'rate': self.rate}
    base_config = super(AlphaDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape


@keras_export('keras.layers.NoisyDense')
class NoisyDense(Dense):
  """Densely-connected NN layer with additive zero-centered Gaussian noise (NoisyNet).

  `NoisyDense` implements the operation:
  `output = activation(dot(input, kernel + kernel_sigma * kernel_epsilon) + bias + bias_sigma * bias_epsilon)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a base weights matrix
  created by the layer, `kernel_sigma` is a noise weights matrix
  created by the layer, `bias` is a base bias vector created by the layer, 
  `bias_sigma` is a noise bias vector created by the layer,
  'kernel_epsilon' and 'bias_epsilon' are noise random variables.
  (biases are only applicable if `use_bias` is `True`)
  
  There are implemented both variants: 
    1. Independent Gaussian noise                                    
    2. Factorised Gaussian noise.

  We can choose between that by 'use_factorised' parameter.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    sigma0: Float, initial sigma parameter (uses only if use_factorised=True)
    use_factorised: Boolean, whether the layer uses independent or factorised Gaussian noise
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    kernel_sigma_regularizer: Regularizer function applied to
      the `kernel_sigma` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    bias_sigma_regularizer: Regularizer function applied to the bias_sigma vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    kernel_sigma_constraint: Constraint function applied to
      the `kernel_sigma` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    bias_sigma_constraint: Constraint function applied to the bias_sigma vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.

  Reference:
    - [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
  """

  def __init__(self,
               units,
               sigma0=0.5,
               use_factorised=True,
               activation=None, 
               use_bias=True, 
               kernel_initializer='glorot_uniform', 
               bias_initializer='zeros',
               kernel_regularizer=None, 
               kernel_sigma_regularizer=None,
               bias_regularizer=None, 
               bias_sigma_regularizer=None,
               activity_regularizer=None, 
               kernel_constraint=None, 
               kernel_sigma_constraint=None,
               bias_constraint=None, 
               bias_sigma_constraint=None,
               **kwargs):
    super(NoisyDense, self).__init__(units=units, 
                                     activation=activation, 
                                     use_bias=use_bias, 
                                     kernel_initializer=kernel_initializer, 
                                     bias_initializer=bias_initializer, 
                                     kernel_regularizer=kernel_regularizer, 
                                     bias_regularizer=bias_regularizer, 
                                     activity_regularizer=activity_regularizer, 
                                     kernel_constraint=kernel_constraint, 
                                     bias_constraint=bias_constraint, 
                                     **kwargs)

    self.sigma0 = sigma0
    self.use_factorised = use_factorised
    
    self.kernel_sigma_regularizer = regularizers.get(kernel_sigma_regularizer)
    self.bias_sigma_regularizer = regularizers.get(bias_sigma_regularizer)
    self.kernel_sigma_constraint = constraints.get(kernel_sigma_constraint)
    self.bias_sigma_constraint = constraints.get(bias_sigma_constraint)

  def build(self, input_shape):
    super(NoisyDense, self).build(input_shape)

    # use factorising Gaussian variables
    if self.use_factorised:
      sigma_init = self.sigma0 / np.sqrt(self.kernel.shape[0])
    # use independent Gaussian variables  
    else:
      sigma_init = 0.017
    
    # create sigma weights
    self.kernel_sigma = self.add_weight(
        'kernel_sigma',
        shape=self.kernel.shape,
        initializer=initializers.initializers_v2.Constant(value=sigma_init),
        regularizer=self.kernel_sigma_regularizer,
        constraint=self.kernel_sigma_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias_sigma = self.add_weight(
          'bias_sigma',
          shape=self.bias.shape,
          initializer=initializers.initializers_v2.Constant(value=sigma_init),
          regularizer=self.bias_sigma_regularizer,
          constraint=self.bias_sigma_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias_sigma = None
    
    # create noise variables
    self.kernel_epsilon = self.add_weight(
          name='kernel_epsilon',
          shape=self.kernel.shape,
          dtype=self.dtype,
          initializer='zeros',
          trainable=False)
    if self.use_bias:
      self.bias_epsilon = self.add_weight(
            name='bias_epsilon',
            shape=self.bias.shape,
            dtype=self.dtype,
            initializer='zeros',
            trainable=False)
    else:
      self.bias_epsilon = None

    # init epsilon parameters
    self.reset_noise()

  def call(self, inputs):
    kernel = math_ops.add(self.kernel, math_ops.mul(self.kernel_sigma, self.kernel_epsilon))
    bias = self.bias
    if bias is not None:
      bias = math_ops.add(self.bias, math_ops.mul(self.bias_sigma, self.bias_epsilon))

    return core_ops.dense(
        inputs,
        kernel,
        bias,
        self.activation,
        dtype=self._compute_dtype_object)

  def get_config(self):
    config = super(NoisyDense, self).get_config()
    config.update({
        'sigma0':
            self.sigma0,
        'use_factorised':
            self.use_factorised,
        'kernel_sigma_regularizer':
            regularizers.serialize(self.kernel_sigma_regularizer),
        'bias_sigma_regularizer':
            regularizers.serialize(self.bias_sigma_regularizer),
        'kernel_sigma_constraint':
            constraints.serialize(self.kernel_sigma_constraint),
        'bias_sigma_constraint':
            constraints.serialize(self.bias_sigma_constraint)
    })
    return config
  
  def _scale_noise(self, size):
    x = K.random_normal(shape=size,
                        mean=0.0,
                        stddev=1.0,
                        dtype=self.dtype)
    return math_ops.mul(math_ops.sign(x), math_ops.sqrt(math_ops.abs(x)))
    
  def reset_noise(self):
    if self.use_factorised:
      in_eps = self._scale_noise((self.kernel_epsilon.shape[0], 1))
      out_eps = self._scale_noise((1, self.units))
      w_eps = math_ops.matmul(in_eps, out_eps)
      b_eps = out_eps[0]
    else:
      # generate independent variables
      w_eps = K.random_normal(shape=self.kernel_epsilon.shape,
                              mean=0.0,
                              stddev=1.0,
                              dtype=self.dtype)
      b_eps = K.random_normal(shape=self.bias_epsilon.shape,
                              mean=0.0,
                              stddev=1.0,
                              dtype=self.dtype)

      self.kernel_epsilon.assign(w_eps)
      self.bias_epsilon.assign(b_eps)
    
  def remove_noise(self):
    self.kernel_epsilon.assign(array_ops.zeros(self.kernel_epsilon.shape, self.dtype))
    self.bias_epsilon.assign(array_ops.zeros(self.bias_epsilon.shape, self.dtype))