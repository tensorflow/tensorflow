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
        kept_idx = math_ops.cast(kept_idx, K.floatx())

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
