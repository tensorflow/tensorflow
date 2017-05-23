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
"""Layers for regularization models via the addition of noise.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.engine import Layer


class GaussianNoise(Layer):
  """Apply additive zero-centered Gaussian noise.

  This is useful to mitigate overfitting
  (you could see it as a form of random data augmentation).
  Gaussian Noise (GS) is a natural choice as corruption process
  for real valued inputs.

  As it is a regularization layer, it is only active at training time.

  Arguments:
      stddev: float, standard deviation of the noise distribution.

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
          shape=K.shape(inputs), mean=0., stddev=self.stddev)

    return K.in_train_phase(noised, inputs, training=training)

  def get_config(self):
    config = {'stddev': self.stddev}
    base_config = super(GaussianNoise, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GaussianDropout(Layer):
  """Apply multiplicative 1-centered Gaussian noise.

  As it is a regularization layer, it is only active at training time.

  Arguments:
      rate: float, drop probability (as with `Dropout`).
          The multiplicative noise will have
          standard deviation `sqrt(rate / (1 - rate))`.

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as input.

  References:
      - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        Srivastava, Hinton, et al.
        2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
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
            shape=K.shape(inputs), mean=1.0, stddev=stddev)

      return K.in_train_phase(noised, inputs, training=training)
    return inputs

  def get_config(self):
    config = {'rate': self.rate}
    base_config = super(GaussianDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
