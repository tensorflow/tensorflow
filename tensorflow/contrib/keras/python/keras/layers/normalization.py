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
"""Normalization layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import constraints
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.engine import Layer
from tensorflow.python.layers import normalization as tf_normalization_layers


class BatchNormalization(tf_normalization_layers.BatchNormalization, Layer):
  """Batch normalization layer (Ioffe and Szegedy, 2014).

  Normalize the activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.

  Arguments:
      axis: Integer, the axis that should be normalized
          (typically the features axis).
          For instance, after a `Conv2D` layer with
          `data_format="channels_first"`,
          set `axis=1` in `BatchNormalization`.
      momentum: Momentum for the moving average.
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor.
          If False, `beta` is ignored.
      scale: If True, multiply by `gamma`.
          If False, `gamma` is not used.
          When the next layer is linear (also e.g. `nn.relu`),
          this can be disabled since the scaling
          will be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      moving_mean_initializer: Initializer for the moving mean.
      moving_variance_initializer: Initializer for the moving variance.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      beta_constraint: Optional constraint for the beta weight.
      gamma_constraint: Optional constraint for the gamma weight.

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as input.

  References:
      - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               **kwargs):
    self.supports_masking = True
    super(BatchNormalization, self).__init__(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=initializers.get(beta_initializer),
        gamma_initializer=initializers.get(gamma_initializer),
        moving_mean_initializer=initializers.get(moving_mean_initializer),
        moving_variance_initializer=initializers.get(
            moving_variance_initializer),
        beta_regularizer=regularizers.get(beta_regularizer),
        gamma_regularizer=regularizers.get(gamma_regularizer),
        beta_constraint=constraints.get(beta_constraint),
        gamma_constraint=constraints.get(gamma_constraint),
        **kwargs
    )

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()
    output = super(BatchNormalization, self).call(inputs, training=training)
    if training is K.learning_phase():
      output._uses_learning_phase = True  # pylint: disable=protected-access
    return output

  def get_config(self):
    config = {
        'axis': self.axis,
        'momentum': self.momentum,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': initializers.serialize(self.beta_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'moving_mean_initializer':
            initializers.serialize(self.moving_mean_initializer),
        'moving_variance_initializer':
            initializers.serialize(self.moving_variance_initializer),
        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
        'beta_constraint': constraints.serialize(self.beta_constraint),
        'gamma_constraint': constraints.serialize(self.gamma_constraint)
    }
    base_config = super(BatchNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
