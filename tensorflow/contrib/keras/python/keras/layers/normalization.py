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
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.engine import Layer
from tensorflow.python.framework import tensor_shape


class BatchNormalization(Layer):
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
    super(BatchNormalization, self).__init__(**kwargs)
    self.supports_masking = True
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.moving_mean_initializer = initializers.get(moving_mean_initializer)
    self.moving_variance_initializer = initializers.get(
        moving_variance_initializer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.beta_constraint = constraints.get(beta_constraint)
    self.gamma_constraint = constraints.get(gamma_constraint)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    dim = input_shape[self.axis]
    if dim is None:
      raise ValueError('Axis ' + str(self.axis) + ' of '
                       'input tensor should have a defined dimension '
                       'but the layer received an input with shape ' + str(
                           input_shape) + '.')
    self.input_spec = InputSpec(ndim=len(input_shape), axes={self.axis: dim})
    shape = (dim,)

    if self.scale:
      self.gamma = self.add_weight(
          shape,
          name='gamma',
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint)
    else:
      self.gamma = None
    if self.center:
      self.beta = self.add_weight(
          shape,
          name='beta',
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint)
    else:
      self.beta = None
    self.moving_mean = self.add_weight(
        shape,
        name='moving_mean',
        initializer=self.moving_mean_initializer,
        trainable=False)
    self.moving_variance = self.add_weight(
        shape,
        name='moving_variance',
        initializer=self.moving_variance_initializer,
        trainable=False)
    self.built = True

  def call(self, inputs, training=None):
    input_shape = inputs.get_shape().as_list()
    # Prepare broadcasting shape.
    ndim = len(input_shape)
    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[self.axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[self.axis] = input_shape[self.axis]

    # Determines whether broadcasting is needed.
    needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

    normed, mean, variance = K.normalize_batch_in_training(
        inputs, self.gamma, self.beta, reduction_axes, epsilon=self.epsilon)

    if training in {0, False}:
      return normed
    else:
      self.add_update([
          K.moving_average_update(self.moving_mean, mean, self.momentum),
          K.moving_average_update(self.moving_variance, variance, self.momentum)
      ], inputs)

      def normalize_inference():
        if needs_broadcasting:
          # In this case we must explictly broadcast all parameters.
          broadcast_moving_mean = K.reshape(self.moving_mean, broadcast_shape)
          broadcast_moving_variance = K.reshape(self.moving_variance,
                                                broadcast_shape)
          if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
          else:
            broadcast_beta = None
          if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
          else:
            broadcast_gamma = None
          return K.batch_normalization(
              inputs,
              broadcast_moving_mean,
              broadcast_moving_variance,
              broadcast_beta,
              broadcast_gamma,
              epsilon=self.epsilon)
        else:
          return K.batch_normalization(
              inputs,
              self.moving_mean,
              self.moving_variance,
              self.beta,
              self.gamma,
              epsilon=self.epsilon)

    # Pick the normalized form corresponding to the training phase.
    return K.in_train_phase(normed, normalize_inference, training=training)

  def get_config(self):
    config = {
        'axis':
            self.axis,
        'momentum':
            self.momentum,
        'epsilon':
            self.epsilon,
        'center':
            self.center,
        'scale':
            self.scale,
        'beta_initializer':
            initializers.serialize(self.beta_initializer),
        'gamma_initializer':
            initializers.serialize(self.gamma_initializer),
        'moving_mean_initializer':
            initializers.serialize(self.moving_mean_initializer),
        'moving_variance_initializer':
            initializers.serialize(self.moving_variance_initializer),
        'beta_regularizer':
            regularizers.serialize(self.beta_regularizer),
        'gamma_regularizer':
            regularizers.serialize(self.gamma_regularizer),
        'beta_constraint':
            constraints.serialize(self.beta_constraint),
        'gamma_constraint':
            constraints.serialize(self.gamma_constraint)
    }
    base_config = super(BatchNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
