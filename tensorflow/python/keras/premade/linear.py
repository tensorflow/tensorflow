# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Built-in linear model classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.experimental.LinearModel')
class LinearModel(training.Model):
  r"""Linear Model for regression and classification problems.

  This model approximates the following function:
  $$y = \beta + \sum_{i=1}^{N} w_{i} * x_{i}$$
  where $$\beta$$ is the bias and $$w_{i}$$ is the weight for each feature.

  Example:

  ```python
  model = LinearModel()
  model.compile(optimizer='sgd', loss='mse')
  model.fit(x, y, epochs)
  ```

  This model accepts sparse float inputs as well:

  Example:
  ```python
  model = LinearModel()
  opt = tf.keras.optimizers.Adam()
  loss_fn = tf.keras.losses.MeanSquaredError()
  with tf.GradientTape() as tape:
    output = model(sparse_input)
    loss = tf.reduce_mean(loss_fn(target, output))
  grads = tape.gradient(loss, model.weights)
  opt.apply_gradients(zip(grads, model.weights))
  ```

  """

  def __init__(self,
               units=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """Create a Linear Model.

    Args:
      units: Positive integer, output dimension without the batch size.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied.
      use_bias: whether to calculate the bias/intercept for this model. If set
        to False, no bias/intercept will be used in calculations, e.g., the data
        is already centered.
      kernel_initializer: Initializer for the `kernel` weights matrices.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: regularizer for kernel vectors.
      bias_regularizer: regularizer for bias vector.
      **kwargs: The keyword arguments that are passed on to BaseLayer.__init__.
    """

    self.units = units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    super(LinearModel, self).__init__(**kwargs)
    base_layer._keras_model_gauge.get_cell('Linear').set(True)  # pylint: disable=protected-access

  def build(self, input_shape):
    self.dense_layers = []
    if isinstance(input_shape, list):
      for shape in input_shape:
        layer = core.Dense(
            units=self.units,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            input_shape=shape)
        self.dense_layers.append(layer)
    else:
      layer = core.Dense(
          units=self.units,
          use_bias=False,
          kernel_initializer=self.kernel_initializer,
          kernel_regularizer=self.kernel_regularizer,
          input_shape=input_shape)
      self.dense_layers.append(layer)

    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=self.units,
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None

  def call(self, inputs):
    if not isinstance(inputs, (tuple, list)):
      inputs = [inputs]
    if len(inputs) != len(self.dense_layers):
      raise ValueError('Expected {} inputs, but got {} inputs'.format(
          len(self.dense_layers), len(inputs)))
    result = None
    for inp, layer in zip(inputs, self.dense_layers):
      output = layer(inp)
      if result is None:
        result = output
      else:
        result += output

    if self.use_bias:
      result = nn.bias_add(result, self.bias)
    if self.activation is not None:
      return self.activation(result)  # pylint: disable=not-callable
    return result

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
    }
    base_config = base_layer.Layer.get_config(self)
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    del custom_objects
    return cls(**config)
