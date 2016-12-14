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
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""Contains the core layers: Dense, Dropout.

Also contains their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.layers import base
from tensorflow.python.layers import utils


class Dense(base._Layer):  # pylint: disable=protected-access
  """Densely-connected layer class.

  This layer implements the operation `outputs = activation(inputs.w + b)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `w` is a weights matrix created by the layer,
  and `b` is a bias vector created by the layer (only if `use_bias` is `True`).

  Note: if the input to the layer has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `w`.

  Arguments:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    weights_initializer: Initializer function for the weight matrix.
    bias_initializer: Initializer function for the bias.
    weights_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    weights_initializer: Initializer instance (or name) for the weight matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    weights_regularizer: Regularizer instance for the weight matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    weights: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
  """

  def __init__(self, units,
               activation=None,
               use_bias=True,
               weights_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               weights_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(Dense, self).__init__(trainable=trainable, name=name, **kwargs)
    self.units = units
    self.activation = activation
    self.use_bias = use_bias
    self.weights_initializer = weights_initializer
    self.bias_initializer = bias_initializer
    self.weights_regularizer = weights_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape.ndims is None:
      raise ValueError('Inputs to `Dense` should have known rank.')
    if len(input_shape) < 2:
      raise ValueError('Inputs to `Dense` should have rank >= 2.')
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    # Note that we set `trainable=True` because this is a trainable
    # weight of the layer. If the layer is not trainable
    # (self.trainable = False), the variable will not be added to
    # tf.trainable_variables(), and self.trainable_weights will be empty.
    self.w = vs.get_variable('weights',
                             shape=[input_shape[-1].value, self.units],
                             initializer=self.weights_initializer,
                             regularizer=self.weights_regularizer,
                             dtype=self.dtype,
                             trainable=True)
    if self.use_bias:
      self.bias = vs.get_variable('bias',
                                  shape=[self.units,],
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  dtype=self.dtype,
                                  trainable=True)
    else:
      self.bias = None

  def call(self, inputs):
    shape = inputs.get_shape().as_list()
    input_dim = shape[-1]
    output_shape = shape[:-1] + [self.units]
    if len(output_shape) > 2:
      # Reshape the input to 2D.
      output_shape_tensors = array_ops.unpack(array_ops.shape(inputs))
      output_shape_tensors[-1] = self.units
      output_shape_tensor = array_ops.pack(output_shape_tensors)
      inputs = array_ops.reshape(inputs, [-1, input_dim])

    outputs = standard_ops.matmul(inputs, self.w)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)

    if len(output_shape) > 2:
      # Reshape the output back to the original ndim of the input.
      outputs = array_ops.reshape(outputs, output_shape_tensor)
      outputs.set_shape(output_shape)

    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs


def dense(
    inputs, units,
    activation=None,
    use_bias=True,
    weights_initializer=None,
    bias_initializer=init_ops.zeros_initializer(),
    weights_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=False):
  """Functional interface for the densely-connected layer.

  This layer implements the operation `outputs = activation(inputs.w + b)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `w` is a weights matrix created by the layer,
  and `b` is a bias vector created by the layer (only if `use_bias` is `True`).

  Note: if the `inputs` tensor has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `w`.

  Arguments:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    weights_initializer: Initializer function for the weight matrix.
    bias_initializer: Initializer function for the bias.
    weights_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.
  """
  layer = Dense(units,
                activation=activation,
                use_bias=use_bias,
                weights_initializer=weights_initializer,
                bias_initializer=bias_initializer,
                weights_regularizer=weights_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                _scope=name,
                _reuse=reuse)
  return layer.apply(inputs)


class Dropout(base._Layer):  # pylint: disable=protected-access
  """Applies Dropout to the input.

  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.

  Arguments:
    rate: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: The name of the layer (string).
  """

  def __init__(self, rate=0.5,
               noise_shape=None,
               seed=None,
               name=None,
               **kwargs):
    super(Dropout, self).__init__(name=name, **kwargs)
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed

  def call(self, inputs, training=False):
    def dropped_inputs():
      return nn.dropout(inputs, 1  - self.rate,
                        noise_shape=self.noise_shape,
                        seed=self.seed)
    return utils.smart_cond(training,
                            dropped_inputs,
                            lambda: array_ops.identity(inputs))


def dropout(inputs,
            rate=0.5,
            noise_shape=None,
            seed=None,
            training=False,
            name=None):
  """Applies Dropout to the input.

  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.

  Arguments:
    inputs: Tensor input.
    rate: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    training: Either a Python boolean, or a TensorFlow boolean scalar tensor
      (e.g. a placeholder). Whether to return the output in training mode
      (apply dropout) or in inference mode (return the input untouched).
    name: The name of the layer (string).

  Returns:
    Output tensor.
  """
  layer = Dropout(rate, noise_shape=noise_shape, seed=seed, name=name)
  return layer.apply(inputs, training=training)


# Aliases

FullyConnected = Dense
fully_connected = dense
