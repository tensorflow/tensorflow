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


class Dense(base.Layer):
  """Densely-connected layer class.

  This layer implements the operation:
  `outputs = activation(inputs.kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

  Note: if the input to the layer has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `kernel`.

  Arguments:
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Properties:
    units: Python integer, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer instance (or name) for the kernel matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    kernel_regularizer: Regularizer instance for the kernel matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    kernel_constraint: Constraint function for the kernel matrix.
    bias_constraint: Constraint function for the bias.
    kernel: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
  """

  def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(Dense, self).__init__(trainable=trainable, name=name, **kwargs)
    self.units = units
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.input_spec = base.InputSpec(min_ndim=2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = base.InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value})
    self.kernel = self.add_variable('kernel',
                                    shape=[input_shape[-1].value, self.units],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
    if self.use_bias:
      self.bias = self.add_variable('bias',
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
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    output_shape = shape[:-1] + [self.units]
    if len(output_shape) > 2:
      # Broadcasting is required for the inputs.
      outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1],
                                                             [0]])
      # Reshape the output back to the original ndim of the input.
      outputs.set_shape(output_shape)
    else:
      outputs = standard_ops.matmul(inputs, self.kernel)
    if self.use_bias:
      outputs = nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1].value is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)


def dense(
    inputs, units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=init_ops.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None):
  """Functional interface for the densely-connected layer.

  This layer implements the operation:
  `outputs = activation(inputs.kernel + bias)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `kernel` is a weights matrix created by the layer,
  and `bias` is a bias vector created by the layer
  (only if `use_bias` is `True`).

  Note: if the `inputs` tensor has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `kernel`.

  Arguments:
    inputs: Tensor input.
    units: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: Initializer function for the weight matrix.
      If `None` (default), weights are initialized using the default
      initializer used by `tf.get_variable`.
    bias_initializer: Initializer function for the bias.
    kernel_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    kernel_constraint: An optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: An optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.
  """
  layer = Dense(units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                _scope=name,
                _reuse=reuse)
  return layer.apply(inputs)


class Dropout(base.Layer):
  """Applies Dropout to the input.

  Dropout consists in randomly setting a fraction `rate` of input units to 0
  at each update during training time, which helps prevent overfitting.
  The units that are kept are scaled by `1 / (1 - rate)`, so that their
  sum is unchanged at training time and inference time.

  Arguments:
    rate: The dropout rate, between 0 and 1. E.g. `rate=0.1` would drop out
      10% of input units.
    noise_shape: 1D tensor of type `int32` representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)`, and you want the dropout mask
      to be the same for all timesteps, you can use
      `noise_shape=[batch_size, 1, features]`.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}.
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

  def _get_noise_shape(self, _):
    # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
    # which will override `self.noise_shape`, and allows for custom noise
    # shapes with dynamically sized inputs.
    return self.noise_shape

  def call(self, inputs, training=False):
    def dropped_inputs():
      return nn.dropout(inputs, 1  - self.rate,
                        noise_shape=self._get_noise_shape(inputs),
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
      @{tf.set_random_seed}
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
