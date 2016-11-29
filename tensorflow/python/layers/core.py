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
"""Contains the core layers: FullyConnected, [Flatten, Dropout].

Also contains their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.layers import base


class FullyConnected(base._Layer):  # pylint: disable=protected-access
  """Fully-connected layer class.

  WARNING: Do not use this class unless you know what you are doing:
  the API is subject to future changes.

  This layer implements the operation `outputs = activation(inputs.w + b)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `w` is a weights matrix created by the layer,
  and `b` is a bias vector created by the layer (only if `use_bias` is `True`).

  Note: if the input to the layer has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `w`.

  Arguments:
    output_dim: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    w_initializer: Initializer function for the weight matrix.
    bias_initializer: Initializer function for the bias.
    w_regularizer: Regularizer function for the weight matrix.
    bias_regularizer: Regularizer function for the bias.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Properties:
    output_dim: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable).
    use_bias: Boolean, whether the layer uses a bias.
    w_initializer: Initializer instance (or name) for the weight matrix.
    bias_initializer: Initializer instance (or name) for the bias.
    w_regularizer: Regularizer instance for the weight matrix (callable)
    bias_regularizer: Regularizer instance for the bias (callable).
    activity_regularizer: Regularizer instance for the output (callable)
    w: Weight matrix (TensorFlow variable or tensor).
    bias: Bias vector, if applicable (TensorFlow variable or tensor).
  """

  def __init__(self, output_dim,
               activation=None,
               use_bias=True,
               w_initializer=None,
               bias_initializer=init_ops.zeros_initializer,
               w_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(FullyConnected, self).__init__(trainable=trainable, name=name,
                                         **kwargs)
    self.output_dim = output_dim
    self.activation = activation
    self.use_bias = use_bias
    self.w_initializer = w_initializer
    self.bias_initializer = bias_initializer
    self.w_regularizer = w_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer

  def build(self, input_shape):
    if len(input_shape) < 2:
      raise ValueError('Inputs to `FullyConnected` should have rank >= 2.')
    if input_shape[-1] is None:
      raise ValueError('The last dimension of the inputs to `FullyConnected` '
                       'should be defined. Found `None`.')
    # Note that we set `trainable=True` because this is a trainable
    # weight of the layer. If the layer is not trainable
    # (self.trainable = False), the variable will not be added to
    # tf.trainable_variables(), and self.trainable_weights will be empty.
    self.w = vs.get_variable('weights',
                             shape=[input_shape[-1], self.output_dim],
                             initializer=self.w_initializer,
                             regularizer=self.w_regularizer,
                             dtype=self._dtype,
                             trainable=True)
    if self.use_bias:
      self.bias = vs.get_variable('biases',
                                  shape=[self.output_dim,],
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  dtype=self._dtype,
                                  trainable=True)
    else:
      self.bias = None

  def call(self, inputs):
    shape = inputs.get_shape().as_list()
    input_dim = shape[-1]
    output_shape = shape[:-1] + [self.output_dim]
    if len(output_shape) > 2:
      # Reshape the input to 2D.
      output_shape_tensors = array_ops.unpack(array_ops.shape(inputs))
      output_shape_tensors[-1] = self.output_dim
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
      return self.activation(outputs)
    return outputs


def fully_connected(
    inputs, output_dim,
    activation=None,
    use_bias=True,
    w_initializer=None,
    bias_initializer=init_ops.zeros_initializer,
    w_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=False):
  """Functional interface for the fully connected layer.

  This layer implements the operation `outputs = activation(inputs.w + b)`
  Where `activation` is the activation function passed as the `activation`
  argument (if not `None`), `w` is a weights matrix created by the layer,
  and `b` is a bias vector created by the layer (only if `use_bias` is `True`).

  Note: if the `inputs` tensor has a rank greater than 2, then it is
  flattened prior to the initial matrix multiply by `w`.

  Arguments:
    inputs: Tensor input.
    output_dim: Integer or Long, dimensionality of the output space.
    activation: Activation function (callable). Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    w_initializer: Initializer function for the weight matrix.
    bias_initializer: Initializer function for the bias.
    w_regularizer: Regularizer function for the weight matrix.
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
  layer = FullyConnected(output_dim,
                         activation=activation,
                         use_bias=use_bias,
                         w_initializer=w_initializer,
                         bias_initializer=bias_initializer,
                         w_regularizer=w_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         trainable=trainable,
                         name=name,
                         dtype=inputs.dtype.base_dtype,
                         _scope=name,
                         _reuse_weights=reuse)
  return layer.apply(inputs)
