"""TensorFlow ops for deep neural networks."""
#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.learn.python.learn.ops import dropout_ops


def dnn(tensor_in, hidden_units, activation=nn.relu, dropout=None):
  """Creates fully connected deep neural network subgraph.

  Args:
    tensor_in: tensor or placeholder for input features.
    hidden_units: list of counts of hidden units in each layer.
    activation: activation function between layers. Can be None. Can be list of
      activation function for each layer
    dropout: if not None, will add a dropout layer with given probability. Can
      be list of dropouts per layer

  Returns:
    A tensor which would be a deep neural network.
  """

  def broadcast_or_noop(f, length, noop=lambda x:x):
    """Creates given length list of functions, broadcasting for single functions

    Args:
      f: function to broadcast. If None, broadcast noop. If list of function
        validate len(f) == length
      length: output list size
      noop: No operation to fill if None

    Returns:
      A list which has given lenght with optionally broadcasted function
    """
    if f is None:
      return([noop for _ in range(length)])
    try:
      f[0]
    except TypeError:
      return [f for _ in range(length)]
    if len(f) != length:
      raise ValueError("invalid supplied argument size")
    return f

  activation = broadcast_or_noop(activation, len(hidden_units))
  dropout = broadcast_or_noop(dropout, len(hidden_units), None)

  with vs.variable_scope('dnn'):
    for i, (n_units, activation_l, dropout_l) in \
      enumerate(zip(hidden_units, activation, dropout)):
      with vs.variable_scope('layer%d' % i):
        # Weight initializer was set to None to replicate the behavior of
        # rnn_cell.linear. Using fully_connected's default initializer gets
        # slightly worse quality results on unit tests.
        tensor_in = layers.legacy_fully_connected(
            tensor_in,
            n_units,
            weight_init=None,
            weight_collections=['dnn_weights'],
            bias_collections=['dnn_biases'])
        tensor_in = activation_l(tensor_in)
        if dropout_l is not None:
          tensor_in = dropout_ops.dropout(tensor_in, prob=(1.0 - dropout_l))
    return tensor_in
