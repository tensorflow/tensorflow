"""TensorFlow ops for autoencoder."""
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

from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.learn.python.learn.ops import dnn_ops


def dnn_autoencoder(tensor_in, hidden_units,
                activation=nn.relu, add_noise=None,
                dropout=None, scope=None):
    """Creates fully connected autoencoder subgraph.

    Args:
        tensor_in: tensor or placeholder for input features.
        hidden_units: list of counts of hidden units in each layer.
        activation: activation function used to map inner latent layer onto
                    reconstruction layer.
        add_noise: a function that adds noise to tensor_in, 
               e.g. def add_noise(x):
                        return(x + np.random.normal(0, 0.1, (len(x), len(x[0]))))
        dropout: if not None, will add a dropout layer with given
                 probability.
        scope: the variable scope for this op.

    Returns:
        Tensors for encoder and decoder.
    """
    with vs.variable_op_scope([tensor_in], scope, "autoencoder"):
        if add_noise is not None:
                    tensor_in = add_noise(tensor_in)
        with vs.variable_scope('encoder'):
            # build DNN encoder
            encoder = dnn_ops.dnn(tensor_in, hidden_units,
                activation=activation, dropout=dropout)
        with vs.variable_scope('decoder'):
            # reverse hidden_units and built DNN decoder
            decoder = dnn_ops.dnn(encoder, hidden_units[::-1],
                activation=activation, dropout=dropout)
        return encoder, decoder

