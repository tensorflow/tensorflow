"""Deep Autoencoder estimators."""
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
from tensorflow.contrib.learn.python.learn.estimators.base import TensorFlowBaseTransformer
from tensorflow.contrib.learn.python.learn import models


class TensorFlowDNNAutoencoder(TensorFlowBaseTransformer):
    """TensorFlow Autoencoder Regressor model.

    Parameters:
        hidden_units: List of hidden units per layer.
        batch_size: Mini batch size.
        activation: activation function used to map inner latent layer onto
                    reconstruction layer.
        add_noise: a function that adds noise to tensor_in, 
               e.g. def add_noise(x):
                        return(x + np.random.normal(0, 0.1, (len(x), len(x[0]))))
        steps: Number of steps to run over data.
        optimizer: Optimizer name (or class), for example "SGD", "Adam",
                   "Adagrad".
        learning_rate: If this is constant float value, no decay function is used.
            Instead, a customized decay function can be passed that accepts
            global_step as parameter and returns a Tensor.
            e.g. exponential decay function:
            def exp_decay(global_step):
                return tf.train.exponential_decay(
                    learning_rate=0.1, global_step,
                    decay_steps=2, decay_rate=0.001)
        continue_training: when continue_training is True, once initialized
            model will be continuely trained on every call of fit.
        config: RunConfig object that controls the configurations of the session,
            e.g. num_cores, gpu_memory_fraction, etc.
        verbose: Controls the verbosity, possible values:
                 0: the algorithm and debug information is muted.
                 1: trainer prints the progress.
                 2: log device placement is printed.
        dropout: When not None, the probability we will drop out a given
                 coordinate.
    """
    def __init__(self, hidden_units, n_classes=0, batch_size=32,
                 steps=200, optimizer="Adagrad", learning_rate=0.1,
                 clip_gradients=5.0, activation=nn.relu, add_noise=None,
                 continue_training=False, config=None,
                 verbose=1, dropout=None):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.activation = activation
        self.add_noise = add_noise
        super(TensorFlowDNNAutoencoder, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, clip_gradients=clip_gradients,
            continue_training=continue_training,
            config=config, verbose=verbose)

    def _model_fn(self, X, y):
        encoder, decoder, autoencoder_estimator = models.get_autoencoder_model(
            self.hidden_units,
            models.linear_regression,
            activation=self.activation,
            add_noise=self.add_noise,
            dropout=self.dropout)(X)
        self.encoder = encoder
        self.decoder = decoder
        return autoencoder_estimator

    def generate(self, hidden=None):
        """Generate new data using trained construction layer"""
        if hidden is None:
            last_layer = len(self.hidden_units) - 1
            bias = self.get_tensor_value('encoder/dnn/layer%d/Linear/Bias:0' % last_layer)
            import numpy as np
            hidden = np.random.normal(size=bias.shape)
            hidden = np.reshape(hidden, (1, len(hidden)))
        return self._session.run(self.decoder, feed_dict={self.encoder: hidden})

    @property
    def weights_(self):
        """Returns weights of the autoencoder's weight layers."""
        weights = []
        for layer in range(len(self.hidden_units)):
            weights.append(self.get_tensor_value('encoder/dnn/layer%d/Linear/Matrix:0' % layer))
        for layer in range(len(self.hidden_units)):
            weights.append(self.get_tensor_value('decoder/dnn/layer%d/Linear/Matrix:0' % layer))
        weights.append(self.get_tensor_value('linear_regression/weights:0'))
        return weights

    @property
    def bias_(self):
        """Returns bias of the autoencoder's bias layers."""
        biases = []
        for layer in range(len(self.hidden_units)):
            biases.append(self.get_tensor_value('encoder/dnn/layer%d/Linear/Bias:0' % layer))
        for layer in range(len(self.hidden_units)):
            biases.append(self.get_tensor_value('decoder/dnn/layer%d/Linear/Bias:0' % layer))
        biases.append(self.get_tensor_value('linear_regression/bias:0'))
        return biases

