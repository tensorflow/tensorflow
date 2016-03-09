"""Recurrent Neural Network estimators."""
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
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

from __future__ import division, print_function, absolute_import

from sklearn.base import ClassifierMixin, RegressorMixin

from skflow.estimators.base import TensorFlowEstimator
from skflow import models


def null_input_op_fn(X):
    """This function does no transformation on the inputs, used as default"""
    return X


class TensorFlowRNNClassifier(TensorFlowEstimator, ClassifierMixin):
    """TensorFlow RNN Classifier model.

    Parameters:
        rnn_size: The size for rnn cell, e.g. size of your word embeddings.
        cell_type: The type of rnn cell, including rnn, gru, and lstm.
        num_layers: The number of layers of the rnn model.
        input_op_fn: Function that will transform the input tensor, such as
                     creating word embeddings, byte list, etc. This takes
                     an argument X for input and returns transformed X.
        bidirectional: boolean, Whether this is a bidirectional rnn.
        sequence_length: If sequence_length is provided, dynamic calculation is performed.
                 This saves computational time when unrolling past max sequence length.
        initial_state: An initial state for the RNN. This must be a tensor of appropriate type
                       and shape [batch_size x cell.state_size].
        n_classes: Number of classes in the target.
        tf_master: TensorFlow master. Empty string is default for local.
        batch_size: Mini batch size.
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
        class_weight: None or list of n_classes floats. Weight associated with
                     classes for loss computation. If not given, all classes are suppose to have
                     weight one.
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
        continue_training: when continue_training is True, once initialized
            model will be continuely trained on every call of fit.
        num_cores: Number of cores to be used. (default: 4)
        max_to_keep: The maximum number of recent checkpoint files to keep.
            As new files are created, older files are deleted.
            If None or 0, all checkpoint files are kept.
            Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)
        keep_checkpoint_every_n_hours: Number of hours between each checkpoint
            to be saved. The default value of 10,000 hours effectively disables the feature.
     """

    def __init__(self, rnn_size, n_classes, cell_type='gru', num_layers=1,
                 input_op_fn=null_input_op_fn,
                 initial_state=None, bidirectional=False,
                 sequence_length=None, tf_master="", batch_size=32,
                 steps=50, optimizer="SGD", learning_rate=0.1,
                 class_weight=None,
                 tf_random_seed=42, continue_training=False,
                 config_addon=None, verbose=1,
                 max_to_keep=5, keep_checkpoint_every_n_hours=10000):

        self.rnn_size = rnn_size
        self.cell_type = cell_type
        self.input_op_fn = input_op_fn
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.initial_state = initial_state
        super(TensorFlowRNNClassifier, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, class_weight=class_weight,
            tf_random_seed=tf_random_seed,
            continue_training=continue_training, config_addon=config_addon,
            verbose=verbose,
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    def _model_fn(self, X, y):
        return models.get_rnn_model(self.rnn_size, self.cell_type,
                                    self.num_layers,
                                    self.input_op_fn, self.bidirectional,
                                    models.logistic_regression,
                                    self.sequence_length,
                                    self.initial_state)(X, y)

    @property
    def bias_(self):
        """Returns bias of the rnn layer."""
        return self.get_tensor_value('logistic_regression/bias:0')

    @property
    def weights_(self):
        """Returns weights of the rnn layer."""
        return self.get_tensor_value('logistic_regression/weights:0')


class TensorFlowRNNRegressor(TensorFlowEstimator, RegressorMixin):
    """TensorFlow RNN Regressor model.

    Parameters:
        rnn_size: The size for rnn cell, e.g. size of your word embeddings.
        cell_type: The type of rnn cell, including rnn, gru, and lstm.
        num_layers: The number of layers of the rnn model.
        input_op_fn: Function that will transform the input tensor, such as
                     creating word embeddings, byte list, etc. This takes
                     an argument X for input and returns transformed X.
        bidirectional: boolean, Whether this is a bidirectional rnn.
        sequence_length: If sequence_length is provided, dynamic calculation is performed.
                 This saves computational time when unrolling past max sequence length.
        initial_state: An initial state for the RNN. This must be a tensor of appropriate type
                       and shape [batch_size x cell.state_size].
        tf_master: TensorFlow master. Empty string is default for local.
        batch_size: Mini batch size.
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
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
        continue_training: when continue_training is True, once initialized
            model will be continuely trained on every call of fit.
        num_cores: Number of cores to be used. (default: 4)
        verbose: Controls the verbosity, possible values:
                 0: the algorithm and debug information is muted.
                 1: trainer prints the progress.
                 2: log device placement is printed.
        max_to_keep: The maximum number of recent checkpoint files to keep.
            As new files are created, older files are deleted.
            If None or 0, all checkpoint files are kept.
            Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)
        keep_checkpoint_every_n_hours: Number of hours between each checkpoint
            to be saved. The default value of 10,000 hours effectively disables the feature.
   """

    def __init__(self, rnn_size, cell_type='gru', num_layers=1,
                 input_op_fn=null_input_op_fn, initial_state=None,
                 bidirectional=False, sequence_length=None,
                 n_classes=0, tf_master="", batch_size=32,
                 steps=50, optimizer="SGD", learning_rate=0.1,
                 tf_random_seed=42, continue_training=False,
                 config_addon=None, verbose=1,
                 max_to_keep=5, keep_checkpoint_every_n_hours=10000):

        self.rnn_size = rnn_size
        self.cell_type = cell_type
        self.input_op_fn = input_op_fn
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.initial_state = initial_state
        super(TensorFlowRNNRegressor, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training, config_addon=config_addon,
            verbose=verbose, max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    def _model_fn(self, X, y):
        return models.get_rnn_model(self.rnn_size, self.cell_type,
                                    self.num_layers,
                                    self.input_op_fn, self.bidirectional,
                                    models.linear_regression,
                                    self.sequence_length,
                                    self.initial_state)(X, y)

    @property
    def bias_(self):
        """Returns bias of the rnn layer."""
        return self.get_tensor_value('linear_regression/bias:0')

    @property
    def weights_(self):
        """Returns weights of the rnn layer."""
        return self.get_tensor_value('linear_regression/weights:0')
