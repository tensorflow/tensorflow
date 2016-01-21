"""Deep Neural Network estimators."""
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


class TensorFlowRNNClassifier(TensorFlowEstimator, ClassifierMixin):
    """TensorFlow RNN Classifier model.

    Parameters:
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
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
        continue_training: when continue_training is True, once initialized
            model will be continuely trained on every call of fit.
        early_stopping_rounds: Activates early stopping if this is not None.
            Loss needs to decrease at least every every <early_stopping_rounds>
            round(s) to continue training. (default: None)
        max_to_keep: The maximum number of recent checkpoint files to keep.
            As new files are created, older files are deleted.
            If None or 0, all checkpoint files are kept.
            Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)
        keep_checkpoint_every_n_hours: Number of hours between each checkpoint
            to be saved. The default value of 10,000 hours effectively disables the feature.
     """

    def __init__(self, rnn_size, cell_type, input_op_fn,
                 n_classes, tf_master="", batch_size=32,
                 steps=50, optimizer="SGD", learning_rate=0.1,
                 tf_random_seed=42, continue_training=False,
                 verbose=1, early_stopping_rounds=None,
                 max_to_keep=5, keep_checkpoint_every_n_hours=10000):
        self.rnn_size = rnn_size
        self.cell_type = cell_type
        self.input_op_fn = input_op_fn
        super(TensorFlowDNNClassifier, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training, verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    def _model_fn(self, X, y):
        return models.get_rnn_model(self.rnn_size, self.cell_type,
                                    self.input_op_fn,
                                    models.logistic_regression)(X, y)


class TensorFlowRNNRegressor(TensorFlowEstimator, RegressorMixin):
    """TensorFlow RNN Regressor model.

    Parameters:
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
        early_stopping_rounds: Activates early stopping if this is not None.
            Loss needs to decrease at least every every <early_stopping_rounds>
            round(s) to continue training. (default: None)
    """

    def __init__(self, rnn_size, cell_type, input_op_fn
                 n_classes=0, tf_master="", batch_size=32,
                 steps=50, optimizer="SGD", learning_rate=0.1,
                 tf_random_seed=42, continue_training=False,
                 verbose=1, early_stopping_rounds=None,
                 max_to_keep=5, keep_checkpoint_every_n_hours=10000):
        self.rnn_size = rnn_size
        self.cell_type = cell_type
        self.input_op_fn = input_op_fn
        super(TensorFlowDNNRegressor, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training, verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    def _model_fn(self, X, y):
        return models.get_rnn_model(self.rnn_size, self.cell_type,
                                    self.input_op_fn,
                                    models.linear_regression)(X, y)
