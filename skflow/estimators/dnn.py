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


class TensorFlowDNNClassifier(TensorFlowEstimator, ClassifierMixin):
    """TensorFlow DNN Classifier model.

    Parameters:
        hidden_units: List of hidden units per layer.
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
        config_addon: ConfigAddon object that controls the configurations of the session,
            e.g. num_cores, gpu_memory_fraction, etc.
        max_to_keep: The maximum number of recent checkpoint files to keep.
            As new files are created, older files are deleted.
            If None or 0, all checkpoint files are kept.
            Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)
        keep_checkpoint_every_n_hours: Number of hours between each checkpoint
            to be saved. The default value of 10,000 hours effectively disables the feature.
     """

    def __init__(self, hidden_units, n_classes, tf_master="", batch_size=32,
                 steps=200, optimizer="SGD", learning_rate=0.1,
                 class_weight=None,
                 tf_random_seed=42, continue_training=False, config_addon=None,
                 verbose=1, max_to_keep=5, keep_checkpoint_every_n_hours=10000):

        self.hidden_units = hidden_units
        super(TensorFlowDNNClassifier, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, class_weight=class_weight,
            tf_random_seed=tf_random_seed,
            continue_training=continue_training,
            config_addon=config_addon, verbose=verbose,
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    def _model_fn(self, X, y):
        return models.get_dnn_model(self.hidden_units,
                                    models.logistic_regression)(X, y)

    @property
    def weights_(self):
        """Returns weights of the DNN weight layers."""
        weights = []
        for layer in range(len(self.hidden_units)):
            weights.append(self.get_tensor_value('dnn/layer%d/Linear/Matrix:0' % layer))
        weights.append(self.get_tensor_value('logistic_regression/weights:0'))
        return weights

    @property
    def bias_(self):
        """Returns bias of the DNN's bias layers."""
        biases = []
        for layer in range(len(self.hidden_units)):
            biases.append(self.get_tensor_value('dnn/layer%d/Linear/Bias:0' % layer))
        biases.append(self.get_tensor_value('logistic_regression/bias:0'))
        return biases


class TensorFlowDNNRegressor(TensorFlowEstimator, RegressorMixin):
    """TensorFlow DNN Regressor model.

    Parameters:
        hidden_units: List of hidden units per layer.
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
        config_addon: ConfigAddon object that controls the configurations of the session,
            e.g. num_cores, gpu_memory_fraction, etc.
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

    def __init__(self, hidden_units, n_classes=0, tf_master="", batch_size=32,
                 steps=200, optimizer="SGD", learning_rate=0.1,
                 tf_random_seed=42, continue_training=False, config_addon=None,
                 verbose=1, max_to_keep=5, keep_checkpoint_every_n_hours=10000):

        self.hidden_units = hidden_units
        super(TensorFlowDNNRegressor, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training,
            config_addon=config_addon, verbose=verbose,
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    def _model_fn(self, X, y):
        return models.get_dnn_model(self.hidden_units,
                                    models.linear_regression)(X, y)

    @property
    def weights_(self):
        """Returns weights of the DNN weight layers."""
        weights = []
        for layer in range(len(self.hidden_units)):
            weights.append(self.get_tensor_value('dnn/layer%d/Linear/Matrix:0' % layer))
        weights.append(self.get_tensor_value('linear_regression/weights:0'))
        return weights

    @property
    def bias_(self):
        """Returns bias of the DNN's bias layers."""
        biases = []
        for layer in range(len(self.hidden_units)):
            biases.append(self.get_tensor_value('dnn/layer%d/Linear/Bias:0' % layer))
        biases.append(self.get_tensor_value('linear_regression/bias:0'))
        return biases
