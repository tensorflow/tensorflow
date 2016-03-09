"""Base estimator class."""
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

import datetime
import json
import os
import shutil
from six import string_types

import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from sklearn.base import BaseEstimator
try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    from sklearn.utils.validation import NotFittedError  # pylint: disable=ungrouped-imports

from skflow.trainer import TensorFlowTrainer, RestoredTrainer
from skflow.io.data_feeder import setup_train_data_feeder
from skflow.io.data_feeder import setup_predict_data_feeder
from skflow.ops.dropout_ops import DROPOUTS
from skflow import monitors

from skflow.addons.config_addon import ConfigAddon


def _write_with_backup(filename, content):
    if os.path.exists(filename):
        shutil.move(filename, filename + '.old')
    with open(filename, 'w') as f:
        f.write(content)


class TensorFlowEstimator(BaseEstimator):
    """Base class for all TensorFlow estimators.

    Parameters:
        model_fn: Model function, that takes input X, y tensors and outputs
                  prediction and loss tensors.
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

    def __init__(self, model_fn, n_classes, tf_master="", batch_size=32,
                 steps=200, optimizer="SGD",
                 learning_rate=0.1, class_weight=None,
                 tf_random_seed=42, continue_training=False,
                 config_addon=None, verbose=1,
                 max_to_keep=5, keep_checkpoint_every_n_hours=10000):

        self.n_classes = n_classes
        self.tf_master = tf_master
        self.batch_size = batch_size
        self.steps = steps
        self.verbose = verbose
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.tf_random_seed = tf_random_seed
        self.model_fn = model_fn
        self.continue_training = continue_training
        self._initialized = False
        self.max_to_keep = max_to_keep
        self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
        self.class_weight = class_weight
        self.config_addon = config_addon

    def _setup_training(self):
        """Sets up graph, model and trainer."""
        self._graph = tf.Graph()
        self._graph.add_to_collection("IS_TRAINING", True)
        with self._graph.as_default():
            tf.set_random_seed(self.tf_random_seed)
            self._global_step = tf.Variable(
                0, name="global_step", trainable=False)

            # Setting up input and output placeholders.
            input_shape = [None] + self._data_feeder.input_shape[1:]
            output_shape = [None] + self._data_feeder.output_shape[1:]
            self._inp = tf.placeholder(
                tf.as_dtype(self._data_feeder.input_dtype), input_shape,
                name="input")
            self._out = tf.placeholder(
                tf.as_dtype(self._data_feeder.output_dtype), output_shape,
                name="output")

            # If class weights are provided, add them to the graph.
            # Different loss functions can use this tensor by name.
            if self.class_weight:
                self._class_weight_node = tf.constant(
                    self.class_weight, name='class_weight')

            # Add histograms for X and y if they are floats.
            if self._data_feeder.input_dtype in (np.float32, np.float64):
                tf.histogram_summary("X", self._inp)
            if self._data_feeder.output_dtype in (np.float32, np.float64):
                tf.histogram_summary("y", self._out)

            # Create model's graph.
            self._model_predictions, self._model_loss = self.model_fn(
                self._inp, self._out)

            # Create summary to monitor loss
            tf.scalar_summary("loss", self._model_loss)

            # Set up a single operator to merge all the summaries
            self._summaries = tf.merge_all_summaries()

            # Create trainer and augment graph with gradients and optimizer.
            # Additionally creates initialization ops.
            self._trainer = TensorFlowTrainer(
                loss=self._model_loss, global_step=self._global_step,
                optimizer=self.optimizer, learning_rate=self.learning_rate)

            # Create model's saver capturing all the nodes created up until now.
            self._saver = tf.train.Saver(
                max_to_keep=self.max_to_keep,
                keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)

            # Enable monitor to create validation data dict with appropriate tf placeholders
            self._monitor.create_val_feed_dict(self._inp, self._out)

            # Create session to run model with.
            if self.config_addon is None:
                self.config_addon = ConfigAddon(verbose=self.verbose)
            self._session = tf.Session(self.tf_master, config=self.config_addon.config)

    def _setup_summary_writer(self, logdir):
        """Sets up the summary writer to prepare for later optional visualization."""
        self._summary_writer = tf.train.SummaryWriter(
            os.path.join(logdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
            graph_def=self._session.graph_def)

    def fit(self, X, y, monitor=None, logdir=None):
        """Builds a neural network model given provided `model_fn` and training
        data X and y.

        Note: called first time constructs the graph and initializers
        variables. Consecutives times it will continue training the same model.
        This logic follows partial_fit() interface in scikit-learn.

        To restart learning, create new estimator.

        Args:
            X: matrix or tensor of shape [n_samples, n_features...]. Can be
            iterator that returns arrays of features. The training input
            samples for fitting the model.
            y: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
            iterator that returns array of targets. The training target values
            (class labels in classification, real numbers in regression).
            monitor: Monitor object to print training progress and invoke early stopping
            logdir: the directory to save the log file that can be used for
            optional visualization.

        Returns:
            Returns self.
        """
        # Sets up data feeder.
        self._data_feeder = setup_train_data_feeder(X, y,
                                                    self.n_classes,
                                                    self.batch_size)

        if monitor is None:
            self._monitor = monitors.default_monitor()
        else:
            self._monitor = monitor

        if not self.continue_training or not self._initialized:
            # Sets up model and trainer.
            self._setup_training()
            # Initialize model parameters.
            self._trainer.initialize(self._session)
            self._initialized = True

        # Sets up summary writer for later optional visualization.
        # Due to not able to setup _summary_writer in __init__ as it's not a
        # parameter of the model, here we need to check if such variable exists
        # and if it's None or not (in case it was setup in a previous run).
        # It is initialized only in the case where it wasn't before and log dir
        # is provided.
        if logdir:
            if (not hasattr(self, "_summary_writer") or
                    (hasattr(self, "_summary_writer") and self._summary_writer is None)):
                self._setup_summary_writer(logdir)
        else:
            self._summary_writer = None

        # Train model for given number of steps.
        self._trainer.train(self._session,
                            self._data_feeder.get_feed_dict_fn(
                                self._inp, self._out),
                            self.steps,
                            self._monitor,
                            self._summary_writer,
                            self._summaries,
                            feed_params_fn=self._data_feeder.get_feed_params)
        return self

    def partial_fit(self, X, y):
        """Incremental fit on a batch of samples.

        This method is expected to be called several times consecutively
        on different or the same chunks of the dataset. This either can
        implement iterative training or out-of-core/online training.

        This is especially useful when the whole dataset is too big to
        fit in memory at the same time. Or when model is taking long time
        to converge, and you want to split up training into subparts.

        Args:
            X: matrix or tensor of shape [n_samples, n_features...]. Can be
            iterator that returns arrays of features. The training input
            samples for fitting the model.
            y: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
            iterator that returns array of targets. The training target values
            (class label in classification, real numbers in regression).

        Returns:
            Returns self.
        """
        return self.fit(X, y)

    def _predict(self, X, axis=-1, batch_size=-1):
        if not self._initialized:
            raise NotFittedError()
        self._graph.add_to_collection("IS_TRAINING", False)
        predict_data_feeder = setup_predict_data_feeder(
            X, batch_size=batch_size)
        preds = []
        dropouts = self._graph.get_collection(DROPOUTS)
        feed_dict = {prob: 1.0 for prob in dropouts}
        for data in predict_data_feeder:
            feed_dict[self._inp] = data
            predictions_for_batch = self._session.run(
                self._model_predictions,
                feed_dict)
            if self.n_classes > 1 and axis != -1:
                preds.append(predictions_for_batch.argmax(axis=axis))
            else:
                preds.append(predictions_for_batch)

        return np.concatenate(preds, axis=0)

    def predict(self, X, axis=1, batch_size=-1):
        """Predict class or regression for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Args:
            X: array-like matrix, [n_samples, n_features...] or iterator.
            axis: Which axis to argmax for classification.
                  By default axis 1 (next after batch) is used.
                  Use 2 for sequence predictions.
            batch_size: If test set is too big, use batch size to split
                        it into mini batches. By default full dataset is used.

        Returns:
            y: array of shape [n_samples]. The predicted classes or predicted
            value.
        """
        return self._predict(X, axis=axis, batch_size=batch_size)

    def predict_proba(self, X, batch_size=-1):
        """Predict class probability of the input samples X.

        Args:
            X: array-like matrix, [n_samples, n_features...] or iterator.
            batch_size: If test set is too big, use batch size to split
                        it into mini batches. By default full dataset is used.

        Returns:
            y: array of shape [n_samples, n_classes]. The predicted
            probabilities for each class.

        """
        return self._predict(X, batch_size=batch_size)

    def get_tensor(self, name):
        """Returns tensor by name.

        Args:
            name: string, name of the tensor.

        Returns:
            Tensor.
        """
        return self._graph.get_tensor_by_name(name)

    def get_tensor_value(self, name):
        """Returns value of the tensor give by name.

        Args:
            name: string, name of the tensor.

        Returns:
            Numpy array - value of the tensor.
        """
        return self._session.run(self.get_tensor(name))

    def save(self, path):
        """Saves checkpoints and graph to given path.

        Args:
            path: Folder to save model to.
        """
        if not self._initialized:
            raise NotFittedError()

        # Currently Saver requires absolute path to work correctly.
        path = os.path.abspath(path)

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.isdir(path):
            raise ValueError("Path %s should be a directory to save"
                             "checkpoints and graph." % path)
        # Save model definition.
        all_params = self.get_params()
        params = {}
        for key, value in all_params.items():
            if not callable(value) and value is not None:
                params[key] = value
        params['class_name'] = type(self).__name__
        model_def = json.dumps(
            params,
            default=lambda o: o.__dict__ if hasattr(o, '__dict__') else None)
        _write_with_backup(os.path.join(path, 'model.def'), model_def)

        # Save checkpoints.
        endpoints = '%s\n%s\n%s\n%s' % (
            self._inp.name,
            self._out.name,
            self._model_predictions.name,
            self._model_loss.name)
        _write_with_backup(os.path.join(path, 'endpoints'), endpoints)

        # Save graph definition.
        _write_with_backup(os.path.join(path, 'graph.pbtxt'), str(self._graph.as_graph_def()))

        # Save saver defintion.
        _write_with_backup(os.path.join(path, 'saver.pbtxt'), str(self._saver.as_saver_def()))

        # Save checkpoints.
        self._saver.save(self._session, os.path.join(path, 'model'),
                         global_step=self._global_step)

    def _restore(self, path):
        """Restores this estimator from given path.

        Note: will rebuild the graph and initialize all parameters,
        and will ignore provided model.

        Args:
            path: Path to checkpoints and other information.
        """
        # Currently Saver requires absolute path to work correctly.
        path = os.path.abspath(path)

        self._graph = tf.Graph()
        with self._graph.as_default():
            endpoints_filename = os.path.join(path, 'endpoints')
            if not os.path.exists(endpoints_filename):
                raise ValueError("Restore folder doesn't contain endpoints.")
            with open(endpoints_filename) as foutputs:
                endpoints = foutputs.read().split('\n')
            graph_filename = os.path.join(path, 'graph.pbtxt')
            if not os.path.exists(graph_filename):
                raise ValueError("Restore folder doesn't contain graph definition.")
            with open(graph_filename) as fgraph:
                graph_def = tf.GraphDef()
                text_format.Merge(fgraph.read(), graph_def)
                (self._inp, self._out,
                 self._model_predictions, self._model_loss) = tf.import_graph_def(
                     graph_def, name='', return_elements=endpoints)
            saver_filename = os.path.join(path, 'saver.pbtxt')
            if not os.path.exists(saver_filename):
                raise ValueError("Restore folder doesn't contain saver defintion.")
            with open(saver_filename) as fsaver:
                saver_def = tf.train.SaverDef()
                text_format.Merge(fsaver.read(), saver_def)
                self._saver = tf.train.Saver(saver_def=saver_def)

            # Restore trainer
            self._global_step = self._graph.get_tensor_by_name('global_step:0')
            trainer_op = self._graph.get_operation_by_name('train')
            self._trainer = RestoredTrainer(
                self._model_loss, self._global_step, trainer_op)

            # Restore summaries.
            self._summaries = self._graph.get_operation_by_name('MergeSummary/MergeSummary')

            # Restore session.
            if not isinstance(self.config_addon, ConfigAddon):
                self.config_addon = ConfigAddon(verbose=self.verbose)
            self._session = tf.Session(
                self.tf_master,
                config=self.config_addon.config)
            checkpoint_path = tf.train.latest_checkpoint(path)
            if checkpoint_path is None:
                raise ValueError("Missing checkpoint files in the %s. Please "
                                 "make sure you are you have checkpoint file that describes "
                                 "latest checkpoints and appropriate checkpoints are there. "
                                 "If you have moved the folder, you at this point need to "
                                 "update manually update the paths in the checkpoint file." % path)
            self._saver.restore(self._session, checkpoint_path)
        # Set to be initialized.
        self._initialized = True

    # pylint: disable=unused-argument
    @classmethod
    def restore(cls, path, config_addon=None):
        """Restores model from give path.

        Args:
            path: Path to the checkpoints and other model information.
            config_addon: ConfigAddon object that controls the configurations of the session,
                e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

        Returns:
            Estiamator, object of the subclass of TensorFlowEstimator.
        """
        model_def_filename = os.path.join(path, 'model.def')
        if not os.path.exists(model_def_filename):
            raise ValueError("Restore folder doesn't contain model definition.")
        # list of parameters that are allowed to be reconfigured
        reconfigurable_params = ['config_addon']
        with open(model_def_filename) as fmodel:
            model_def = json.loads(fmodel.read())
            # TensorFlow binding requires parameters to be strings not unicode.
            # Only issue in Python2.
            for key, value in model_def.items():
                if (isinstance(value, string_types) and
                        not isinstance(value, str)):
                    model_def[key] = str(value)
                if key in reconfigurable_params:
                    newValue = locals()[key]
                    if newValue is not None:
                        model_def[key] = newValue
        class_name = model_def.pop('class_name')
        if class_name == 'TensorFlowEstimator':
            custom_estimator = TensorFlowEstimator(model_fn=None, **model_def)
            custom_estimator._restore(path)
            return custom_estimator

        # To avoid cyclical dependencies, import inside the function instead of
        # the beginning of the file.
        from skflow import estimators
        # Estimator must be one of the defined estimators in the __init__ file.
        estimator = getattr(estimators, class_name)(**model_def)
        estimator._restore(path)
        return estimator
