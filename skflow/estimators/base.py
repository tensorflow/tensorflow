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

import json
import os
import datetime
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
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
        continue_training: when continue_training is True, once initialized
            model will be continuely trained on every call of fit.
        num_cores: Number of cores to be used. (default: 4)
        verbose: Controls the verbosity, possible values:
                 0: the algorithm and debug information is muted.
                 1: trainer prints the progress.
                 2: log device placement is printed.
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

    def __init__(self, model_fn, n_classes, tf_master="", batch_size=32, steps=50, optimizer="SGD",
                 learning_rate=0.1, tf_random_seed=42, continue_training=False,
                 num_cores=4, verbose=1, early_stopping_rounds=None,
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
        self.num_cores = num_cores
        self._initialized = False
        self._early_stopping_rounds = early_stopping_rounds
        self.max_to_keep = max_to_keep
        self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

    def _setup_training(self):
        """Sets up graph, model and trainer."""
        self._graph = tf.Graph()
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

            # Create session to run model with.
            self._session = tf.Session(self.tf_master,
                                       config=tf.ConfigProto(
                                           log_device_placement=self.verbose > 1,
                                           inter_op_parallelism_threads=self.num_cores,
                                           intra_op_parallelism_threads=self.num_cores))

    def _setup_summary_writer(self, logdir):
        """Sets up the summary writer to prepare for later optional visualization."""
        self._summary_writer = tf.train.SummaryWriter(
            os.path.join(logdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
            graph_def=self._session.graph_def)

    def fit(self, X, y, logdir=None):
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
            logdir: the directory to save the log file that can be used for
            optional visualization.

        Returns:
            Returns self.
        """
        # Sets up data feeder.
        self._data_feeder = setup_train_data_feeder(X, y,
                                                    self.n_classes,
                                                    self.batch_size)
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
        if (logdir and (not hasattr(self, "_summary_writer") or
                        (hasattr(self, "_summary_writer") and self._summary_writer is None))):
            self._setup_summary_writer(logdir)
        else:
            self._summary_writer = None

        # Train model for given number of steps.
        self._trainer.train(self._session,
                            self._data_feeder.get_feed_dict_fn(
                                self._inp, self._out),
                            self.steps,
                            self._summary_writer,
                            self._summaries,
                            verbose=self.verbose,
                            early_stopping_rounds=self._early_stopping_rounds)
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

    def _predict(self, X):
        if not self._initialized:
            raise NotFittedError()
        predict_data_feeder = setup_predict_data_feeder(X)
        preds = []
        for data in predict_data_feeder:
            preds.append(self._session.run(
                self._model_predictions,
                feed_dict={
                    self._inp.name: data
                }))
        return np.concatenate(preds, axis=0)

    def predict(self, X, axis=1):
        """Predict class or regression for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Args:
            X: array-like matrix, [n_samples, n_features...] or iterator.

        Returns:
            y: array of shape [n_samples]. The predicted classes or predicted
            value.
        """
        pred = self._predict(X)
        if self.n_classes < 2:
            return pred
        return pred.argmax(axis=axis)

    def predict_proba(self, X):
        """Predict class probability of the input samples X.

        Args:
            X: array-like matrix, [n_samples, n_features...] or iterator.

        Returns:
            y: array of shape [n_samples, n_classes]. The predicted
            probabilities for each class.
        """
        return self._predict(X)

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
        with open(os.path.join(path, 'model.def'), 'w') as fmodel:
            all_params = self.get_params()
            params = {}
            for key, value in all_params.items():
                if not callable(value) and value is not None:
                    params[key] = value
            params['class_name'] = type(self).__name__
            fmodel.write(json.dumps(params))
        with open(os.path.join(path, 'endpoints'), 'w') as foutputs:
            foutputs.write('%s\n%s\n%s\n%s' % (
                self._inp.name,
                self._out.name,
                self._model_predictions.name,
                self._model_loss.name))
        with open(os.path.join(path, 'graph.pbtxt'), 'w') as fgraph:
            fgraph.write(str(self._graph.as_graph_def()))
        with open(os.path.join(path, 'saver.pbtxt'), 'w') as fsaver:
            fsaver.write(str(self._saver.as_saver_def()))
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
                from tensorflow.python.training import saver_pb2
                saver_def = saver_pb2.SaverDef()
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
            self._session = tf.Session(self.tf_master,
                                       config=tf.ConfigProto(
                                           log_device_placement=self.verbose > 1,
                                           inter_op_parallelism_threads=self.num_cores,
                                           intra_op_parallelism_threads=self.num_cores))
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

    @classmethod
    def restore(cls, path):
        """Restores model from give path.

        Args:
            path: Path to the checkpoints and other model information.

        Returns:
            Estiamator, object of the subclass of TensorFlowEstimator.
        """
        model_def_filename = os.path.join(path, 'model.def')
        if not os.path.exists(model_def_filename):
            raise ValueError("Restore folder doesn't contain model definition.")
        with open(model_def_filename) as fmodel:
            model_def = json.loads(fmodel.read())
            # TensorFlow binding requires parameters to be strings not unicode.
            # Only issue in Python2.
            for key, value in model_def.items():
                if (isinstance(value, string_types) and
                        not isinstance(value, str)):
                    model_def[key] = str(value)
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

