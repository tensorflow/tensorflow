"""Main Scikit Flow module."""
#  Copyright 2015 Google Inc. All Rights Reserved.
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

import collections
import random

import json
import os
import datetime

import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import NotFittedError

from skflow.trainer import TensorFlowTrainer
from skflow import models
from skflow import data_feeder
from skflow import preprocessing
from skflow import ops
from skflow.io import *

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
        learning_rate: Learning rate for optimizer.
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
        continue_training: when continue_training is True, once initialized
            model will be continuely trained on every call of fit.
        num_cores: Number of cores to be used. (default: 4)
        verbose: Controls the verbosity, possible values:
                 0: the algorithm and debug information is muted.
                 1: trainer prints the progress.
                 2: log device placement is printed.
    """

    def __init__(self, model_fn, n_classes, tf_master="", batch_size=32, steps=50, optimizer="SGD",
                 learning_rate=0.1, tf_random_seed=42, continue_training=False,
                 num_cores=4, verbose=1):
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

    @staticmethod
    def _data_type_filter(X, y):
        """Filter data types into acceptable format"""
        if HAS_PANDAS:
            X = extract_pandas_data(X)
            y = extract_pandas_labels(y)
        return X, y

    def _setup_data_feeder(self, X, y):
        """Create data feeder, to sample inputs from dataset.
        If X and y are iterators, use StreamingDataFeeder.
        """
        data_feeder_cls = data_feeder.DataFeeder
        if hasattr(X, 'next'):
            assert hasattr(y, 'next')
            data_feeder_cls = data_feeder.StreamingDataFeeder
        self._data_feeder = data_feeder_cls(X, y, self.n_classes, self.batch_size)

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

            # Create model's graph.
            self._model_predictions, self._model_loss = self.model_fn(
                self._inp, self._out)

            # Create trainer and augment graph with gradients and optimizer.
            # Additionally creates initialization ops.
            self._trainer = TensorFlowTrainer(self._model_loss,
                                              self._global_step, self.optimizer, self.learning_rate)

            # Create model's saver capturing all the nodes created up until now.
            self._saver = tf.train.Saver()

            # Create session to run model with.
            self._session = tf.Session(self.tf_master,
                                       config=tf.ConfigProto(
                                           log_device_placement=self.verbose > 1,
                                           inter_op_parallelism_threads=self.num_cores,
                                           intra_op_parallelism_threads=self.num_cores))

    def _setup_summary_writer(self, logdir):
        """Sets up the summary writer to prepare for later optional visualization."""
        # Create summary to monitor loss
        tf.scalar_summary("loss", self._model_loss)
        # Set up a single operator to merge all the summaries
        tf.merge_all_summaries()
        # Set up summary writer to the specified log directory
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
        X, y = self._data_type_filter(X, y)
        # Sets up data feeder.
        self._setup_data_feeder(X, y)
        if not self.continue_training or not self._initialized:
            # Sets up model and trainer.
            self._setup_training()
            # Initialize model parameters.
            self._trainer.initialize(self._session)
            self._initialized = True
            # Sets up summary writer for later optional visualization
            if logdir:
                self._setup_summary_writer(logdir)

        # Train model for given number of steps.
        self._trainer.train(self._session,
                            self._data_feeder.get_feed_dict_fn(
                                self._inp, self._out),
                            self.steps,
                            verbose=self.verbose)
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
        if HAS_PANDAS:
            X = extract_pandas_data(X)
        pred = self._session.run(self._model_predictions,
                                 feed_dict={
                                     self._inp.name: X
                                 })
        return pred

    def predict(self, X):
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
        return pred.argmax(axis=1)

    def predict_proba(self, X):
        """Predict class probability of the input samples X.

        Args:
            X: array-like matrix, [n_samples, n_features...] or iterator.

        Returns:
            y: array of shape [n_samples, n_classes]. The predicted
            probabilities for each class.
        """
        return self._predict(X)

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
            params = self.get_params()
            for key, value in params.items():
                if callable(value):
                    params.pop(key)
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
                     graph_def, return_elements=endpoints)
            saver_filename = os.path.join(path, 'saver.pbtxt')
            if not os.path.exists(saver_filename):
                raise ValueError("Restore folder doesn't contain saver defintion.")
            with open(saver_filename) as fsaver:
                from tensorflow.python.training import saver_pb2
                saver_def = saver_pb2.SaverDef()
                text_format.Merge(fsaver.read(), saver_def)
                # ??? For some reason the saver def doesn't have import/ prefix.
                saver_def.filename_tensor_name = 'import/' + saver_def.filename_tensor_name
                saver_def.restore_op_name = 'import/' + saver_def.restore_op_name
                self._saver = tf.train.Saver(saver_def=saver_def)
            self._session = tf.Session(self.tf_master,
                                       config=tf.ConfigProto(
                                           log_device_placement=self.verbose > 1,
                                           inter_op_parallelism_threads=self.num_cores,
                                           intra_op_parallelism_threads=self.num_cores))
            self._graph.get_operation_by_name('import/save/restore_all')
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
            for key, value in model_def.items():
                if isinstance(value, unicode):
                    model_def[key] = str(value)
        class_name = model_def.pop('class_name')
        if class_name == 'TensorFlowEstimator':
            custom_estimator = TensorFlowEstimator(model_fn=None, **model_def)
            custom_estimator._restore(path)
            return custom_estimator

        # XXX(ilblackdragon): Using eval here is bad, should use lookup!!!!
        estimator = eval(class_name)(**model_def) # pylint: disable=eval-used
        estimator._restore(path)
        return estimator


class TensorFlowLinearRegressor(TensorFlowEstimator, RegressorMixin):
    """TensorFlow Linear Regression model."""

    def __init__(self, n_classes=0, tf_master="", batch_size=32, steps=50, optimizer="SGD",
                 learning_rate=0.1, tf_random_seed=42, continue_training=False,
                 verbose=1):
        super(TensorFlowLinearRegressor, self).__init__(
            model_fn=models.linear_regression, n_classes=n_classes,
            tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training,
            verbose=verbose)


class TensorFlowLinearClassifier(TensorFlowEstimator, ClassifierMixin):
    """TensorFlow Linear Classifier model."""

    def __init__(self, n_classes, tf_master="", batch_size=32, steps=50, optimizer="SGD",
                 learning_rate=0.1, tf_random_seed=42, continue_training=False,
                 verbose=1):
        super(TensorFlowLinearClassifier, self).__init__(
            model_fn=models.logistic_regression, n_classes=n_classes,
            tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training,
            verbose=verbose)


TensorFlowRegressor = TensorFlowLinearRegressor
TensorFlowClassifier = TensorFlowLinearClassifier


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
        learning_rate: Learning rate for optimizer.
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
        continue_training: when continue_training is True, once initialized
            model will be continuely trained on every call of fit.
     """

    def __init__(self, hidden_units, n_classes, tf_master="", batch_size=32,
                 steps=50, optimizer="SGD", learning_rate=0.1,
                 tf_random_seed=42, continue_training=False,
                 verbose=1):
        self.hidden_units = hidden_units
        super(TensorFlowDNNClassifier, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training,
            verbose=verbose)

    def _model_fn(self, X, y):
        return models.get_dnn_model(self.hidden_units,
                                    models.logistic_regression)(X, y)


class TensorFlowDNNRegressor(TensorFlowEstimator, ClassifierMixin):
    """TensorFlow DNN Regressor model.

    Parameters:
        hidden_units: List of hidden units per layer.
        tf_master: TensorFlow master. Empty string is default for local.
        batch_size: Mini batch size.
        steps: Number of steps to run over data.
        optimizer: Optimizer name (or class), for example "SGD", "Adam",
                   "Adagrad".
        learning_rate: Learning rate for optimizer.
        tf_random_seed: Random seed for TensorFlow initializers.
            Setting this value, allows consistency between reruns.
        continue_training: when continue_training is True, once initialized
            model will be continuely trained on every call of fit.
    """

    def __init__(self, hidden_units, n_classes=0, tf_master="", batch_size=32,
                 steps=50, optimizer="SGD", learning_rate=0.1,
                 tf_random_seed=42, continue_training=False,
                 verbose=1):
        self.hidden_units = hidden_units
        super(TensorFlowDNNRegressor, self).__init__(
            model_fn=self._model_fn,
            n_classes=n_classes, tf_master=tf_master,
            batch_size=batch_size, steps=steps, optimizer=optimizer,
            learning_rate=learning_rate, tf_random_seed=tf_random_seed,
            continue_training=continue_training,
            verbose=verbose)

    def _model_fn(self, X, y):
        return models.get_dnn_model(self.hidden_units,
                                    models.linear_regression)(X, y) 
