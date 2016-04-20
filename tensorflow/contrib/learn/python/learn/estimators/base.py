"""Base estimator class."""
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

import datetime
import json
import os
import shutil
from six import string_types

import numpy as np

from google.protobuf import text_format
from tensorflow.python.platform.default import _gfile as gfile

from tensorflow.python.client import session
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops as array_ops_
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training import training as train

from tensorflow.contrib.layers import optimizers
from tensorflow.contrib.learn.python.learn import trainer
from tensorflow.contrib.learn.python.learn.io.data_feeder import setup_train_data_feeder
from tensorflow.contrib.learn.python.learn.io.data_feeder import setup_predict_data_feeder
from tensorflow.contrib.learn.python.learn.ops.dropout_ops import DROPOUTS
from tensorflow.contrib.learn.python.learn import monitors

from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators._sklearn import NotFittedError
from tensorflow.contrib.learn.python.learn.estimators.run_config import RunConfig


def _write_with_backup(filename, content):
    if gfile.Exists(filename):
        gfile.Rename(filename, filename + '.old', overwrite=True)
    with gfile.Open(filename, 'w') as f:
        f.write(content)


class TensorFlowEstimator(_sklearn.BaseEstimator):
    """Base class for all TensorFlow estimators.

    Parameters:
        model_fn: Model function, that takes input X, y tensors and outputs
                  prediction and loss tensors.
        n_classes: Number of classes in the target.
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
        clip_gradients: Clip norm of the gradients to this value to stop
                        gradient explosion.
        class_weight: None or list of n_classes floats. Weight associated with
                     classes for loss computation. If not given, all classes are suppose to have
                     weight one.
        continue_training: when continue_training is True, once initialized
            model will be continuely trained on every call of fit.
        config: RunConfig object that controls the configurations of the session,
            e.g. num_cores, gpu_memory_fraction, etc.
        verbose: Controls the verbosity, possible values:
                 0: the algorithm and debug information is muted.
                 1: trainer prints the progress.
                 2: log device placement is printed.
    """

    def __init__(self, model_fn, n_classes, batch_size=32,
                 steps=200, optimizer="Adagrad",
                 learning_rate=0.1, clip_gradients=5.0, class_weight=None,
                 continue_training=False,
                 config=None, verbose=1):
        self.model_fn = model_fn
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.steps = steps
        self.verbose = verbose
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.clip_gradients = clip_gradients
        self.continue_training = continue_training
        self._initialized = False
        self.class_weight = class_weight
        self._config = config

    def _setup_training(self):
        """Sets up graph, model and trainer."""
        # Create config if not given.
        if self._config is None:
            self._config = RunConfig(verbose=self.verbose)
        # Create new graph.
        self._graph = ops.Graph()
        self._graph.add_to_collection("IS_TRAINING", True)
        with self._graph.as_default():
            random_seed.set_random_seed(self._config.tf_random_seed)
            self._global_step = variables.Variable(
                0, name="global_step", trainable=False)

            # Setting up inputs and outputs.
            self._inp, self._out = self._data_feeder.input_builder()

            # If class weights are provided, add them to the graph.
            # Different loss functions can use this tensor by name.
            if self.class_weight:
                self._class_weight_node = constant_op.constant(
                    self.class_weight, name='class_weight')

            # Add histograms for X and y if they are floats.
            if self._data_feeder.input_dtype in (np.float32, np.float64):
                logging_ops.histogram_summary("X", self._inp)
            if self._data_feeder.output_dtype in (np.float32, np.float64):
                logging_ops.histogram_summary("y", self._out)

            # Create model's graph.
            self._model_predictions, self._model_loss = self.model_fn(
                self._inp, self._out)

            # Set up a single operator to merge all the summaries
            self._summaries = logging_ops.merge_all_summaries()

            # Create trainer and augment graph with gradients and optimizer.
            # Additionally creates initialization ops.
            learning_rate = self.learning_rate
            optimizer = self.optimizer
            if callable(learning_rate):
                learning_rate = learning_rate(self._global_step)
            if callable(optimizer):
                optimizer = optimizer(learning_rate)
            self._train = optimizers.optimize_loss(self._model_loss, self._global_step,
                learning_rate=learning_rate,
                optimizer=optimizer, clip_gradients=self.clip_gradients)

            # Update ops during training, e.g. batch_norm_ops
            self._train = control_flow_ops.group(self._train, *ops.get_collection('update_ops'))

            # Get all initializers for all trainable variables.
            self._initializers = variables.initialize_all_variables()

            # Create model's saver capturing all the nodes created up until now.
            self._saver = train.Saver(
                max_to_keep=self._config.keep_checkpoint_max,
                keep_checkpoint_every_n_hours=self._config.keep_checkpoint_every_n_hours)

            # Enable monitor to create validation data dict with appropriate tf placeholders
            self._monitor.create_val_feed_dict(self._inp, self._out)

            # Create session to run model with.
            self._session = session.Session(self._config.tf_master, config=self._config.tf_config)

            # Run parameter initializers.
            self._session.run(self._initializers)

    def _setup_summary_writer(self, logdir):
        """Sets up the summary writer to prepare for later optional visualization."""
        self._summary_writer = train.SummaryWriter(
            os.path.join(logdir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
            graph=self._session.graph)

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
            self._monitor = monitors.default_monitor(verbose=self.verbose)
        else:
            self._monitor = monitor

        if not self.continue_training or not self._initialized:
            # Sets up model and trainer.
            self._setup_training()
            self._initialized = True
        else:
            self._data_feeder.set_placeholders(self._inp, self._out)

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
        trainer.train(
            self._session, self._train, 
            self._model_loss, self._global_step,
            self._data_feeder.get_feed_dict_fn(),
            steps=self.steps,
            monitor=self._monitor,
            summary_writer=self._summary_writer,
            summaries=self._summaries,
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

    def _predict(self, X, axis=-1, batch_size=None):
        if not self._initialized:
            raise _sklearn.NotFittedError()

        # Use the batch size for fitting if the user did not specify one.
        if batch_size is None:
            batch_size = self.batch_size

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

    def predict(self, X, axis=1, batch_size=None):
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
                        it into mini batches. By default the batch_size member
                        variable is used.

        Returns:
            y: array of shape [n_samples]. The predicted classes or predicted
            value.
        """
        return self._predict(X, axis=axis, batch_size=batch_size)

    def predict_proba(self, X, batch_size=None):
        """Predict class probability of the input samples X.

        Args:
            X: array-like matrix, [n_samples, n_features...] or iterator.
            batch_size: If test set is too big, use batch size to split
                        it into mini batches. By default the batch_size
                        member variable is used.

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
            raise _sklearn.NotFittedError()

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

        # Save saver definition.
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

        self._graph = ops.Graph()
        with self._graph.as_default():
            endpoints_filename = os.path.join(path, 'endpoints')
            if not os.path.exists(endpoints_filename):
                raise ValueError("Restore folder doesn't contain endpoints.")
            with gfile.Open(endpoints_filename) as foutputs:
                endpoints = foutputs.read().split('\n')
            graph_filename = os.path.join(path, 'graph.pbtxt')
            if not os.path.exists(graph_filename):
                raise ValueError("Restore folder doesn't contain graph definition.")
            with gfile.Open(graph_filename) as fgraph:
                graph_def = graph_pb2.GraphDef()
                text_format.Merge(fgraph.read(), graph_def)
                (self._inp, self._out,
                 self._model_predictions, self._model_loss) = importer.import_graph_def(
                     graph_def, name='', return_elements=endpoints)
            saver_filename = os.path.join(path, 'saver.pbtxt')
            if not os.path.exists(saver_filename):
                raise ValueError("Restore folder doesn't contain saver definition.")
            with gfile.Open(saver_filename) as fsaver:
                saver_def = train.SaverDef()
                text_format.Merge(fsaver.read(), saver_def)
                self._saver = train.Saver(saver_def=saver_def)

            # Restore trainer
            self._global_step = self._graph.get_tensor_by_name('global_step:0')
            self._train = self._graph.get_operation_by_name('train')

            # Restore summaries.
            self._summaries = self._graph.get_operation_by_name('MergeSummary/MergeSummary')

            # Restore session.
            if not isinstance(self._config, RunConfig):
                self._config = RunConfig(verbose=self.verbose)
            self._session = session.Session(
                self._config.tf_master,
                config=self._config.tf_config)
            checkpoint_path = train.latest_checkpoint(path)
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
    def restore(cls, path, config=None):
        """Restores model from give path.

        Args:
            path: Path to the checkpoints and other model information.
            config: RunConfig object that controls the configurations of the session,
                e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be reconfigured.

        Returns:
            Estiamator, object of the subclass of TensorFlowEstimator.
        """
        model_def_filename = os.path.join(path, 'model.def')
        if not os.path.exists(model_def_filename):
            raise ValueError("Restore folder doesn't contain model definition.")
        # list of parameters that are allowed to be reconfigured
        reconfigurable_params = ['_config']
        _config = config
        with gfile.Open(model_def_filename) as fmodel:
            model_def = json.loads(fmodel.read())
            # TensorFlow binding requires parameters to be strings not unicode.
            # Only issue in Python2.
            for key, value in model_def.items():
                if (isinstance(value, string_types) and
                        not isinstance(value, str)):
                    model_def[key] = str(value)
                if key in reconfigurable_params:
                    new_value = locals()[key]
                    if new_value is not None:
                        model_def[key] = new_value
        class_name = model_def.pop('class_name')
        if class_name == 'TensorFlowEstimator':
            custom_estimator = TensorFlowEstimator(model_fn=None, **model_def)
            custom_estimator._restore(path)
            return custom_estimator

        # To avoid cyclical dependencies, import inside the function instead of
        # the beginning of the file.
        from tensorflow.contrib.learn.python.learn import estimators
        # Estimator must be one of the defined estimators in the __init__ file.
        estimator = getattr(estimators, class_name)(**model_def)
        estimator._restore(path)
        return estimator

