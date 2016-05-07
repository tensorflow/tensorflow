"""Recurrent Neural Network estimators."""
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

from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators.base import TensorFlowEstimator
from tensorflow.contrib.learn.python.learn import models


def null_input_op_fn(X):
  """This function does no transformation on the inputs, used as default"""
  return X


class TensorFlowRNNClassifier(TensorFlowEstimator, _sklearn.ClassifierMixin):
  """TensorFlow RNN Classifier model.

  Parameters:
    rnn_size: The size for rnn cell, e.g. size of your word embeddings.
    cell_type: The type of rnn cell, including rnn, gru, and lstm.
    num_layers: The number of layers of the rnn model.
    input_op_fn: Function that will transform the input tensor, such as
      creating word embeddings, byte list, etc. This takes
      an argument X for input and returns transformed X.
    bidirectional: boolean, Whether this is a bidirectional rnn.
    sequence_length: If sequence_length is provided, dynamic calculation is
      performed. This saves computational time when unrolling past max sequence
      length.
    initial_state: An initial state for the RNN. This must be a tensor of
      appropriate type and shape [batch_size x cell.state_size].
    n_classes: Number of classes in the target.
    batch_size: Mini batch size.
    steps: Number of steps to run over data.
    optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
    learning_rate: If this is constant float value, no decay function is
      used. Instead, a customized decay function can be passed that accepts
      global_step as parameter and returns a Tensor.
      e.g. exponential decay function:
      def exp_decay(global_step):
          return tf.train.exponential_decay(
              learning_rate=0.1, global_step,
              decay_steps=2, decay_rate=0.001)
    class_weight: None or list of n_classes floats. Weight associated with
      classes for loss computation. If not given, all classes are
      supposed to have weight one.
    continue_training: when continue_training is True, once initialized
      model will be continuely trained on every call of fit.
    config: RunConfig object that controls the configurations of the session,
      e.g. num_cores, gpu_memory_fraction, etc.
  """

  def __init__(self,
               rnn_size,
               n_classes,
               cell_type='gru',
               num_layers=1,
               input_op_fn=null_input_op_fn,
               initial_state=None,
               bidirectional=False,
               sequence_length=None,
               batch_size=32,
               steps=50,
               optimizer='Adagrad',
               learning_rate=0.1,
               class_weight=None,
               clip_gradients=5.0,
               continue_training=False,
               config=None,
               verbose=1):

    self.rnn_size = rnn_size
    self.cell_type = cell_type
    self.input_op_fn = input_op_fn
    self.bidirectional = bidirectional
    self.num_layers = num_layers
    self.sequence_length = sequence_length
    self.initial_state = initial_state
    super(TensorFlowRNNClassifier, self).__init__(
        model_fn=self._model_fn,
        n_classes=n_classes,
        batch_size=batch_size,
        steps=steps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        class_weight=class_weight,
        clip_gradients=clip_gradients,
        continue_training=continue_training,
        config=config,
        verbose=verbose)

  def _model_fn(self, X, y):
    return models.get_rnn_model(self.rnn_size, self.cell_type, self.num_layers,
                                self.input_op_fn, self.bidirectional,
                                models.logistic_regression,
                                self.sequence_length, self.initial_state)(X, y)

  @property
  def bias_(self):
    """Returns bias of the rnn layer."""
    return self.get_tensor_value('logistic_regression/bias:0')

  @property
  def weights_(self):
    """Returns weights of the rnn layer."""
    return self.get_tensor_value('logistic_regression/weights:0')


class TensorFlowRNNRegressor(TensorFlowEstimator, _sklearn.RegressorMixin):
  """TensorFlow RNN Regressor model.

  Parameters:
    rnn_size: The size for rnn cell, e.g. size of your word embeddings.
    cell_type: The type of rnn cell, including rnn, gru, and lstm.
    num_layers: The number of layers of the rnn model.
    input_op_fn: Function that will transform the input tensor, such as
      creating word embeddings, byte list, etc. This takes
      an argument X for input and returns transformed X.
    bidirectional: boolean, Whether this is a bidirectional rnn.
    sequence_length: If sequence_length is provided, dynamic calculation is
      performed. This saves computational time when unrolling past max sequence
      length.
    initial_state: An initial state for the RNN. This must be a tensor of
      appropriate type and shape [batch_size x cell.state_size].
    batch_size: Mini batch size.
    steps: Number of steps to run over data.
    optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
    learning_rate: If this is constant float value, no decay function is
      used. Instead, a customized decay function can be passed that accepts
      global_step as parameter and returns a Tensor.
      e.g. exponential decay function:
      def exp_decay(global_step):
          return tf.train.exponential_decay(
              learning_rate=0.1, global_step,
              decay_steps=2, decay_rate=0.001)
    continue_training: when continue_training is True, once initialized
      model will be continuely trained on every call of fit.
    config: RunConfig object that controls the configurations of the
      session, e.g. num_cores, gpu_memory_fraction, etc.
    verbose: Controls the verbosity, possible values:
      0: the algorithm and debug information is muted.
      1: trainer prints the progress.
      2: log device placement is printed.
  """

  def __init__(self,
               rnn_size,
               cell_type='gru',
               num_layers=1,
               input_op_fn=null_input_op_fn,
               initial_state=None,
               bidirectional=False,
               sequence_length=None,
               n_classes=0,
               batch_size=32,
               steps=50,
               optimizer='Adagrad',
               learning_rate=0.1,
               clip_gradients=5.0,
               continue_training=False,
               config=None,
               verbose=1):
    self.rnn_size = rnn_size
    self.cell_type = cell_type
    self.input_op_fn = input_op_fn
    self.bidirectional = bidirectional
    self.num_layers = num_layers
    self.sequence_length = sequence_length
    self.initial_state = initial_state
    super(TensorFlowRNNRegressor, self).__init__(
        model_fn=self._model_fn,
        n_classes=n_classes,
        batch_size=batch_size,
        steps=steps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        clip_gradients=clip_gradients,
        continue_training=continue_training,
        config=config,
        verbose=verbose)

  def _model_fn(self, X, y):
    return models.get_rnn_model(self.rnn_size, self.cell_type, self.num_layers,
                                self.input_op_fn, self.bidirectional,
                                models.linear_regression, self.sequence_length,
                                self.initial_state)(X, y)

  @property
  def bias_(self):
    """Returns bias of the rnn layer."""
    return self.get_tensor_value('linear_regression/bias:0')

  @property
  def weights_(self):
    """Returns weights of the rnn layer."""
    return self.get_tensor_value('linear_regression/weights:0')
