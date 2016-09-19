# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Estimator for Dynamic RNNs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.training import momentum as momentum_opt
from tensorflow.python.training import optimizer as opt


# TODO(jamieas): move `_padding_mask` to array_ops.
def _padding_mask(sequence_lengths, padded_length):
  """Creates a mask used for calculating losses with padded input.

  Args:
    sequence_lengths: a `Tensor` of shape `[batch_size]` containing the unpadded
      length of  each sequence.
    padded_length: a scalar `Tensor` indicating the length of the sequences
      after padding
  Returns:
    A boolean `Tensor` M of shape `[batch_size, padded_length]` where
    `M[i, j] == True` when `lengths[i] > j`.

  """
  range_tensor = math_ops.range(padded_length)
  return math_ops.less(array_ops.expand_dims(range_tensor, 0),
                       array_ops.expand_dims(sequence_lengths, 1))


def _mask_activations_and_targets(activations, targets, sequence_lengths):
  """Remove entries outside `sequence_lengths` and returned flattened results.

  Args:
    activations: output of the RNN, shape `[batch_size, padded_length, k]`.
    targets: target values, shape `[batch_size, padded_length]`.
    sequence_lengths: a `Tensor` of shape `[batch_size]` with the unpadded
      length of each sequence. If `None`, then each sequence is unpadded.

  Returns:
    activations_masked: `logit` values with those beyond `sequence_lengths`
    removed for each batch. Batches are then concatenated. Shape
      `[tf.sum(sequence_lengths), k]` if `sequence_lengths` is not `None` and
      shape `[batch_size * padded_length, k]` otherwise.
    targets_masked: target values after removing unneeded entries. Shape
      `[tf.sum(sequence_lengths)]` if `sequence_lengths` is not `None` and shape
      `[batch_size * padded_length]` otherwise.
  """
  with ops.name_scope('mask_activations_and_targets',
                      values=[activations, targets, sequence_lengths]):
    targets_shape = array_ops.shape(targets)
    batch_size = targets_shape[0]
    padded_length = targets_shape[1]
    if sequence_lengths is None:
      flattened_dimension = padded_length * batch_size
      activations_masked = array_ops.reshape(activations,
                                             [flattened_dimension, -1])
      targets_masked = array_ops.reshape(targets, [flattened_dimension])
    else:
      mask = _padding_mask(sequence_lengths, padded_length)
      activations_masked = array_ops.boolean_mask(activations, mask)
      targets_masked = array_ops.boolean_mask(targets, mask)
    return activations_masked, targets_masked


def _select_last_activations(activations, sequence_lengths):
  """Selects the nth set of activations for each n in `sequence_length`.

  Reuturns a `Tensor` of shape `[batch_size, k]`. If `sequence_length` is not
  `None`, then `output[i, :] = activations[i, sequence_length[i], :]`. If
  `sequence_length` is `None`, then `output[i, :] = activations[i, -1, :]`.

  Args:
    activations: a `Tensor` with shape `[batch_size, padded_length, k]`.
    sequence_lengths: a `Tensor` with shape `[batch_size]` or `None`.
  Returns:
    A `Tensor` of shape `[batch_size, k]`.
  """
  with ops.name_scope('select_last_activations',
                      values=[activations, sequence_lengths]):
    activations_shape = array_ops.shape(activations)
    batch_size = activations_shape[0]
    padded_length = activations_shape[1]
    num_label_columns = activations_shape[2]
    if sequence_lengths is None:
      sequence_lengths = padded_length
    reshaped_activations = array_ops.reshape(activations,
                                             [-1, num_label_columns])
    indices = math_ops.range(batch_size) * padded_length + sequence_lengths - 1
    last_activations = array_ops.gather(reshaped_activations, indices)
    last_activations.set_shape(
        [activations.get_shape()[0], activations.get_shape()[2]])
    return last_activations


@six.add_metaclass(abc.ABCMeta)
class _DynamicRNNEstimator(estimator.BaseEstimator):
  """Estimator that uses a dynamic RNN for sequences."""

  def __init__(self,
               cell,
               target_column,
               optimizer,
               model_dir=None,
               config=None,
               gradient_clipping_norm=None,
               inputs_key='inputs',
               sequence_length_key='sequence_length',
               initial_state_key='initial_state',
               dtype=None,
               parallel_iterations=None,
               swap_memory=False,
               name=None):
    """Initialize `DynamicRNNEstimator`.

    Args:
      cell: an initialized `RNNCell` to be used in the RNN.
      target_column: an initialized `TargetColumn`, used to calculate loss and
        metrics.
      optimizer: an initialized `tensorflow.Optimizer`.
      model_dir: The directory in which to save and restore the model graph,
        parameters, etc.
      config: A `RunConfig` instance.
      gradient_clipping_norm: parameter used for gradient clipping. If `None`,
        then no clipping is performed.
      inputs_key: the key for input values in the features dict passed to
        `fit()`.
      sequence_length_key: the key for the sequence length tensor in the
        features dict passed to `fit()`.
      initial_state_key: the key for input values in the features dict passed to
        `fit()`.
      dtype: Parameter passed ot `dynamic_rnn`. The dtype of the state and
        output returned by `RNNCell`.
      parallel_iterations: Parameter passed ot `dynamic_rnn`. The number of
        iterations to run in parallel.
      swap_memory: Parameter passed ot `dynamic_rnn`. Transparently swap the
        tensors produced in forward inference but needed for back prop from GPU
        to CPU.
      name: Optional name for the `Estimator`.
    """
    super(_DynamicRNNEstimator, self).__init__(
        model_dir=model_dir, config=config)
    self._cell = cell
    self._target_column = target_column
    self._optimizer = optimizer
    self._gradient_clipping_norm = gradient_clipping_norm
    self._inputs_key = inputs_key
    self._sequence_length_key = sequence_length_key
    self._initial_state_key = initial_state_key
    self._dtype = dtype or dtypes.float32
    self._parallel_iterations = parallel_iterations
    self._swap_memory = swap_memory
    self._name = name or 'DynamicRnnEstimator'

  def _construct_rnn(self, features):
    """Apply an RNN to `features`.

    The `features` dict must contain `self._inputs_key`, and the corresponding
    input should be a `Tensor` of shape `[batch_size, padded_length, k]`
    where `k` is the dimension of the input for each element of a sequence.

    `activations` has shape `[batch_size, sequence_length, n]` where `n` is
    `self._target_column.num_label_columns`. In the case of a multiclass
    classifier, `n` is the number of classes.

    `final_state` has shape determined by `self._cell` and its dtype must match
    `self._dtype`.

    Args:
      features: a `dict` containing the input for the RNN and (optionally) an
        initial state and information about sequence lengths.

    Returns:
      activations: the output of the RNN, projected to the appropriate number of
        dimensions.
      final_state: the final state output by the RNN.

    Raises:
      KeyError: if `features` does not contain `self._inputs_key`.
    """
    with ops.name_scope('RNN'):
      inputs = features.get(self._inputs_key)
      if inputs is None:
        raise KeyError('features must contain the key {}'.format(
            self._inputs_key))
      if inputs.dtype != self._dtype:
        inputs = math_ops.cast(inputs, self._dtype)
      initial_state = features.get(self._initial_state_key)
      rnn_outputs, final_state = rnn.dynamic_rnn(
          cell=self._cell,
          inputs=inputs,
          initial_state=initial_state,
          dtype=self._dtype,
          parallel_iterations=self._parallel_iterations,
          swap_memory=self._swap_memory,
          time_major=False)
      activations = layers.fully_connected(
          inputs=rnn_outputs,
          num_outputs=self._target_column.num_label_columns,
          activation_fn=None,
          trainable=False)
      return activations, final_state

  @abc.abstractmethod
  def _activations_to_loss(self, features, activations, targets):
    """Map `activations` and `targets` to a loss `Tensor`.

    `activations` has shape `[batch_size, padded_length,
     self._target_column.num_label_columns]`. It is the output of
    `_construct_rnn`.

    `targets` is a `Tensor` of shape `[batch_size, padded_length]`. The type
    of `targets` depends on what type of `TargetColumn` is being used.

    Args:
      features: a `dict` containing the input and (optionally) sequence length
        information and initial state. This is the same `features` passed to
        `_construct_rnn`.
      activations: a `Tensor` of activations representing the output of the RNN.
      targets: a `Tensor` of target values.

    Returns:
      loss: A scalar `Tensor` representing the aggregated loss for the batch.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _activations_to_predictions(self, features, activations):
    """Map `activations` to predictions.

    `activations` has shape [batch_size, time, num_labels]. `TargetColumn`s
    require shape [n, num_labels]. `activations` is flattened before being
    converted to labels. Afterwards, its shape is reconstituted.

    Args:
      features: a `dict` containing the input and (optionally) sequence length
        information and initial state.
      activations: logit values returned by `_construct_rnn`.

    Returns:
      A set of predictions. The type of prediction is dependent on
      `_target_column`.
    """
    raise NotImplementedError()

  def _process_gradients(self, gradients_vars):
    """Process gradients (e.g. clipping) before applying them to weights."""
    with ops.name_scope('process_gradients'):
      gradients, variables = zip(*gradients_vars)
      if self._gradient_clipping_norm is not None:
        gradients, _ = clip_ops.clip_by_global_norm(
            gradients, self._gradient_clipping_norm)
      return zip(gradients, variables)

  def _loss_to_train_op(self, loss):
    """Map `loss` to a training op."""
    with ops.name_scope('loss_to_train_op'):
      trainable_variables = ops.get_default_graph().get_collection(
          ops.GraphKeys.TRAINABLE_VARIABLES)
      global_step = contrib_framework.get_global_step()
      gradients = self._optimizer.compute_gradients(
          loss=loss, var_list=trainable_variables)
      processed_gradients = self._process_gradients(gradients)
      return self._optimizer.apply_gradients(
          processed_gradients, global_step=global_step)

  @abc.abstractmethod
  def _activations_to_eval_ops(self, features, activations, targets, metrics):
    """Map `activations` to eval operations.

    `activations` has shape [batch_size, time, num_labels]. `TargetColumn`s
    require shape [n, num_labels]. `activations` is flattened before being
    converted to labels. Afterwards, its shape is reconstituted.

    Args:
      features: a `dict` containing the input and (optionally) sequence length
        information and initial state.
      activations: logit values returned by `_construct_rnn`.
      targets: a `Tensor` of target values.
      metrics: a list of `Metric`s to evaluate. Possibly `None`.

    Returns:
      A dict of named eval ops.
    """
    raise NotImplementedError()

  def _get_train_ops(self, features, targets):
    with ops.name_scope(self._name):
      if isinstance(features, ops.Tensor):
        features = {self._inputs_key: features}
      activations, _ = self._construct_rnn(features)
      loss = self._activations_to_loss(features, activations, targets)
      train_op = self._loss_to_train_op(loss)
      return train_op, loss

  def _get_eval_ops(self, features, targets, metrics):
    with ops.name_scope(self._name):
      if isinstance(features, ops.Tensor):
        features = {self._inputs_key: features}
      activations, _ = self._construct_rnn(features)
      return self._activations_to_eval_ops(features, activations, targets,
                                           metrics)

  def _get_predict_ops(self, features):
    with ops.name_scope(self._name):
      if isinstance(features, ops.Tensor):
        features = {self._inputs_key: features}
      activations, state = self._construct_rnn(features)
      predictions = self._activations_to_predictions(features, activations)
      return {'predictions': predictions, 'state': state}


class _MultiValueRNNEstimator(_DynamicRNNEstimator):
  """An `Estimator` that maps sequences of inputs to sequences of outputs."""

  def _activations_to_loss(self, features, activations, targets):
    sequence_length = features.get(self._sequence_length_key)
    # Mask the activations and targets past `sequence_length`. Note that the
    # `Tensor`s returned by `_mask_activations_and_targets` are flattened.
    with ops.name_scope('activations_to_loss'):
      activations_masked, targets_masked = _mask_activations_and_targets(
          activations, targets, sequence_length)
      return self._target_column.loss(activations_masked, targets_masked,
                                      features)

  def _activations_to_predictions(self, unused_features, activations):
    with ops.name_scope('activations_to_predictions'):
      activations_shape = array_ops.shape(activations)
      flattened_activations = array_ops.reshape(activations,
                                                [-1, activations_shape[2]])
      predictions = self._target_column.activations_to_predictions(
          flattened_activations, proba=False)
      reshaped_predictions = array_ops.reshape(
          predictions, [activations_shape[0], activations_shape[1], -1])
      return array_ops.squeeze(reshaped_predictions, [2])

  def _activations_to_eval_ops(self, features, activations, targets, metrics):
    with ops.name_scope('activations_to_eval_ops'):
      activations_masked, targets_masked = _mask_activations_and_targets(
          activations, targets, features.get(self._sequence_length_key))

      return self._target_column.get_eval_ops(features=features,
                                              logits=activations_masked,
                                              targets=targets_masked,
                                              metrics=metrics)


class _SingleValueRNNEstimator(_DynamicRNNEstimator):
  """An `Estimator` that maps sequences of inputs to single outputs."""

  def _activations_to_loss(self, features, activations, targets):
    with ops.name_scope('activations_to_loss'):
      sequence_lengths = features.get(self._sequence_length_key)
      last_activations = _select_last_activations(activations, sequence_lengths)
      return self._target_column.loss(last_activations, targets, features)

  def _activations_to_predictions(self, features, activations):
    with ops.name_scope('activations_to_predictions'):
      sequence_lengths = features.get(self._sequence_length_key)
      last_activations = _select_last_activations(activations, sequence_lengths)
      return self._target_column.activations_to_predictions(
          last_activations, proba=False)

  def _activations_to_eval_ops(self, features, activations, targets, metrics):
    with ops.name_scope('activations_to_eval_ops'):
      sequence_lengths = features.get(self._sequence_length_key)
      last_activations = _select_last_activations(activations, sequence_lengths)
      return self._target_column.get_eval_ops(features=features,
                                              logits=last_activations,
                                              targets=targets,
                                              metrics=metrics)


def _get_optimizer(optimizer_type, learning_rate, momentum):
  """Constructs and returns an `Optimizer`.

  Args:
    optimizer_type: either a string identifying the `Optimizer` type, or a
      subclass of `Optimizer`.
    learning_rate: the learning rate used to initialize the `Optimizer`.
    momentum: used only when `optimizer_type` is 'Momentum'.
  Returns:
    An initialized `Optimizer`.
  Raises:
    ValueError: `optimizer_type` is an invalid optimizer name.
    TypeError: `optimizer_type` is not a string or a subclass of `Optimizer`.
  """
  if isinstance(optimizer_type, str):
    optimizer_type = layers.OPTIMIZER_CLS_NAMES.get(optimizer_type)
  if optimizer_type is None:
    raise ValueError('optimizer must be one of {}; got "{}".'.format(
        list(layers.OPTIMIZER_CLS_NAMES.keys()), optimizer_type))
  if not issubclass(optimizer_type, opt.Optimizer):
    raise TypeError(
        'optimizer_type must be a subclass of Optimizer or one of {}'.format(
            list(layers.OPTIMZIER.keys())))
  if optimizer_type == momentum_opt.MomentumOptimizer:
    return optimizer_type(learning_rate, momentum)
  return optimizer_type(learning_rate)


_CELL_TYPES = {'basic_rnn': rnn_cell.BasicRNNCell,
               'lstm': rnn_cell.LSTMCell,
               'gru': rnn_cell.GRUCell,}


def _get_rnn_cell(cell_type, num_units, num_layers):
  """Constructs and return an `RNNCell`.

  Args:
    cell_type: either a string identifying the `RNNCell` type, or a subclass of
      `RNNCell`.
    num_units: the number of units in the `RNNCell`.
    num_layers: the number of layers in the RNN.
  Returns:
    An initialized `RNNCell`.
  Raises:
    ValueError: `cell_type` is an invalid `RNNCell` name.
    TypeError: `cell_type` is not a string or a subclass of `RNNCell`.
  """
  if isinstance(cell_type, str):
    cell_type = _CELL_TYPES.get(cell_type)
    if cell_type is None:
      raise ValueError('The supported cell types are {}; got {}'.format(
          list(_CELL_TYPES.keys()), cell_type))
  if not issubclass(cell_type, rnn_cell.RNNCell):
    raise TypeError(
        'cell_type must be a subclass of RNNCell or one of {}.'.format(
            list(_CELL_TYPES.keys())))
  cell = cell_type(num_units=num_units)
  if num_layers > 1:
    cell = rnn_cell.MultiRNNCell(
        [cell] * num_layers, state_is_tuple=True)
  return cell


def multi_value_rnn_regressor(num_units,
                              cell_type='basic_rnn',
                              cell_dtype=dtypes.float32,
                              num_rnn_layers=1,
                              optimizer_type='SGD',
                              learning_rate=0.1,
                              momentum=None,
                              gradient_clipping_norm=10.0,
                              model_dir=None,
                              config=None):
  """Creates a RNN `Estimator` that predicts sequences of values.

  Args:
    num_units: the size of the RNN cells.
    cell_type: subclass of `RNNCell` or one of 'basic_rnn,' 'lstm' or 'gru'.
    cell_dtype: the dtype of the state and output for the given `cell_type`.
    num_rnn_layers: number of RNN layers.
    optimizer_type: the type of optimizer to use. Either a subclass of
      `Optimizer` or a string.
    learning_rate: learning rate.
    momentum: momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    model_dir: directory to use for The directory in which to save and restore
      the model graph, parameters, etc.
    config: A `RunConfig` instance.
  Returns:
    An initialized instance of `_MultiValueRNNEstimator`.
  """
  optimizer = _get_optimizer(optimizer_type, learning_rate, momentum)
  cell = _get_rnn_cell(cell_type, num_units, num_rnn_layers)
  target_column = layers.regression_target()
  return _MultiValueRNNEstimator(cell,
                                 target_column,
                                 optimizer,
                                 model_dir,
                                 config,
                                 gradient_clipping_norm,
                                 dtype=cell_dtype)


def multi_value_rnn_classifier(num_classes,
                               num_units,
                               cell_type='basic_rnn',
                               cell_dtype=dtypes.float32,
                               num_rnn_layers=1,
                               optimizer_type='SGD',
                               learning_rate=0.1,
                               momentum=None,
                               gradient_clipping_norm=10.0,
                               model_dir=None,
                               config=None):
  """Creates a RNN `Estimator` that predicts sequences of labels.

  Args:
    num_classes: the number of classes for categorization.
    num_units: the size of the RNN cells.
    cell_type: subclass of `RNNCell` or one of 'basic_rnn,' 'lstm' or 'gru'.
    cell_dtype: the dtype of the state and output for the given `cell_type`.
    num_rnn_layers: number of RNN layers.
    optimizer_type: the type of optimizer to use. Either a subclass of
      `Optimizer` or a string.
    learning_rate: learning rate.
    momentum: momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    model_dir: directory to use for The directory in which to save and restore
      the model graph, parameters, etc.
    config: A `RunConfig` instance.
  Returns:
    An initialized instance of `_MultiValueRNNEstimator`.
  """
  optimizer = _get_optimizer(optimizer_type, learning_rate, momentum)
  cell = _get_rnn_cell(cell_type, num_units, num_rnn_layers)
  target_column = layers.multi_class_target(n_classes=num_classes)
  return _MultiValueRNNEstimator(cell,
                                 target_column,
                                 optimizer,
                                 model_dir,
                                 config,
                                 gradient_clipping_norm,
                                 dtype=cell_dtype)


def single_value_rnn_regressor(num_units,
                               cell_type='basic_rnn',
                               cell_dtype=dtypes.float32,
                               num_rnn_layers=1,
                               optimizer_type='SGD',
                               learning_rate=0.1,
                               momentum=None,
                               gradient_clipping_norm=10.0,
                               model_dir=None,
                               config=None):
  """Create a RNN `Estimator` that predicts single values.

  Args:
    num_units: the size of the RNN cells.
    cell_type: subclass of `RNNCell` or one of 'basic_rnn,' 'lstm' or 'gru'.
    cell_dtype: the dtype of the state and output for the given `cell_type`.
    num_rnn_layers: number of RNN layers.
    optimizer_type: the type of optimizer to use. Either a subclass of
      `Optimizer` or a string.
    learning_rate: learning rate.
    momentum: momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    model_dir: directory to use for The directory in which to save and restore
      the model graph, parameters, etc.
    config: A `RunConfig` instance.
  Returns:
    An initialized instance of `_MultiValueRNNEstimator`.
  """
  optimizer = _get_optimizer(optimizer_type, learning_rate, momentum)
  cell = _get_rnn_cell(cell_type, num_units, num_rnn_layers)
  target_column = layers.regression_target()
  return _SingleValueRNNEstimator(cell,
                                  target_column,
                                  optimizer,
                                  model_dir,
                                  config,
                                  gradient_clipping_norm,
                                  dtype=cell_dtype)


def single_value_rnn_classifier(num_classes,
                                num_units,
                                cell_type='basic_rnn',
                                cell_dtype=dtypes.float32,
                                num_rnn_layers=1,
                                optimizer_type='SGD',
                                learning_rate=0.1,
                                momentum=None,
                                gradient_clipping_norm=10.0,
                                model_dir=None,
                                config=None):
  """Creates a RNN `Estimator` that predicts single labels.

  Args:
    num_classes: the number of classes for categorization.
    num_units: the size of the RNN cells.
    cell_type: subclass of `RNNCell` or one of 'basic_rnn,' 'lstm' or 'gru'.
    cell_dtype: the dtype of the state and output for the given `cell_type`.
    num_rnn_layers: number of RNN layers.
    optimizer_type: the type of optimizer to use. Either a subclass of
      `Optimizer` or a string.
    learning_rate: learning rate.
    momentum: momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    model_dir: directory to use for The directory in which to save and restore
      the model graph, parameters, etc.
    config: A `RunConfig` instance.
  Returns:
    An initialized instance of `_MultiValueRNNEstimator`.
  """
  optimizer = _get_optimizer(optimizer_type, learning_rate, momentum)
  cell = _get_rnn_cell(cell_type, num_units, num_rnn_layers)
  target_column = layers.multi_class_target(n_classes=num_classes)
  return _SingleValueRNNEstimator(cell,
                                  target_column,
                                  optimizer,
                                  model_dir,
                                  config,
                                  gradient_clipping_norm,
                                  dtype=cell_dtype)
