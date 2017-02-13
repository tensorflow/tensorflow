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

from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib.framework.python.framework import deprecated
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.training import momentum as momentum_opt
from tensorflow.python.util import nest


class PredictionType(object):
  SINGLE_VALUE = 1
  MULTIPLE_VALUE = 2


# NOTE(jamieas): As of February 7, 2017, some of the `RNNKeys` have been removed
# and replaced with values from `prediction_key.PredictionKey`. The key
# `RNNKeys.PREDICTIONS_KEY` has been replaced by
# `prediction_key.PredictionKey.SCORES` for regression and
# `prediction_key.PredictionKey.CLASSES` for classification. The key
# `RNNKeys.PROBABILITIES_KEY` has been replaced by
# `prediction_key.PredictionKey.PROBABILITIES`.
class RNNKeys(object):
  SEQUENCE_LENGTH_KEY = 'sequence_length'
  STATE_PREFIX = 'rnn_cell_state'

_CELL_TYPES = {'basic_rnn': contrib_rnn.BasicRNNCell,
               'lstm': contrib_rnn.LSTMCell,
               'gru': contrib_rnn.GRUCell,}


def mask_activations_and_labels(activations, labels, sequence_lengths):
  """Remove entries outside `sequence_lengths` and returned flattened results.

  Args:
    activations: Output of the RNN, shape `[batch_size, padded_length, k]`.
    labels: Label values, shape `[batch_size, padded_length]`.
    sequence_lengths: A `Tensor` of shape `[batch_size]` with the unpadded
      length of each sequence. If `None`, then each sequence is unpadded.

  Returns:
    activations_masked: `logit` values with those beyond `sequence_lengths`
      removed for each batch. Batches are then concatenated. Shape
      `[tf.sum(sequence_lengths), k]` if `sequence_lengths` is not `None` and
      shape `[batch_size * padded_length, k]` otherwise.
    labels_masked: Label values after removing unneeded entries. Shape
      `[tf.sum(sequence_lengths)]` if `sequence_lengths` is not `None` and shape
      `[batch_size * padded_length]` otherwise.
  """
  with ops.name_scope('mask_activations_and_labels',
                      values=[activations, labels, sequence_lengths]):
    labels_shape = array_ops.shape(labels)
    batch_size = labels_shape[0]
    padded_length = labels_shape[1]
    if sequence_lengths is None:
      flattened_dimension = padded_length * batch_size
      activations_masked = array_ops.reshape(activations,
                                             [flattened_dimension, -1])
      labels_masked = array_ops.reshape(labels, [flattened_dimension])
    else:
      mask = array_ops.sequence_mask(sequence_lengths, padded_length)
      activations_masked = array_ops.boolean_mask(activations, mask)
      labels_masked = array_ops.boolean_mask(labels, mask)
    return activations_masked, labels_masked


def select_last_activations(activations, sequence_lengths):
  """Selects the nth set of activations for each n in `sequence_length`.

  Reuturns a `Tensor` of shape `[batch_size, k]`. If `sequence_length` is not
  `None`, then `output[i, :] = activations[i, sequence_length[i], :]`. If
  `sequence_length` is `None`, then `output[i, :] = activations[i, -1, :]`.

  Args:
    activations: A `Tensor` with shape `[batch_size, padded_length, k]`.
    sequence_lengths: A `Tensor` with shape `[batch_size]` or `None`.
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


def _get_state_name(i):
  """Constructs the name string for state component `i`."""
  return '{}_{}'.format(RNNKeys.STATE_PREFIX, i)


def state_tuple_to_dict(state):
  """Returns a dict containing flattened `state`.

  Args:
    state: A `Tensor` or a nested tuple of `Tensors`. All of the `Tensor`s must
    have the same rank and agree on all dimensions except the last.

  Returns:
    A dict containing the `Tensor`s that make up `state`. The keys of the dict
    are of the form "STATE_PREFIX_i" where `i` is the place of this `Tensor`
    in a depth-first traversal of `state`.
  """
  with ops.name_scope('state_tuple_to_dict'):
    flat_state = nest.flatten(state)
    state_dict = {}
    for i, state_component in enumerate(flat_state):
      state_name = _get_state_name(i)
      state_value = (None if state_component is None
                     else array_ops.identity(state_component, name=state_name))
      state_dict[state_name] = state_value
  return state_dict


def dict_to_state_tuple(input_dict, cell):
  """Reconstructs nested `state` from a dict containing state `Tensor`s.

  Args:
    input_dict: A dict of `Tensor`s.
    cell: An instance of `RNNCell`.
  Returns:
    If `input_dict` does not contain keys 'STATE_PREFIX_i' for `0 <= i < n`
    where `n` is the number of nested entries in `cell.state_size`, this
    function returns `None`. Otherwise, returns a `Tensor` if `cell.state_size`
    is an `int` or a nested tuple of `Tensor`s if `cell.state_size` is a nested
    tuple.
  Raises:
    ValueError: State is partially specified. The `input_dict` must contain
      values for all state components or none at all.
  """
  flat_state_sizes = nest.flatten(cell.state_size)
  state_tensors = []
  with ops.name_scope('dict_to_state_tuple'):
    for i, state_size in enumerate(flat_state_sizes):
      state_name = _get_state_name(i)
      state_tensor = input_dict.get(state_name)
      if state_tensor is not None:
        rank_check = check_ops.assert_rank(
            state_tensor, 2, name='check_state_{}_rank'.format(i))
        shape_check = check_ops.assert_equal(
            array_ops.shape(state_tensor)[1],
            state_size,
            name='check_state_{}_shape'.format(i))
        with ops.control_dependencies([rank_check, shape_check]):
          state_tensor = array_ops.identity(state_tensor, name=state_name)
        state_tensors.append(state_tensor)
    if not state_tensors:
      return None
    elif len(state_tensors) == len(flat_state_sizes):
      dummy_state = cell.zero_state(batch_size=1, dtype=dtypes.bool)
      return nest.pack_sequence_as(dummy_state, state_tensors)
    else:
      raise ValueError(
          'RNN state was partially specified.'
          'Expected zero or {} state Tensors; got {}'.
          format(len(flat_state_sizes), len(state_tensors)))


def _concatenate_context_input(sequence_input, context_input):
  """Replicates `context_input` accross all timesteps of `sequence_input`.

  Expands dimension 1 of `context_input` then tiles it `sequence_length` times.
  This value is appended to `sequence_input` on dimension 2 and the result is
  returned.

  Args:
    sequence_input: A `Tensor` of dtype `float32` and shape `[batch_size,
      padded_length, d0]`.
    context_input: A `Tensor` of dtype `float32` and shape `[batch_size, d1]`.

  Returns:
    A `Tensor` of dtype `float32` and shape `[batch_size, padded_length,
    d0 + d1]`.

  Raises:
    ValueError: If `sequence_input` does not have rank 3 or `context_input` does
      not have rank 2.
  """
  seq_rank_check = check_ops.assert_rank(
      sequence_input,
      3,
      message='sequence_input must have rank 3',
      data=[array_ops.shape(sequence_input)])
  seq_type_check = check_ops.assert_type(
      sequence_input,
      dtypes.float32,
      message='sequence_input must have dtype float32; got {}.'.format(
          sequence_input.dtype))
  ctx_rank_check = check_ops.assert_rank(
      context_input,
      2,
      message='context_input must have rank 2',
      data=[array_ops.shape(context_input)])
  ctx_type_check = check_ops.assert_type(
      context_input,
      dtypes.float32,
      message='context_input must have dtype float32; got {}.'.format(
          context_input.dtype))
  with ops.control_dependencies(
      [seq_rank_check, seq_type_check, ctx_rank_check, ctx_type_check]):
    padded_length = array_ops.shape(sequence_input)[1]
    tiled_context_input = array_ops.tile(
        array_ops.expand_dims(context_input, 1),
        array_ops.concat([[1], [padded_length], [1]], 0))
  return array_ops.concat([sequence_input, tiled_context_input], 2)


def build_sequence_input(features,
                         sequence_feature_columns,
                         context_feature_columns,
                         weight_collections=None,
                         scope=None):
  """Combine sequence and context features into input for an RNN.

  Args:
    features: A `dict` containing the input and (optionally) sequence length
      information and initial state.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features i.e. features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    weight_collections: List of graph collections to which weights are added.
    scope: Optional scope, passed through to parsing ops.
  Returns:
    A `Tensor` of dtype `float32` and shape `[batch_size, padded_length, ?]`.
    This will be used as input to an RNN.
  """
  sequence_input = layers.sequence_input_from_feature_columns(
      columns_to_tensors=features,
      feature_columns=sequence_feature_columns,
      weight_collections=weight_collections,
      scope=scope)
  if context_feature_columns is not None:
    context_input = layers.input_from_feature_columns(
        columns_to_tensors=features,
        feature_columns=context_feature_columns,
        weight_collections=weight_collections,
        scope=scope)
    sequence_input = _concatenate_context_input(sequence_input, context_input)
  return sequence_input


def construct_rnn(initial_state,
                  sequence_input,
                  cell,
                  num_label_columns,
                  dtype=dtypes.float32,
                  parallel_iterations=32,
                  swap_memory=True):
  """Build an RNN and apply a fully connected layer to get the desired output.

  Args:
    initial_state: The initial state to pass the RNN. If `None`, the
      default starting state for `self._cell` is used.
    sequence_input: A `Tensor` with shape `[batch_size, padded_length, d]`
      that will be passed as input to the RNN.
    cell: An initialized `RNNCell`.
    num_label_columns: The desired output dimension.
    dtype: dtype of `cell`.
    parallel_iterations: Number of iterations to run in parallel. Values >> 1
      use more memory but take less time, while smaller values use less memory
      but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
  Returns:
    activations: The output of the RNN, projected to `num_label_columns`
      dimensions.
    final_state: A `Tensor` or nested tuple of `Tensor`s representing the final
      state output by the RNN.
  """
  with ops.name_scope('RNN'):
    rnn_outputs, final_state = rnn.dynamic_rnn(
        cell=cell,
        inputs=sequence_input,
        initial_state=initial_state,
        dtype=dtype,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        time_major=False)
    activations = layers.fully_connected(
        inputs=rnn_outputs,
        num_outputs=num_label_columns,
        activation_fn=None,
        trainable=True)
    return activations, final_state


def _get_eval_metric_ops(problem_type, prediction_type, sequence_length,
                         prediction_dict, labels):
  """Returns eval metric ops for given `problem_type` and `prediction_type`.

  Args:
    problem_type: `ProblemType.CLASSIFICATION` or
      `ProblemType.LINEAR_REGRESSION`.
    prediction_type: `PredictionType.SINGLE_VALUE` or
      `PredictionType.MULTIPLE_VALUE`.
    sequence_length: A `Tensor` with shape `[batch_size]` and dtype `int32`
      containing the length of each sequence in the batch. If `None`, sequences
      are assumed to be unpadded.
    prediction_dict: A dict of prediction tensors.
    labels: The label `Tensor`.

  Returns:
    A `dict` mapping strings to the result of calling the metric_fn.
  """
  eval_metric_ops = {}
  if problem_type == constants.ProblemType.CLASSIFICATION:
    # Multi value classification
    if prediction_type == PredictionType.MULTIPLE_VALUE:
      masked_predictions, masked_labels = mask_activations_and_labels(
          prediction_dict[prediction_key.PredictionKey.CLASSES],
          labels,
          sequence_length)
      eval_metric_ops['accuracy'] = metrics.streaming_accuracy(
          predictions=masked_predictions,
          labels=masked_labels)
    # Single value classification
    elif prediction_type == PredictionType.SINGLE_VALUE:
      eval_metric_ops['accuracy'] = metrics.streaming_accuracy(
          predictions=prediction_dict[prediction_key.PredictionKey.CLASSES],
          labels=labels)
  elif problem_type == constants.ProblemType.LINEAR_REGRESSION:
    # Multi value regression
    if prediction_type == PredictionType.MULTIPLE_VALUE:
      pass
    # Single value regression
    elif prediction_type == PredictionType.SINGLE_VALUE:
      pass
  return eval_metric_ops


def _multi_value_predictions(
    activations, target_column, problem_type, predict_probabilities):
  """Maps `activations` from the RNN to predictions for multi value models.

  If `predict_probabilities` is `False`, this function returns a `dict`
  containing single entry with key `PREDICTIONS_KEY`. If `predict_probabilities`
  is `True`, it will contain a second entry with key `PROBABILITIES_KEY`. The
  value of this entry is a `Tensor` of probabilities with shape
  `[batch_size, padded_length, num_classes]`.

  Note that variable length inputs will yield some predictions that don't have
  meaning. For example, if `sequence_length = [3, 2]`, then prediction `[1, 2]`
  has no meaningful interpretation.

  Args:
    activations: Output from an RNN. Should have dtype `float32` and shape
      `[batch_size, padded_length, ?]`.
    target_column: An initialized `TargetColumn`, calculate predictions.
    problem_type: Either `ProblemType.CLASSIFICATION` or
      `ProblemType.LINEAR_REGRESSION`.
    predict_probabilities: A Python boolean, indicating whether probabilities
      should be returned. Should only be set to `True` for
      classification/logistic regression problems.
  Returns:
    A `dict` mapping strings to `Tensors`.
  """
  with ops.name_scope('MultiValuePrediction'):
    activations_shape = array_ops.shape(activations)
    flattened_activations = array_ops.reshape(activations,
                                              [-1, activations_shape[2]])
    prediction_dict = {}
    if predict_probabilities:
      flat_probabilities = target_column.logits_to_predictions(
          flattened_activations, proba=True)
      flat_predictions = math_ops.argmax(flat_probabilities, 1)
      if target_column.num_label_columns == 1:
        probability_shape = array_ops.concat([activations_shape[:2], [2]], 0)
      else:
        probability_shape = activations_shape
      probabilities = array_ops.reshape(
          flat_probabilities,
          probability_shape,
          name=prediction_key.PredictionKey.PROBABILITIES)
      prediction_dict[
          prediction_key.PredictionKey.PROBABILITIES] = probabilities
    else:
      flat_predictions = target_column.logits_to_predictions(
          flattened_activations, proba=False)
    predictions_name = (prediction_key.PredictionKey.CLASSES
                        if problem_type == constants.ProblemType.CLASSIFICATION
                        else prediction_key.PredictionKey.SCORES)
    predictions = array_ops.reshape(
        flat_predictions, [activations_shape[0], activations_shape[1]],
        name=predictions_name)
    prediction_dict[predictions_name] = predictions
    return prediction_dict


def _single_value_predictions(activations,
                              sequence_length,
                              target_column,
                              problem_type,
                              predict_probabilities):
  """Maps `activations` from the RNN to predictions for single value models.

  If `predict_probabilities` is `False`, this function returns a `dict`
  containing single entry with key `PREDICTIONS_KEY`. If `predict_probabilities`
  is `True`, it will contain a second entry with key `PROBABILITIES_KEY`. The
  value of this entry is a `Tensor` of probabilities with shape
  `[batch_size, num_classes]`.

  Args:
    activations: Output from an RNN. Should have dtype `float32` and shape
      `[batch_size, padded_length, ?]`.
    sequence_length: A `Tensor` with shape `[batch_size]` and dtype `int32`
      containing the length of each sequence in the batch. If `None`, sequences
      are assumed to be unpadded.
    target_column: An initialized `TargetColumn`, calculate predictions.
    problem_type: Either `ProblemType.CLASSIFICATION` or
      `ProblemType.LINEAR_REGRESSION`.
    predict_probabilities: A Python boolean, indicating whether probabilities
      should be returned. Should only be set to `True` for
      classification/logistic regression problems.
  Returns:
    A `dict` mapping strings to `Tensors`.
  """
  with ops.name_scope('SingleValuePrediction'):
    last_activations = select_last_activations(activations, sequence_length)
    predictions_name = (prediction_key.PredictionKey.CLASSES
                        if problem_type == constants.ProblemType.CLASSIFICATION
                        else prediction_key.PredictionKey.SCORES)
    if predict_probabilities:
      probabilities = target_column.logits_to_predictions(
          last_activations, proba=True)
      prediction_dict = {
          prediction_key.PredictionKey.PROBABILITIES: probabilities,
          predictions_name: math_ops.argmax(probabilities, 1)}
    else:
      predictions = target_column.logits_to_predictions(
          last_activations, proba=False)
      prediction_dict = {predictions_name: predictions}
    return prediction_dict


def _multi_value_loss(
    activations, labels, sequence_length, target_column, features):
  """Maps `activations` from the RNN to loss for multi value models.

  Args:
    activations: Output from an RNN. Should have dtype `float32` and shape
      `[batch_size, padded_length, ?]`.
    labels: A `Tensor` with length `[batch_size, padded_length]`.
    sequence_length: A `Tensor` with shape `[batch_size]` and dtype `int32`
      containing the length of each sequence in the batch. If `None`, sequences
      are assumed to be unpadded.
    target_column: An initialized `TargetColumn`, calculate predictions.
    features: A `dict` containing the input and (optionally) sequence length
      information and initial state.
  Returns:
    A scalar `Tensor` containing the loss.
  """
  with ops.name_scope('MultiValueLoss'):
    activations_masked, labels_masked = mask_activations_and_labels(
        activations, labels, sequence_length)
    return target_column.loss(activations_masked, labels_masked, features)


def _single_value_loss(
    activations, labels, sequence_length, target_column, features):
  """Maps `activations` from the RNN to loss for multi value models.

  Args:
    activations: Output from an RNN. Should have dtype `float32` and shape
      `[batch_size, padded_length, ?]`.
    labels: A `Tensor` with length `[batch_size]`.
    sequence_length: A `Tensor` with shape `[batch_size]` and dtype `int32`
      containing the length of each sequence in the batch. If `None`, sequences
      are assumed to be unpadded.
    target_column: An initialized `TargetColumn`, calculate predictions.
    features: A `dict` containing the input and (optionally) sequence length
      information and initial state.
  Returns:
    A scalar `Tensor` containing the loss.
  """

  with ops.name_scope('SingleValueLoss'):
    last_activations = select_last_activations(activations, sequence_length)
    return target_column.loss(last_activations, labels, features)


def _get_dynamic_rnn_model_fn(cell_type,
                              num_units,
                              target_column,
                              problem_type,
                              prediction_type,
                              optimizer,
                              sequence_feature_columns,
                              context_feature_columns=None,
                              predict_probabilities=False,
                              learning_rate=None,
                              gradient_clipping_norm=None,
                              dropout_keep_probabilities=None,
                              sequence_length_key=RNNKeys.SEQUENCE_LENGTH_KEY,
                              dtype=dtypes.float32,
                              parallel_iterations=None,
                              swap_memory=True,
                              name='DynamicRNNModel'):
  """Creates an RNN model function for an `Estimator`.

  Args:
    cell_type: A string, a subclass of `RNNCell` or an instance of an `RNNCell`.
    num_units: A single `int` or a list of `int`s. The size of the `RNNCell`s.
    target_column: An initialized `TargetColumn`, used to calculate prediction
      and loss.
    problem_type: `ProblemType.CLASSIFICATION` or
      `ProblemType.LINEAR_REGRESSION`.
    prediction_type: `PredictionType.SINGLE_VALUE` or
      `PredictionType.MULTIPLE_VALUE`.
    optimizer: A subclass of `Optimizer`, an instance of an `Optimizer` or a
      string.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    predict_probabilities: A boolean indicating whether to predict probabilities
      for all classes. Must only be used with
      `ProblemType.CLASSIFICATION`.
    learning_rate: Learning rate used for optimization. This argument has no
      effect if `optimizer` is an instance of an `Optimizer`.
    gradient_clipping_norm: A float. Gradients will be clipped to this value.
    dropout_keep_probabilities: a list of dropout keep probabilities or `None`.
      If a list is given, it must have length `len(num_units) + 1`.
    sequence_length_key: The key that will be used to look up sequence length in
      the `features` dict.
    dtype: The dtype of the state and output of the given `cell`.
    parallel_iterations: Number of iterations to run in parallel. Values >> 1
      use more memory but take less time, while smaller values use less memory
      but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    name: A string that will be used to create a scope for the RNN.

  Returns:
    A model function to be passed to an `Estimator`.

  Raises:
    ValueError: `problem_type` is not one of
      `ProblemType.LINEAR_REGRESSION` or `ProblemType.CLASSIFICATION`.
    ValueError: `prediction_type` is not one of `PredictionType.SINGLE_VALUE`
      or `PredictionType.MULTIPLE_VALUE`.
    ValueError: `predict_probabilities` is `True` for `problem_type` other
      than `ProblemType.CLASSIFICATION`.
    ValueError: `len(dropout_keep_probabilities)` is not `len(num_units) + 1`.
  """
  if problem_type not in (constants.ProblemType.CLASSIFICATION,
                          constants.ProblemType.LINEAR_REGRESSION):
    raise ValueError(
        'problem_type must be ProblemType.LINEAR_REGRESSION or '
        'ProblemType.CLASSIFICATION; got {}'.
        format(problem_type))
  if prediction_type not in (
      PredictionType.SINGLE_VALUE, PredictionType.MULTIPLE_VALUE):
    raise ValueError(
        'prediction_type must be PredictionType.MULTIPLE_VALUEs or '
        'PredictionType.SINGLE_VALUE; got {}'.
        format(prediction_type))
  if (problem_type != constants.ProblemType.CLASSIFICATION
      and predict_probabilities):
    raise ValueError(
        'predict_probabilities can only be set to True for problem_type'
        ' ProblemType.CLASSIFICATION; got {}.'.format(problem_type))
  def _dynamic_rnn_model_fn(features, labels, mode):
    """The model to be passed to an `Estimator`."""
    with ops.name_scope(name):
      sequence_length = features.get(sequence_length_key)
      sequence_input = build_sequence_input(features,
                                            sequence_feature_columns,
                                            context_feature_columns)
      dropout = (dropout_keep_probabilities
                 if mode == model_fn.ModeKeys.TRAIN
                 else None)
      cell = _construct_rnn_cell(cell_type, num_units, dropout)
      initial_state = dict_to_state_tuple(features, cell)
      rnn_activations, final_state = construct_rnn(
          initial_state,
          sequence_input,
          cell,
          target_column.num_label_columns,
          dtype=dtype,
          parallel_iterations=parallel_iterations,
          swap_memory=swap_memory)

      loss = None  # Created below for modes TRAIN and EVAL.
      if prediction_type == PredictionType.MULTIPLE_VALUE:
        prediction_dict = _multi_value_predictions(
            rnn_activations, target_column, problem_type, predict_probabilities)
        if mode != model_fn.ModeKeys.INFER:
          loss = _multi_value_loss(
              rnn_activations, labels, sequence_length, target_column, features)
      elif prediction_type == PredictionType.SINGLE_VALUE:
        prediction_dict = _single_value_predictions(
            rnn_activations, sequence_length, target_column,
            problem_type, predict_probabilities)
        if mode != model_fn.ModeKeys.INFER:
          loss = _single_value_loss(
              rnn_activations, labels, sequence_length, target_column, features)
      state_dict = state_tuple_to_dict(final_state)
      prediction_dict.update(state_dict)

      eval_metric_ops = None
      if mode != model_fn.ModeKeys.INFER:
        eval_metric_ops = _get_eval_metric_ops(
            problem_type, prediction_type, sequence_length, prediction_dict,
            labels)

      train_op = None
      if mode == model_fn.ModeKeys.TRAIN:
        train_op = optimizers.optimize_loss(
            loss=loss,
            global_step=None,  # Get it internally.
            learning_rate=learning_rate,
            optimizer=optimizer,
            clip_gradients=gradient_clipping_norm,
            summaries=optimizers.OPTIMIZER_SUMMARIES)

    return model_fn.ModelFnOps(mode=mode,
                               predictions=prediction_dict,
                               loss=loss,
                               train_op=train_op,
                               eval_metric_ops=eval_metric_ops)
  return _dynamic_rnn_model_fn


def _apply_dropout(
    cells, dropout_keep_probabilities, random_seed=None):
  """Applies dropout to the outputs and inputs of `cell`.

  Args:
    cells: A list of `RNNCell`s.
    dropout_keep_probabilities: a list whose elements are either floats in
    `[0.0, 1.0]` or `None`. It must have length one greater than `cells`.
    random_seed: Seed for random dropout.

  Returns:
    A list of `RNNCell`s, the result of applying the supplied dropouts.

  Raises:
    ValueError: If `len(dropout_keep_probabilities) != len(cells) + 1`.
  """
  if len(dropout_keep_probabilities) != len(cells) + 1:
    raise ValueError(
        'The number of dropout probabilites must be one greater than the '
        'number of cells. Got {} cells and {} dropout probabilities.'.format(
            len(cells), len(dropout_keep_probabilities)))
  wrapped_cells = [
      contrib_rnn.DropoutWrapper(cell, prob, 1.0, random_seed)
      for cell, prob in zip(cells[:-1], dropout_keep_probabilities[:-2])
  ]
  wrapped_cells.append(contrib_rnn.DropoutWrapper(
      cells[-1],
      dropout_keep_probabilities[-2],
      dropout_keep_probabilities[-1]))
  return wrapped_cells


def _get_single_cell(cell_type, num_units):
  """Constructs and return an single `RNNCell`.

  Args:
    cell_type: Either a string identifying the `RNNCell` type, a subclass of
      `RNNCell` or an instance of an `RNNCell`.
    num_units: The number of units in the `RNNCell`.
  Returns:
    An initialized `RNNCell`.
  Raises:
    ValueError: `cell_type` is an invalid `RNNCell` name.
    TypeError: `cell_type` is not a string or a subclass of `RNNCell`.
  """
  if isinstance(cell_type, contrib_rnn.RNNCell):
    return cell_type
  if isinstance(cell_type, str):
    cell_type = _CELL_TYPES.get(cell_type)
    if cell_type is None:
      raise ValueError('The supported cell types are {}; got {}'.format(
          list(_CELL_TYPES.keys()), cell_type))
  if not issubclass(cell_type, contrib_rnn.RNNCell):
    raise TypeError(
        'cell_type must be a subclass of RNNCell or one of {}.'.format(
            list(_CELL_TYPES.keys())))
  return cell_type(num_units=num_units)


def _construct_rnn_cell(cell_type, num_units, dropout_keep_probabilities):
  """Constructs cells, applies dropout and assembles a `MultiRNNCell`.

  Args:
    cell_type: A string identifying the `RNNCell` type, a subclass of
      `RNNCell` or an instance of an `RNNCell`.
    num_units: A single `int` or a list/tuple of `int`s. The size of the
      `RNNCell`s.
    dropout_keep_probabilities: a list of dropout probabilities or `None`. If a
      list is given, it must have length `len(cell_type) + 1`.

  Returns:
    An initialized `RNNCell`.
  """
  if not isinstance(num_units, (list, tuple)):
    num_units = (num_units,)

  cells = [_get_single_cell(cell_type, n) for n in num_units]
  if dropout_keep_probabilities:
    cells = _apply_dropout(cells, dropout_keep_probabilities)
  if len(cells) == 1:
    return cells[0]
  return contrib_rnn.MultiRNNCell(cells)


def _get_dropout_and_num_units(cell_type,
                               num_units,
                               num_rnn_layers,
                               input_keep_probability,
                               output_keep_probability):
  """Helper function for deprecated factory functions."""
  dropout_keep_probabilities = None
  if isinstance(cell_type, contrib_rnn.RNNCell):
    num_units = None
  else:
    num_units = [num_units for _ in range(num_rnn_layers)]
    if input_keep_probability or output_keep_probability:
      dropout_keep_probabilities = ([input_keep_probability]
                                    + [1.0] * (num_rnn_layers - 1)
                                    + [output_keep_probability])
  return dropout_keep_probabilities, num_units


class DynamicRnnEstimator(estimator.Estimator):

  def __init__(self,
               problem_type,
               prediction_type,
               sequence_feature_columns,
               context_feature_columns=None,
               num_classes=None,
               num_units=None,
               cell_type='basic_rnn',
               optimizer='SGD',
               learning_rate=0.1,
               predict_probabilities=False,
               momentum=None,
               gradient_clipping_norm=5.0,
               dropout_keep_probabilities=None,
               model_dir=None,
               feature_engineering_fn=None,
               config=None):
    """Initializes a `DynamicRnnEstimator`.

    The input function passed to this `Estimator` optionally contains keys
    `RNNKeys.SEQUENCE_LENGTH_KEY`. The value corresponding to
    `RNNKeys.SEQUENCE_LENGTH_KEY` must be vector of size `batch_size` where
    entry `n` corresponds to the length of the `n`th sequence in the batch. The
    sequence length feature is required for batches of varying sizes. It will be
    used to calculate loss and evaluation metrics. If
    `RNNKeys.SEQUENCE_LENGTH_KEY` is not included, all sequences are assumed to
    have length equal to the size of dimension 1 of the input to the RNN.

    In order to specify an initial state, the input function must include keys
    `STATE_PREFIX_i` for all `0 <= i < n` where `n` is the number of nested
    elements in `cell.state_size`. The input function must contain values for
    all state components or none of them. If none are included, then the default
    (zero) state is used as an initial state. See the documentation for
    `dict_to_state_tuple` and `state_tuple_to_dict` for further details.

    The `predict()` method of the `Estimator` returns a dictionary with keys
    `STATE_PREFIX_i` for `0 <= i < n` where `n` is the number of nested elements
    in `cell.state_size`, along with `PredictionKey.CLASSES` for problem type
    `CLASSIFICATION` or `PredictionKey.SCORES` for problem type
    `LINEAR_REGRESSION`.  The value keyed by
    `PredictionKey.CLASSES` or `PredictionKey.SCORES` has shape
    `[batch_size, padded_length]` in the multi-value case and shape
    `[batch_size]` in the single-value case.  Here, `padded_length` is the
    largest value in the `RNNKeys.SEQUENCE_LENGTH` `Tensor` passed as input.
    Entry `[i, j]` is the prediction associated with sequence `i` and time step
    `j`. If the problem type is `CLASSIFICATION` and `predict_probabilities` is
    `True`, it will also include key`PredictionKey.PROBABILITIES`.

    Args:
      problem_type: whether the `Estimator` is intended for a regression or
        classification problem. Value must be one of
        `ProblemType.CLASSIFICATION` or `ProblemType.LINEAR_REGRESSION`.
      prediction_type: whether the `Estimator` should return a value for each
        step in the sequence, or just a single value for the final time step.
        Must be one of `ProblemType.SINGLE_VALUE` or
        `ProblemType.MULTIPLE_VALUE`.
      sequence_feature_columns: An iterable containing all the feature columns
        describing sequence features. All items in the iterable should be
        instances of classes derived from `FeatureColumn`.
      context_feature_columns: An iterable containing all the feature columns
        describing context features, i.e., features that apply accross all time
        steps. All items in the set should be instances of classes derived from
        `FeatureColumn`.
      num_classes: the number of classes for a classification problem. Only
        used when `problem_type=ProblemType.CLASSIFICATION`.
      num_units: A list of integers indicating the number of units in the
        `RNNCell`s in each layer. Either `num_units` is specified or `cell_type`
        is an instance of `RNNCell`.
      cell_type: A subclass of `RNNCell`, an instance of an `RNNCell` or one of
        'basic_rnn,' 'lstm' or 'gru'.
      optimizer: The type of optimizer to use. Either a subclass of
        `Optimizer`, an instance of an `Optimizer`, a callback that returns an
        optimizer, or a string. Strings must be one of 'Adagrad', 'Adam',
        'Ftrl', 'Momentum', 'RMSProp' or 'SGD. See `layers.optimize_loss` for
        more details.
      learning_rate: Learning rate. This argument has no effect if `optimizer`
        is an instance of an `Optimizer`.
      predict_probabilities: A boolean indicating whether to predict
        probabilities for all classes. Used only if `problem_type` is
        `ProblemType.CLASSIFICATION`
      momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
      gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
        then no clipping is performed.
      dropout_keep_probabilities: a list of dropout probabilities or `None`.
        If a list is given, it must have length `len(num_units) + 1`. If
        `None`, then no dropout is applied.
      model_dir: The directory in which to save and restore the model graph,
        parameters, etc.
      feature_engineering_fn: Takes features and labels which are the output of
        `input_fn` and returns features and labels which will be fed into
        `model_fn`. Please check `model_fn` for a definition of features and
        labels.
      config: A `RunConfig` instance.

    Raises:
      ValueError: Both or neither of the following are true: (a) `num_units` is
        specified and (b) `cell_type` is an instance of `RNNCell`.
      ValueError: `problem_type` is not one of
        `ProblemType.LINEAR_REGRESSION` or `ProblemType.CLASSIFICATION`.
      ValueError: `problem_type` is `ProblemType.CLASSIFICATION` but
        `num_classes` is not specifieProblemType
      ValueError: `prediction_type` is not one of
        `PredictionType.MULTIPLE_VALUE` or `PredictionType.SINGLE_VALUE`.
    """
    if (num_units is not None) == isinstance(cell_type, contrib_rnn.RNNCell):
      raise ValueError(
          'Either num_units is specified OR cell_type is an instance of '
          'RNNCell. Got num_units = {} and cell_type = {}.'.format(
              num_units, cell_type))

    if prediction_type == PredictionType.MULTIPLE_VALUE:
      name = 'MultiValueDynamicRNN'
    elif prediction_type == PredictionType.SINGLE_VALUE:
      name = 'SingleValueDynamicRNN'
    else:
      raise ValueError(
          'prediction_type must be one of PredictionType.MULTIPLE_VALUE or '
          'PredictionType.SINGLE_VALUE; got {}'.format(prediction_type))

    if problem_type == constants.ProblemType.LINEAR_REGRESSION:
      name += 'Regressor'
      target_column = layers.regression_target()
    elif problem_type == constants.ProblemType.CLASSIFICATION:
      if not num_classes:
        raise ValueError('For CLASSIFICATION problem_type, num_classes must be '
                         'specified.')
      target_column = layers.multi_class_target(n_classes=num_classes)
      name += 'Classifier'
    else:
      raise ValueError(
          'problem_type must be either ProblemType.LINEAR_REGRESSION '
          'or ProblemType.CLASSIFICATION; got {}'.format(
              problem_type))

    if optimizer == 'Momentum':
      optimizer = momentum_opt.MomentumOptimizer(learning_rate, momentum)
    dynamic_rnn_model_fn = _get_dynamic_rnn_model_fn(
        cell_type=cell_type,
        num_units=num_units,
        target_column=target_column,
        problem_type=problem_type,
        prediction_type=prediction_type,
        optimizer=optimizer,
        sequence_feature_columns=sequence_feature_columns,
        context_feature_columns=context_feature_columns,
        predict_probabilities=predict_probabilities,
        learning_rate=learning_rate,
        gradient_clipping_norm=gradient_clipping_norm,
        dropout_keep_probabilities=dropout_keep_probabilities,
        name=name)

    super(DynamicRnnEstimator, self).__init__(
        model_fn=dynamic_rnn_model_fn,
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)


@deprecated('2017-04-01',
            'multi_value_rnn_regressor is deprecated. '
            'Please construct a DynamicRnnEstimator directly.')
def multi_value_rnn_regressor(num_units,
                              sequence_feature_columns,
                              context_feature_columns=None,
                              cell_type='basic_rnn',
                              num_rnn_layers=1,
                              optimizer_type='SGD',
                              learning_rate=0.1,
                              momentum=None,
                              gradient_clipping_norm=5.0,
                              input_keep_probability=None,
                              output_keep_probability=None,
                              model_dir=None,
                              config=None,
                              feature_engineering_fn=None):
  """Creates a `DynamicRnnEstimator` for multi-value regression.

  Returns an `Estimator` that given input sequences, processes them in a dynamic
  recurrent network and outputs a sequence of continuous values.

  Args:
    num_units: The size of the RNN cells. This argument has no effect
      if `cell_type` is an instance of `RNNCell`.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    cell_type: A subclass of `RNNCell`, an instance of an `RNNCell` or one of
      'basic_rnn,' 'lstm' or 'gru'.
    num_rnn_layers: Number of RNN layers. Leave this at its default value 1
      if passing a `cell_type` that is already a MultiRNNCell.
    optimizer_type: The type of optimizer to use. Either a subclass of
      `Optimizer`, an instance of an `Optimizer`, a callback that returns an
      optimizer, or a string. Strings must be one of 'Adagrad', 'Adam',
      'Ftrl', 'Momentum', 'RMSProp' or 'SGD. See `layers.optimize_loss` for
      more details.
    learning_rate: Learning rate. This argument has no effect if `optimizer`
      is an instance of an `Optimizer`.
    momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs of `cell`. If `None`,
      no dropout is applied.
    model_dir: The directory in which to save and restore the model graph,
      parameters, etc.
    config: A `RunConfig` instance.
    feature_engineering_fn: Takes features and labels which are the output of
      `input_fn` and returns features and labels which will be fed into
      `model_fn`. Please check `model_fn` for a definition of features and
      labels.
  Returns:
    An initialized `Estimator`.
  """
  dropout_keep_probabilities, num_units = _get_dropout_and_num_units(
      cell_type,
      num_units,
      num_rnn_layers,
      input_keep_probability,
      output_keep_probability)
  return DynamicRnnEstimator(
      problem_type=constants.ProblemType.LINEAR_REGRESSION,
      prediction_type=PredictionType.MULTIPLE_VALUE,
      sequence_feature_columns=sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      num_units=num_units,
      cell_type=cell_type,
      optimizer=optimizer_type,
      learning_rate=learning_rate,
      momentum=momentum,
      gradient_clipping_norm=gradient_clipping_norm,
      dropout_keep_probabilities=dropout_keep_probabilities,
      model_dir=model_dir,
      feature_engineering_fn=feature_engineering_fn,
      config=config)


@deprecated('2017-04-01',
            'multi_value_rnn_classifier is deprecated. '
            'Please construct a DynamicRNNEstimator directly.')
def multi_value_rnn_classifier(num_classes,
                               num_units,
                               sequence_feature_columns,
                               context_feature_columns=None,
                               cell_type='basic_rnn',
                               num_rnn_layers=1,
                               optimizer_type='SGD',
                               learning_rate=0.1,
                               predict_probabilities=False,
                               momentum=None,
                               gradient_clipping_norm=5.0,
                               input_keep_probability=None,
                               output_keep_probability=None,
                               model_dir=None,
                               config=None,
                               feature_engineering_fn=None):
  """Creates a `DynamicRNNEstimator` for multi-value classification.

  Returns an `Estimator` that given input sequences, processes them in a dynamic
  recurrent network and outputs a sequence of classifications, along with
  (optionally) a probability distribution over classes.

  Args:
    num_classes: The number of classes for categorization.
    num_units: The size of the RNN cells. This argument has no effect
      if `cell_type` is an instance of `RNNCell`.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    cell_type: A subclass of `RNNCell`, an instance of an `RNNCell or one of
      'basic_rnn,' 'lstm' or 'gru'.
    num_rnn_layers: Number of RNN layers. Leave this at its default value 1
      if passing a `cell_type` that is already a MultiRNNCell.
    optimizer_type: The type of optimizer to use. Either a subclass of
      `Optimizer`, an instance of an `Optimizer`, a callback that returns an
      optimizer, or a string. Strings must be one of 'Adagrad', 'Adam',
      'Ftrl', 'Momentum', 'RMSProp' or 'SGD. See `layers.optimize_loss` for
      more details.
    learning_rate: Learning rate. This argument has no effect if `optimizer`
      is an instance of an `Optimizer`.
    predict_probabilities: A boolean indicating whether to predict probabilities
      for all classes.
    momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs of `cell`. If `None`,
      no dropout is applied.
    model_dir: The directory in which to save and restore the model graph,
      parameters, etc.
    config: A `RunConfig` instance.
    feature_engineering_fn: Takes features and labels which are the output of
      `input_fn` and returns features and labels which will be fed into
      `model_fn`. Please check `model_fn` for a definition of features and
      labels.
  Returns:
    An initialized `Estimator`.
  """
  dropout_keep_probabilities, num_units = _get_dropout_and_num_units(
      cell_type,
      num_units,
      num_rnn_layers,
      input_keep_probability,
      output_keep_probability)
  return DynamicRnnEstimator(
      problem_type=constants.ProblemType.CLASSIFICATION,
      prediction_type=PredictionType.MULTIPLE_VALUE,
      num_classes=num_classes,
      sequence_feature_columns=sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      num_units=num_units,
      cell_type=cell_type,
      optimizer=optimizer_type,
      learning_rate=learning_rate,
      predict_probabilities=predict_probabilities,
      momentum=momentum,
      gradient_clipping_norm=gradient_clipping_norm,
      dropout_keep_probabilities=dropout_keep_probabilities,
      model_dir=model_dir,
      feature_engineering_fn=feature_engineering_fn,
      config=config)


@deprecated('2017-04-01',
            'single_value_rnn_regressor is deprecated. '
            'Please construct a DynamicRnnEstimator directly.')
def single_value_rnn_regressor(num_units,
                               sequence_feature_columns,
                               context_feature_columns=None,
                               cell_type='basic_rnn',
                               num_rnn_layers=1,
                               optimizer_type='SGD',
                               learning_rate=0.1,
                               momentum=None,
                               gradient_clipping_norm=5.0,
                               input_keep_probability=None,
                               output_keep_probability=None,
                               model_dir=None,
                               config=None,
                               feature_engineering_fn=None):
  """Creates a `DynamicRnnEstimator` for single-value regression.

  Returns an `Estimator` that given input sequences, processes them in a dynamic
  recurrent network and outputs a single continuous values.

  Args:
    num_units: The size of the RNN cells. This argument has no effect
      if `cell_type` is an instance of `RNNCell`.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    cell_type: A subclass of `RNNCell`, an instance of an `RNNCell` or one of
      'basic_rnn,' 'lstm' or 'gru'.
    num_rnn_layers: Number of RNN layers. Leave this at its default value 1
      if passing a `cell_type` that is already a MultiRNNCell.
    optimizer_type: The type of optimizer to use. Either a subclass of
      `Optimizer`, an instance of an `Optimizer`, a callback that returns an
      optimizer, or a string. Strings must be one of 'Adagrad', 'Adam',
      'Ftrl', 'Momentum', 'RMSProp' or 'SGD. See `layers.optimize_loss` for
      more details.
    learning_rate: Learning rate. This argument has no effect if `optimizer`
      is an instance of an `Optimizer`.
    momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs of `cell`. If `None`,
      no dropout is applied.
    model_dir: The directory in which to save and restore the model graph,
      parameters, etc.
    config: A `RunConfig` instance.
    feature_engineering_fn: Takes features and labels which are the output of
      `input_fn` and returns features and labels which will be fed into
      `model_fn`. Please check `model_fn` for a definition of features and
      labels.
  Returns:
    An initialized `Estimator`.
  """
  dropout_keep_probabilities, num_units = _get_dropout_and_num_units(
      cell_type,
      num_units,
      num_rnn_layers,
      input_keep_probability,
      output_keep_probability)
  return DynamicRnnEstimator(
      problem_type=constants.ProblemType.LINEAR_REGRESSION,
      prediction_type=PredictionType.SINGLE_VALUE,
      sequence_feature_columns=sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      num_units=num_units,
      cell_type=cell_type,
      optimizer=optimizer_type,
      learning_rate=learning_rate,
      momentum=momentum,
      gradient_clipping_norm=gradient_clipping_norm,
      dropout_keep_probabilities=dropout_keep_probabilities,
      model_dir=model_dir,
      feature_engineering_fn=feature_engineering_fn,
      config=config)


@deprecated('2017-04-01',
            'single_value_rnn_classifier is deprecated. '
            'Please construct a DynamicRnnEstimator directly.')
def single_value_rnn_classifier(num_classes,
                                num_units,
                                sequence_feature_columns,
                                context_feature_columns=None,
                                cell_type='basic_rnn',
                                num_rnn_layers=1,
                                optimizer_type='SGD',
                                learning_rate=0.1,
                                predict_probabilities=False,
                                momentum=None,
                                gradient_clipping_norm=5.0,
                                input_keep_probability=None,
                                output_keep_probability=None,
                                model_dir=None,
                                config=None,
                                feature_engineering_fn=None):
  """Creates a `DynamicRnnEstimator` for single-value classification.

  Returns an `Estimator` that given input sequences, processes them in a dynamic
  recurrent network and outputs a single classifications, along with
  (optionally) a probability distribution over classes.

  Args:
    num_classes: The number of classes for categorization.
    num_units: The size of the RNN cells. This argument has no effect
      if `cell_type` is an instance of `RNNCell`.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    cell_type: A subclass of `RNNCell`, an instance of an `RNNCell or one of
      'basic_rnn,' 'lstm' or 'gru'.
    num_rnn_layers: Number of RNN layers. Leave this at its default value 1
      if passing a `cell_type` that is already a MultiRNNCell.
    optimizer_type: The type of optimizer to use. Either a subclass of
      `Optimizer`, an instance of an `Optimizer`, a callback that returns an
      optimizer, or a string. Strings must be one of 'Adagrad', 'Adam',
      'Ftrl', 'Momentum', 'RMSProp' or 'SGD. See `layers.optimize_loss` for
      more details.
    learning_rate: Learning rate. This argument has no effect if `optimizer`
      is an instance of an `Optimizer`.
    predict_probabilities: A boolean indicating whether to predict probabilities
      for all classes.
    momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs of `cell`. If `None`,
      no dropout is applied.
    model_dir: The directory in which to save and restore the model graph,
      parameters, etc.
    config: A `RunConfig` instance.
    feature_engineering_fn: Takes features and labels which are the output of
      `input_fn` and returns features and labels which will be fed into
      `model_fn`. Please check `model_fn` for a definition of features and
      labels.
  Returns:
    An initialized `Estimator`.
  """
  dropout_keep_probabilities, num_units = _get_dropout_and_num_units(
      cell_type,
      num_units,
      num_rnn_layers,
      input_keep_probability,
      output_keep_probability)
  return DynamicRnnEstimator(
      problem_type=constants.ProblemType.CLASSIFICATION,
      prediction_type=PredictionType.SINGLE_VALUE,
      num_classes=num_classes,
      sequence_feature_columns=sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      num_units=num_units,
      cell_type=cell_type,
      optimizer=optimizer_type,
      learning_rate=learning_rate,
      predict_probabilities=predict_probabilities,
      momentum=momentum,
      gradient_clipping_norm=gradient_clipping_norm,
      dropout_keep_probabilities=dropout_keep_probabilities,
      model_dir=model_dir,
      feature_engineering_fn=feature_engineering_fn,
      config=config)
