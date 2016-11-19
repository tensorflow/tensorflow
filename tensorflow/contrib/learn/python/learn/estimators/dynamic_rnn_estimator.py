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

import functools

from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.training import momentum as momentum_opt


class ProblemType(object):
  REGRESSION = 1
  CLASSIFICATION = 2


class PredictionType(object):
  SINGLE_VALUE = 1
  MULTIPLE_VALUE = 2


class RNNKeys(object):
  SEQUENCE_LENGTH_KEY = 'sequence_length'
  INITIAL_STATE_KEY = 'initial_state'
  PREDICTIONS_KEY = 'predictions'
  PROBABILITIES_KEY = 'probabilities'
  FINAL_STATE_KEY = 'final_state'

_CELL_TYPES = {'basic_rnn': rnn_cell.BasicRNNCell,
               'lstm': rnn_cell.LSTMCell,
               'gru': rnn_cell.GRUCell,}


# TODO(jamieas): move `padding_mask` to array_ops.
def padding_mask(sequence_lengths, padded_length):
  """Creates a mask used for calculating losses with padded input.

  Args:
    sequence_lengths: A `Tensor` of shape `[batch_size]` containing the unpadded
      length of  each sequence.
    padded_length: A scalar `Tensor` indicating the length of the sequences
      after padding
  Returns:
    A boolean `Tensor` M of shape `[batch_size, padded_length]` where
    `M[i, j] == True` when `lengths[i] > j`.

  """
  range_tensor = math_ops.range(padded_length)
  return math_ops.less(array_ops.expand_dims(range_tensor, 0),
                       array_ops.expand_dims(sequence_lengths, 1))


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
      mask = padding_mask(sequence_lengths, padded_length)
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
        array_ops.concat(0, [[1], [padded_length], [1]]))
  return array_ops.concat(2, [sequence_input, tiled_context_input])


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
                  swap_memory=False):
  """Build an RNN and apply a fully connected layer to get the desired output.

  Args:
    initial_state: The initial state to pass the the RNN. If `None`, the
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
    final_state: The final state output by the RNN.
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


def _mask_multivalue(sequence_length, metric):
  """Wrapper function that masks values by `sequence_length`.

  Args:
    sequence_length: A `Tensor` with shape `[batch_size]` and dtype `int32`
      containing the length of each sequence in the batch. If `None`, sequences
      are assumed to be unpadded.
    metric: A metric function. Its signature must contain `predictions` and
      `labels`.

  Returns:
    A metric function that masks `predictions` and `labels` using
    `sequence_length` and then applies `metric` to the results.
  """
  @functools.wraps(metric)
  def _metric(predictions, labels, *args, **kwargs):
    predictions, labels = mask_activations_and_labels(
        predictions, labels, sequence_length)
    return metric(predictions, labels, *args, **kwargs)
  return _metric


def _get_default_metrics(problem_type, prediction_type, sequence_length):
  """Returns default `MetricSpec`s for `problem_type` and `prediction_type`.

  Args:
    problem_type: `ProblemType.CLASSIFICATION` or`ProblemType.REGRESSION`.
    prediction_type: `PredictionType.SINGLE_VALUE` or
      `PredictionType.MULTIPLE_VALUE`.
    sequence_length: A `Tensor` with shape `[batch_size]` and dtype `int32`
      containing the length of each sequence in the batch. If `None`, sequences
      are assumed to be unpadded.
  Returns:
    A `dict` mapping strings to `MetricSpec`s.
  """
  default_metrics = {}
  if problem_type == ProblemType.CLASSIFICATION:
    # Multi value classification
    if prediction_type == PredictionType.MULTIPLE_VALUE:
      default_metrics['accuracy'] = metric_spec.MetricSpec(
          metric_fn=_mask_multivalue(
              sequence_length, metrics.streaming_accuracy),
          prediction_key=RNNKeys.PREDICTIONS_KEY)
    # Single value classification
    elif prediction_type == PredictionType.SINGLE_VALUE:
      default_metrics['accuracy'] = metric_spec.MetricSpec(
          metric_fn=metrics.streaming_accuracy,
          prediction_key=RNNKeys.PREDICTIONS_KEY)
  elif problem_type == ProblemType.REGRESSION:
    # Multi value regression
    if prediction_type == PredictionType.MULTIPLE_VALUE:
      pass
    # Single value regression
    elif prediction_type == PredictionType.SINGLE_VALUE:
      pass
  return default_metrics


def _multi_value_predictions(
    activations, target_column, predict_probabilities):
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
        probability_shape = array_ops.concat(0, [activations_shape[:2], [2]])
      else:
        probability_shape = activations_shape
      probabilities = array_ops.reshape(flat_probabilities, probability_shape)
      prediction_dict[RNNKeys.PROBABILITIES_KEY] = probabilities
    else:
      flat_predictions = target_column.logits_to_predictions(
          flattened_activations, proba=False)
    predictions = array_ops.reshape(
        flat_predictions, [activations_shape[0], activations_shape[1]])
    prediction_dict[RNNKeys.PREDICTIONS_KEY] = predictions
    return prediction_dict


def _single_value_predictions(
    activations, sequence_length, target_column, predict_probabilities):
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
    predict_probabilities: A Python boolean, indicating whether probabilities
      should be returned. Should only be set to `True` for
      classification/logistic regression problems.
  Returns:
    A `dict` mapping strings to `Tensors`.
  """
  with ops.name_scope('SingleValuePrediction'):
    last_activations = select_last_activations(activations, sequence_length)
    if predict_probabilities:
      probabilities = target_column.logits_to_predictions(
          last_activations, proba=True)
      prediction_dict = {
          RNNKeys.PROBABILITIES_KEY: probabilities,
          RNNKeys.PREDICTIONS_KEY: math_ops.argmax(probabilities, 1)}
    else:
      predictions = target_column.logits_to_predictions(
          last_activations, proba=False)
      prediction_dict = {RNNKeys.PREDICTIONS_KEY: predictions}
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


def apply_dropout(
    cell, input_keep_probability, output_keep_probability, random_seed=None):
  """Apply dropout to the outputs and inputs of `cell`.

  Args:
    cell: An `RNNCell`.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs to `cell`. If `None`,
      no dropout is applied.
    random_seed: Seed for random dropout.

  Returns:
    An `RNNCell`, the result of applying the supplied dropouts to `cell`.
  """
  input_prob_none = input_keep_probability is None
  output_prob_none = output_keep_probability is None
  if input_prob_none and output_prob_none:
    return cell
  if input_prob_none:
    input_keep_probability = 1.0
  if output_prob_none:
    output_keep_probability = 1.0
  return rnn_cell.DropoutWrapper(
      cell, input_keep_probability, output_keep_probability, random_seed)


def _get_dynamic_rnn_model_fn(cell,
                              target_column,
                              problem_type,
                              prediction_type,
                              optimizer,
                              sequence_feature_columns,
                              context_feature_columns=None,
                              predict_probabilities=False,
                              learning_rate=None,
                              gradient_clipping_norm=None,
                              input_keep_probability=None,
                              output_keep_probability=None,
                              sequence_length_key=RNNKeys.SEQUENCE_LENGTH_KEY,
                              initial_state_key=RNNKeys.INITIAL_STATE_KEY,
                              dtype=dtypes.float32,
                              parallel_iterations=None,
                              swap_memory=False,
                              name='DynamicRNNModel'):
  """Creates an RNN model function for an `Estimator`.

  Args:
    cell: An initialized `RNNCell` to be used in the RNN.
      'basic_rnn,' 'lstm' or 'gru'.
    target_column: An initialized `TargetColumn`, used to calculate prediction
      and loss.
    problem_type: `ProblemType.CLASSIFICATION` or`ProblemType.REGRESSION`.
    prediction_type: `PredictionType.SINGLE_VALUE` or
      `PredictionType.MULTIPLE_VALUE`.
    optimizer: A subclass of `Optimizer`, an instance of an `Optimizer` or a
      string.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features i.e. features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    predict_probabilities: A boolean indicating whether to predict probabilities
      for all classes. Should only be used with `ProblemType.CLASSIFICATION`.
    learning_rate: Learning rate used for optimization. This argument has no
      effect if `optimizer` is an instance of an `Optimizer`.
    gradient_clipping_norm: A float. Gradients will be clipped to this value.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs to `cell`. If `None`,
      no dropout is applied.
    sequence_length_key: The key that will be used to look up sequence length in
      the `features` dict.
    initial_state_key: The key that will be used to look up initial_state in
      the `features` dict.
    dtype: The dtype of the state and output for the given `cell_num`
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
    ValueError: `problem_type` is not one of `ProblemType.REGRESSION` or
      `ProblemType.CLASSIFICATION`.
    ValueError: `prediction_type` is not one of `PredictionType.SINGLE_VALUE`
      or `PredictionType.MULTIPLE_VALUE`.
    ValueError: `predict_probabilities` is `True` for `problem_type` other
      than `ProblemType.CLASSIFICATION`.
  """
  if problem_type not in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
    raise ValueError(
        'problem_type must be ProblemType.REGRESSION or '
        'ProblemType.CLASSIFICATION; got {}'.
        format(problem_type))
  if prediction_type not in (
      PredictionType.SINGLE_VALUE, PredictionType.MULTIPLE_VALUE):
    raise ValueError(
        'prediction_type must be PredictionType.MULTIPLE_VALUEs or '
        'PredictionType.SINGLE_VALUE; got {}'.
        format(prediction_type))
  if problem_type != ProblemType.CLASSIFICATION and predict_probabilities:
    raise ValueError(
        'predict_probabilities can only be set to True for problem_type'
        ' ProblemType.CLASSIFICATION; got {}.'.format(problem_type))

  def _dynamic_rnn_model_fn(features, labels, mode):
    """The model to be passed to an `Estimator`."""
    with ops.name_scope(name):
      initial_state = features.get(initial_state_key)
      sequence_length = features.get(sequence_length_key)
      sequence_input = build_sequence_input(features,
                                            sequence_feature_columns,
                                            context_feature_columns)
      if mode == model_fn.ModeKeys.TRAIN:
        cell_for_mode = apply_dropout(
            cell, input_keep_probability, output_keep_probability)
      else:
        cell_for_mode = cell
      rnn_activations, final_state = construct_rnn(
          initial_state,
          sequence_input,
          cell_for_mode,
          target_column.num_label_columns,
          dtype=dtype,
          parallel_iterations=parallel_iterations,
          swap_memory=swap_memory)
      if prediction_type == PredictionType.MULTIPLE_VALUE:
        prediction_dict = _multi_value_predictions(
            rnn_activations, target_column, predict_probabilities)
        loss = _multi_value_loss(
            rnn_activations, labels, sequence_length, target_column, features)
      elif prediction_type == PredictionType.SINGLE_VALUE:
        prediction_dict = _single_value_predictions(
            rnn_activations, sequence_length, target_column,
            predict_probabilities)
        loss = _single_value_loss(
            rnn_activations, labels, sequence_length, target_column, features)
      # TODO(roumposg): Return eval_metric_ops here, instead of default_metrics.
      default_metrics = _get_default_metrics(
          problem_type, prediction_type, sequence_length)
      prediction_dict[RNNKeys.FINAL_STATE_KEY] = final_state
      eval_metric_ops = estimator._make_metrics_ops(  # pylint: disable=protected-access
          default_metrics, features, labels, prediction_dict)
      train_op = optimizers.optimize_loss(
          loss=loss,
          global_step=None,
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


def _to_rnn_cell(cell_or_type, num_units, num_layers):
  """Constructs and return an `RNNCell`.

  Args:
    cell_or_type: Either a string identifying the `RNNCell` type, a subclass of
      `RNNCell` or an instance of an `RNNCell`.
    num_units: The number of units in the `RNNCell`.
    num_layers: The number of layers in the RNN.
  Returns:
    An initialized `RNNCell`.
  Raises:
    ValueError: `cell_or_type` is an invalid `RNNCell` name.
    TypeError: `cell_or_type` is not a string or a subclass of `RNNCell`.
  """
  if isinstance(cell_or_type, rnn_cell.RNNCell):
    return cell_or_type
  if isinstance(cell_or_type, str):
    cell_or_type = _CELL_TYPES.get(cell_or_type)
    if cell_or_type is None:
      raise ValueError('The supported cell types are {}; got {}'.format(
          list(_CELL_TYPES.keys()), cell_or_type))
  if not issubclass(cell_or_type, rnn_cell.RNNCell):
    raise TypeError(
        'cell_or_type must be a subclass of RNNCell or one of {}.'.format(
            list(_CELL_TYPES.keys())))
  cell = cell_or_type(num_units=num_units)
  if num_layers > 1:
    cell = rnn_cell.MultiRNNCell(
        [cell] * num_layers, state_is_tuple=True)
  return cell


@experimental
def multi_value_rnn_regressor(num_units,
                              sequence_feature_columns,
                              context_feature_columns=None,
                              cell_type='basic_rnn',
                              num_rnn_layers=1,
                              optimizer_type='SGD',
                              learning_rate=0.1,
                              momentum=None,
                              gradient_clipping_norm=10.0,
                              input_keep_probability=None,
                              output_keep_probability=None,
                              model_dir=None,
                              config=None,
                              params=None,
                              feature_engineering_fn=None):

  """Creates a RNN `Estimator` that predicts sequences of values.

  Args:
    num_units: The size of the RNN cells.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features i.e. features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    cell_type: A subclass of `RNNCell`, an instance of an `RNNCell or one of
      'basic_rnn,' 'lstm' or 'gru'.
    num_rnn_layers: Number of RNN layers.
    optimizer_type: The type of optimizer to use. Either a subclass of
      `Optimizer`, an instance of an `Optimizer` or a string. Strings must be
      one of 'Adagrad', 'Momentum' or 'SGD'.
    learning_rate: Learning rate.
    momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs to `cell`. If `None`,
      no dropout is applied.
    model_dir: Directory to use for The directory in which to save and restore
      the model graph, parameters, etc.
    config: A `RunConfig` instance.
    params: `dict` of hyperparameters. Passed through to `Estimator`.
    feature_engineering_fn: Takes features and labels which are the output of
      `input_fn` and returns features and labels which will be fed into
      `model_fn`. Please check `model_fn` for a definition of features and
      labels.
  Returns:
    An initialized `Estimator`.
  """
  cell = _to_rnn_cell(cell_type, num_units, num_rnn_layers)
  target_column = layers.regression_target()
  if optimizer_type == 'Momentum':
    optimizer_type = momentum_opt.MomentumOptimizer(learning_rate, momentum)
  dynamic_rnn_model_fn = _get_dynamic_rnn_model_fn(
      cell=cell,
      target_column=target_column,
      problem_type=ProblemType.REGRESSION,
      prediction_type=PredictionType.MULTIPLE_VALUE,
      optimizer=optimizer_type,
      sequence_feature_columns=sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      learning_rate=learning_rate,
      gradient_clipping_norm=gradient_clipping_norm,
      input_keep_probability=input_keep_probability,
      output_keep_probability=output_keep_probability,
      name='MultiValueRnnRegressor')

  return estimator.Estimator(model_fn=dynamic_rnn_model_fn,
                             model_dir=model_dir,
                             config=config,
                             params=params,
                             feature_engineering_fn=feature_engineering_fn)


@experimental
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
                               gradient_clipping_norm=10.0,
                               input_keep_probability=None,
                               output_keep_probability=None,
                               model_dir=None,
                               config=None,
                               params=None,
                               feature_engineering_fn=None):
  """Creates a RNN `Estimator` that predicts sequences of labels.

  Args:
    num_classes: The number of classes for categorization.
    num_units: The size of the RNN cells.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features i.e. features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    cell_type: A subclass of `RNNCell`, an instance of an `RNNCell or one of
      'basic_rnn,' 'lstm' or 'gru'.
    num_rnn_layers: Number of RNN layers.
    optimizer_type: The type of optimizer to use. Either a subclass of
      `Optimizer`, an instance of an `Optimizer` or a string. Strings must be
      one of 'Adagrad', 'Momentum' or 'SGD'.
    learning_rate: Learning rate.
    predict_probabilities: A boolean indicating whether to predict probabilities
      for all classes.
    momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs to `cell`. If `None`,
      no dropout is applied.
    model_dir: Directory to use for The directory in which to save and restore
      the model graph, parameters, etc.
    config: A `RunConfig` instance.
    params: `dict` of hyperparameters. Passed through to `Estimator`.
    feature_engineering_fn: Takes features and labels which are the output of
      `input_fn` and returns features and labels which will be fed into
      `model_fn`. Please check `model_fn` for a definition of features and
      labels.
  Returns:
    An initialized `Estimator`.
  """
  cell = _to_rnn_cell(cell_type, num_units, num_rnn_layers)
  target_column = layers.multi_class_target(n_classes=num_classes)
  if optimizer_type == 'Momentum':
    optimizer_type = momentum_opt.MomentumOptimizer(learning_rate, momentum)
  dynamic_rnn_model_fn = _get_dynamic_rnn_model_fn(
      cell=cell,
      target_column=target_column,
      problem_type=ProblemType.CLASSIFICATION,
      prediction_type=PredictionType.MULTIPLE_VALUE,
      optimizer=optimizer_type,
      sequence_feature_columns=sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      predict_probabilities=predict_probabilities,
      learning_rate=learning_rate,
      gradient_clipping_norm=gradient_clipping_norm,
      input_keep_probability=input_keep_probability,
      output_keep_probability=output_keep_probability,
      name='MultiValueRnnClassifier')

  return estimator.Estimator(model_fn=dynamic_rnn_model_fn,
                             model_dir=model_dir,
                             config=config,
                             params=params,
                             feature_engineering_fn=feature_engineering_fn)


@experimental
def single_value_rnn_regressor(num_units,
                               sequence_feature_columns,
                               context_feature_columns=None,
                               cell_type='basic_rnn',
                               num_rnn_layers=1,
                               optimizer_type='SGD',
                               learning_rate=0.1,
                               momentum=None,
                               gradient_clipping_norm=10.0,
                               input_keep_probability=None,
                               output_keep_probability=None,
                               model_dir=None,
                               config=None,
                               params=None,
                               feature_engineering_fn=None):
  """Create a RNN `Estimator` that predicts single values.

  Args:
    num_units: The size of the RNN cells.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features i.e. features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    cell_type: A subclass of `RNNCell`, an instance of an `RNNCell or one of
      'basic_rnn,' 'lstm' or 'gru'.
    num_rnn_layers: Number of RNN layers.
    optimizer_type: The type of optimizer to use. Either a subclass of
      `Optimizer`, an instance of an `Optimizer` or a string. Strings must be
      one of 'Adagrad', 'Momentum' or 'SGD'.
    learning_rate: Learning rate.
    momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs to `cell`. If `None`,
      no dropout is applied.
    model_dir: Directory to use for The directory in which to save and restore
      the model graph, parameters, etc.
    config: A `RunConfig` instance.
    params: `dict` of hyperparameters. Passed through to `Estimator`.
    feature_engineering_fn: Takes features and labels which are the output of
      `input_fn` and returns features and labels which will be fed into
      `model_fn`. Please check `model_fn` for a definition of features and
      labels.
  Returns:
    An initialized `Estimator`.
  """
  cell = _to_rnn_cell(cell_type, num_units, num_rnn_layers)
  target_column = layers.regression_target()
  if optimizer_type == 'Momentum':
    optimizer_type = momentum_opt.MomentumOptimizer(learning_rate, momentum)
  dynamic_rnn_model_fn = _get_dynamic_rnn_model_fn(
      cell=cell,
      target_column=target_column,
      problem_type=ProblemType.REGRESSION,
      prediction_type=PredictionType.SINGLE_VALUE,
      optimizer=optimizer_type,
      sequence_feature_columns=sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      learning_rate=learning_rate,
      gradient_clipping_norm=gradient_clipping_norm,
      input_keep_probability=input_keep_probability,
      output_keep_probability=output_keep_probability,
      name='SingleValueRnnRegressor')

  return estimator.Estimator(model_fn=dynamic_rnn_model_fn,
                             model_dir=model_dir,
                             config=config,
                             params=params,
                             feature_engineering_fn=feature_engineering_fn)


@experimental
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
                                gradient_clipping_norm=10.0,
                                input_keep_probability=None,
                                output_keep_probability=None,
                                model_dir=None,
                                config=None,
                                params=None,
                                feature_engineering_fn=None):
  """Creates a RNN `Estimator` that predicts single labels.

  Args:
    num_classes: The number of classes for categorization.
    num_units: The size of the RNN cells.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features i.e. features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    cell_type: A subclass of `RNNCell`, an instance of an `RNNCell or one of
      'basic_rnn,' 'lstm' or 'gru'.
    num_rnn_layers: Number of RNN layers.
    optimizer_type: The type of optimizer to use. Either a subclass of
      `Optimizer`, an instance of an `Optimizer` or a string. Strings must be
      one of 'Adagrad', 'Momentum' or 'SGD'.
    learning_rate: Learning rate.
    predict_probabilities: A boolean indicating whether to predict probabilities
      for all classes.
    momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs to `cell`. If `None`,
      no dropout is applied.
    model_dir: Directory to use for The directory in which to save and restore
      the model graph, parameters, etc.
    config: A `RunConfig` instance.
    params: `dict` of hyperparameters. Passed through to `Estimator`.
    feature_engineering_fn: Takes features and labels which are the output of
      `input_fn` and returns features and labels which will be fed into
      `model_fn`. Please check `model_fn` for a definition of features and
      labels.
  Returns:
    An initialized `Estimator`.
  """
  cell = _to_rnn_cell(cell_type, num_units, num_rnn_layers)
  target_column = layers.multi_class_target(n_classes=num_classes)
  if optimizer_type == 'Momentum':
    optimizer_type = momentum_opt.MomentumOptimizer(learning_rate, momentum)
  dynamic_rnn_model_fn = _get_dynamic_rnn_model_fn(
      cell=cell,
      target_column=target_column,
      problem_type=ProblemType.CLASSIFICATION,
      prediction_type=PredictionType.SINGLE_VALUE,
      optimizer=optimizer_type,
      sequence_feature_columns=sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      predict_probabilities=predict_probabilities,
      learning_rate=learning_rate,
      gradient_clipping_norm=gradient_clipping_norm,
      input_keep_probability=input_keep_probability,
      output_keep_probability=output_keep_probability,
      name='SingleValueRnnClassifier')

  return estimator.Estimator(model_fn=dynamic_rnn_model_fn,
                             model_dir=model_dir,
                             config=config,
                             params=params,
                             feature_engineering_fn=feature_engineering_fn)
