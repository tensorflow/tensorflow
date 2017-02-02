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
"""Estimator for State Saving RNNs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.contrib import layers
from tensorflow.contrib import metrics
from tensorflow.contrib import rnn as rnn_cell
from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.training.python.training import sequence_queueing_state_saver as sqss
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import momentum as momentum_opt
from tensorflow.python.util import nest


class ProblemType(object):
  REGRESSION = 1
  CLASSIFICATION = 2


class RNNKeys(object):
  PREDICTIONS_KEY = 'predictions'
  PROBABILITIES_KEY = 'probabilities'
  FINAL_STATE_KEY = 'final_state'
  LABELS_KEY = '__labels__'
  STATE_PREFIX = 'rnn_cell_state'


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


def construct_state_saving_rnn(cell,
                               inputs,
                               num_label_columns,
                               state_saver,
                               state_name,
                               scope='rnn'):
  """Build a state saving RNN and apply a fully connected layer.

  Args:
    cell: An instance of `RNNCell`.
    inputs: A length `T` list of inputs, each a `Tensor` of shape
      `[batch_size, input_size, ...]`.
    num_label_columns: The desired output dimension.
    state_saver: A state saver object with methods `state` and `save_state`.
    state_name: Python string or tuple of strings.  The name to use with the
      state_saver. If the cell returns tuples of states (i.e.,
      `cell.state_size` is a tuple) then `state_name` should be a tuple of
      strings having the same length as `cell.state_size`.  Otherwise it should
      be a single string.
    scope: `VariableScope` for the created subgraph; defaults to "rnn".

  Returns:
    activations: The output of the RNN, projected to `num_label_columns`
      dimensions, a `Tensor` of shape `[batch_size, T, num_label_columns]`.
    final_state: The final state output by the RNN
  """
  with ops.name_scope(scope):
    rnn_outputs, final_state = core_rnn.static_state_saving_rnn(
        cell=cell,
        inputs=inputs,
        state_saver=state_saver,
        state_name=state_name,
        scope=scope)
    # Convert rnn_outputs from a list of time-major order Tensors to a single
    # Tensor of batch-major order.
    rnn_outputs = array_ops.stack(rnn_outputs, axis=1)
    activations = layers.fully_connected(
        inputs=rnn_outputs,
        num_outputs=num_label_columns,
        activation_fn=None,
        trainable=True)
    # Use `identity` to rename `final_state`.
    final_state = array_ops.identity(final_state, name=RNNKeys.FINAL_STATE_KEY)
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


def _get_default_metrics(problem_type, sequence_length):
  """Returns default `MetricSpec`s for `problem_type`.

  Args:
    problem_type: `ProblemType.CLASSIFICATION` or`ProblemType.REGRESSION`.
    sequence_length: A `Tensor` with shape `[batch_size]` and dtype `int32`
      containing the length of each sequence in the batch. If `None`, sequences
      are assumed to be unpadded.
  Returns:
    A `dict` mapping strings to `MetricSpec`s.
  """
  default_metrics = {}
  if problem_type == ProblemType.CLASSIFICATION:
    default_metrics['accuracy'] = metric_spec.MetricSpec(
        metric_fn=_mask_multivalue(sequence_length, metrics.streaming_accuracy),
        prediction_key=RNNKeys.PREDICTIONS_KEY)
  elif problem_type == ProblemType.REGRESSION:
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
        probability_shape = array_ops.concat([activations_shape[:2], [2]], 0)
      else:
        probability_shape = activations_shape
      probabilities = array_ops.reshape(
          flat_probabilities, probability_shape, name=RNNKeys.PROBABILITIES_KEY)
      prediction_dict[RNNKeys.PROBABILITIES_KEY] = probabilities
    else:
      flat_predictions = target_column.logits_to_predictions(
          flattened_activations, proba=False)
    predictions = array_ops.reshape(
        flat_predictions, [activations_shape[0], activations_shape[1]],
        name=RNNKeys.PREDICTIONS_KEY)
    prediction_dict[RNNKeys.PREDICTIONS_KEY] = predictions
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


def _get_name_or_parent_names(column):
  """Gets the name of a column or its parent columns' names.

  Args:
    column: A sequence feature column derived from `FeatureColumn`.

  Returns:
    A list of the name of `column` or the names of its parent columns,
    if any exist.
  """
  # pylint: disable=protected-access
  parent_columns = feature_column_ops._get_parent_columns(column)
  if parent_columns:
    return [x.name for x in parent_columns]
  return [column.name]


def _prepare_features_for_sqss(features, labels, mode, input_key_column_name,
                               sequence_feature_columns,
                               context_feature_columns):
  """Prepares features for batching by the SQSS.

  In preparation for batching by the SQSS, this function:
  - Extracts the input key from the features dict.
  - Separates sequence and context features dicts from the features dict.
  - Adds the labels tensor to the sequence features dict.

  Args:
    features: A dict of Python string to an iterable of `Tensor` or
      `SparseTensor` of rank 2, the `features` argument of a TF.Learn model_fn.
    labels: An iterable of `Tensor`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    input_key_column_name: Python string, the name of the feature column
      containing a string scalar `Tensor` that serves as a unique key to
      identify the input sequence across minibatches.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.

  Returns:
    input_key: The string scalar `Tensor` that serves as a unique key to
      identify the input sequence across minibatches.
    sequence_features: A dict mapping feature names to sequence features.
    context_features: A dict mapping feature names to context features.

  Raises:
    ValueError: If `features` does not contain a value for
      `input_key_column_name`.
    ValueError: If `features` does not contain a value for every key in
      `sequence_feature_columns` or `context_feature_columns`.
  """
  # Pop the input key from the features dict.
  input_key = features.pop(input_key_column_name, None)
  if input_key is None:
    raise ValueError('No key in features for input_key_column_name: ' +
                     input_key_column_name)

  # Extract sequence features.

  feature_column_ops._check_supported_sequence_columns(sequence_feature_columns)  # pylint: disable=protected-access
  sequence_features = {}
  for column in sequence_feature_columns:
    for name in _get_name_or_parent_names(column):
      feature = features.get(name, None)
      if feature is None:
        raise ValueError('No key in features for sequence feature: ' + name)
      sequence_features[name] = feature

  # Extract context features.
  context_features = {}
  if context_feature_columns is not None:
    for column in context_feature_columns:
      name = column.name
      feature = features.get(name, None)
      if feature is None:
        raise ValueError('No key in features for context feature: ' + name)
      context_features[name] = feature

  # Add labels to the resulting sequence features dict.
  if mode != model_fn.ModeKeys.INFER:
    sequence_features[RNNKeys.LABELS_KEY] = labels

  return input_key, sequence_features, context_features


def _read_batch(cell,
                features,
                labels,
                mode,
                num_unroll,
                num_layers,
                batch_size,
                input_key_column_name,
                sequence_feature_columns,
                context_feature_columns=None,
                num_threads=3,
                queue_capacity=1000):
  """Reads a batch from a state saving sequence queue.

  Args:
    cell: An initialized `RNNCell` to be used in the RNN.
    features: A dict of Python string to an iterable of `Tensor`, the
      `features` argument of a TF.Learn model_fn.
    labels: An iterable of `Tensor`, the `labels` argument of a
      TF.Learn model_fn.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    num_unroll: Python integer, how many time steps to unroll at a time.
      The input sequences of length `k` are then split into `k / num_unroll`
      many segments.
    num_layers: Python integer, number of layers in the RNN.
    batch_size: Python integer, the size of the minibatch produced by the SQSS.
    input_key_column_name: Python string, the name of the feature column
      containing a string scalar `Tensor` that serves as a unique key to
      identify input sequence across minibatches.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    num_threads: The Python integer number of threads enqueuing input examples
      into a queue. Defaults to 3.
    queue_capacity: The max capacity of the queue in number of examples.
      Needs to be at least `batch_size`. Defaults to 1000. When iterating
      over the same input example multiple times reusing their keys the
      `queue_capacity` must be smaller than the number of examples.

  Returns:
    batch: A `NextQueuedSequenceBatch` containing batch_size `SequenceExample`
      values and their saved internal states.
  """
  # Set batch_size=1 to initialize SQSS with cell's zero state.
  values = cell.zero_state(batch_size=1, dtype=dtypes.float32)

  # Set up stateful queue reader.
  states = {}
  state_names = _get_lstm_state_names(num_layers)
  for i in range(num_layers):
    states[state_names[i][0]] = array_ops.squeeze(values[i][0], axis=0)
    states[state_names[i][1]] = array_ops.squeeze(values[i][1], axis=0)

  input_key, sequences, context = _prepare_features_for_sqss(
      features, labels, mode, input_key_column_name, sequence_feature_columns,
      context_feature_columns)

  return sqss.batch_sequences_with_states(
      input_key=input_key,
      input_sequences=sequences,
      input_context=context,
      input_length=None,  # infer sequence lengths
      initial_states=states,
      num_unroll=num_unroll,
      batch_size=batch_size,
      pad=True,  # pad to a multiple of num_unroll
      num_threads=num_threads,
      capacity=queue_capacity)


def apply_dropout(
    cell, input_keep_probability, output_keep_probability, random_seed=None):
  """Apply dropout to the outputs and inputs of `cell`.

  Args:
    cell: An `RNNCell`.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs of `cell`. If `None`,
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
      state_value = (None if state_component is None else array_ops.identity(
          state_component, name=state_name))
      state_dict[state_name] = state_value
  return state_dict


def _prepare_inputs_for_rnn(sequence_features, context_features,
                            sequence_feature_columns, num_unroll):
  """Prepares features batched by the SQSS for input to a state-saving RNN.

  Args:
    sequence_features: A dict of sequence feature name to `Tensor` or
      `SparseTensor`, with `Tensor`s of shape `[batch_size, num_unroll, ...]`
      or `SparseTensors` of dense shape `[batch_size, num_unroll, d]`.
    context_features: A dict of context feature name to `Tensor`, with
      tensors of shape `[batch_size, 1, ...]` and type float32.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    num_unroll: Python integer, how many time steps to unroll at a time.
      The input sequences of length `k` are then split into `k / num_unroll`
      many segments.

  Returns:
    features_by_time: A list of length `num_unroll` with `Tensor` entries of
      shape `[batch_size, sum(sequence_features dimensions) +
      sum(context_features dimensions)]` of type float32.
      Context features are copied into each time step.
  """

  def _tile(feature):
    return array_ops.squeeze(
        array_ops.tile(array_ops.expand_dims(feature, 1), [1, num_unroll, 1]),
        axis=2)
  for feature in sequence_features.values():
    if isinstance(feature, sparse_tensor.SparseTensor):
      # Explicitly set dense_shape's shape to 3 ([batch_size, num_unroll, d])
      # since it can't be statically inferred.
      feature.dense_shape.set_shape([3])
  sequence_features = layers.sequence_input_from_feature_columns(
      columns_to_tensors=sequence_features,
      feature_columns=sequence_feature_columns,
      weight_collections=None,
      scope=None)
  # Explicitly set shape along dimension 1 to num_unroll for the unstack op.
  sequence_features.set_shape([None, num_unroll, None])

  if not context_features:
    return array_ops.unstack(sequence_features, axis=1)
  # TODO(jtbates): Call layers.input_from_feature_columns for context features.
  context_features = [
      _tile(context_features[k]) for k in sorted(context_features)
  ]
  return array_ops.unstack(
      array_ops.concat(
          [sequence_features, array_ops.stack(context_features, 2)], axis=2),
      axis=1)


def _get_rnn_model_fn(cell,
                      target_column,
                      problem_type,
                      optimizer,
                      num_unroll,
                      num_layers,
                      num_threads,
                      queue_capacity,
                      batch_size,
                      input_key_column_name,
                      sequence_feature_columns,
                      context_feature_columns=None,
                      predict_probabilities=False,
                      learning_rate=None,
                      gradient_clipping_norm=None,
                      input_keep_probability=None,
                      output_keep_probability=None,
                      name='StateSavingRNNModel'):
  """Creates a state saving RNN model function for an `Estimator`.

  Args:
    cell: An initialized `RNNCell` to be used in the RNN.
    target_column: An initialized `TargetColumn`, used to calculate prediction
      and loss.
    problem_type: `ProblemType.CLASSIFICATION` or`ProblemType.REGRESSION`.
    optimizer: A subclass of `Optimizer`, an instance of an `Optimizer` or a
      string.
    num_unroll: Python integer, how many time steps to unroll at a time.
      The input sequences of length `k` are then split into `k / num_unroll`
      many segments.
    num_layers: Python integer, number of layers in the RNN.
    num_threads: The Python integer number of threads enqueuing input examples
      into a queue.
    queue_capacity: The max capacity of the queue in number of examples.
      Needs to be at least `batch_size`. When iterating over the same input
      example multiple times reusing their keys the `queue_capacity` must be
      smaller than the number of examples.
    batch_size: Python integer, the size of the minibatch produced by the SQSS.
    input_key_column_name: Python string, the name of the feature column
      containing a string scalar `Tensor` that serves as a unique key to
      identify input sequence across minibatches.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    predict_probabilities: A boolean indicating whether to predict probabilities
      for all classes. Must only be used with `ProblemType.CLASSIFICATION`.
    learning_rate: Learning rate used for optimization. This argument has no
      effect if `optimizer` is an instance of an `Optimizer`.
    gradient_clipping_norm: A float. Gradients will be clipped to this value.
    input_keep_probability: Probability to keep inputs to `cell`. If `None`,
      no dropout is applied.
    output_keep_probability: Probability to keep outputs of `cell`. If `None`,
      no dropout is applied.
    name: A string that will be used to create a scope for the RNN.

  Returns:
    A model function to be passed to an `Estimator`.

  Raises:
    ValueError: `problem_type` is not one of `ProblemType.REGRESSION` or
      `ProblemType.CLASSIFICATION`.
    ValueError: `predict_probabilities` is `True` for `problem_type` other
      than `ProblemType.CLASSIFICATION`.
    ValueError: `num_unroll` is not positive.
    ValueError: `input_key_column_name` is empty.
  """
  if problem_type not in (ProblemType.CLASSIFICATION, ProblemType.REGRESSION):
    raise ValueError(
        'problem_type must be ProblemType.REGRESSION or '
        'ProblemType.CLASSIFICATION; got {}'.
        format(problem_type))
  if problem_type != ProblemType.CLASSIFICATION and predict_probabilities:
    raise ValueError(
        'predict_probabilities can only be set to True for problem_type'
        ' ProblemType.CLASSIFICATION; got {}.'.format(problem_type))
  if num_unroll <= 0:
    raise ValueError('num_unroll must be positive; got {}.'.format(num_unroll))
  if not input_key_column_name:
    raise ValueError('input_key_column_name must not be empty')

  def _rnn_model_fn(features, labels, mode):
    """The model to be passed to an `Estimator`."""
    with ops.name_scope(name):
      if mode == model_fn.ModeKeys.TRAIN:
        cell_for_mode = apply_dropout(
            cell, input_keep_probability, output_keep_probability)
      else:
        cell_for_mode = cell

      batch = _read_batch(
          cell=cell_for_mode,
          features=features,
          labels=labels,
          mode=mode,
          num_unroll=num_unroll,
          num_layers=num_layers,
          batch_size=batch_size,
          input_key_column_name=input_key_column_name,
          sequence_feature_columns=sequence_feature_columns,
          context_feature_columns=context_feature_columns,
          num_threads=num_threads,
          queue_capacity=queue_capacity)
      sequence_features = batch.sequences
      context_features = batch.context
      if mode != model_fn.ModeKeys.INFER:
        labels = sequence_features.pop(RNNKeys.LABELS_KEY)
      inputs = _prepare_inputs_for_rnn(sequence_features, context_features,
                                       sequence_feature_columns, num_unroll)
      state_name = _get_lstm_state_names(num_layers)
      rnn_activations, final_state = construct_state_saving_rnn(
          cell=cell_for_mode,
          inputs=inputs,
          num_label_columns=target_column.num_label_columns,
          state_saver=batch,
          state_name=state_name)

      loss = None  # Created below for modes TRAIN and EVAL.
      prediction_dict = _multi_value_predictions(rnn_activations, target_column,
                                                 predict_probabilities)
      if mode != model_fn.ModeKeys.INFER:
        loss = _multi_value_loss(rnn_activations, labels, batch.length,
                                 target_column, features)

      eval_metric_ops = None
      if mode != model_fn.ModeKeys.INFER:
        default_metrics = _get_default_metrics(problem_type, batch.length)
        eval_metric_ops = estimator._make_metrics_ops(  # pylint: disable=protected-access
            default_metrics, features, labels, prediction_dict)
      state_dict = state_tuple_to_dict(final_state)
      prediction_dict.update(state_dict)

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
  return _rnn_model_fn


def _get_lstm_state_names(num_layers):
  """Returns a num_layers long list of lstm state name pairs.

  Args:
    num_layers: The number of layers in the RNN.

  Returns:
     A num_layers long list of lstm state name pairs of the form:
     ['lstm_state_cN', 'lstm_state_mN'] for all N from 0 to num_layers.
  """
  return [['lstm_state_c' + str(i), 'lstm_state_m' + str(i)]
          for i in range(num_layers)]


# TODO(jtbates): Allow users to specify cell types other than LSTM.
def lstm_cell(num_units, num_layers):
  """Constructs a `MultiRNNCell` with num_layers `BasicLSTMCell`s.

  Args:
    num_units: The number of units in the `RNNCell`.
    num_layers: The number of layers in the RNN.

  Returns:
    An intiialized `MultiRNNCell`.
  """
  return rnn_cell.MultiRNNCell([
      rnn_cell.BasicLSTMCell(
          num_units=num_units, state_is_tuple=True) for _ in range(num_layers)
  ])


@experimental
def multi_value_rnn_regressor(num_units,
                              num_unroll,
                              batch_size,
                              input_key_column_name,
                              sequence_feature_columns,
                              context_feature_columns=None,
                              num_rnn_layers=1,
                              optimizer_type='SGD',
                              learning_rate=0.1,
                              momentum=None,
                              gradient_clipping_norm=5.0,
                              input_keep_probability=None,
                              output_keep_probability=None,
                              model_dir=None,
                              config=None,
                              feature_engineering_fn=None,
                              num_threads=3,
                              queue_capacity=1000):
  """Creates a RNN `Estimator` that predicts sequences of values.

  Args:
    num_units: The size of the RNN cells.
    num_unroll: Python integer, how many time steps to unroll at a time.
      The input sequences of length `k` are then split into `k / num_unroll`
      many segments.
    batch_size: Python integer, the size of the minibatch.
    input_key_column_name: Python string, the name of the feature column
      containing a string scalar `Tensor` that serves as a unique key to
      identify input sequence across minibatches.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    num_rnn_layers: Number of RNN layers. Leave this at its default value 1
      if passing a `cell_type` that is already a MultiRNNCell.
    optimizer_type: The type of optimizer to use. Either a subclass of
      `Optimizer`, an instance of an `Optimizer` or a string. Strings must be
      one of 'Adagrad', 'Momentum' or 'SGD'.
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
    num_threads: The Python integer number of threads enqueuing input examples
      into a queue. Defaults to 3.
    queue_capacity: The max capacity of the queue in number of examples.
      Needs to be at least `batch_size`. Defaults to 1000. When iterating
      over the same input example multiple times reusing their keys the
      `queue_capacity` must be smaller than the number of examples.
  Returns:
    An initialized `Estimator`.
  """
  cell = lstm_cell(num_units, num_rnn_layers)
  target_column = layers.regression_target()
  if optimizer_type == 'Momentum':
    optimizer_type = momentum_opt.MomentumOptimizer(learning_rate, momentum)
  rnn_model_fn = _get_rnn_model_fn(
      cell=cell,
      target_column=target_column,
      problem_type=ProblemType.REGRESSION,
      optimizer=optimizer_type,
      num_unroll=num_unroll,
      num_layers=num_rnn_layers,
      num_threads=num_threads,
      queue_capacity=queue_capacity,
      batch_size=batch_size,
      input_key_column_name=input_key_column_name,
      sequence_feature_columns=sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      learning_rate=learning_rate,
      gradient_clipping_norm=gradient_clipping_norm,
      input_keep_probability=input_keep_probability,
      output_keep_probability=output_keep_probability,
      name='MultiValueRnnRegressor')

  return estimator.Estimator(
      model_fn=rnn_model_fn,
      model_dir=model_dir,
      config=config,
      feature_engineering_fn=feature_engineering_fn)


@experimental
def multi_value_rnn_classifier(num_classes,
                               num_units,
                               num_unroll,
                               batch_size,
                               input_key_column_name,
                               sequence_feature_columns,
                               context_feature_columns=None,
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
                               feature_engineering_fn=None,
                               num_threads=3,
                               queue_capacity=1000):
  """Creates a RNN `Estimator` that predicts sequences of labels.

  Args:
    num_classes: The number of classes for categorization.
    num_units: The size of the RNN cells.
    num_unroll: Python integer, how many time steps to unroll at a time.
      The input sequences of length `k` are then split into `k / num_unroll`
      many segments.
    batch_size: Python integer, the size of the minibatch.
    input_key_column_name: Python string, the name of the feature column
      containing a string scalar `Tensor` that serves as a unique key to
      identify input sequence across minibatches.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    num_rnn_layers: Number of RNN layers.
    optimizer_type: The type of optimizer to use. Either a subclass of
      `Optimizer`, an instance of an `Optimizer` or a string. Strings must be
      one of 'Adagrad', 'Momentum' or 'SGD'.
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
    num_threads: The Python integer number of threads enqueuing input examples
      into a queue. Defaults to 3.
    queue_capacity: The max capacity of the queue in number of examples.
      Needs to be at least `batch_size`. Defaults to 1000. When iterating
      over the same input example multiple times reusing their keys the
      `queue_capacity` must be smaller than the number of examples.
  Returns:
    An initialized `Estimator`.
  """
  cell = lstm_cell(num_units, num_rnn_layers)
  target_column = layers.multi_class_target(n_classes=num_classes)
  if optimizer_type == 'Momentum':
    optimizer_type = momentum_opt.MomentumOptimizer(learning_rate, momentum)
  rnn_model_fn = _get_rnn_model_fn(
      cell=cell,
      target_column=target_column,
      problem_type=ProblemType.CLASSIFICATION,
      optimizer=optimizer_type,
      num_unroll=num_unroll,
      num_layers=num_rnn_layers,
      num_threads=num_threads,
      queue_capacity=queue_capacity,
      batch_size=batch_size,
      input_key_column_name=input_key_column_name,
      sequence_feature_columns=sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      predict_probabilities=predict_probabilities,
      learning_rate=learning_rate,
      gradient_clipping_norm=gradient_clipping_norm,
      input_keep_probability=input_keep_probability,
      output_keep_probability=output_keep_probability,
      name='MultiValueRnnClassifier')

  return estimator.Estimator(
      model_fn=rnn_model_fn,
      model_dir=model_dir,
      config=config,
      feature_engineering_fn=feature_engineering_fn)
