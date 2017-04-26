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

from tensorflow.contrib import layers
from tensorflow.contrib import rnn as rnn_cell
from tensorflow.contrib.framework.python.framework import deprecated
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.training.python.training import sequence_queueing_state_saver as sqss
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.training import momentum as momentum_opt
from tensorflow.python.util import nest


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
    final_state = array_ops.identity(
        final_state, name=rnn_common.RNNKeys.FINAL_STATE_KEY)
    return activations, final_state


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
    activations_masked, labels_masked = rnn_common.mask_activations_and_labels(
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


def _prepare_features_for_sqss(features, labels, mode,
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
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.

  Returns:
    sequence_features: A dict mapping feature names to sequence features.
    context_features: A dict mapping feature names to context features.

  Raises:
    ValueError: If `features` does not contain a value for every key in
      `sequence_feature_columns` or `context_feature_columns`.
  """

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
    sequence_features[rnn_common.RNNKeys.LABELS_KEY] = labels

  return sequence_features, context_features


def _get_state_names(cell):
  """Gets the state names for an `RNNCell`.

  Args:
    cell: A `RNNCell` to be used in the RNN.

  Returns:
    State names in the form of a string, a list of strings, or a list of
    string pairs, depending on the type of `cell.state_size`.

  Raises:
    TypeError: If cell.state_size is of type TensorShape.
  """
  state_size = cell.state_size
  if isinstance(state_size, tensor_shape.TensorShape):
    raise TypeError('cell.state_size of type TensorShape is not supported.')
  if isinstance(state_size, int):
    return '{}_{}'.format(rnn_common.RNNKeys.STATE_PREFIX, 0)
  if isinstance(state_size, rnn_cell.LSTMStateTuple):
    return [
        '{}_{}_c'.format(rnn_common.RNNKeys.STATE_PREFIX, 0),
        '{}_{}_h'.format(rnn_common.RNNKeys.STATE_PREFIX, 0),
    ]
  if isinstance(state_size[0], rnn_cell.LSTMStateTuple):
    return [[
        '{}_{}_c'.format(rnn_common.RNNKeys.STATE_PREFIX, i),
        '{}_{}_h'.format(rnn_common.RNNKeys.STATE_PREFIX, i),
    ] for i in range(len(state_size))]
  return [
      '{}_{}'.format(rnn_common.RNNKeys.STATE_PREFIX, i)
      for i in range(len(state_size))]


def _get_initial_states(cell):
  """Gets the initial state of the `RNNCell` used in the RNN.

  Args:
    cell: A `RNNCell` to be used in the RNN.

  Returns:
    A Python dict mapping state names to the `RNNCell`'s initial state for
    consumption by the SQSS.
  """
  names = nest.flatten(_get_state_names(cell))
  values = nest.flatten(cell.zero_state(1, dtype=dtypes.float32))
  return {n: array_ops.squeeze(v, axis=0) for [n, v] in zip(names, values)}


def _read_batch(cell,
                features,
                labels,
                mode,
                num_unroll,
                batch_size,
                sequence_feature_columns,
                context_feature_columns=None,
                num_threads=3,
                queue_capacity=1000,
                seed=None):
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
    batch_size: Python integer, the size of the minibatch produced by the SQSS.
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
    seed: Fixes the random seed used for generating input keys by the SQSS.

  Returns:
    batch: A `NextQueuedSequenceBatch` containing batch_size `SequenceExample`
      values and their saved internal states.
  """
  states = _get_initial_states(cell)

  sequences, context = _prepare_features_for_sqss(
      features, labels, mode, sequence_feature_columns,
      context_feature_columns)

  return sqss.batch_sequences_with_states(
      input_key='key',
      input_sequences=sequences,
      input_context=context,
      input_length=None,  # infer sequence lengths
      initial_states=states,
      num_unroll=num_unroll,
      batch_size=batch_size,
      pad=True,  # pad to a multiple of num_unroll
      make_keys_unique=True,
      make_keys_unique_seed=seed,
      num_threads=num_threads,
      capacity=queue_capacity)


def _get_state_name(i):
  """Constructs the name string for state component `i`."""
  return '{}_{}'.format(rnn_common.RNNKeys.STATE_PREFIX, i)


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


def _get_rnn_model_fn(cell_type,
                      target_column,
                      problem_type,
                      optimizer,
                      num_unroll,
                      num_units,
                      num_threads,
                      queue_capacity,
                      batch_size,
                      sequence_feature_columns,
                      context_feature_columns=None,
                      predict_probabilities=False,
                      learning_rate=None,
                      gradient_clipping_norm=None,
                      dropout_keep_probabilities=None,
                      name='StateSavingRNNModel',
                      seed=None):
  """Creates a state saving RNN model function for an `Estimator`.

  Args:
    cell_type: A subclass of `RNNCell` or one of 'basic_rnn,' 'lstm' or 'gru'.
    target_column: An initialized `TargetColumn`, used to calculate prediction
      and loss.
    problem_type: `ProblemType.CLASSIFICATION` or
    `ProblemType.LINEAR_REGRESSION`.
    optimizer: A subclass of `Optimizer`, an instance of an `Optimizer` or a
      string.
    num_unroll: Python integer, how many time steps to unroll at a time.
      The input sequences of length `k` are then split into `k / num_unroll`
      many segments.
    num_units: The number of units in the `RNNCell`.
    num_threads: The Python integer number of threads enqueuing input examples
      into a queue.
    queue_capacity: The max capacity of the queue in number of examples.
      Needs to be at least `batch_size`. When iterating over the same input
      example multiple times reusing their keys the `queue_capacity` must be
      smaller than the number of examples.
    batch_size: Python integer, the size of the minibatch produced by the SQSS.
    sequence_feature_columns: An iterable containing all the feature columns
      describing sequence features. All items in the set should be instances
      of classes derived from `FeatureColumn`.
    context_feature_columns: An iterable containing all the feature columns
      describing context features, i.e., features that apply accross all time
      steps. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    predict_probabilities: A boolean indicating whether to predict probabilities
      for all classes.
      Must only be used with `ProblemType.CLASSIFICATION`.
    learning_rate: Learning rate used for optimization. This argument has no
      effect if `optimizer` is an instance of an `Optimizer`.
    gradient_clipping_norm: A float. Gradients will be clipped to this value.
    dropout_keep_probabilities: a list of dropout keep probabilities or `None`.
      If given a list, it must have length `len(num_units) + 1`.
    name: A string that will be used to create a scope for the RNN.
    seed: Fixes the random seed used for generating input keys by the SQSS.

  Returns:
    A model function to be passed to an `Estimator`.

  Raises:
    ValueError: `problem_type` is not one of
      `ProblemType.LINEAR_REGRESSION`
      or `ProblemType.CLASSIFICATION`.
    ValueError: `predict_probabilities` is `True` for `problem_type` other
      than `ProblemType.CLASSIFICATION`.
    ValueError: `num_unroll` is not positive.
  """
  if problem_type not in (constants.ProblemType.CLASSIFICATION,
                          constants.ProblemType.LINEAR_REGRESSION):
    raise ValueError(
        'problem_type must be ProblemType.LINEAR_REGRESSION or '
        'ProblemType.CLASSIFICATION; got {}'.
        format(problem_type))
  if (problem_type != constants.ProblemType.CLASSIFICATION and
      predict_probabilities):
    raise ValueError(
        'predict_probabilities can only be set to True for problem_type'
        ' ProblemType.CLASSIFICATION; got {}.'.format(problem_type))
  if num_unroll <= 0:
    raise ValueError('num_unroll must be positive; got {}.'.format(num_unroll))

  def _rnn_model_fn(features, labels, mode):
    """The model to be passed to an `Estimator`."""
    with ops.name_scope(name):
      dropout = (dropout_keep_probabilities
                 if mode == model_fn.ModeKeys.TRAIN
                 else None)
      cell = rnn_common.construct_rnn_cell(num_units, cell_type, dropout)

      batch = _read_batch(
          cell=cell,
          features=features,
          labels=labels,
          mode=mode,
          num_unroll=num_unroll,
          batch_size=batch_size,
          sequence_feature_columns=sequence_feature_columns,
          context_feature_columns=context_feature_columns,
          num_threads=num_threads,
          queue_capacity=queue_capacity,
          seed=seed)
      sequence_features = batch.sequences
      context_features = batch.context
      if mode != model_fn.ModeKeys.INFER:
        labels = sequence_features.pop(rnn_common.RNNKeys.LABELS_KEY)
      inputs = _prepare_inputs_for_rnn(sequence_features, context_features,
                                       sequence_feature_columns, num_unroll)
      state_name = _get_state_names(cell)
      rnn_activations, final_state = construct_state_saving_rnn(
          cell=cell,
          inputs=inputs,
          num_label_columns=target_column.num_label_columns,
          state_saver=batch,
          state_name=state_name)

      loss = None  # Created below for modes TRAIN and EVAL.
      prediction_dict = rnn_common.multi_value_predictions(
          rnn_activations, target_column, problem_type, predict_probabilities)
      if mode != model_fn.ModeKeys.INFER:
        loss = _multi_value_loss(rnn_activations, labels, batch.length,
                                 target_column, features)

      eval_metric_ops = None
      if mode != model_fn.ModeKeys.INFER:
        eval_metric_ops = rnn_common.get_eval_metric_ops(
            problem_type, rnn_common.PredictionType.MULTIPLE_VALUE,
            batch.length, prediction_dict, labels)

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


class StateSavingRnnEstimator(estimator.Estimator):

  def __init__(self,
               problem_type,
               num_unroll,
               batch_size,
               sequence_feature_columns,
               context_feature_columns=None,
               num_classes=None,
               num_units=None,
               cell_type='basic_rnn',
               optimizer_type='SGD',
               learning_rate=0.1,
               predict_probabilities=False,
               momentum=None,
               gradient_clipping_norm=5.0,
               dropout_keep_probabilities=None,
               model_dir=None,
               config=None,
               feature_engineering_fn=None,
               num_threads=3,
               queue_capacity=1000,
               seed=None):
    """Initializes a StateSavingRnnEstimator.

    Args:
      problem_type: `ProblemType.CLASSIFICATION` or
        `ProblemType.LINEAR_REGRESSION`.
      num_unroll: Python integer, how many time steps to unroll at a time.
        The input sequences of length `k` are then split into `k / num_unroll`
        many segments.
      batch_size: Python integer, the size of the minibatch.
      sequence_feature_columns: An iterable containing all the feature columns
        describing sequence features. All items in the set should be instances
        of classes derived from `FeatureColumn`.
      context_feature_columns: An iterable containing all the feature columns
        describing context features, i.e., features that apply accross all time
        steps. All items in the set should be instances of classes derived from
        `FeatureColumn`.
      num_classes: The number of classes for categorization. Used only and
        required if `problem_type` is `ProblemType.CLASSIFICATION`.
      num_units: A list of integers indicating the number of units in the
        `RNNCell`s in each layer. Either `num_units` is specified or `cell_type`
        is an instance of `RNNCell`.
      cell_type: A subclass of `RNNCell` or one of 'basic_rnn,' 'lstm' or 'gru'.
      optimizer_type: The type of optimizer to use. Either a subclass of
        `Optimizer`, an instance of an `Optimizer` or a string. Strings must be
        one of 'Adagrad', 'Adam', 'Ftrl', Momentum', 'RMSProp', or 'SGD'.
      learning_rate: Learning rate. This argument has no effect if `optimizer`
        is an instance of an `Optimizer`.
      predict_probabilities: A boolean indicating whether to predict
        probabilities for all classes. Used only if `problem_type` is
        `ProblemType.CLASSIFICATION`.
      momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
      gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
        then no clipping is performed.
      dropout_keep_probabilities: a list of dropout keep probabilities or
        `None`. If given a list, it must have length `len(num_units) + 1`.
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
      seed: Fixes the random seed used for generating input keys by the SQSS.

    Raises:
      ValueError: Both or neither of the following are true: (a) `num_units` is
        specified and (b) `cell_type` is an instance of `RNNCell`.
      ValueError: `problem_type` is not one of
        `ProblemType.LINEAR_REGRESSION` or `ProblemType.CLASSIFICATION`.
      ValueError: `problem_type` is `ProblemType.CLASSIFICATION` but
        `num_classes` is not specified.
    """
    name = 'MultiValueStateSavingRNN'
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

    if optimizer_type == 'Momentum':
      optimizer_type = momentum_opt.MomentumOptimizer(learning_rate, momentum)

    rnn_model_fn = _get_rnn_model_fn(
        cell_type=cell_type,
        target_column=target_column,
        problem_type=problem_type,
        optimizer=optimizer_type,
        num_unroll=num_unroll,
        num_units=num_units,
        num_threads=num_threads,
        queue_capacity=queue_capacity,
        batch_size=batch_size,
        sequence_feature_columns=sequence_feature_columns,
        context_feature_columns=context_feature_columns,
        predict_probabilities=predict_probabilities,
        learning_rate=learning_rate,
        gradient_clipping_norm=gradient_clipping_norm,
        dropout_keep_probabilities=dropout_keep_probabilities,
        name=name,
        seed=seed)

    super(StateSavingRnnEstimator, self).__init__(
        model_fn=rnn_model_fn,
        model_dir=model_dir,
        config=config,
        feature_engineering_fn=feature_engineering_fn)


@deprecated('2017-04-01', 'multi_value_rnn_regressor is deprecated. '
            'Please construct a StateSavingRnnEstimator directly.')
def multi_value_rnn_regressor(num_units,
                              num_unroll,
                              batch_size,
                              sequence_feature_columns,
                              context_feature_columns=None,
                              num_rnn_layers=1,
                              optimizer_type='SGD',
                              learning_rate=0.1,
                              momentum=None,
                              gradient_clipping_norm=5.0,
                              dropout_keep_probabilities=None,
                              model_dir=None,
                              config=None,
                              feature_engineering_fn=None,
                              num_threads=3,
                              queue_capacity=1000,
                              seed=None):
  """Creates a RNN `Estimator` that predicts sequences of values.

  Args:
    num_units: The size of the RNN cells.
    num_unroll: Python integer, how many time steps to unroll at a time.
      The input sequences of length `k` are then split into `k / num_unroll`
      many segments.
    batch_size: Python integer, the size of the minibatch.
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
    momentum: Momentum value. Only used if `optimizer_type` is 'Momentum'.
    gradient_clipping_norm: Parameter used for gradient clipping. If `None`,
      then no clipping is performed.
    dropout_keep_probabilities: a list of dropout keep probabilities or `None`.
        If given a list, it must have length `num_rnn_layers + 1`.
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
    seed: Fixes the random seed used for generating input keys by the SQSS.
  Returns:
    An initialized `Estimator`.
  """
  num_units = [num_units for _ in range(num_rnn_layers)]
  return StateSavingRnnEstimator(
      constants.ProblemType.LINEAR_REGRESSION,
      num_unroll,
      batch_size,
      sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      num_classes=None,
      num_units=num_units,
      cell_type='lstm',
      optimizer_type=optimizer_type,
      learning_rate=learning_rate,
      predict_probabilities=False,
      momentum=momentum,
      gradient_clipping_norm=gradient_clipping_norm,
      dropout_keep_probabilities=dropout_keep_probabilities,
      model_dir=model_dir,
      config=config,
      feature_engineering_fn=feature_engineering_fn,
      num_threads=num_threads,
      queue_capacity=queue_capacity,
      seed=seed)


@deprecated('2017-04-01', 'multi_value_rnn_classifier is deprecated. '
            'Please construct a StateSavingRnnEstimator directly.')
def multi_value_rnn_classifier(num_classes,
                               num_units,
                               num_unroll,
                               batch_size,
                               sequence_feature_columns,
                               context_feature_columns=None,
                               num_rnn_layers=1,
                               optimizer_type='SGD',
                               learning_rate=0.1,
                               predict_probabilities=False,
                               momentum=None,
                               gradient_clipping_norm=5.0,
                               dropout_keep_probabilities=None,
                               model_dir=None,
                               config=None,
                               feature_engineering_fn=None,
                               num_threads=3,
                               queue_capacity=1000,
                               seed=None):
  """Creates a RNN `Estimator` that predicts sequences of labels.

  Args:
    num_classes: The number of classes for categorization.
    num_units: The size of the RNN cells.
    num_unroll: Python integer, how many time steps to unroll at a time.
      The input sequences of length `k` are then split into `k / num_unroll`
      many segments.
    batch_size: Python integer, the size of the minibatch.
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
    dropout_keep_probabilities: a list of dropout keep probabilities or `None`.
        If given a list, it must have length `num_rnn_layers + 1`.
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
    seed: Fixes the random seed used for generating input keys by the SQSS.
  Returns:
    An initialized `Estimator`.
  """
  num_units = [num_units for _ in range(num_rnn_layers)]
  return StateSavingRnnEstimator(
      constants.ProblemType.CLASSIFICATION,
      num_unroll,
      batch_size,
      sequence_feature_columns,
      context_feature_columns=context_feature_columns,
      num_classes=num_classes,
      num_units=num_units,
      cell_type='lstm',
      optimizer_type=optimizer_type,
      learning_rate=learning_rate,
      predict_probabilities=predict_probabilities,
      momentum=momentum,
      gradient_clipping_norm=gradient_clipping_norm,
      dropout_keep_probabilities=dropout_keep_probabilities,
      model_dir=model_dir,
      config=config,
      feature_engineering_fn=feature_engineering_fn,
      num_threads=num_threads,
      queue_capacity=queue_capacity,
      seed=seed)
