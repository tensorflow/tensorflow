# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Recurrent Neural Network estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.estimator.python.estimator import extenders
from tensorflow.contrib.feature_column.python.feature_column import sequence_feature_column as seq_fc
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import training_util


# The defaults are historical artifacts of the initial implementation, but seem
# reasonable choices.
_DEFAULT_LEARNING_RATE = 0.05
_DEFAULT_CLIP_NORM = 5.0

_CELL_TYPES = {'basic_rnn': rnn_cell.BasicRNNCell,
               'lstm': rnn_cell.BasicLSTMCell,
               'gru': rnn_cell.GRUCell}

# Indicates no value was provided by the user to a kwarg.
USE_DEFAULT = object()


def _single_rnn_cell(num_units, cell_type):
  cell_type = _CELL_TYPES.get(cell_type, cell_type)
  if not cell_type or not issubclass(cell_type, rnn_cell.RNNCell):
    raise ValueError('Supported cell types are {}; got {}'.format(
        list(_CELL_TYPES.keys()), cell_type))
  return cell_type(num_units=num_units)


def _make_rnn_cell_fn(num_units, cell_type='basic_rnn'):
  """Convenience function to create `rnn_cell_fn` for canned RNN Estimators.

  Args:
    num_units: Iterable of integer number of hidden units per RNN layer.
    cell_type: A subclass of `tf.nn.rnn_cell.RNNCell` or a string specifying
      the cell type. Supported strings are: `'basic_rnn'`, `'lstm'`, and
      `'gru'`.

  Returns:
    A function that takes a single argument, an instance of
    `tf.estimator.ModeKeys`, and returns an instance derived from
    `tf.nn.rnn_cell.RNNCell`.

  Raises:
    ValueError: If cell_type is not supported.
  """
  def rnn_cell_fn(mode):
    # Unused. Part of the rnn_cell_fn interface since user specified functions
    # may need different behavior across modes (e.g. dropout).
    del mode
    cells = [_single_rnn_cell(n, cell_type) for n in num_units]
    if len(cells) == 1:
      return cells[0]
    return rnn_cell.MultiRNNCell(cells)
  return rnn_cell_fn


def _concatenate_context_input(sequence_input, context_input):
  """Replicates `context_input` across all timesteps of `sequence_input`.

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


def _select_last_activations(activations, sequence_lengths):
  """Selects the nth set of activations for each n in `sequence_length`.

  Returns a `Tensor` of shape `[batch_size, k]`. If `sequence_length` is not
  `None`, then `output[i, :] = activations[i, sequence_length[i] - 1, :]`. If
  `sequence_length` is `None`, then `output[i, :] = activations[i, -1, :]`.

  Args:
    activations: A `Tensor` with shape `[batch_size, padded_length, k]`.
    sequence_lengths: A `Tensor` with shape `[batch_size]` or `None`.
  Returns:
    A `Tensor` of shape `[batch_size, k]`.
  """
  with ops.name_scope(
      'select_last_activations', values=[activations, sequence_lengths]):
    activations_shape = array_ops.shape(activations)
    batch_size = activations_shape[0]
    padded_length = activations_shape[1]
    output_units = activations_shape[2]
    if sequence_lengths is None:
      sequence_lengths = padded_length
    start_indices = math_ops.to_int64(
        math_ops.range(batch_size) * padded_length)
    last_indices = start_indices + sequence_lengths - 1
    reshaped_activations = array_ops.reshape(
        activations, [batch_size * padded_length, output_units])

    last_activations = array_ops.gather(reshaped_activations, last_indices)
    last_activations.set_shape([activations.shape[0], activations.shape[2]])
    return last_activations


def _rnn_logit_fn_builder(output_units, rnn_cell_fn, sequence_feature_columns,
                          context_feature_columns, input_layer_partitioner):
  """Function builder for a rnn logit_fn.

  Args:
    output_units: An int indicating the dimension of the logit layer.
    rnn_cell_fn: A function with one argument, a `tf.estimator.ModeKeys`, and
      returns an object of type `tf.nn.rnn_cell.RNNCell`.
    sequence_feature_columns: An iterable containing the `FeatureColumn`s
      that represent sequential input.
    context_feature_columns: An iterable containing the `FeatureColumn`s
      that represent contextual input.
    input_layer_partitioner: Partitioner for input layer.

  Returns:
    A logit_fn (see below).

  Raises:
    ValueError: If output_units is not an int.
  """
  if not isinstance(output_units, int):
    raise ValueError('output_units must be an int.  Given type: {}'.format(
        type(output_units)))

  def rnn_logit_fn(features, mode):
    """Recurrent Neural Network logit_fn.

    Args:
      features: This is the first item returned from the `input_fn`
                passed to `train`, `evaluate`, and `predict`. This should be a
                single `Tensor` or `dict` of same.
      mode: Optional. Specifies if this training, evaluation or prediction. See
            `ModeKeys`.

    Returns:
      A `Tensor` representing the logits.
    """
    with variable_scope.variable_scope(
        'sequence_input_layer',
        values=tuple(six.itervalues(features)),
        partitioner=input_layer_partitioner):
      sequence_input, sequence_length = seq_fc.sequence_input_layer(
          features=features, feature_columns=sequence_feature_columns)
      summary.histogram('sequence_length', sequence_length)

      if context_feature_columns:
        context_input = feature_column_lib.input_layer(
            features=features,
            feature_columns=context_feature_columns)
        sequence_input = _concatenate_context_input(sequence_input,
                                                    context_input)

    cell = rnn_cell_fn(mode)
    # Ignore output state.
    rnn_outputs, _ = rnn.dynamic_rnn(
        cell=cell,
        inputs=sequence_input,
        sequence_length=sequence_length,
        dtype=dtypes.float32,
        time_major=False)
    last_activations = _select_last_activations(rnn_outputs, sequence_length)

    with variable_scope.variable_scope('logits', values=(rnn_outputs,)):
      logits = core_layers.dense(
          last_activations,
          units=output_units,
          activation=None,
          kernel_initializer=init_ops.glorot_uniform_initializer())
    return logits

  return rnn_logit_fn


def _rnn_model_fn(features,
                  labels,
                  mode,
                  head,
                  rnn_cell_fn,
                  sequence_feature_columns,
                  context_feature_columns,
                  optimizer='Adagrad',
                  input_layer_partitioner=None,
                  config=None):
  """Recurrent Neural Net model_fn.

  Args:
    features: dict of `Tensor` and `SparseTensor` objects returned from
      `input_fn`.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] with labels.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    head: A `head_lib._Head` instance.
    rnn_cell_fn: A function with one argument, a `tf.estimator.ModeKeys`, and
      returns an object of type `tf.nn.rnn_cell.RNNCell`.
    sequence_feature_columns: Iterable containing `FeatureColumn`s that
      represent sequential model inputs.
    context_feature_columns: Iterable containing `FeatureColumn`s that
      represent model inputs not associated with a specific timestep.
    optimizer: String, `tf.Optimizer` object, or callable that creates the
      optimizer to use for training. If not specified, will use the Adagrad
      optimizer with a default learning rate of 0.05 and gradient clip norm of
      5.0.
    input_layer_partitioner: Partitioner for input layer. Defaults
      to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: If mode or optimizer is invalid, or features has the wrong type.
  """
  if not isinstance(features, dict):
    raise ValueError('features should be a dictionary of `Tensor`s. '
                     'Given type: {}'.format(type(features)))

  # If user does not provide an optimizer instance, use the optimizer specified
  # by the string with default learning rate and gradient clipping.
  if not isinstance(optimizer, optimizer_lib.Optimizer):
    optimizer = optimizers.get_optimizer_instance(
        optimizer, learning_rate=_DEFAULT_LEARNING_RATE)
    optimizer = extenders.clip_gradients_by_norm(optimizer, _DEFAULT_CLIP_NORM)

  num_ps_replicas = config.num_ps_replicas if config else 0
  partitioner = partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas)
  with variable_scope.variable_scope(
      'rnn',
      values=tuple(six.itervalues(features)),
      partitioner=partitioner):
    input_layer_partitioner = input_layer_partitioner or (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20))

    logit_fn = _rnn_logit_fn_builder(
        output_units=head.logits_dimension,
        rnn_cell_fn=rnn_cell_fn,
        sequence_feature_columns=sequence_feature_columns,
        context_feature_columns=context_feature_columns,
        input_layer_partitioner=input_layer_partitioner)
    logits = logit_fn(features=features, mode=mode)

    def _train_op_fn(loss):
      """Returns the op to optimize the loss."""
      return optimizer.minimize(
          loss,
          global_step=training_util.get_global_step())

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)


def _assert_rnn_cell_fn(rnn_cell_fn, num_units, cell_type):
  """Assert arguments are valid and return rnn_cell_fn."""
  if rnn_cell_fn and (num_units or cell_type != USE_DEFAULT):
    raise ValueError(
        'num_units and cell_type must not be specified when using rnn_cell_fn'
    )
  if not rnn_cell_fn:
    if cell_type == USE_DEFAULT:
      cell_type = 'basic_rnn'
    rnn_cell_fn = _make_rnn_cell_fn(num_units, cell_type)
  return rnn_cell_fn


class RNNClassifier(estimator.Estimator):
  """A classifier for TensorFlow RNN models.

  Trains a recurrent neural network model to classify instances into one of
  multiple classes.

  Example:

  ```python
  token_sequence = sequence_categorical_column_with_hash_bucket(...)
  token_emb = embedding_column(categorical_column=token_sequence, ...)

  estimator = RNNClassifier(
      sequence_feature_columns=[token_emb],
      num_units=[32, 16], cell_type='lstm')

  # Input builders
  def input_fn_train: # returns x, y
    pass
  estimator.train(input_fn=input_fn_train, steps=100)

  def input_fn_eval: # returns x, y
    pass
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  def input_fn_predict: # returns x, None
    pass
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * if `weight_column` is not `None`, a feature with
    `key=weight_column` whose value is a `Tensor`.
  * for each `column` in `sequence_feature_columns`:
    - a feature with `key=column.name` whose `value` is a `SparseTensor`.
  * for each `column` in `context_feature_columns`:
    - if `column` is a `_CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `_WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `_DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss is calculated by using softmax cross entropy.

  @compatibility(eager)
  Estimators are not compatible with eager execution.
  @end_compatibility
  """

  def __init__(self,
               sequence_feature_columns,
               context_feature_columns=None,
               num_units=None,
               cell_type=USE_DEFAULT,
               rnn_cell_fn=None,
               model_dir=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               optimizer='Adagrad',
               loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE,
               input_layer_partitioner=None,
               config=None):
    """Initializes a `RNNClassifier` instance.

    Args:
      sequence_feature_columns: An iterable containing the `FeatureColumn`s
        that represent sequential input. All items in the set should either be
        sequence columns (e.g. `sequence_numeric_column`) or constructed from
        one (e.g. `embedding_column` with `sequence_categorical_column_*` as
        input).
      context_feature_columns: An iterable containing the `FeatureColumn`s
        for contextual input. The data represented by these columns will be
        replicated and given to the RNN at each timestep. These columns must be
        instances of classes derived from `_DenseColumn` such as
        `numeric_column`, not the sequential variants.
      num_units: Iterable of integer number of hidden units per RNN layer. If
        set, `cell_type` must also be specified and `rnn_cell_fn` must be
        `None`.
      cell_type: A subclass of `tf.nn.rnn_cell.RNNCell` or a string specifying
        the cell type. Supported strings are: `'basic_rnn'`, `'lstm'`, and
        `'gru'`. If set, `num_units` must also be specified and `rnn_cell_fn`
        must be `None`.
      rnn_cell_fn: A function with one argument, a `tf.estimator.ModeKeys`, and
        returns an object of type `tf.nn.rnn_cell.RNNCell` that will be used to
        construct the RNN. If set, `num_units` and `cell_type` cannot be set.
        This is for advanced users who need additional customization beyond
        `num_units` and `cell_type`. Note that `tf.nn.rnn_cell.MultiRNNCell` is
        needed for stacked RNNs.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      n_classes: Number of label classes. Defaults to 2, namely binary
        classification. Must be > 1.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      label_vocabulary: A list of strings represents possible label values. If
        given, labels must be string type and have any value in
        `label_vocabulary`. If it is not given, that means labels are
        already encoded as integer or float within [0, 1] for `n_classes=2` and
        encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
        Also there will be errors if vocabulary is not provided and labels are
        string.
      optimizer: An instance of `tf.Optimizer` or string specifying optimizer
        type. Defaults to Adagrad optimizer.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.
      input_layer_partitioner: Optional. Partitioner for input layer. Defaults
        to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
      config: `RunConfig` object to configure the runtime settings.

    Raises:
      ValueError: If `num_units`, `cell_type`, and `rnn_cell_fn` are not
        compatible.
    """
    rnn_cell_fn = _assert_rnn_cell_fn(rnn_cell_fn, num_units, cell_type)

    if n_classes == 2:
      head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction)
    else:
      head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
          n_classes,
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction)

    def _model_fn(features, labels, mode, config):
      return _rnn_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          rnn_cell_fn=rnn_cell_fn,
          sequence_feature_columns=tuple(sequence_feature_columns or []),
          context_feature_columns=tuple(context_feature_columns or []),
          optimizer=optimizer,
          input_layer_partitioner=input_layer_partitioner,
          config=config)
    super(RNNClassifier, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)


class RNNEstimator(estimator.Estimator):
  """An Estimator for TensorFlow RNN models with user-specified head.

  Example:

  ```python
  token_sequence = sequence_categorical_column_with_hash_bucket(...)
  token_emb = embedding_column(categorical_column=token_sequence, ...)

  estimator = RNNEstimator(
      head=tf.contrib.estimator.regression_head(),
      sequence_feature_columns=[token_emb],
      num_units=[32, 16], cell_type='lstm')

  # Or with custom RNN cell:
  def rnn_cell_fn(mode):
    cells = [ tf.contrib.rnn.LSTMCell(size) for size in [32, 16] ]
    if mode == tf.estimator.ModeKeys.TRAIN:
      cells = [ tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.5)
                    for cell in cells ]
    return tf.contrib.rnn.MultiRNNCell(cells)

  estimator = RNNEstimator(
      head=tf.contrib.estimator.regression_head(),
      sequence_feature_columns=[token_emb],
      rnn_cell_fn=rnn_cell_fn)

  # Input builders
  def input_fn_train: # returns x, y
    pass
  estimator.train(input_fn=input_fn_train, steps=100)

  def input_fn_eval: # returns x, y
    pass
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  def input_fn_predict: # returns x, None
    pass
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * if the head's `weight_column` is not `None`, a feature with
    `key=weight_column` whose value is a `Tensor`.
  * for each `column` in `sequence_feature_columns`:
    - a feature with `key=column.name` whose `value` is a `SparseTensor`.
  * for each `column` in `context_feature_columns`:
    - if `column` is a `_CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `_WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `_DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss and predicted output are determined by the specified head.

  @compatibility(eager)
  Estimators are not compatible with eager execution.
  @end_compatibility
  """

  def __init__(self,
               head,
               sequence_feature_columns,
               context_feature_columns=None,
               num_units=None,
               cell_type=USE_DEFAULT,
               rnn_cell_fn=None,
               model_dir=None,
               optimizer='Adagrad',
               input_layer_partitioner=None,
               config=None):
    """Initializes a `RNNClassifier` instance.

    Args:
      head: A `_Head` instance constructed with a method such as
        `tf.contrib.estimator.multi_label_head`. This specifies the model's
        output and loss function to be optimized.
      sequence_feature_columns: An iterable containing the `FeatureColumn`s
        that represent sequential input. All items in the set should either be
        sequence columns (e.g. `sequence_numeric_column`) or constructed from
        one (e.g. `embedding_column` with `sequence_categorical_column_*` as
        input).
      context_feature_columns: An iterable containing the `FeatureColumn`s
        for contextual input. The data represented by these columns will be
        replicated and given to the RNN at each timestep. These columns must be
        instances of classes derived from `_DenseColumn` such as
        `numeric_column`, not the sequential variants.
      num_units: Iterable of integer number of hidden units per RNN layer. If
        set, `cell_type` must also be specified and `rnn_cell_fn` must be
        `None`.
      cell_type: A subclass of `tf.nn.rnn_cell.RNNCell` or a string specifying
        the cell type. Supported strings are: `'basic_rnn'`, `'lstm'`, and
        `'gru'`. If set, `num_units` must also be specified and `rnn_cell_fn`
        must be `None`.
      rnn_cell_fn: A function with one argument, a `tf.estimator.ModeKeys`, and
        returns an object of type `tf.nn.rnn_cell.RNNCell` that will be used to
        construct the RNN. If set, `num_units` and `cell_type` cannot be set.
        This is for advanced users who need additional customization beyond
        `num_units` and `cell_type`. Note that `tf.nn.rnn_cell.MultiRNNCell` is
        needed for stacked RNNs.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      optimizer: An instance of `tf.Optimizer` or string specifying optimizer
        type. Defaults to Adagrad optimizer.
      input_layer_partitioner: Optional. Partitioner for input layer. Defaults
        to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
      config: `RunConfig` object to configure the runtime settings.

    Raises:
      ValueError: If `num_units`, `cell_type`, and `rnn_cell_fn` are not
        compatible.
    """
    rnn_cell_fn = _assert_rnn_cell_fn(rnn_cell_fn, num_units, cell_type)

    def _model_fn(features, labels, mode, config):
      return _rnn_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          rnn_cell_fn=rnn_cell_fn,
          sequence_feature_columns=tuple(sequence_feature_columns or []),
          context_feature_columns=tuple(context_feature_columns or []),
          optimizer=optimizer,
          input_layer_partitioner=input_layer_partitioner,
          config=config)
    super(RNNEstimator, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)
