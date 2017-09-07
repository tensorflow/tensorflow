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
"""Common operations for RNN Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import metrics
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


# NOTE(jtbates): As of February 10, 2017, some of the `RNNKeys` have been
# removed and replaced with values from `prediction_key.PredictionKey`. The key
# `RNNKeys.PREDICTIONS_KEY` has been replaced by
# `prediction_key.PredictionKey.SCORES` for regression and
# `prediction_key.PredictionKey.CLASSES` for classification. The key
# `RNNKeys.PROBABILITIES_KEY` has been replaced by
# `prediction_key.PredictionKey.PROBABILITIES`.
class RNNKeys(object):
  FINAL_STATE_KEY = 'final_state'
  LABELS_KEY = '__labels__'
  SEQUENCE_LENGTH_KEY = 'sequence_length'
  STATE_PREFIX = 'rnn_cell_state'


class PredictionType(object):
  """Enum-like values for the type of prediction that the model makes.
  """
  SINGLE_VALUE = 1
  MULTIPLE_VALUE = 2


_CELL_TYPES = {'basic_rnn': contrib_rnn.BasicRNNCell,
               'lstm': contrib_rnn.LSTMCell,
               'gru': contrib_rnn.GRUCell,}


def _get_single_cell(cell_type, num_units):
  """Constructs and return a single `RNNCell`.

  Args:
    cell_type: Either a string identifying the `RNNCell` type or a subclass of
      `RNNCell`.
    num_units: The number of units in the `RNNCell`.
  Returns:
    An initialized `RNNCell`.
  Raises:
    ValueError: `cell_type` is an invalid `RNNCell` name.
    TypeError: `cell_type` is not a string or a subclass of `RNNCell`.
  """
  cell_type = _CELL_TYPES.get(cell_type, cell_type)
  if not cell_type or not issubclass(cell_type, contrib_rnn.RNNCell):
    raise ValueError('The supported cell types are {}; got {}'.format(
        list(_CELL_TYPES.keys()), cell_type))
  return cell_type(num_units=num_units)


def construct_rnn_cell(num_units, cell_type='basic_rnn',
                       dropout_keep_probabilities=None):
  """Constructs cells, applies dropout and assembles a `MultiRNNCell`.

  The cell type chosen by DynamicRNNEstimator.__init__() is the same as
  returned by this function when called with the same arguments.

  Args:
    num_units: A single `int` or a list/tuple of `int`s. The size of the
      `RNNCell`s.
    cell_type: A string identifying the `RNNCell` type or a subclass of
      `RNNCell`.
    dropout_keep_probabilities: a list of dropout probabilities or `None`. If a
      list is given, it must have length `len(cell_type) + 1`.

  Returns:
    An initialized `RNNCell`.
  """
  if not isinstance(num_units, (list, tuple)):
    num_units = (num_units,)

  cells = [_get_single_cell(cell_type, n) for n in num_units]
  if dropout_keep_probabilities:
    cells = apply_dropout(cells, dropout_keep_probabilities)
  if len(cells) == 1:
    return cells[0]
  return contrib_rnn.MultiRNNCell(cells)


def apply_dropout(cells, dropout_keep_probabilities, random_seed=None):
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
        'The number of dropout probabilities must be one greater than the '
        'number of cells. Got {} cells and {} dropout probabilities.'.format(
            len(cells), len(dropout_keep_probabilities)))
  wrapped_cells = [
      contrib_rnn.DropoutWrapper(cell, prob, 1.0, seed=random_seed)
      for cell, prob in zip(cells[:-1], dropout_keep_probabilities[:-2])
  ]
  wrapped_cells.append(
      contrib_rnn.DropoutWrapper(cells[-1], dropout_keep_probabilities[-2],
                                 dropout_keep_probabilities[-1]))
  return wrapped_cells


def get_eval_metric_ops(problem_type, prediction_type, sequence_length,
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
      mask_predictions, mask_labels = mask_activations_and_labels(
          prediction_dict[prediction_key.PredictionKey.CLASSES], labels,
          sequence_length)
      eval_metric_ops['accuracy'] = metrics.streaming_accuracy(
          predictions=mask_predictions, labels=mask_labels)
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


def select_last_activations(activations, sequence_lengths):
  """Selects the nth set of activations for each n in `sequence_length`.

  Reuturns a `Tensor` of shape `[batch_size, k]`. If `sequence_length` is not
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
  with ops.name_scope(
      'mask_activations_and_labels',
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


def multi_value_predictions(activations, target_column, problem_type,
                            predict_probabilities):
  """Maps `activations` from the RNN to predictions for multi value models.

  If `predict_probabilities` is `False`, this function returns a `dict`
  containing single entry with key `prediction_key.PredictionKey.CLASSES` for
  `problem_type` `ProblemType.CLASSIFICATION` or
  `prediction_key.PredictionKey.SCORE` for `problem_type`
  `ProblemType.LINEAR_REGRESSION`.

  If `predict_probabilities` is `True`, it will contain a second entry with key
  `prediction_key.PredictionKey.PROBABILITIES`. The
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
