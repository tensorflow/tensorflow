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
"""Abstractions for the head(s) of a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export_output
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.summary import summary

_DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


def multi_class_head(n_classes,
                     weight_column=None,
                     label_vocabulary=None,
                     loss_reduction=losses.Reduction.SUM,
                     loss_fn=None,
                     name=None):
  """Creates a `_Head` for multi class classification.

  Uses `sparse_softmax_cross_entropy` loss.

  The head expects `logits` with shape `[D0, D1, ... DN, n_classes]`.
  In many applications, the shape is `[batch_size, n_classes]`.

  `labels` must be a dense `Tensor` with shape matching `logits`, namely
  `[D0, D1, ... DN, 1]`. If `label_vocabulary` given, `labels` must be a string
  `Tensor` with values from the vocabulary. If `label_vocabulary` is not given,
  `labels` must be an integer `Tensor` with values specifying the class index.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, or `[D0, D1, ... DN, 1]`.

  The loss is the weighted sum over the input dimensions. Namely, if the input
  labels have shape `[batch_size, 1]`, the loss is the weighted sum over
  `batch_size`.

  Also supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or
  `(labels, logits, features)` as arguments and returns unreduced loss with
  shape `[D0, D1, ... DN, 1]`. `loss_fn` must support integer `labels` with
  shape `[D0, D1, ... DN, 1]`. Namely, the head applies `label_vocabulary` to
  the input labels before passing them to `loss_fn`.

  Args:
    n_classes: Number of classes, must be greater than 2 (for 2 classes, use
      `binary_classification_head`).
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_vocabulary: A list or tuple of strings representing possible label
      values. If it is not given, that means labels are already encoded as an
      integer within [0, n_classes). If given, labels must be of string type and
      have any value in `label_vocabulary`. Note that errors will be raised if
      `label_vocabulary` is not provided but labels are strings.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.
    loss_fn: Optional loss function.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `_Head` for multi class classification.

  Raises:
    ValueError: if `n_classes`, `label_vocabulary` or `loss_reduction` is
      invalid.
  """
  return head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint:disable=protected-access
      n_classes=n_classes,
      weight_column=weight_column,
      label_vocabulary=label_vocabulary,
      loss_reduction=loss_reduction,
      loss_fn=loss_fn,
      name=name)


def binary_classification_head(
    weight_column=None,
    thresholds=None,
    label_vocabulary=None,
    loss_reduction=losses.Reduction.SUM,
    loss_fn=None,
    name=None):
  """Creates a `_Head` for single label binary classification.

  This head uses `sigmoid_cross_entropy_with_logits` loss.

  The head expects `logits` with shape `[D0, D1, ... DN, 1]`.
  In many applications, the shape is `[batch_size, 1]`.

  `labels` must be a dense `Tensor` with shape matching `logits`, namely
  `[D0, D1, ... DN, 1]`. If `label_vocabulary` given, `labels` must be a string
  `Tensor` with values from the vocabulary. If `label_vocabulary` is not given,
  `labels` must be float `Tensor` with values in the interval `[0, 1]`.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, or `[D0, D1, ... DN, 1]`.

  The loss is the weighted sum over the input dimensions. Namely, if the input
  labels have shape `[batch_size, 1]`, the loss is the weighted sum over
  `batch_size`.

  Also supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or
  `(labels, logits, features)` as arguments and returns unreduced loss with
  shape `[D0, D1, ... DN, 1]`. `loss_fn` must support float `labels` with
  shape `[D0, D1, ... DN, 1]`. Namely, the head applies `label_vocabulary` to
  the input labels before passing them to `loss_fn`.

  Args:
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    thresholds: Iterable of floats in the range `(0, 1)`. For binary
      classification metrics such as precision and recall, an eval metric is
      generated for each threshold value. This threshold is applied to the
      logistic values to determine the binary classification (i.e., above the
      threshold is `true`, below is `false`.
    label_vocabulary: A list or tuple of strings representing possible label
      values. If it is not given, labels must be float with values within
      [0, 1]. If given, labels must be string type and have any value in
      `label_vocabulary`. Note that errors will be raised if `label_vocabulary`
      is not provided but labels are strings.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.
    loss_fn: Optional loss function.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `_Head` for binary classification.

  Raises:
    ValueError: If `thresholds` contains a value outside of `(0, 1)`.
    ValueError: If `loss_reduction` is invalid.
  """
  return head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint:disable=protected-access
      weight_column=weight_column,
      thresholds=thresholds,
      label_vocabulary=label_vocabulary,
      loss_reduction=loss_reduction,
      loss_fn=loss_fn,
      name=name)


def regression_head(weight_column=None,
                    label_dimension=1,
                    loss_reduction=losses.Reduction.SUM,
                    loss_fn=None,
                    inverse_link_fn=None,
                    name=None):
  """Creates a `_Head` for regression using the `mean_squared_error` loss.

  The loss is the weighted sum over all input dimensions. Namely, if the input
  labels have shape `[batch_size, label_dimension]`, the loss is the weighted
  sum over both `batch_size` and `label_dimension`.

  The head expects `logits` with shape `[D0, D1, ... DN, label_dimension]`.
  In many applications, the shape is `[batch_size, label_dimension]`.

  The `labels` shape must match `logits`, namely
  `[D0, D1, ... DN, label_dimension]`. If `label_dimension=1`, shape
  `[D0, D1, ... DN]` is also supported.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, `[D0, D1, ... DN, 1]` or
  `[D0, D1, ... DN, label_dimension]`.

  Supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or
  `(labels, logits, features)` as arguments and returns unreduced loss with
  shape `[D0, D1, ... DN, label_dimension]`.

  Also supports custom `inverse_link_fn`, also known as 'mean function'.
  `inverse_link_fn` takes `logits` as argument and returns predicted values.
  This function is the inverse of the link function defined in
  https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function
  Namely, for poisson regression, set `inverse_link_fn=tf.exp`.

  Args:
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_dimension: Number of regression labels per example. This is the size
      of the last dimension of the labels `Tensor` (typically, this has shape
      `[batch_size, label_dimension]`).
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.
    loss_fn: Optional loss function. Defaults to `mean_squared_error`.
    inverse_link_fn: Optional inverse link function, also known as 'mean
      function'. Defaults to identity.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `_Head` for linear regression.

  Raises:
    ValueError: If `label_dimension` or `loss_reduction` is invalid.
  """
  return head_lib._regression_head_with_mean_squared_error_loss(  # pylint:disable=protected-access
      weight_column=weight_column,
      label_dimension=label_dimension,
      loss_reduction=loss_reduction,
      loss_fn=loss_fn,
      inverse_link_fn=inverse_link_fn,
      name=name)


def poisson_regression_head(
    weight_column=None,
    label_dimension=1,
    loss_reduction=losses.Reduction.SUM,
    compute_full_loss=True,
    name=None):
  """Creates a `_Head` for poisson regression using `tf.nn.log_poisson_loss`.

  The loss is the weighted sum over all input dimensions. Namely, if the input
  labels have shape `[batch_size, label_dimension]`, the loss is the weighted
  sum over both `batch_size` and `label_dimension`.

  The head expects `logits` with shape `[D0, D1, ... DN, label_dimension]`.
  In many applications, the shape is `[batch_size, label_dimension]`.

  The `labels` shape must match `logits`, namely
  `[D0, D1, ... DN, label_dimension]`. If `label_dimension=1`, shape
  `[D0, D1, ... DN]` is also supported.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, `[D0, D1, ... DN, 1]` or
  `[D0, D1, ... DN, label_dimension]`.

  This is implemented as a generalized linear model, see
  https://en.wikipedia.org/wiki/Generalized_linear_model.

  Args:
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    label_dimension: Number of regression labels per example. This is the size
      of the last dimension of the labels `Tensor` (typically, this has shape
      `[batch_size, label_dimension]`).
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.
    compute_full_loss: Whether to include the constant `log(z!)` term in
      computing the poisson loss. See `tf.nn.log_poisson_loss` for the full
      documentation.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `_Head` for poisson regression.

  Raises:
    ValueError: If `label_dimension` or `loss_reduction` is invalid.
  """
  def _poisson_loss(labels, logits):
    return nn.log_poisson_loss(
        targets=labels, log_input=logits, compute_full_loss=compute_full_loss)
  return head_lib._regression_head_with_mean_squared_error_loss(  # pylint:disable=protected-access
      weight_column=weight_column,
      label_dimension=label_dimension,
      loss_reduction=loss_reduction,
      loss_fn=_poisson_loss,
      inverse_link_fn=math_ops.exp,
      name=name)


def multi_label_head(n_classes,
                     weight_column=None,
                     thresholds=None,
                     label_vocabulary=None,
                     loss_reduction=losses.Reduction.SUM,
                     loss_fn=None,
                     name=None):
  """Creates a `_Head` for multi-label classification.

  Multi-label classification handles the case where each example may have zero
  or more associated labels, from a discrete set. This is distinct from
  `multi_class_head` which has exactly one label per example.

  Uses `sigmoid_cross_entropy` loss average over classes and weighted sum over
  the batch. Namely, if the input logits have shape `[batch_size, n_classes]`,
  the loss is the average over `n_classes` and the weighted sum over
  `batch_size`.

  The head expects `logits` with shape `[D0, D1, ... DN, n_classes]`. In many
  applications, the shape is `[batch_size, n_classes]`.

  Labels can be:
  * A multi-hot tensor of shape `[D0, D1, ... DN, n_classes]`
  * An integer `SparseTensor` of class indices. The `dense_shape` must be
    `[D0, D1, ... DN, ?]` and the values within `[0, n_classes)`.
  * If `label_vocabulary` is given, a string `SparseTensor`. The `dense_shape`
    must be `[D0, D1, ... DN, ?]` and the values within `label_vocabulary`.

  If `weight_column` is specified, weights must be of shape
  `[D0, D1, ... DN]`, or `[D0, D1, ... DN, 1]`.

  Also supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or
  `(labels, logits, features)` as arguments and returns unreduced loss with
  shape `[D0, D1, ... DN, 1]`. `loss_fn` must support indicator `labels` with
  shape `[D0, D1, ... DN, n_classes]`. Namely, the head applies
  `label_vocabulary` to the input labels before passing them to `loss_fn`.

  Args:
    n_classes: Number of classes, must be greater than 1 (for 1 class, use
      `binary_classification_head`).
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.  Per-class weighting is
      not supported.
    thresholds: Iterable of floats in the range `(0, 1)`. Accuracy, precision
      and recall metrics are evaluated for each threshold value. The threshold
      is applied to the predicted probabilities, i.e. above the threshold is
      `true`, below is `false`.
    label_vocabulary: A list of strings represents possible label values. If it
      is not given, that means labels are already encoded as integer within
      [0, n_classes) or multi-hot Tensor. If given, labels must be SparseTensor
      string type and have any value in `label_vocabulary`. Also there will be
      errors if vocabulary is not provided and labels are string.
    loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
      reduce training loss over batch. Defaults to `SUM`.
    loss_fn: Optional loss function.
    name: name of the head. If provided, summary and metrics keys will be
      suffixed by `"/" + name`. Also used as `name_scope` when creating ops.

  Returns:
    An instance of `_Head` for multi-label classification.

  Raises:
    ValueError: if `n_classes`, `thresholds`, `loss_reduction` or `loss_fn` is
    invalid.
  """
  thresholds = tuple(thresholds) if thresholds else tuple()
  if n_classes is None or n_classes < 2:
    raise ValueError(
        'n_classes must be > 1 for multi-class classification. '
        'Given: {}'.format(n_classes))
  for threshold in thresholds:
    if (threshold <= 0.0) or (threshold >= 1.0):
      raise ValueError(
          'thresholds must be in (0, 1) range. Given: {}'.format(threshold))
  if label_vocabulary is not None:
    if not isinstance(label_vocabulary, (list, tuple)):
      raise ValueError(
          'label_vocabulary must be a list or tuple. '
          'Given type: {}'.format(type(label_vocabulary)))
    if len(label_vocabulary) != n_classes:
      raise ValueError(
          'Length of label_vocabulary must be n_classes ({}). '
          'Given: {}'.format(n_classes, len(label_vocabulary)))
  if loss_fn:
    head_lib._validate_loss_fn_args(loss_fn)  # pylint:disable=protected-access
  if (loss_reduction not in losses.Reduction.all() or
      loss_reduction == losses.Reduction.NONE):
    raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
  return _MultiLabelHead(
      n_classes=n_classes, weight_column=weight_column, thresholds=thresholds,
      label_vocabulary=label_vocabulary, loss_reduction=loss_reduction,
      loss_fn=loss_fn, name=name)


class _MultiLabelHead(head_lib._Head):  # pylint:disable=protected-access
  """`_Head` for multi-label classification."""

  def __init__(self,
               n_classes,
               weight_column=None,
               thresholds=None,
               label_vocabulary=None,
               loss_reduction=losses.Reduction.SUM,
               loss_fn=None,
               name=None):
    self._n_classes = n_classes
    self._weight_column = weight_column
    self._thresholds = thresholds
    self._label_vocabulary = label_vocabulary
    self._loss_reduction = loss_reduction
    self._loss_fn = loss_fn
    self._name = name

  @property
  def name(self):
    return self._name

  @property
  def logits_dimension(self):
    return self._n_classes

  def _process_labels(self, labels):
    if labels is None:
      raise ValueError(
          'You must provide a labels Tensor. Given: None. '
          'Suggested troubleshooting steps: Check that your data contain '
          'your label feature. Check that your input_fn properly parses and '
          'returns labels.')
    if isinstance(labels, sparse_tensor.SparseTensor):
      if labels.dtype == dtypes.string:
        label_ids_values = lookup_ops.index_table_from_tensor(
            vocabulary_list=tuple(self._label_vocabulary),
            name='class_id_lookup').lookup(labels.values)
        label_ids = sparse_tensor.SparseTensor(
            indices=labels.indices,
            values=label_ids_values,
            dense_shape=labels.dense_shape)
        return math_ops.to_int64(
            sparse_ops.sparse_to_indicator(label_ids, self._n_classes))
      else:
        err_msg = (
            r'labels must be an integer SparseTensor with values in '
            r'[0, {})'.format(self._n_classes))
        assert_int = check_ops.assert_integer(
            labels.values, message=err_msg)
        assert_less = check_ops.assert_less(
            labels.values,
            ops.convert_to_tensor(self._n_classes, dtype=labels.dtype),
            message=err_msg)
        assert_greater = check_ops.assert_non_negative(
            labels.values, message=err_msg)
        with ops.control_dependencies(
            [assert_int, assert_less, assert_greater]):
          return math_ops.to_int64(
              sparse_ops.sparse_to_indicator(labels, self._n_classes))
    err_msg = (
        r'labels must be an integer indicator Tensor with values in [0, 1]')
    return head_lib._assert_range(labels, 2, message=err_msg)  # pylint:disable=protected-access,

  def create_loss(self, features, mode, logits, labels):
    """See `Head`."""
    del mode  # Unused for this head.
    logits = ops.convert_to_tensor(logits)
    processed_labels = self._process_labels(labels)
    processed_labels = head_lib._check_dense_labels_match_logits_and_reshape(  # pylint:disable=protected-access
        labels=processed_labels, logits=logits,
        expected_labels_dimension=self.logits_dimension)
    if self._loss_fn:
      unweighted_loss = head_lib._call_loss_fn(  # pylint:disable=protected-access
          loss_fn=self._loss_fn, labels=processed_labels, logits=logits,
          features=features, expected_loss_dim=1)
    else:
      unweighted_loss = losses.sigmoid_cross_entropy(
          multi_class_labels=processed_labels, logits=logits,
          reduction=losses.Reduction.NONE)
      # Averages loss over classes.
      unweighted_loss = math_ops.reduce_mean(
          unweighted_loss, axis=-1, keep_dims=True)
    weights = head_lib._get_weights_and_check_match_logits(  # pylint:disable=protected-access,
        features=features, weight_column=self._weight_column, logits=logits)
    training_loss = losses.compute_weighted_loss(
        unweighted_loss, weights=weights, reduction=self._loss_reduction)
    return head_lib.LossSpec(
        training_loss=training_loss,
        unreduced_loss=unweighted_loss,
        weights=weights,
        processed_labels=processed_labels)

  def create_estimator_spec(
      self, features, mode, logits, labels=None, train_op_fn=None,
      regularization_losses=None):
    """Returns an `EstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` with shape `[D0, D1, ... DN, n_classes]`.
        For many applications, the shape is `[batch_size, n_classes]`.
      labels: Labels with shape matching `logits`. Can be multi-hot `Tensor`
        with shape `[D0, D1, ... DN, n_classes]` or `SparseTensor` with
        `dense_shape` `[D0, D1, ... DN, ?]`. `labels` is required argument when
        `mode` equals `TRAIN` or `EVAL`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Required in TRAIN mode.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` or
        `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when creating the head to
        avoid scaling errors.
    Returns:
      `EstimatorSpec`.
    Raises:
      ValueError: If `train_op_fn` is `None` in TRAIN mode.
    """
    with ops.name_scope(self._name, 'head'):
      logits = head_lib._check_logits_final_dim(logits, self.logits_dimension)  # pylint:disable=protected-access

      # Predict.
      pred_keys = prediction_keys.PredictionKeys
      with ops.name_scope(None, 'predictions', (logits,)):
        probabilities = math_ops.sigmoid(logits, name=pred_keys.PROBABILITIES)
        predictions = {
            pred_keys.LOGITS: logits,
            pred_keys.PROBABILITIES: probabilities,
        }
      if mode == model_fn.ModeKeys.PREDICT:
        classifier_output = head_lib._classification_output(  # pylint:disable=protected-access
            scores=probabilities, n_classes=self._n_classes,
            label_vocabulary=self._label_vocabulary)
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                _DEFAULT_SERVING_KEY: classifier_output,
                head_lib._CLASSIFY_SERVING_KEY: classifier_output,  # pylint:disable=protected-access
                head_lib._PREDICT_SERVING_KEY: (  # pylint:disable=protected-access
                    export_output.PredictOutput(predictions))
            })

      (training_loss, unreduced_loss, weights,
       processed_labels) = self.create_loss(
           features=features, mode=mode, logits=logits, labels=labels)
      if regularization_losses:
        regularization_loss = math_ops.add_n(regularization_losses)
        regularized_training_loss = math_ops.add_n(
            [training_loss, regularization_loss])
      else:
        regularization_loss = None
        regularized_training_loss = training_loss

      # Eval.
      if mode == model_fn.ModeKeys.EVAL:
        return model_fn.EstimatorSpec(
            mode=model_fn.ModeKeys.EVAL,
            predictions=predictions,
            loss=regularized_training_loss,
            eval_metric_ops=self._eval_metric_ops(
                labels=processed_labels,
                probabilities=probabilities,
                weights=weights,
                unreduced_loss=unreduced_loss,
                regularization_loss=regularization_loss))

      # Train.
      if train_op_fn is None:
        raise ValueError('train_op_fn can not be None.')
      # Only summarize mean_loss for SUM reduction to preserve backwards
      # compatibility. Otherwise skip it to avoid unnecessary computation.
      if self._loss_reduction == losses.Reduction.SUM:
        example_weight_sum = math_ops.reduce_sum(
            weights * array_ops.ones_like(unreduced_loss))
        mean_loss = training_loss / example_weight_sum
      else:
        mean_loss = None
    with ops.name_scope(''):
      keys = metric_keys.MetricKeys
      summary.scalar(
          head_lib._summary_key(self._name, keys.LOSS),  # pylint:disable=protected-access
          regularized_training_loss)
      if mean_loss is not None:
        summary.scalar(
            head_lib._summary_key(self._name, keys.LOSS_MEAN),  # pylint:disable=protected-access
            mean_loss)
      if regularization_loss is not None:
        summary.scalar(
            head_lib._summary_key(self._name, keys.LOSS_REGULARIZATION),  # pylint:disable=protected-access
            regularization_loss)
    return model_fn.EstimatorSpec(
        mode=model_fn.ModeKeys.TRAIN,
        predictions=predictions,
        loss=regularized_training_loss,
        train_op=train_op_fn(regularized_training_loss))

  def _eval_metric_ops(
      self, labels, probabilities, weights, unreduced_loss,
      regularization_loss):
    """Returns a dict of metrics for eval_metric_ops."""
    with ops.name_scope(
        None, 'metrics',
        [labels, probabilities, weights, unreduced_loss, regularization_loss]):
      keys = metric_keys.MetricKeys
      metric_ops = {
          # Estimator already adds a metric for loss.
          head_lib._summary_key(self._name, keys.LOSS_MEAN):  # pylint:disable=protected-access
              metrics_lib.mean(
                  values=unreduced_loss,
                  weights=weights,
                  name=keys.LOSS_MEAN),
          head_lib._summary_key(self._name, keys.AUC):  # pylint:disable=protected-access
              metrics_lib.auc(labels=labels, predictions=probabilities,
                              weights=weights, name=keys.AUC),
          head_lib._summary_key(self._name, keys.AUC_PR):  # pylint:disable=protected-access
              metrics_lib.auc(labels=labels, predictions=probabilities,
                              weights=weights, curve='PR',
                              name=keys.AUC_PR),
      }
      if regularization_loss is not None:
        loss_regularization_key = head_lib._summary_key(  # pylint:disable=protected-access
            self._name, keys.LOSS_REGULARIZATION)
        metric_ops[loss_regularization_key] = (
            metrics_lib.mean(
                values=regularization_loss,
                name=keys.LOSS_REGULARIZATION))
      for threshold in self._thresholds:
        accuracy_key = keys.ACCURACY_AT_THRESHOLD % threshold
        metric_ops[head_lib._summary_key(self._name, accuracy_key)] = (  # pylint:disable=protected-access
            head_lib._accuracy_at_threshold(  # pylint:disable=protected-access
                labels=labels,
                predictions=probabilities,
                weights=weights,
                threshold=threshold,
                name=accuracy_key))
        # Precision for positive examples.
        precision_key = keys.PRECISION_AT_THRESHOLD % threshold
        metric_ops[head_lib._summary_key(self._name, precision_key)] = (  # pylint:disable=protected-access
            head_lib._precision_at_threshold(  # pylint:disable=protected-access
                labels=labels,
                predictions=probabilities,
                weights=weights,
                threshold=threshold,
                name=precision_key))
        # Recall for positive examples.
        recall_key = keys.RECALL_AT_THRESHOLD % threshold
        metric_ops[head_lib._summary_key(self._name, recall_key)] = (  # pylint:disable=protected-access
            head_lib._recall_at_threshold(  # pylint:disable=protected-access
                labels=labels,
                predictions=probabilities,
                weights=weights,
                threshold=threshold,
                name=recall_key))
    return metric_ops
