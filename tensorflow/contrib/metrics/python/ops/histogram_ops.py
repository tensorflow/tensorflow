# Copyright 2016 Google Inc. All Rights Reserved.
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
# pylint: disable=g-short-docstring-punctuation
"""## Metrics that use histograms.

@@auc_using_histogram
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import histogram_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope


def auc_using_histogram(boolean_labels,
                        scores,
                        score_range,
                        nbins=100,
                        collections=None,
                        check_shape=True,
                        name=None):
  """AUC computed by maintaining histograms.

  Rather than computing AUC directly, this Op maintains Variables containing
  histograms of the scores associated with `True` and `False` labels.  By
  comparing these the AUC is generated, with some discretization error.
  See: "Efficient AUC Learning Curve Calculation" by Bouckaert.

  This AUC Op updates in `O(batch_size + nbins)` time and works well even with
  large class imbalance.  The accuracy is limited by discretization error due
  to finite number of bins.  If scores are concentrated in a fewer bins,
  accuracy is lower.  If this is a concern, we recommend trying different
  numbers of bins and comparing results.

  Args:
    boolean_labels:  1-D boolean `Tensor`.  Entry is `True` if the corresponding
      record is in class.
    scores:  1-D numeric `Tensor`, same shape as boolean_labels.
    score_range:  `Tensor` of shape `[2]`, same dtype as `scores`.  The min/max
      values of score that we expect.  Scores outside range will be clipped.
    nbins:  Integer number of bins to use.  Accuracy strictly increases as the
      number of bins increases.
    collections: List of graph collections keys. Internal histogram Variables
      are added to these collections. Defaults to `[GraphKeys.LOCAL_VARIABLES]`.
    check_shape:  Boolean.  If `True`, do a runtime shape check on the scores
      and labels.
    name:  A name for this Op.  Defaults to "auc_using_histogram".

  Returns:
    auc:  `float32` scalar `Tensor`.  Fetching this converts internal histograms
      to auc value.
    update_op:  `Op`, when run, updates internal histograms.
  """
  if collections is None:
    collections = [ops.GraphKeys.LOCAL_VARIABLES]
  with variable_scope.variable_op_scope(
      [boolean_labels, scores, score_range], name, 'auc_using_histogram'):
    score_range = ops.convert_to_tensor(score_range, name='score_range')
    boolean_labels, scores = _check_labels_and_scores(
        boolean_labels, scores, check_shape)
    hist_true, hist_false = _make_auc_histograms(boolean_labels, scores,
                                                 score_range, nbins)
    hist_true_acc, hist_false_acc, update_op = _auc_hist_accumulate(hist_true,
                                                                    hist_false,
                                                                    nbins,
                                                                    collections)
    auc = _auc_convert_hist_to_auc(hist_true_acc, hist_false_acc, nbins)
    return auc, update_op


def _check_labels_and_scores(boolean_labels, scores, check_shape):
  """Check the rank of labels/scores, return tensor versions."""
  with ops.op_scope([boolean_labels, scores], '_check_labels_and_scores'):
    boolean_labels = ops.convert_to_tensor(boolean_labels,
                                           name='boolean_labels')
    scores = ops.convert_to_tensor(scores, name='scores')

    if boolean_labels.dtype != dtypes.bool:
      raise ValueError(
          'Argument boolean_labels should have dtype bool.  Found: %s',
          boolean_labels.dtype)

    if check_shape:
      labels_rank_1 = logging_ops.Assert(
          math_ops.equal(1, array_ops.rank(boolean_labels)),
          ['Argument boolean_labels should have rank 1.  Found: ',
           boolean_labels.name, array_ops.shape(boolean_labels)])

      scores_rank_1 = logging_ops.Assert(
          math_ops.equal(1, array_ops.rank(scores)),
          ['Argument scores should have rank 1.  Found: ', scores.name,
           array_ops.shape(scores)])

      with ops.control_dependencies([labels_rank_1, scores_rank_1]):
        return boolean_labels, scores
    else:
      return boolean_labels, scores


def _make_auc_histograms(boolean_labels, scores, score_range, nbins):
  """Create histogram tensors from one batch of labels/scores."""

  with variable_scope.variable_op_scope(
      [boolean_labels, scores, nbins], None, 'make_auc_histograms'):
    # Histogram of scores for records in this batch with True label.
    hist_true = histogram_ops.histogram_fixed_width(
        array_ops.boolean_mask(scores, boolean_labels),
        score_range,
        nbins=nbins,
        dtype=dtypes.int64,
        name='hist_true')
    # Histogram of scores for records in this batch with False label.
    hist_false = histogram_ops.histogram_fixed_width(
        array_ops.boolean_mask(scores, math_ops.logical_not(boolean_labels)),
        score_range,
        nbins=nbins,
        dtype=dtypes.int64,
        name='hist_false')
    return hist_true, hist_false


def _auc_hist_accumulate(hist_true, hist_false, nbins, collections):
  """Accumulate histograms in new variables."""
  with variable_scope.variable_op_scope(
      [hist_true, hist_false], None, 'hist_accumulate'):
    # Holds running total histogram of scores for records labeled True.
    hist_true_acc = variable_scope.get_variable(
        'hist_true_acc',
        initializer=array_ops.zeros_initializer(
            [nbins],
            dtype=hist_true.dtype),
        collections=collections,
        trainable=False)
    # Holds running total histogram of scores for records labeled False.
    hist_false_acc = variable_scope.get_variable(
        'hist_false_acc',
        initializer=array_ops.zeros_initializer(
            [nbins],
            dtype=hist_false.dtype),
        collections=collections,
        trainable=False)

    update_op = control_flow_ops.group(
        hist_true_acc.assign_add(hist_true),
        hist_false_acc.assign_add(hist_false),
        name='update_op')

    return hist_true_acc, hist_false_acc, update_op


def _auc_convert_hist_to_auc(hist_true_acc, hist_false_acc, nbins):
  """Convert histograms to auc.

  Args:
    hist_true_acc:  `Tensor` holding accumulated histogram of scores for records
      that were `True`.
    hist_false_acc:  `Tensor` holding accumulated histogram of scores for
      records that were `False`.
    nbins:  Integer number of bins in the histograms.

  Returns:
    Scalar `Tensor` estimating AUC.
  """
  # Note that this follows the "Approximating AUC" section in:
  # Efficient AUC learning curve calculation, R. R. Bouckaert,
  # AI'06 Proceedings of the 19th Australian joint conference on Artificial
  # Intelligence: advances in Artificial Intelligence
  # Pages 181-191.
  # Note that the above paper has an error, and we need to re-order our bins to
  # go from high to low score.

  # Normalize histogram so we get fraction in each bin.
  normed_hist_true = math_ops.truediv(hist_true_acc,
                                      math_ops.reduce_sum(hist_true_acc))
  normed_hist_false = math_ops.truediv(hist_false_acc,
                                       math_ops.reduce_sum(hist_false_acc))

  # These become delta x, delta y from the paper.
  delta_y_t = array_ops.reverse(normed_hist_true, [True], name='delta_y_t')
  delta_x_t = array_ops.reverse(normed_hist_false, [True], name='delta_x_t')

  # strict_1d_cumsum requires float32 args.
  delta_y_t = math_ops.cast(delta_y_t, dtypes.float32)
  delta_x_t = math_ops.cast(delta_x_t, dtypes.float32)

  # Trapezoidal integration, \int_0^1 0.5 * (y_t + y_{t-1}) dx_t
  y_t = _strict_1d_cumsum(delta_y_t, nbins)
  first_trap = delta_x_t[0] * y_t[0] / 2.0
  other_traps = delta_x_t[1:] * (y_t[1:] + y_t[:nbins - 1]) / 2.0
  return math_ops.add(first_trap, math_ops.reduce_sum(other_traps), name='auc')


# TODO(langmore) Remove once a faster cumsum (accumulate_sum) Op is available.
# Also see if cast to float32 above can be removed with new cumsum.
# See:  https://github.com/tensorflow/tensorflow/issues/813
def _strict_1d_cumsum(tensor, len_tensor):
  """Cumsum of a 1D tensor with defined shape by padding and convolving."""
  # Assumes tensor shape is fully defined.
  with ops.op_scope([tensor], 'strict_1d_cumsum'):
    if len_tensor == 0:
      return constant_op.constant([])
    len_pad = len_tensor - 1
    x = array_ops.pad(tensor, [[len_pad, 0]])
    h = array_ops.ones_like(x)
    return _strict_conv1d(x, h)[:len_tensor]


# TODO(langmore) Remove once a faster cumsum (accumulate_sum) Op is available.
# See:  https://github.com/tensorflow/tensorflow/issues/813
def _strict_conv1d(x, h):
  """Return x * h for rank 1 tensors x and h."""
  with ops.op_scope([x, h], 'strict_conv1d'):
    x = array_ops.reshape(x, (1, -1, 1, 1))
    h = array_ops.reshape(h, (-1, 1, 1, 1))
    result = nn_ops.conv2d(x, h, [1, 1, 1, 1], 'SAME')
    return array_ops.reshape(result, [-1])
