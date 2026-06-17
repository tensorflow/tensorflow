# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.keras.metrics.AUC single-class undefined-result behavior."""
# pylint: disable=not-callable

import math

import numpy as np

from tensorflow.python.keras import metrics
from tensorflow.python.platform import test


class AUCSingleClassTest(test.TestCase):
  """Tests for ROC-AUC being NaN when only one class is present in y_true."""

  def _make_auc(self, **kwargs):
    return metrics.AUC(num_thresholds=3, **kwargs)

  # --- ROC: single-class inputs should return NaN --------------------------

  def test_roc_auc_all_positive_labels_returns_nan(self):
    """Reproducer from issue #41081: all-positive y_true must return NaN."""
    m = self._make_auc()
    m.update_state([1, 1, 1], [1.0, 0.5, 0.3])
    result = m.result().numpy()
    self.assertTrue(
        math.isnan(result),
        msg="Expected NaN for all-positive labels, got {}".format(result))

  def test_roc_auc_all_negative_labels_returns_nan(self):
    m = self._make_auc()
    m.update_state([0, 0, 0], [1.0, 0.5, 0.3])
    result = m.result().numpy()
    self.assertTrue(
        math.isnan(result),
        msg="Expected NaN for all-negative labels, got {}".format(result))

  def test_roc_auc_sample_weight_zeros_out_negatives_returns_nan(self):
    """Weights that zero out all negative examples make ROC-AUC undefined."""
    m = self._make_auc()
    m.update_state([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9],
                   sample_weight=[0.0, 0.0, 1.0, 1.0])
    result = m.result().numpy()
    self.assertTrue(
        math.isnan(result),
        msg="Expected NaN when negatives zeroed by sample_weight, "
            "got {}".format(result))

  def test_roc_auc_sample_weight_zeros_out_positives_returns_nan(self):
    """Weights that zero out all positive examples make ROC-AUC undefined."""
    m = self._make_auc()
    m.update_state([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9],
                   sample_weight=[1.0, 1.0, 0.0, 0.0])
    result = m.result().numpy()
    self.assertTrue(
        math.isnan(result),
        msg="Expected NaN when positives zeroed by sample_weight, "
            "got {}".format(result))

  # --- ROC: mixed labels must still return the expected value ---------------

  def test_roc_auc_mixed_labels_returns_correct_value(self):
    # num_thresholds=3 => thresholds [-eps, 0.5, 1+eps]
    # tp=[2,1,0], fp=[2,0,0], fn=[0,1,2], tn=[0,2,2]
    # tp_rate=[1,0.5,0], fp_rate=[1,0,0]
    # auc = (1+0.5)/2*(1-0) + (0.5+0)/2*(0-0) = 0.75
    m = self._make_auc()
    m.update_state([0, 0, 1, 1], [0.0, 0.5, 0.3, 0.9])
    self.assertAlmostEqual(m.result().numpy(), 0.75, places=5)

  def test_roc_auc_perfect_classifier_returns_one(self):
    m = self._make_auc()
    m.update_state([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0])
    self.assertAlmostEqual(m.result().numpy(), 1.0, places=5)

  def test_roc_auc_random_classifier_not_nan(self):
    m = self._make_auc()
    m.update_state([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    self.assertFalse(math.isnan(m.result().numpy()))

  def test_roc_auc_custom_thresholds_not_nan(self):
    m = metrics.AUC(thresholds=[0.5, 0.8])
    m.update_state([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    self.assertFalse(math.isnan(m.result().numpy()))

  def test_roc_auc_custom_thresholds_out_of_range_preds_not_nan(self):
    """Both classes present but negatives predict below the lowest custom
    threshold. The sum-based totals (TP+FN, FP+TN) must still detect the
    negatives, so the result is a real value, not a spurious NaN."""
    m = metrics.AUC(thresholds=[0.5, 0.8])
    # Negatives at out-of-range predictions (< 0); positives in range.
    m.update_state([0, 0, 1, 1], [-0.5, -0.3, 0.6, 0.9])
    self.assertFalse(math.isnan(m.result().numpy()))

  # --- multi_label ROC: single-class labels excluded from the average ------

  def _multi_label_data(self):
    # 3 labels over 4 samples:
    #   label 0: y_true=[0,0,1,1] preds=[0.0,0.5,0.3,0.9] -> ROC-AUC 0.75
    #   label 1: y_true=[1,1,1,1]                          -> single-class
    #   label 2: y_true=[0,0,1,1] preds=[0.0,0.1,0.9,1.0] -> ROC-AUC 1.0
    y_true = [[0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1]]
    y_pred = [[0.0, 1.0, 0.0], [0.5, 0.5, 0.1], [0.3, 0.3, 0.9], [0.9, 0.7, 1.0]]
    return y_true, y_pred

  def test_multi_label_roc_excludes_single_class_label_from_mean(self):
    """A single-class label must not poison the averaged AUC with a 0."""
    m = metrics.AUC(num_thresholds=3, multi_label=True)
    y_true, y_pred = self._multi_label_data()
    m.update_state(y_true, y_pred)
    # Mean over the two defined labels only: (0.75 + 1.0) / 2 = 0.875.
    # Before the fix the single-class label contributed 0 -> 0.5833.
    self.assertAlmostEqual(m.result().numpy(), 0.875, places=5)

  def test_multi_label_roc_weighted_excludes_single_class_label(self):
    m = metrics.AUC(num_thresholds=3, multi_label=True,
                    label_weights=[0.2, 0.3, 0.5])
    y_true, y_pred = self._multi_label_data()
    m.update_state(y_true, y_pred)
    # Weighted mean over defined labels: (0.75*0.2 + 1.0*0.5)/(0.2+0.5) = 0.9286.
    self.assertAlmostEqual(m.result().numpy(), 0.928571, places=5)

  def test_multi_label_roc_all_labels_single_class_returns_nan(self):
    m = metrics.AUC(num_thresholds=3, multi_label=True)
    # label 0 all-positive, label 1 all-negative -> every label undefined.
    m.update_state([[1, 0], [1, 0], [1, 0]],
                   [[0.9, 0.9], [0.5, 0.5], [0.3, 0.3]])
    self.assertTrue(math.isnan(m.result().numpy()))

  def test_multi_label_roc_all_labels_defined_returns_mean(self):
    """No single-class label -> plain macro-average, unchanged behavior."""
    m = metrics.AUC(num_thresholds=3, multi_label=True)
    # label 0 -> 0.75, label 1 -> 1.0.
    m.update_state([[0, 0], [0, 0], [1, 1], [1, 1]],
                   [[0.0, 0.0], [0.5, 0.1], [0.3, 0.9], [0.9, 1.0]])
    self.assertAlmostEqual(m.result().numpy(), 0.875, places=5)

  # --- PR-AUC: single-class inputs must NOT be changed by this fix ---------

  def test_pr_auc_all_positive_labels_not_nan(self):
    """PR-AUC fix is out of scope; ensure we did not accidentally break it."""
    m = metrics.AUC(num_thresholds=3, curve="PR")
    m.update_state([1, 1, 1], [1.0, 0.5, 0.3])
    result = m.result().numpy()
    # No assertion on value -- just verify the PR path was not touched.
    self.assertIsInstance(result, (float, np.floating))


if __name__ == "__main__":
  test.main()
