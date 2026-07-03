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

from tensorflow.python.framework import ops
from tensorflow.python.keras import metrics
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


class AUCSingleClassTest(test.TestCase):
  """Tests for ROC-AUC being NaN when only one class is present in y_true."""

  def _make_auc(self, **kwargs):
    return metrics.AUC(num_thresholds=3, **kwargs)

  # --- ROC: single-class inputs should return NaN --------------------------

  def test_roc_auc_all_positive_labels_returns_nan(self):
    """Reproducer from issue #41081: all-positive y_true must return NaN."""
    m = self._make_auc()
    update_op = m.update_state([1, 1, 1], [1.0, 0.5, 0.3])
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(update_op)
    result = self.evaluate(m.result())
    self.assertTrue(
        math.isnan(result),
        msg="Expected NaN for all-positive labels, got {}".format(result))

  def test_roc_auc_all_negative_labels_returns_nan(self):
    m = self._make_auc()
    update_op = m.update_state([0, 0, 0], [1.0, 0.5, 0.3])
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(update_op)
    result = self.evaluate(m.result())
    self.assertTrue(
        math.isnan(result),
        msg="Expected NaN for all-negative labels, got {}".format(result))

  def test_roc_auc_sample_weight_zeros_out_negatives_returns_nan(self):
    """Weights that zero out all negative examples make ROC-AUC undefined."""
    m = self._make_auc()
    update_op = m.update_state([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9],
                               sample_weight=[0.0, 0.0, 1.0, 1.0])
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(update_op)
    result = self.evaluate(m.result())
    self.assertTrue(
        math.isnan(result),
        msg="Expected NaN when negatives zeroed by sample_weight, "
            "got {}".format(result))

  def test_roc_auc_sample_weight_zeros_out_positives_returns_nan(self):
    """Weights that zero out all positive examples make ROC-AUC undefined."""
    m = self._make_auc()
    update_op = m.update_state([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9],
                               sample_weight=[1.0, 1.0, 0.0, 0.0])
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(update_op)
    result = self.evaluate(m.result())
    self.assertTrue(
        math.isnan(result),
        msg="Expected NaN when positives zeroed by sample_weight, "
            "got {}".format(result))

  # --- ROC: mixed labels must still return the expected value ---------------

  def test_roc_auc_mixed_labels_returns_correct_value(self):
    m = self._make_auc()
    update_op = m.update_state([0, 0, 1, 1], [0.0, 0.5, 0.3, 0.9])
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(update_op)
    self.assertAlmostEqual(self.evaluate(m.result()), 0.75, places=5)

  def test_roc_auc_perfect_classifier_returns_one(self):
    m = self._make_auc()
    update_op = m.update_state([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0])
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(update_op)
    self.assertAlmostEqual(self.evaluate(m.result()), 1.0, places=5)

  def test_roc_auc_random_classifier_not_nan(self):
    m = self._make_auc()
    update_op = m.update_state([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(update_op)
    self.assertFalse(math.isnan(self.evaluate(m.result())))

  def test_roc_auc_custom_thresholds_not_nan(self):
    m = metrics.AUC(thresholds=[0.5, 0.8])
    update_op = m.update_state([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(update_op)
    self.assertFalse(math.isnan(self.evaluate(m.result())))

  # --- PR-AUC: single-class inputs must NOT be changed by this fix ---------

  def test_pr_auc_all_positive_labels_not_nan(self):
    """PR-AUC fix is out of scope; ensure we did not accidentally break it."""
    m = metrics.AUC(num_thresholds=3, curve="PR")
    update_op = m.update_state([1, 1, 1], [1.0, 0.5, 0.3])
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(update_op)
    result = self.evaluate(m.result())
    self.assertIsInstance(result, (float, np.floating))


ops.enable_eager_execution()
if __name__ == "__main__":
  test.main()
