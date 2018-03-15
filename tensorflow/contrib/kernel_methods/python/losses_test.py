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
"""Tests for third_party.tensorflow.contrib.kernel_methods.python.losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.kernel_methods.python import losses
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SparseMulticlassHingeLossTest(test.TestCase):

  def testInvalidLogitsShape(self):
    """An error is raised when logits have invalid shape."""
    with self.test_session():
      logits = constant_op.constant([-1.0, 2.1], shape=(2,))
      labels = constant_op.constant([0, 1])
      with self.assertRaises(ValueError):
        _ = losses.sparse_multiclass_hinge_loss(labels, logits)

  def testInvalidLabelsShape(self):
    """An error is raised when labels have invalid shape."""
    with self.test_session():
      logits = constant_op.constant([-1.0, 2.1], shape=(2, 1))
      labels = constant_op.constant([1, 0], shape=(1, 1, 2))
      with self.assertRaises(ValueError):
        _ = losses.sparse_multiclass_hinge_loss(labels, logits)

  def testInvalidWeightsShape(self):
    """An error is raised when weights have invalid shape."""
    with self.test_session():
      logits = constant_op.constant([-1.0, 2.1], shape=(2, 1))
      labels = constant_op.constant([1, 0], shape=(2,))
      weights = constant_op.constant([1.5, 0.2], shape=(2, 1, 1))
      with self.assertRaises(ValueError):
        _ = losses.sparse_multiclass_hinge_loss(labels, logits, weights)

  def testInvalidLabelsDtype(self):
    """An error is raised when labels have invalid shape."""
    with self.test_session():
      logits = constant_op.constant([-1.0, 2.1], shape=(2, 1))
      labels = constant_op.constant([1, 0], dtype=dtypes.float32)
      with self.assertRaises(ValueError):
        _ = losses.sparse_multiclass_hinge_loss(labels, logits)

  def testNoneWeightRaisesValueError(self):
    """An error is raised when weights are None."""
    with self.test_session():
      logits = constant_op.constant([-1.0, 2.1], shape=(2, 1))
      labels = constant_op.constant([1, 0])
      with self.assertRaises(ValueError):
        _ = losses.sparse_multiclass_hinge_loss(labels, logits, weights=None)

  def testInconsistentLabelsAndWeightsShapesSameRank(self):
    """Error raised when weights and labels have same ranks, different sizes."""
    with self.test_session():
      logits = constant_op.constant([-1.0, 2.1, 4.1], shape=(3, 1))
      labels = constant_op.constant([1, 0, 2], shape=(3, 1))
      weights = constant_op.constant([1.1, 2.0], shape=(2, 1))
      with self.assertRaises(ValueError):
        _ = losses.sparse_multiclass_hinge_loss(labels, logits, weights)

  def testInconsistentLabelsAndWeightsShapesDifferentRank(self):
    """Error raised when weights and labels have different ranks and sizes."""
    with self.test_session():
      logits = constant_op.constant([-1.0, 2.1], shape=(2, 1))
      labels = constant_op.constant([1, 0], shape=(2, 1))
      weights = constant_op.constant([1.1, 2.0, 2.8], shape=(3,))
      with self.assertRaises(ValueError):
        _ = losses.sparse_multiclass_hinge_loss(labels, logits, weights)

  def testOutOfRangeLabels(self):
    """An error is raised when labels are not in [0, num_classes)."""
    with self.test_session():
      logits = constant_op.constant([[1.2, -1.4, -1.0], [1.4, 1.8, 4.0],
                                     [0.5, 1.8, -1.0]])
      labels = constant_op.constant([1, 0, 4])
      loss = losses.sparse_multiclass_hinge_loss(labels, logits)
      with self.assertRaises(errors.InvalidArgumentError):
        loss.eval()

  def testZeroLossInt32Labels(self):
    """Loss is 0 if true class logits sufficiently higher than other classes."""
    with self.test_session():
      logits = constant_op.constant([[1.2, -1.4, -1.0], [1.4, 1.8, 4.0],
                                     [0.5, 1.8, -1.0]])
      labels = constant_op.constant([0, 2, 1], dtype=dtypes.int32)
      loss = losses.sparse_multiclass_hinge_loss(labels, logits)
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testZeroLossInt64Labels(self):
    """Loss is 0 if true class logits sufficiently higher than other classes."""
    with self.test_session():
      logits = constant_op.constant([[2.1, -0.4, -1.0], [1.4, 2.8, 4.0],
                                     [-0.5, 0.8, -1.0]])
      labels = constant_op.constant([0, 2, 1], dtype=dtypes.int64)
      loss = losses.sparse_multiclass_hinge_loss(labels, logits)
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testUnknownShape(self):
    """Result keeps same with `testZeroLossInt32Labels`"""
    logits_np = np.array([[1.2, -1.4, -1.0], [1.4, 1.8, 4.0], [0.5, 1.8, -1.0]])
    labels_np = np.array([0, 2, 1], dtype=np.int32)

    logits_shapes = [
        [3, 3],  # batch_size, num_classes
        [None, 3],
        [3, None],
        [None, None]
    ]

    for batch_size, num_classes in logits_shapes:
      with self.test_session():
        logits = array_ops.placeholder(
            dtypes.float32, shape=(batch_size, num_classes))
        labels = array_ops.placeholder(dtypes.int32, shape=(batch_size,))
        loss = losses.sparse_multiclass_hinge_loss(labels, logits)
        result = loss.eval(feed_dict={logits: logits_np, labels: labels_np})
        self.assertAlmostEqual(result, 0.0, 3)

  def testCorrectPredictionsSomeClassesInsideMargin(self):
    """Loss is > 0 even if true class logits are higher than other classes."""
    with self.test_session():
      logits = constant_op.constant([[1.2, -1.4, 0.8], [1.4, 1.8, 4.0],
                                     [1.5, 1.8, -1.0]])
      labels = constant_op.constant([0, 2, 1])
      loss = losses.sparse_multiclass_hinge_loss(labels, logits)
      # The first and third samples incur some loss (0.6 and 0.7 respectively).
      self.assertAlmostEqual(loss.eval(), 0.4333, 3)

  def testIncorrectPredictions(self):
    """Loss is >0 when an incorrect class has higher logits than true class."""
    with self.test_session():
      logits = constant_op.constant([[2.6, 0.4, 0.8], [1.4, 0.8, -1.0],
                                     [0.5, -1.8, 2.0]])
      labels = constant_op.constant([1, 0, 2])
      loss = losses.sparse_multiclass_hinge_loss(labels, logits)
      # The first examples incurs a high loss (3.2) since the logits of an
      # incorrect class (0) are higher than the logits of the ground truth. The
      # second example also incures a (smaller) loss (0.4).
      self.assertAlmostEqual(loss.eval(), 1.2, 3)

  def testIncorrectPredictionsColumnLabels(self):
    """Same as above but labels is a rank-2 tensor."""
    with self.test_session():
      logits = constant_op.constant([[1.6, -0.4, 0.8], [1.5, 0.8, -1.0],
                                     [0.2, -1.8, 4.0]])
      labels = constant_op.constant([1, 0, 2], shape=(3, 1))
      loss = losses.sparse_multiclass_hinge_loss(labels, logits)
      # The first examples incurs a high loss (3.0) since the logits of an
      # incorrect class (0) are higher than the logits of the ground truth. The
      # second example also incures a (smaller) loss (0.3).
      self.assertAlmostEqual(loss.eval(), 1.1, 3)

  def testIncorrectPredictionsZeroWeights(self):
    """Loss is 0 when all weights are missing even if predictions are wrong."""
    with self.test_session():
      logits = constant_op.constant([[1.6, -0.4, 0.8], [1.5, 0.8, -1.0],
                                     [0.2, -1.8, 4.0]])
      labels = constant_op.constant([1, 0, 2], shape=(3, 1))
      weights = constant_op.constant([0.0, 0.0, 0.0], shape=(3, 1))
      loss = losses.sparse_multiclass_hinge_loss(labels, logits, weights)
      # No overall loss since all weights are 0.
      self.assertAlmostEqual(loss.eval(), 0.0, 3)

  def testNonZeroLossWithPythonScalarWeights(self):
    """Weighted loss is correctly computed when weights is a python scalar."""
    with self.test_session():
      logits = constant_op.constant([[1.6, -0.4, 0.8], [1.5, 0.8, -1.0],
                                     [0.2, -1.8, 4.0]])
      labels = constant_op.constant([1, 0, 2], shape=(3, 1))
      weights = 10.0
      loss = losses.sparse_multiclass_hinge_loss(labels, logits, weights)
      self.assertAlmostEqual(loss.eval(), 11.0, 3)

  def testNonZeroLossWithScalarTensorWeights(self):
    """Weighted loss is correctly computed when weights is a rank-0 tensor."""
    with self.test_session():
      logits = constant_op.constant([[1.6, -0.4, 0.8], [1.5, 0.8, -1.0],
                                     [0.2, -1.8, 4.0]])
      labels = constant_op.constant([1, 0, 2], shape=(3, 1))
      weights = constant_op.constant(5.0)
      loss = losses.sparse_multiclass_hinge_loss(labels, logits, weights)
      self.assertAlmostEqual(loss.eval(), 5.5, 3)

  def testNonZeroLossWith1DTensorWeightsColumnLabels(self):
    """Weighted loss is correctly computed when weights is a rank-0 tensor."""
    with self.test_session():
      logits = constant_op.constant([[1.6, -0.4, 0.8], [1.5, 0.8, -1.0],
                                     [0.2, -1.8, 4.0]])
      labels = constant_op.constant([1, 0, 2], shape=(3, 1))
      weights = constant_op.constant([1.0, 0.5, 2.0], shape=(3,))
      loss = losses.sparse_multiclass_hinge_loss(labels, logits, weights)
      # The overall loss is 1/3 *(3.0*1.0 + 0.5*0.3+ 2.0*0.0) = 1.05
      self.assertAlmostEqual(loss.eval(), 1.05, 3)

  def testNonZeroLossWith2DTensorWeights1DLabelsSomeWeightsMissing(self):
    """Weighted loss is correctly computed when weights is a rank-0 tensor."""
    with self.test_session():
      logits = constant_op.constant([[1.6, -0.4, 0.8], [1.5, 0.8, -1.0],
                                     [0.2, -1.8, 4.0], [1.6, 1.8, -4.0]])
      labels = constant_op.constant([1, 0, 2, 1])
      weights = constant_op.constant([[1.0], [0.0], [2.0], [4.0]])
      loss = losses.sparse_multiclass_hinge_loss(labels, logits, weights)
      # The overall loss is 1/3 *(3.0*1.0 + 0.0*0.3+ 2.0*0.0 + 4.0*0.8) = 6.2/3.
      self.assertAlmostEqual(loss.eval(), 2.06666, 3)


if __name__ == '__main__':
  test.main()
