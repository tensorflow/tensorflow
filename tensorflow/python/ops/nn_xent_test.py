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
"""Tests for cross entropy related functionality in tensorflow.ops.nn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_impl
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test

exp = math.exp
log = math.log


class SigmoidCrossEntropyWithLogitsTest(test.TestCase):

  def _SigmoidCrossEntropyWithLogits(self, logits, targets):
    assert len(logits) == len(targets)
    pred = [1 / (1 + exp(-x)) for x in logits]
    eps = 0.0001
    pred = [min(max(p, eps), 1 - eps) for p in pred]
    return [-z * log(y) - (1 - z) * log(1 - y) for y, z in zip(pred, targets)]

  def _Inputs(self, x=None, y=None, dtype=dtypes.float64, sizes=None):
    x = [-100, -2, -2, 0, 2, 2, 2, 100] if x is None else x
    y = [0, 0, 1, 0, 0, 1, 0.5, 1] if y is None else y
    assert len(x) == len(y)
    sizes = sizes if sizes else [len(x)]
    logits = constant_op.constant(x, shape=sizes, dtype=dtype, name="logits")
    targets = constant_op.constant(y, shape=sizes, dtype=dtype, name="targets")
    losses = np.array(self._SigmoidCrossEntropyWithLogits(x, y)).reshape(*sizes)
    return logits, targets, losses

  def testConstructionNamed(self):
    with self.test_session():
      logits, targets, _ = self._Inputs()
      loss = nn_impl.sigmoid_cross_entropy_with_logits(
          logits, targets, name="mylogistic")
    self.assertEqual("mylogistic", loss.op.name)

  def testLogisticOutput(self):
    for use_gpu in [True, False]:
      for dtype in [dtypes.float32, dtypes.float16]:
        with self.test_session(use_gpu=use_gpu):
          logits, targets, losses = self._Inputs(dtype=dtype)
          loss = nn_impl.sigmoid_cross_entropy_with_logits(logits, targets)
          np_loss = np.array(losses).astype(np.float32)
          tf_loss = loss.eval()
        self.assertAllClose(np_loss, tf_loss, atol=0.001)

  def testLogisticOutputMultiDim(self):
    for use_gpu in [True, False]:
      for dtype in [dtypes.float32, dtypes.float16]:
        with self.test_session(use_gpu=use_gpu):
          logits, targets, losses = self._Inputs(dtype=dtype, sizes=[2, 2, 2])
          loss = nn_impl.sigmoid_cross_entropy_with_logits(logits, targets)
          np_loss = np.array(losses).astype(np.float32)
          tf_loss = loss.eval()
        self.assertAllClose(np_loss, tf_loss, atol=0.001)

  def testGradient(self):
    sizes = [4, 2]
    with self.test_session():
      logits, targets, _ = self._Inputs(sizes=sizes)
      loss = nn_impl.sigmoid_cross_entropy_with_logits(logits, targets)
      err = gradient_checker.compute_gradient_error(logits, sizes, loss, sizes)
    print("logistic loss gradient err = ", err)
    self.assertLess(err, 1e-7)

  def testGradientAtZero(self):
    with self.test_session():
      logits = constant_op.constant([0.0, 0.0], dtype=dtypes.float64)
      targets = constant_op.constant([0.0, 1.0], dtype=dtypes.float64)
      loss = nn_impl.sigmoid_cross_entropy_with_logits(logits, targets)
      grads = gradients_impl.gradients(loss, logits)[0].eval()
    self.assertAllClose(grads, [0.5, -0.5])

  def testShapeError(self):
    with self.assertRaisesRegexp(ValueError, "must have the same shape"):
      nn_impl.sigmoid_cross_entropy_with_logits([[2, 1]], [1, 2, 3])


class WeightedCrossEntropyTest(test.TestCase):

  def _WeightedCrossEntropy(self, logits, targets, pos_coeff):
    assert len(logits) == len(targets)
    pred = [1 / (1 + exp(-x)) for x in logits]
    eps = 0.0001
    pred = [min(max(p, eps), 1 - eps) for p in pred]
    return [
        -z * pos_coeff * log(y) - (1 - z) * log(1 - y)
        for y, z in zip(pred, targets)
    ]

  def _Inputs(self, x=None, y=None, q=3.0, dtype=dtypes.float64, sizes=None):
    x = [-100, -2, -2, 0, 2, 2, 2, 100] if x is None else x
    y = [0, 0, 1, 0, 0, 1, 0.5, 1] if y is None else y
    assert len(x) == len(y)
    sizes = sizes if sizes else [len(x)]
    logits = constant_op.constant(x, shape=sizes, dtype=dtype, name="logits")
    targets = constant_op.constant(y, shape=sizes, dtype=dtype, name="targets")
    losses = np.array(self._WeightedCrossEntropy(x, y, q)).reshape(*sizes)
    return logits, targets, q, losses

  def testConstructionNamed(self):
    with self.test_session():
      logits, targets, pos_weight, _ = self._Inputs()
      loss = nn_impl.weighted_cross_entropy_with_logits(
          targets, logits, pos_weight, name="mybce")
    self.assertEqual("mybce", loss.op.name)

  def testOutput(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        logits, targets, pos_weight, losses = self._Inputs(dtype=dtypes.float32)
        loss = nn_impl.weighted_cross_entropy_with_logits(targets, logits,
                                                          pos_weight)
        np_loss = np.array(losses).astype(np.float32)
        tf_loss = loss.eval()
      self.assertAllClose(np_loss, tf_loss, atol=0.001)

  def testOutputMultiDim(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        logits, targets, pos_weight, losses = self._Inputs(
            dtype=dtypes.float32, sizes=[2, 2, 2])
        loss = nn_impl.weighted_cross_entropy_with_logits(targets, logits,
                                                          pos_weight)
        np_loss = np.array(losses).astype(np.float32)
        tf_loss = loss.eval()
      self.assertAllClose(np_loss, tf_loss, atol=0.001)

  def testGradient(self):
    sizes = [4, 2]
    with self.test_session():
      logits, targets, pos_weight, _ = self._Inputs(sizes=sizes)
      loss = nn_impl.weighted_cross_entropy_with_logits(targets, logits,
                                                        pos_weight)
      err = gradient_checker.compute_gradient_error(logits, sizes, loss, sizes)
    print("logistic loss gradient err = ", err)
    self.assertLess(err, 1e-7)

  def testShapeError(self):
    with self.assertRaisesRegexp(ValueError, "must have the same shape"):
      nn_impl.weighted_cross_entropy_with_logits([1, 2, 3], [[2, 1]], 2.0)


if __name__ == "__main__":
  test.main()
