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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.distributions.python.ops import deterministic as deterministic_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

rng = np.random.RandomState(0)


class DeterministicTest(test.TestCase):

  def testShape(self):
    with self.test_session():
      loc = rng.rand(2, 3, 4)
      deterministic = deterministic_lib.Deterministic(loc)

      self.assertAllEqual(deterministic.batch_shape_tensor().eval(), (2, 3, 4))
      self.assertAllEqual(deterministic.batch_shape, (2, 3, 4))
      self.assertAllEqual(deterministic.event_shape_tensor().eval(), [])
      self.assertEqual(deterministic.event_shape, tensor_shape.TensorShape([]))

  def testInvalidTolRaises(self):
    loc = rng.rand(2, 3, 4).astype(np.float32)
    deterministic = deterministic_lib.Deterministic(
        loc, atol=-1, validate_args=True)
    with self.test_session():
      with self.assertRaisesOpError("Condition x >= 0"):
        deterministic.prob(0.).eval()

  def testProbWithNoBatchDimsIntegerType(self):
    deterministic = deterministic_lib.Deterministic(0)
    with self.test_session():
      self.assertAllClose(1, deterministic.prob(0).eval())
      self.assertAllClose(0, deterministic.prob(2).eval())
      self.assertAllClose([1, 0], deterministic.prob([0, 2]).eval())

  def testProbWithNoBatchDims(self):
    deterministic = deterministic_lib.Deterministic(0.)
    with self.test_session():
      self.assertAllClose(1., deterministic.prob(0.).eval())
      self.assertAllClose(0., deterministic.prob(2.).eval())
      self.assertAllClose([1., 0.], deterministic.prob([0., 2.]).eval())

  def testProbWithDefaultTol(self):
    loc = [[0., 1.], [2., 3.]]
    x = [[0., 1.1], [1.99, 3.]]
    deterministic = deterministic_lib.Deterministic(loc)
    expected_prob = [[1., 0.], [0., 1.]]
    with self.test_session():
      prob = deterministic.prob(x)
      self.assertAllEqual((2, 2), prob.get_shape())
      self.assertAllEqual(expected_prob, prob.eval())

  def testProbWithNonzeroATol(self):
    loc = [[0., 1.], [2., 3.]]
    x = [[0., 1.1], [1.99, 3.]]
    deterministic = deterministic_lib.Deterministic(loc, atol=0.05)
    expected_prob = [[1., 0.], [1., 1.]]
    with self.test_session():
      prob = deterministic.prob(x)
      self.assertAllEqual((2, 2), prob.get_shape())
      self.assertAllEqual(expected_prob, prob.eval())

  def testProbWithNonzeroATolIntegerType(self):
    loc = [[0, 1], [2, 3]]
    x = [[0, 2], [4, 2]]
    deterministic = deterministic_lib.Deterministic(loc, atol=1)
    expected_prob = [[1, 1], [0, 1]]
    with self.test_session():
      prob = deterministic.prob(x)
      self.assertAllEqual((2, 2), prob.get_shape())
      self.assertAllEqual(expected_prob, prob.eval())

  def testProbWithNonzeroRTol(self):
    loc = [[0., 1.], [100., 100.]]
    x = [[0., 1.1], [100.1, 103.]]
    deterministic = deterministic_lib.Deterministic(loc, rtol=0.01)
    expected_prob = [[1., 0.], [1., 0.]]
    with self.test_session():
      prob = deterministic.prob(x)
      self.assertAllEqual((2, 2), prob.get_shape())
      self.assertAllEqual(expected_prob, prob.eval())

  def testProbWithNonzeroRTolIntegerType(self):
    loc = [[10, 10, 10], [10, 10, 10]]
    x = [[10, 20, 30], [10, 20, 30]]
    # Batch 0 will have rtol = 0
    # Batch 1 will have rtol = 1 (100% slack allowed)
    deterministic = deterministic_lib.Deterministic(loc, rtol=[[0], [1]])
    expected_prob = [[1, 0, 0], [1, 1, 0]]
    with self.test_session():
      prob = deterministic.prob(x)
      self.assertAllEqual((2, 3), prob.get_shape())
      self.assertAllEqual(expected_prob, prob.eval())

  def testCdfWithDefaultTol(self):
    loc = [[0., 0.], [0., 0.]]
    x = [[-1., -0.1], [-0.01, 1.000001]]
    deterministic = deterministic_lib.Deterministic(loc)
    expected_cdf = [[0., 0.], [0., 1.]]
    with self.test_session():
      cdf = deterministic.cdf(x)
      self.assertAllEqual((2, 2), cdf.get_shape())
      self.assertAllEqual(expected_cdf, cdf.eval())

  def testCdfWithNonzeroATol(self):
    loc = [[0., 0.], [0., 0.]]
    x = [[-1., -0.1], [-0.01, 1.000001]]
    deterministic = deterministic_lib.Deterministic(loc, atol=0.05)
    expected_cdf = [[0., 0.], [1., 1.]]
    with self.test_session():
      cdf = deterministic.cdf(x)
      self.assertAllEqual((2, 2), cdf.get_shape())
      self.assertAllEqual(expected_cdf, cdf.eval())

  def testCdfWithNonzeroRTol(self):
    loc = [[1., 1.], [100., 100.]]
    x = [[0.9, 1.], [99.9, 97]]
    deterministic = deterministic_lib.Deterministic(loc, rtol=0.01)
    expected_cdf = [[0., 1.], [1., 0.]]
    with self.test_session():
      cdf = deterministic.cdf(x)
      self.assertAllEqual((2, 2), cdf.get_shape())
      self.assertAllEqual(expected_cdf, cdf.eval())

  def testSampleNoBatchDims(self):
    deterministic = deterministic_lib.Deterministic(0.)
    for sample_shape in [(), (4,)]:
      with self.test_session():
        sample = deterministic.sample(sample_shape)
        self.assertAllEqual(sample_shape, sample.get_shape())
        self.assertAllClose(
            np.zeros(sample_shape).astype(np.float32), sample.eval())

  def testSampleWithBatchDims(self):
    deterministic = deterministic_lib.Deterministic([0., 0.])
    for sample_shape in [(), (4,)]:
      with self.test_session():
        sample = deterministic.sample(sample_shape)
        self.assertAllEqual(sample_shape + (2,), sample.get_shape())
        self.assertAllClose(
            np.zeros(sample_shape + (2,)).astype(np.float32), sample.eval())

  def testSampleDynamicWithBatchDims(self):
    loc = array_ops.placeholder(np.float32)
    sample_shape = array_ops.placeholder(np.int32)

    deterministic = deterministic_lib.Deterministic(loc)
    for sample_shape_ in [(), (4,)]:
      with self.test_session():
        sample_ = deterministic.sample(sample_shape).eval(
            feed_dict={loc: [0., 0.],
                       sample_shape: sample_shape_})
        self.assertAllClose(
            np.zeros(sample_shape_ + (2,)).astype(np.float32), sample_)


class VectorDeterministicTest(test.TestCase):

  def testShape(self):
    with self.test_session():
      loc = rng.rand(2, 3, 4)
      deterministic = deterministic_lib.VectorDeterministic(loc)

      self.assertAllEqual(deterministic.batch_shape_tensor().eval(), (2, 3))
      self.assertAllEqual(deterministic.batch_shape, (2, 3))
      self.assertAllEqual(deterministic.event_shape_tensor().eval(), [4])
      self.assertEqual(deterministic.event_shape, tensor_shape.TensorShape([4]))

  def testInvalidTolRaises(self):
    loc = rng.rand(2, 3, 4).astype(np.float32)
    deterministic = deterministic_lib.VectorDeterministic(
        loc, atol=-1, validate_args=True)
    with self.test_session():
      with self.assertRaisesOpError("Condition x >= 0"):
        deterministic.prob(loc).eval()

  def testInvalidXRaises(self):
    loc = rng.rand(2, 3, 4).astype(np.float32)
    deterministic = deterministic_lib.VectorDeterministic(
        loc, atol=-1, validate_args=True)
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "must have rank at least 1"):
        deterministic.prob(0.).eval()

  def testProbVectorDeterministicWithNoBatchDims(self):
    # 0 batch of deterministics on R^1.
    deterministic = deterministic_lib.VectorDeterministic([0.])
    with self.test_session():
      self.assertAllClose(1., deterministic.prob([0.]).eval())
      self.assertAllClose(0., deterministic.prob([2.]).eval())
      self.assertAllClose([1., 0.], deterministic.prob([[0.], [2.]]).eval())

  def testProbWithDefaultTol(self):
    # 3 batch of deterministics on R^2.
    loc = [[0., 1.], [2., 3.], [4., 5.]]
    x = [[0., 1.], [1.9, 3.], [3.99, 5.]]
    deterministic = deterministic_lib.VectorDeterministic(loc)
    expected_prob = [1., 0., 0.]
    with self.test_session():
      prob = deterministic.prob(x)
      self.assertAllEqual((3,), prob.get_shape())
      self.assertAllEqual(expected_prob, prob.eval())

  def testProbWithNonzeroATol(self):
    # 3 batch of deterministics on R^2.
    loc = [[0., 1.], [2., 3.], [4., 5.]]
    x = [[0., 1.], [1.9, 3.], [3.99, 5.]]
    deterministic = deterministic_lib.VectorDeterministic(loc, atol=0.05)
    expected_prob = [1., 0., 1.]
    with self.test_session():
      prob = deterministic.prob(x)
      self.assertAllEqual((3,), prob.get_shape())
      self.assertAllEqual(expected_prob, prob.eval())

  def testProbWithNonzeroRTol(self):
    # 3 batch of deterministics on R^2.
    loc = [[0., 1.], [1., 1.], [100., 100.]]
    x = [[0., 1.], [0.9, 1.], [99.9, 100.1]]
    deterministic = deterministic_lib.VectorDeterministic(loc, rtol=0.01)
    expected_prob = [1., 0., 1.]
    with self.test_session():
      prob = deterministic.prob(x)
      self.assertAllEqual((3,), prob.get_shape())
      self.assertAllEqual(expected_prob, prob.eval())

  def testProbVectorDeterministicWithNoBatchDimsOnRZero(self):
    # 0 batch of deterministics on R^0.
    deterministic = deterministic_lib.VectorDeterministic(
        [], validate_args=True)
    with self.test_session():
      self.assertAllClose(1., deterministic.prob([]).eval())

  def testProbVectorDeterministicWithNoBatchDimsOnRZeroRaisesIfXNotInSameRk(
      self):
    # 0 batch of deterministics on R^0.
    deterministic = deterministic_lib.VectorDeterministic(
        [], validate_args=True)
    with self.test_session():
      with self.assertRaisesOpError("not defined in the same space"):
        deterministic.prob([1.]).eval()

  def testSampleNoBatchDims(self):
    deterministic = deterministic_lib.VectorDeterministic([0.])
    for sample_shape in [(), (4,)]:
      with self.test_session():
        sample = deterministic.sample(sample_shape)
        self.assertAllEqual(sample_shape + (1,), sample.get_shape())
        self.assertAllClose(
            np.zeros(sample_shape + (1,)).astype(np.float32), sample.eval())

  def testSampleWithBatchDims(self):
    deterministic = deterministic_lib.VectorDeterministic([[0.], [0.]])
    for sample_shape in [(), (4,)]:
      with self.test_session():
        sample = deterministic.sample(sample_shape)
        self.assertAllEqual(sample_shape + (2, 1), sample.get_shape())
        self.assertAllClose(
            np.zeros(sample_shape + (2, 1)).astype(np.float32), sample.eval())

  def testSampleDynamicWithBatchDims(self):
    loc = array_ops.placeholder(np.float32)
    sample_shape = array_ops.placeholder(np.int32)

    deterministic = deterministic_lib.VectorDeterministic(loc)
    for sample_shape_ in [(), (4,)]:
      with self.test_session():
        sample_ = deterministic.sample(sample_shape).eval(
            feed_dict={loc: [[0.], [0.]],
                       sample_shape: sample_shape_})
        self.assertAllClose(
            np.zeros(sample_shape_ + (2, 1)).astype(np.float32), sample_)


if __name__ == "__main__":
  test.main()
