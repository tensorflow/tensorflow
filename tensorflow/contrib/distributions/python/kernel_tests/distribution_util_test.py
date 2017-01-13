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
"""Tests for utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from scipy import special

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class AssertCloseTest(test.TestCase):

  def testAssertCloseIntegerDtype(self):
    x = [1, 5, 10, 15, 20]
    y = x
    z = [2, 5, 10, 15, 20]
    with self.test_session():
      with ops.control_dependencies([distribution_util.assert_close(x, y)]):
        array_ops.identity(x).eval()

      with ops.control_dependencies([distribution_util.assert_close(y, x)]):
        array_ops.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with ops.control_dependencies([distribution_util.assert_close(x, z)]):
          array_ops.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with ops.control_dependencies([distribution_util.assert_close(y, z)]):
          array_ops.identity(y).eval()

  def testAssertCloseNonIntegerDtype(self):
    x = np.array([1., 5, 10, 15, 20], dtype=np.float32)
    y = x + 1e-8
    z = [2., 5, 10, 15, 20]
    with self.test_session():
      with ops.control_dependencies([distribution_util.assert_close(x, y)]):
        array_ops.identity(x).eval()

      with ops.control_dependencies([distribution_util.assert_close(y, x)]):
        array_ops.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with ops.control_dependencies([distribution_util.assert_close(x, z)]):
          array_ops.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with ops.control_dependencies([distribution_util.assert_close(y, z)]):
          array_ops.identity(y).eval()

  def testAssertCloseEpsilon(self):
    x = [0., 5, 10, 15, 20]
    # x != y
    y = [0.1, 5, 10, 15, 20]
    # x = z
    z = [1e-8, 5, 10, 15, 20]
    with self.test_session():
      with ops.control_dependencies([distribution_util.assert_close(x, z)]):
        array_ops.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with ops.control_dependencies([distribution_util.assert_close(x, y)]):
          array_ops.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with ops.control_dependencies([distribution_util.assert_close(y, z)]):
          array_ops.identity(y).eval()

  def testAssertIntegerForm(self):
    # This should only be detected as an integer.
    x = [1., 5, 10, 15, 20]
    y = [1.1, 5, 10, 15, 20]
    # First component isn't less than float32.eps = 1e-7
    z = [1.0001, 5, 10, 15, 20]
    # This shouldn"t be detected as an integer.
    w = [1e-8, 5, 10, 15, 20]
    with self.test_session():
      with ops.control_dependencies([distribution_util.assert_integer_form(x)]):
        array_ops.identity(x).eval()

      with self.assertRaisesOpError("x has non-integer components"):
        with ops.control_dependencies(
            [distribution_util.assert_integer_form(y)]):
          array_ops.identity(y).eval()

      with self.assertRaisesOpError("x has non-integer components"):
        with ops.control_dependencies(
            [distribution_util.assert_integer_form(z)]):
          array_ops.identity(z).eval()

      with self.assertRaisesOpError("x has non-integer components"):
        with ops.control_dependencies(
            [distribution_util.assert_integer_form(w)]):
          array_ops.identity(w).eval()


class GetLogitsAndProbTest(test.TestCase):

  def testGetLogitsAndProbImproperArguments(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        distribution_util.get_logits_and_prob(logits=None, p=None)

      with self.assertRaises(ValueError):
        distribution_util.get_logits_and_prob(logits=[0.1], p=[0.1])

  def testGetLogitsAndProbLogits(self):
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)
    logits = special.logit(p)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_prob(
          logits=logits, validate_args=True)

      self.assertAllClose(p, new_p.eval())
      self.assertAllClose(logits, new_logits.eval())

  def testGetLogitsAndProbLogitsMultidimensional(self):
    p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    logits = np.log(p)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_prob(
          logits=logits, multidimensional=True, validate_args=True)

      self.assertAllClose(new_p.eval(), p)
      self.assertAllClose(new_logits.eval(), logits)

  def testGetLogitsAndProbProbability(self):
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_prob(
          p=p, validate_args=True)

      self.assertAllClose(special.logit(p), new_logits.eval())
      self.assertAllClose(p, new_p.eval())

  def testGetLogitsAndProbProbabilityMultidimensional(self):
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_prob(
          p=p, multidimensional=True, validate_args=True)

      self.assertAllClose(np.log(p), new_logits.eval())
      self.assertAllClose(p, new_p.eval())

  def testGetLogitsAndProbProbabilityValidateArgs(self):
    p = [0.01, 0.2, 0.5, 0.7, .99]
    # Component less than 0.
    p2 = [-1, 0.2, 0.5, 0.3, .2]
    # Component greater than 1.
    p3 = [2, 0.2, 0.5, 0.3, .2]

    with self.test_session():
      _, prob = distribution_util.get_logits_and_prob(p=p, validate_args=True)
      prob.eval()

      with self.assertRaisesOpError("Condition x >= 0"):
        _, prob = distribution_util.get_logits_and_prob(
            p=p2, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_prob(p=p2, validate_args=False)
      prob.eval()

      with self.assertRaisesOpError("p has components greater than 1"):
        _, prob = distribution_util.get_logits_and_prob(
            p=p3, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_prob(p=p3, validate_args=False)
      prob.eval()

  def testGetLogitsAndProbProbabilityValidateArgsMultidimensional(self):
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component less than 0. Still sums to 1.
    p2 = np.array([[-.3, 0.4, 0.9], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component greater than 1. Does not sum to 1.
    p3 = np.array([[1.3, 0.0, 0.0], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Does not sum to 1.
    p4 = np.array([[1.1, 0.3, 0.4], [0.1, 0.5, 0.4]], dtype=np.float32)

    with self.test_session():
      _, prob = distribution_util.get_logits_and_prob(
          p=p, multidimensional=True)
      prob.eval()

      with self.assertRaisesOpError("Condition x >= 0"):
        _, prob = distribution_util.get_logits_and_prob(
            p=p2, multidimensional=True, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_prob(
          p=p2, multidimensional=True, validate_args=False)
      prob.eval()

      with self.assertRaisesOpError(
          "(p has components greater than 1|p does not sum to 1)"):
        _, prob = distribution_util.get_logits_and_prob(
            p=p3, multidimensional=True, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_prob(
          p=p3, multidimensional=True, validate_args=False)
      prob.eval()

      with self.assertRaisesOpError("p does not sum to 1"):
        _, prob = distribution_util.get_logits_and_prob(
            p=p4, multidimensional=True, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_prob(
          p=p4, multidimensional=True, validate_args=False)
      prob.eval()


class LogCombinationsTest(test.TestCase):

  def testLogCombinationsBinomial(self):
    n = [2, 5, 12, 15]
    k = [1, 2, 4, 11]
    log_combs = np.log(special.binom(n, k))

    with self.test_session():
      n = np.array(n, dtype=np.float32)
      counts = [[1., 1], [2., 3], [4., 8], [11, 4]]
      log_binom = distribution_util.log_combinations(n, counts)
      self.assertEqual([4], log_binom.get_shape())
      self.assertAllClose(log_combs, log_binom.eval())

  def testLogCombinationsShape(self):
    # Shape [2, 2]
    n = [[2, 5], [12, 15]]

    with self.test_session():
      n = np.array(n, dtype=np.float32)
      # Shape [2, 2, 4]
      counts = [[[1., 1, 0, 0], [2., 2, 1, 0]], [[4., 4, 1, 3], [10, 1, 1, 4]]]
      log_binom = distribution_util.log_combinations(n, counts)
      self.assertEqual([2, 2], log_binom.get_shape())


class DynamicShapeTest(test.TestCase):

  def testSameDynamicShape(self):
    with self.test_session():
      scalar = constant_op.constant(2.0)
      scalar1 = array_ops.placeholder(dtype=dtypes.float32)

      vector = [0.3, 0.4, 0.5]
      vector1 = array_ops.placeholder(dtype=dtypes.float32, shape=[None])
      vector2 = array_ops.placeholder(dtype=dtypes.float32, shape=[None])

      multidimensional = [[0.3, 0.4], [0.2, 0.6]]
      multidimensional1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, None])
      multidimensional2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, None])

      # Scalar
      self.assertTrue(
          distribution_util.same_dynamic_shape(scalar, scalar1).eval({
              scalar1: 2.0
          }))

      # Vector

      self.assertTrue(
          distribution_util.same_dynamic_shape(vector, vector1).eval({
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertTrue(
          distribution_util.same_dynamic_shape(vector1, vector2).eval({
              vector1: [2.0, 3.0, 4.0],
              vector2: [2.0, 3.5, 6.0]
          }))

      # Multidimensional
      self.assertTrue(
          distribution_util.same_dynamic_shape(
              multidimensional, multidimensional1).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))
      self.assertTrue(
          distribution_util.same_dynamic_shape(
              multidimensional1, multidimensional2).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]],
                  multidimensional2: [[1.0, 3.5], [6.3, 2.3]]
              }))

      # Scalar, X
      self.assertFalse(
          distribution_util.same_dynamic_shape(scalar, vector1).eval({
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(scalar1, vector1).eval({
              scalar1: 2.0,
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(scalar, multidimensional1).eval({
              multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(scalar1, multidimensional1).eval(
              {
                  scalar1: 2.0,
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))

      # Vector, X
      self.assertFalse(
          distribution_util.same_dynamic_shape(vector, vector1).eval({
              vector1: [2.0, 3.0]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(vector1, vector2).eval({
              vector1: [2.0, 3.0, 4.0],
              vector2: [6.0]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(vector, multidimensional1).eval({
              multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
          }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(vector1, multidimensional1).eval(
              {
                  vector1: [2.0, 3.0, 4.0],
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))

      # Multidimensional, X
      self.assertFalse(
          distribution_util.same_dynamic_shape(
              multidimensional, multidimensional1).eval({
                  multidimensional1: [[1.0, 3.5, 5.0], [6.3, 2.3, 7.1]]
              }))
      self.assertFalse(
          distribution_util.same_dynamic_shape(
              multidimensional1, multidimensional2).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]],
                  multidimensional2: [[1.0, 3.5, 5.0], [6.3, 2.3, 7.1]]
              }))


class RotateTransposeTest(test.TestCase):

  def _np_rotate_transpose(self, x, shift):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    return np.transpose(x, np.roll(np.arange(len(x.shape)), shift))

  def testRollStatic(self):
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "None values not supported."):
        distribution_util.rotate_transpose(None, 1)
      for x in (np.ones(1), np.ones((2, 1)), np.ones((3, 2, 1))):
        for shift in np.arange(-5, 5):
          y = distribution_util.rotate_transpose(x, shift)
          self.assertAllEqual(self._np_rotate_transpose(x, shift), y.eval())
          self.assertAllEqual(np.roll(x.shape, shift), y.get_shape().as_list())

  def testRollDynamic(self):
    with self.test_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      shift = array_ops.placeholder(dtypes.int32)
      for x_value in (np.ones(
          1, dtype=x.dtype.as_numpy_dtype()), np.ones(
              (2, 1), dtype=x.dtype.as_numpy_dtype()), np.ones(
                  (3, 2, 1), dtype=x.dtype.as_numpy_dtype())):
        for shift_value in np.arange(-5, 5):
          self.assertAllEqual(
              self._np_rotate_transpose(x_value, shift_value),
              sess.run(distribution_util.rotate_transpose(x, shift),
                       feed_dict={x: x_value,
                                  shift: shift_value}))


class PickVectorTest(test.TestCase):

  def testCorrectlyPicksVector(self):
    with self.test_session():
      x = np.arange(10, 12)
      y = np.arange(15, 18)
      self.assertAllEqual(x,
                          distribution_util.pick_vector(
                              math_ops.less(0, 5), x, y).eval())
      self.assertAllEqual(y,
                          distribution_util.pick_vector(
                              math_ops.less(5, 0), x, y).eval())
      self.assertAllEqual(x,
                          distribution_util.pick_vector(
                              constant_op.constant(True), x, y))  # No eval.
      self.assertAllEqual(y,
                          distribution_util.pick_vector(
                              constant_op.constant(False), x, y))  # No eval.


class FillLowerTriangularTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _fill_lower_triangular(self, x):
    """Numpy implementation of `fill_lower_triangular`."""
    x = np.asarray(x)
    d = x.shape[-1]
    # d = n(n+1)/2 implies n is:
    n = int(0.5 * (math.sqrt(1. + 8. * d) - 1.))
    ids = np.tril_indices(n)
    y = np.zeros(list(x.shape[:-1]) + [n, n], dtype=x.dtype)
    y[..., ids[0], ids[1]] = x
    return y

  def testCorrectlyMakes1x1LowerTril(self):
    with self.test_session():
      x = ops.convert_to_tensor(self._rng.randn(3, 1))
      expected = self._fill_lower_triangular(tensor_util.constant_value(x))
      actual = distribution_util.fill_lower_triangular(x, validate_args=True)
      self.assertAllEqual(expected.shape, actual.get_shape())
      self.assertAllEqual(expected, actual.eval())

  def testCorrectlyMakesNoBatchLowerTril(self):
    with self.test_session():
      x = ops.convert_to_tensor(self._rng.randn(10))
      expected = self._fill_lower_triangular(tensor_util.constant_value(x))
      actual = distribution_util.fill_lower_triangular(x, validate_args=True)
      self.assertAllEqual(expected.shape, actual.get_shape())
      self.assertAllEqual(expected, actual.eval())
      g = gradients_impl.gradients(
          distribution_util.fill_lower_triangular(x), x)
      self.assertAllEqual(np.tri(4).reshape(-1), g[0].values.eval())

  def testCorrectlyMakesBatchLowerTril(self):
    with self.test_session():
      x = ops.convert_to_tensor(self._rng.randn(2, 2, 6))
      expected = self._fill_lower_triangular(tensor_util.constant_value(x))
      actual = distribution_util.fill_lower_triangular(x, validate_args=True)
      self.assertAllEqual(expected.shape, actual.get_shape())
      self.assertAllEqual(expected, actual.eval())
      self.assertAllEqual(
          np.ones((2, 2, 6)),
          gradients_impl.gradients(
              distribution_util.fill_lower_triangular(x), x)[0].eval())


class GenNewSeedTest(test.TestCase):

  def testOnlyNoneReturnsNone(self):
    self.assertFalse(distribution_util.gen_new_seed(0, "salt") is None)
    self.assertTrue(distribution_util.gen_new_seed(None, "salt") is None)


if __name__ == "__main__":
  test.main()
