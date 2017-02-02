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
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
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


class GetLogitsAndProbsTest(test.TestCase):

  def testGetLogitsAndProbsImproperArguments(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        distribution_util.get_logits_and_probs(logits=None, probs=None)

      with self.assertRaises(ValueError):
        distribution_util.get_logits_and_probs(logits=[0.1], probs=[0.1])

  def testGetLogitsAndProbsLogits(self):
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)
    logits = special.logit(p)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_probs(
          logits=logits, validate_args=True)

      self.assertAllClose(p, new_p.eval())
      self.assertAllClose(logits, new_logits.eval())

  def testGetLogitsAndProbsLogitsMultidimensional(self):
    p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    logits = np.log(p)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_probs(
          logits=logits, multidimensional=True, validate_args=True)

      self.assertAllClose(new_p.eval(), p)
      self.assertAllClose(new_logits.eval(), logits)

  def testGetLogitsAndProbsProbability(self):
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_probs(
          probs=p, validate_args=True)

      self.assertAllClose(special.logit(p), new_logits.eval())
      self.assertAllClose(p, new_p.eval())

  def testGetLogitsAndProbsProbabilityMultidimensional(self):
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_probs(
          probs=p, multidimensional=True, validate_args=True)

      self.assertAllClose(np.log(p), new_logits.eval())
      self.assertAllClose(p, new_p.eval())

  def testGetLogitsAndProbsProbabilityValidateArgs(self):
    p = [0.01, 0.2, 0.5, 0.7, .99]
    # Component less than 0.
    p2 = [-1, 0.2, 0.5, 0.3, .2]
    # Component greater than 1.
    p3 = [2, 0.2, 0.5, 0.3, .2]

    with self.test_session():
      _, prob = distribution_util.get_logits_and_probs(
          probs=p, validate_args=True)
      prob.eval()

      with self.assertRaisesOpError("Condition x >= 0"):
        _, prob = distribution_util.get_logits_and_probs(
            probs=p2, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_probs(
          probs=p2, validate_args=False)
      prob.eval()

      with self.assertRaisesOpError("probs has components greater than 1"):
        _, prob = distribution_util.get_logits_and_probs(
            probs=p3, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_probs(
          probs=p3, validate_args=False)
      prob.eval()

  def testGetLogitsAndProbsProbabilityValidateArgsMultidimensional(self):
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component less than 0. Still sums to 1.
    p2 = np.array([[-.3, 0.4, 0.9], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component greater than 1. Does not sum to 1.
    p3 = np.array([[1.3, 0.0, 0.0], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Does not sum to 1.
    p4 = np.array([[1.1, 0.3, 0.4], [0.1, 0.5, 0.4]], dtype=np.float32)

    with self.test_session():
      _, prob = distribution_util.get_logits_and_probs(
          probs=p, multidimensional=True)
      prob.eval()

      with self.assertRaisesOpError("Condition x >= 0"):
        _, prob = distribution_util.get_logits_and_probs(
            probs=p2, multidimensional=True, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_probs(
          probs=p2, multidimensional=True, validate_args=False)
      prob.eval()

      with self.assertRaisesOpError(
          "(probs has components greater than 1|probs does not sum to 1)"):
        _, prob = distribution_util.get_logits_and_probs(
            probs=p3, multidimensional=True, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_probs(
          probs=p3, multidimensional=True, validate_args=False)
      prob.eval()

      with self.assertRaisesOpError("probs does not sum to 1"):
        _, prob = distribution_util.get_logits_and_probs(
            probs=p4, multidimensional=True, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_probs(
          probs=p4, multidimensional=True, validate_args=False)
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


# TODO(jvdillon): Merge this test back into:
# tensorflow/python/kernel_tests/softplus_op_test.py
# once TF core is accepting new ops.
class SoftplusTest(test.TestCase):

  def _npSoftplus(self, np_features):
    np_features = np.asarray(np_features)
    zero = np.asarray(0).astype(np_features.dtype)
    return np.logaddexp(zero, np_features)

  def _testSoftplus(self, np_features, use_gpu=False):
    np_features = np.asarray(np_features)
    np_softplus = self._npSoftplus(np_features)
    with self.test_session(use_gpu=use_gpu) as sess:
      softplus = nn_ops.softplus(np_features)
      softplus_inverse = distribution_util.softplus_inverse(softplus)
      [tf_softplus, tf_softplus_inverse] = sess.run([
          softplus, softplus_inverse])
    self.assertAllCloseAccordingToType(np_softplus, tf_softplus)
    rtol = {"float16": 0.07, "float32": 0.003, "float64": 0.002}.get(
        str(np_features.dtype), 1e-6)
    # This will test that we correctly computed the inverse by verifying we
    # recovered the original input.
    self.assertAllCloseAccordingToType(
        np_features, tf_softplus_inverse,
        atol=0., rtol=rtol)
    self.assertAllEqual(np.ones_like(tf_softplus).astype(np.bool),
                        tf_softplus > 0)

    self.assertShapeEqual(np_softplus, softplus)
    self.assertShapeEqual(np_softplus, softplus_inverse)

    self.assertAllEqual(np.ones_like(tf_softplus).astype(np.bool),
                        np.isfinite(tf_softplus))
    self.assertAllEqual(np.ones_like(tf_softplus_inverse).astype(np.bool),
                        np.isfinite(tf_softplus_inverse))

  def testNumbers(self):
    for t in [np.float16, np.float32, np.float64]:
      lower = {np.float16: -15, np.float32: -50, np.float64: -50}.get(t, -100)
      upper = {np.float16: 50, np.float32: 50, np.float64: 50}.get(t, 100)
      self._testSoftplus(
          np.array(np.linspace(lower, upper, int(1e3)).astype(t)).reshape(
              [2, -1]),
          use_gpu=False)
      self._testSoftplus(
          np.array(np.linspace(lower, upper, int(1e3)).astype(t)).reshape(
              [2, -1]),
          use_gpu=True)
      log_eps = np.log(np.finfo(t).eps)
      one = t(1)
      ten = t(10)
      self._testSoftplus(
          [
              log_eps, log_eps - one, log_eps + one, log_eps - ten,
              log_eps + ten, -log_eps, -log_eps - one, -log_eps + one,
              -log_eps - ten, -log_eps + ten
          ],
          use_gpu=False)
      self._testSoftplus(
          [
              log_eps, log_eps - one, log_eps + one, log_eps - ten,
              log_eps + ten - log_eps, -log_eps - one, -log_eps + one,
              -log_eps - ten, -log_eps + ten
          ],
          use_gpu=True)

  def testGradient(self):
    with self.test_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.softplus(x, name="softplus")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    print("softplus (float) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testInverseSoftplusGradientNeverNan(self):
    with self.test_session():
      # Note that this range contains both zero and inf.
      x = constant_op.constant(np.logspace(-8, 6).astype(np.float16))
      y = distribution_util.softplus_inverse(x)
      grads = gradients_impl.gradients(y, x)[0].eval()
      # Equivalent to `assertAllFalse` (if it existed).
      self.assertAllEqual(np.zeros_like(grads).astype(np.bool), np.isnan(grads))

if __name__ == "__main__":
  test.main()
