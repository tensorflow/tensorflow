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

import importlib

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import util as distribution_util
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

du = distribution_util


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


special = try_import("scipy.special")


def _logit(x):
  x = np.asarray(x)
  return np.log(x) - np.log1p(-x)


class AssertCloseTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testAssertIntegerForm(self):
    # This should only be detected as an integer.
    x = array_ops.placeholder(dtypes.float32)
    y = array_ops.placeholder(dtypes.float32)
    # First component isn't less than float32.eps = 1e-7
    z = array_ops.placeholder(dtypes.float32)
    # This shouldn"t be detected as an integer.
    w = array_ops.placeholder(dtypes.float32)
    feed_dict = {x: [1., 5, 10, 15, 20], y: [1.1, 5, 10, 15, 20],
                 z: [1.0001, 5, 10, 15, 20], w: [1e-8, 5, 10, 15, 20]}
    with self.cached_session():
      with ops.control_dependencies([du.assert_integer_form(x)]):
        array_ops.identity(x).eval(feed_dict=feed_dict)

      with self.assertRaisesOpError("has non-integer components"):
        with ops.control_dependencies(
            [du.assert_integer_form(y)]):
          array_ops.identity(y).eval(feed_dict=feed_dict)

      with self.assertRaisesOpError("has non-integer components"):
        with ops.control_dependencies(
            [du.assert_integer_form(z)]):
          array_ops.identity(z).eval(feed_dict=feed_dict)

      with self.assertRaisesOpError("has non-integer components"):
        with ops.control_dependencies(
            [du.assert_integer_form(w)]):
          array_ops.identity(w).eval(feed_dict=feed_dict)


class MaybeGetStaticTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testGetStaticInt(self):
    x = 2
    self.assertEqual(x, du.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.), du.maybe_get_static_value(x, dtype=np.float64))

  @test_util.run_in_graph_and_eager_modes
  def testGetStaticNumpyArray(self):
    x = np.array(2, dtype=np.int32)
    self.assertEqual(x, du.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.), du.maybe_get_static_value(x, dtype=np.float64))

  @test_util.run_in_graph_and_eager_modes
  def testGetStaticConstant(self):
    x = constant_op.constant(2, dtype=dtypes.int32)
    self.assertEqual(np.array(2, dtype=np.int32), du.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.), du.maybe_get_static_value(x, dtype=np.float64))

  @test_util.run_deprecated_v1
  def testGetStaticPlaceholder(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
    self.assertEqual(None, du.maybe_get_static_value(x))
    self.assertEqual(None, du.maybe_get_static_value(x, dtype=np.float64))


class GetLogitsAndProbsTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testImproperArguments(self):
    with self.assertRaises(ValueError):
      du.get_logits_and_probs(logits=None, probs=None)

    with self.assertRaises(ValueError):
      du.get_logits_and_probs(logits=[0.1], probs=[0.1])

  @test_util.run_in_graph_and_eager_modes
  def testLogits(self):
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)
    logits = _logit(p)

    new_logits, new_p = du.get_logits_and_probs(
        logits=logits, validate_args=True)

    self.assertAllClose(p, self.evaluate(new_p), rtol=1e-5, atol=0.)
    self.assertAllClose(logits, self.evaluate(new_logits), rtol=1e-5, atol=0.)

  @test_util.run_in_graph_and_eager_modes
  def testLogitsMultidimensional(self):
    p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    logits = np.log(p)

    new_logits, new_p = du.get_logits_and_probs(
        logits=logits, multidimensional=True, validate_args=True)

    self.assertAllClose(self.evaluate(new_p), p)
    self.assertAllClose(self.evaluate(new_logits), logits)

  @test_util.run_in_graph_and_eager_modes
  def testProbability(self):
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)

    new_logits, new_p = du.get_logits_and_probs(probs=p, validate_args=True)

    self.assertAllClose(_logit(p), self.evaluate(new_logits))
    self.assertAllClose(p, self.evaluate(new_p))

  @test_util.run_in_graph_and_eager_modes
  def testProbabilityMultidimensional(self):
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)

    new_logits, new_p = du.get_logits_and_probs(
        probs=p, multidimensional=True, validate_args=True)

    self.assertAllClose(np.log(p), self.evaluate(new_logits))
    self.assertAllClose(p, self.evaluate(new_p))

  @test_util.run_in_graph_and_eager_modes
  def testProbabilityValidateArgs(self):
    p = [0.01, 0.2, 0.5, 0.7, .99]
    # Component less than 0.
    p2 = [-1, 0.2, 0.5, 0.3, .2]
    # Component greater than 1.
    p3 = [2, 0.2, 0.5, 0.3, .2]

    _, prob = du.get_logits_and_probs(probs=p, validate_args=True)
    self.evaluate(prob)

    with self.assertRaisesOpError("Condition x >= 0"):
      _, prob = du.get_logits_and_probs(probs=p2, validate_args=True)
      self.evaluate(prob)

    _, prob = du.get_logits_and_probs(probs=p2, validate_args=False)
    self.evaluate(prob)

    with self.assertRaisesOpError("probs has components greater than 1"):
      _, prob = du.get_logits_and_probs(probs=p3, validate_args=True)
      self.evaluate(prob)

    _, prob = du.get_logits_and_probs(probs=p3, validate_args=False)
    self.evaluate(prob)

  @test_util.run_in_graph_and_eager_modes
  def testProbabilityValidateArgsMultidimensional(self):
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component less than 0. Still sums to 1.
    p2 = np.array([[-.3, 0.4, 0.9], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component greater than 1. Does not sum to 1.
    p3 = np.array([[1.3, 0.0, 0.0], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Does not sum to 1.
    p4 = np.array([[1.1, 0.3, 0.4], [0.1, 0.5, 0.4]], dtype=np.float32)

    _, prob = du.get_logits_and_probs(probs=p, multidimensional=True)
    self.evaluate(prob)

    with self.assertRaisesOpError("Condition x >= 0"):
      _, prob = du.get_logits_and_probs(
          probs=p2, multidimensional=True, validate_args=True)
      self.evaluate(prob)

    _, prob = du.get_logits_and_probs(
        probs=p2, multidimensional=True, validate_args=False)
    self.evaluate(prob)

    with self.assertRaisesOpError(
        "(probs has components greater than 1|probs does not sum to 1)"):
      _, prob = du.get_logits_and_probs(
          probs=p3, multidimensional=True, validate_args=True)
      self.evaluate(prob)

    _, prob = du.get_logits_and_probs(
        probs=p3, multidimensional=True, validate_args=False)
    self.evaluate(prob)

    with self.assertRaisesOpError("probs does not sum to 1"):
      _, prob = du.get_logits_and_probs(
          probs=p4, multidimensional=True, validate_args=True)
      self.evaluate(prob)

    _, prob = du.get_logits_and_probs(
        probs=p4, multidimensional=True, validate_args=False)
    self.evaluate(prob)

  @test_util.run_deprecated_v1
  def testProbsMultidimShape(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        p = array_ops.ones([int(2**11+1)], dtype=np.float16)
        du.get_logits_and_probs(
            probs=p, multidimensional=True, validate_args=True)

      with self.assertRaisesOpError(
          "Number of classes exceeds `dtype` precision"):
        p = array_ops.placeholder(dtype=dtypes.float16)
        _, prob = du.get_logits_and_probs(
            probs=p, multidimensional=True, validate_args=True)
        prob.eval(feed_dict={p: np.ones([int(2**11+1)])})

  @test_util.run_deprecated_v1
  def testLogitsMultidimShape(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        l = array_ops.ones([int(2**11+1)], dtype=np.float16)
        du.get_logits_and_probs(
            logits=l, multidimensional=True, validate_args=True)

      with self.assertRaisesOpError(
          "Number of classes exceeds `dtype` precision"):
        l = array_ops.placeholder(dtype=dtypes.float16)
        logit, _ = du.get_logits_and_probs(
            logits=l, multidimensional=True, validate_args=True)
        logit.eval(feed_dict={l: np.ones([int(2**11+1)])})


class EmbedCheckCategoricalEventShapeTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testTooSmall(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        param = array_ops.ones([1], dtype=np.float16)
        checked_param = du.embed_check_categorical_event_shape(
            param)

      with self.assertRaisesOpError(
          "must have at least 2 events"):
        param = array_ops.placeholder(dtype=dtypes.float16)
        checked_param = du.embed_check_categorical_event_shape(
            param)
        checked_param.eval(feed_dict={param: np.ones([1])})

  @test_util.run_deprecated_v1
  def testTooLarge(self):
    with self.cached_session():
      with self.assertRaises(ValueError):
        param = array_ops.ones([int(2**11+1)], dtype=dtypes.float16)
        checked_param = du.embed_check_categorical_event_shape(
            param)

      with self.assertRaisesOpError(
          "Number of classes exceeds `dtype` precision"):
        param = array_ops.placeholder(dtype=dtypes.float16)
        checked_param = du.embed_check_categorical_event_shape(
            param)
        checked_param.eval(feed_dict={param: np.ones([int(2**11+1)])})

  @test_util.disable_tfrt("b/169901260")
  @test_util.run_in_graph_and_eager_modes
  def testUnsupportedDtype(self):
    param = ops.convert_to_tensor(
        np.ones([2**11 + 1]).astype(dtypes.qint16.as_numpy_dtype),
        dtype=dtypes.qint16)
    with self.assertRaises(TypeError):
      du.embed_check_categorical_event_shape(param)


class EmbedCheckIntegerCastingClosedTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testCorrectlyAssertsNonnegative(self):
    with self.cached_session():
      with self.assertRaisesOpError("Elements must be non-negative"):
        x = array_ops.placeholder(dtype=dtypes.float16)
        x_checked = du.embed_check_integer_casting_closed(
            x, target_dtype=dtypes.int16)
        x_checked.eval(feed_dict={x: np.array([1, -1], dtype=np.float16)})

  @test_util.run_deprecated_v1
  def testCorrectlyAssersIntegerForm(self):
    with self.cached_session():
      with self.assertRaisesOpError("Elements must be int16-equivalent."):
        x = array_ops.placeholder(dtype=dtypes.float16)
        x_checked = du.embed_check_integer_casting_closed(
            x, target_dtype=dtypes.int16)
        x_checked.eval(feed_dict={x: np.array([1, 1.5], dtype=np.float16)})

  @test_util.run_deprecated_v1
  def testCorrectlyAssertsLargestPossibleInteger(self):
    with self.cached_session():
      with self.assertRaisesOpError("Elements cannot exceed 32767."):
        x = array_ops.placeholder(dtype=dtypes.int32)
        x_checked = du.embed_check_integer_casting_closed(
            x, target_dtype=dtypes.int16)
        x_checked.eval(feed_dict={x: np.array([1, 2**15], dtype=np.int32)})

  @test_util.run_deprecated_v1
  def testCorrectlyAssertsSmallestPossibleInteger(self):
    with self.cached_session():
      with self.assertRaisesOpError("Elements cannot be smaller than 0."):
        x = array_ops.placeholder(dtype=dtypes.int32)
        x_checked = du.embed_check_integer_casting_closed(
            x, target_dtype=dtypes.uint16, assert_nonnegative=False)
        x_checked.eval(feed_dict={x: np.array([1, -1], dtype=np.int32)})


@test_util.run_all_in_graph_and_eager_modes
class LogCombinationsTest(test.TestCase):

  def testLogCombinationsBinomial(self):
    n = [2, 5, 12, 15]
    k = [1, 2, 4, 11]

    if not special:
      return

    log_combs = np.log(special.binom(n, k))

    n = np.array(n, dtype=np.float32)
    counts = [[1., 1], [2., 3], [4., 8], [11, 4]]
    log_binom = du.log_combinations(n, counts)
    self.assertEqual([4], log_binom.get_shape())
    self.assertAllClose(log_combs, self.evaluate(log_binom))

  def testLogCombinationsShape(self):
    # Shape [2, 2]
    n = [[2, 5], [12, 15]]

    n = np.array(n, dtype=np.float32)
    # Shape [2, 2, 4]
    counts = [[[1., 1, 0, 0], [2., 2, 1, 0]], [[4., 4, 1, 3], [10, 1, 1, 4]]]
    log_binom = du.log_combinations(n, counts)
    self.assertEqual([2, 2], log_binom.get_shape())


class DynamicShapeTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testSameDynamicShape(self):
    with self.cached_session():
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
          du.same_dynamic_shape(scalar, scalar1).eval({
              scalar1: 2.0
          }))

      # Vector

      self.assertTrue(
          du.same_dynamic_shape(vector, vector1).eval({
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertTrue(
          du.same_dynamic_shape(vector1, vector2).eval({
              vector1: [2.0, 3.0, 4.0],
              vector2: [2.0, 3.5, 6.0]
          }))

      # Multidimensional
      self.assertTrue(
          du.same_dynamic_shape(
              multidimensional, multidimensional1).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))
      self.assertTrue(
          du.same_dynamic_shape(
              multidimensional1, multidimensional2).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]],
                  multidimensional2: [[1.0, 3.5], [6.3, 2.3]]
              }))

      # Scalar, X
      self.assertFalse(
          du.same_dynamic_shape(scalar, vector1).eval({
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertFalse(
          du.same_dynamic_shape(scalar1, vector1).eval({
              scalar1: 2.0,
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertFalse(
          du.same_dynamic_shape(scalar, multidimensional1).eval({
              multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
          }))
      self.assertFalse(
          du.same_dynamic_shape(scalar1, multidimensional1).eval(
              {
                  scalar1: 2.0,
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))

      # Vector, X
      self.assertFalse(
          du.same_dynamic_shape(vector, vector1).eval({
              vector1: [2.0, 3.0]
          }))
      self.assertFalse(
          du.same_dynamic_shape(vector1, vector2).eval({
              vector1: [2.0, 3.0, 4.0],
              vector2: [6.0]
          }))
      self.assertFalse(
          du.same_dynamic_shape(vector, multidimensional1).eval({
              multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
          }))
      self.assertFalse(
          du.same_dynamic_shape(vector1, multidimensional1).eval(
              {
                  vector1: [2.0, 3.0, 4.0],
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))

      # Multidimensional, X
      self.assertFalse(
          du.same_dynamic_shape(
              multidimensional, multidimensional1).eval({
                  multidimensional1: [[1.0, 3.5, 5.0], [6.3, 2.3, 7.1]]
              }))
      self.assertFalse(
          du.same_dynamic_shape(
              multidimensional1, multidimensional2).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]],
                  multidimensional2: [[1.0, 3.5, 5.0], [6.3, 2.3, 7.1]]
              }))


class RotateTransposeTest(test.TestCase):

  def _np_rotate_transpose(self, x, shift):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    return np.transpose(x, np.roll(np.arange(len(x.shape)), shift))

  @test_util.run_in_graph_and_eager_modes
  def testRollStatic(self):
    if context.executing_eagerly():
      error_message = r"Attempt to convert a value \(None\)"
    else:
      error_message = "None values not supported."
    with self.assertRaisesRegex(ValueError, error_message):
      du.rotate_transpose(None, 1)
    for x in (np.ones(1), np.ones((2, 1)), np.ones((3, 2, 1))):
      for shift in np.arange(-5, 5):
        y = du.rotate_transpose(x, shift)
        self.assertAllEqual(
            self._np_rotate_transpose(x, shift), self.evaluate(y))
        self.assertAllEqual(np.roll(x.shape, shift), y.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testRollDynamic(self):
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      shift = array_ops.placeholder(dtypes.int32)
      for x_value in (np.ones(
          1, dtype=x.dtype.as_numpy_dtype()), np.ones(
              (2, 1), dtype=x.dtype.as_numpy_dtype()), np.ones(
                  (3, 2, 1), dtype=x.dtype.as_numpy_dtype())):
        for shift_value in np.arange(-5, 5):
          self.assertAllEqual(
              self._np_rotate_transpose(x_value, shift_value),
              sess.run(du.rotate_transpose(x, shift),
                       feed_dict={x: x_value,
                                  shift: shift_value}))


class PickVectorTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testCorrectlyPicksVector(self):
    with self.cached_session():
      x = np.arange(10, 12)
      y = np.arange(15, 18)
      self.assertAllEqual(
          x, self.evaluate(du.pick_vector(math_ops.less(0, 5), x, y)))
      self.assertAllEqual(
          y, self.evaluate(du.pick_vector(math_ops.less(5, 0), x, y)))
      self.assertAllEqual(x,
                          du.pick_vector(
                              constant_op.constant(True), x, y))  # No eval.
      self.assertAllEqual(y,
                          du.pick_vector(
                              constant_op.constant(False), x, y))  # No eval.


class PreferStaticRankTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testNonEmptyConstantTensor(self):
    x = array_ops.zeros((2, 3, 4))
    rank = du.prefer_static_rank(x)
    self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(3, rank)

  @test_util.run_deprecated_v1
  def testEmptyConstantTensor(self):
    x = constant_op.constant([])
    rank = du.prefer_static_rank(x)
    self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(1, rank)

  @test_util.run_deprecated_v1
  def testScalarTensor(self):
    x = constant_op.constant(1.)
    rank = du.prefer_static_rank(x)
    self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(0, rank)

  @test_util.run_deprecated_v1
  def testDynamicRankEndsUpBeingNonEmpty(self):
    x = array_ops.placeholder(np.float64, shape=None)
    rank = du.prefer_static_rank(x)
    with self.cached_session():
      self.assertAllEqual(2, rank.eval(feed_dict={x: np.zeros((2, 3))}))

  @test_util.run_deprecated_v1
  def testDynamicRankEndsUpBeingEmpty(self):
    x = array_ops.placeholder(np.int32, shape=None)
    rank = du.prefer_static_rank(x)
    with self.cached_session():
      self.assertAllEqual(1, rank.eval(feed_dict={x: []}))

  @test_util.run_deprecated_v1
  def testDynamicRankEndsUpBeingScalar(self):
    x = array_ops.placeholder(np.int32, shape=None)
    rank = du.prefer_static_rank(x)
    with self.cached_session():
      self.assertAllEqual(0, rank.eval(feed_dict={x: 1}))


class PreferStaticShapeTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testNonEmptyConstantTensor(self):
    x = array_ops.zeros((2, 3, 4))
    shape = du.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual(np.array([2, 3, 4]), shape)

  @test_util.run_deprecated_v1
  def testEmptyConstantTensor(self):
    x = constant_op.constant([])
    shape = du.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual(np.array([0]), shape)

  @test_util.run_deprecated_v1
  def testScalarTensor(self):
    x = constant_op.constant(1.)
    shape = du.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual(np.array([]), shape)

  @test_util.run_deprecated_v1
  def testDynamicShapeEndsUpBeingNonEmpty(self):
    x = array_ops.placeholder(np.float64, shape=None)
    shape = du.prefer_static_shape(x)
    with self.cached_session():
      self.assertAllEqual((2, 3), shape.eval(feed_dict={x: np.zeros((2, 3))}))

  @test_util.run_deprecated_v1
  def testDynamicShapeEndsUpBeingEmpty(self):
    x = array_ops.placeholder(np.int32, shape=None)
    shape = du.prefer_static_shape(x)
    with self.cached_session():
      self.assertAllEqual(np.array([0]), shape.eval(feed_dict={x: []}))

  @test_util.run_deprecated_v1
  def testDynamicShapeEndsUpBeingScalar(self):
    x = array_ops.placeholder(np.int32, shape=None)
    shape = du.prefer_static_shape(x)
    with self.cached_session():
      self.assertAllEqual(np.array([]), shape.eval(feed_dict={x: 1}))


class PreferStaticValueTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testNonEmptyConstantTensor(self):
    x = array_ops.zeros((2, 3, 4))
    value = du.prefer_static_value(x)
    self.assertIsInstance(value, np.ndarray)
    self.assertAllEqual(np.zeros((2, 3, 4)), value)

  @test_util.run_deprecated_v1
  def testEmptyConstantTensor(self):
    x = constant_op.constant([])
    value = du.prefer_static_value(x)
    self.assertIsInstance(value, np.ndarray)
    self.assertAllEqual(np.array([]), value)

  @test_util.run_deprecated_v1
  def testScalarTensor(self):
    x = constant_op.constant(1.)
    value = du.prefer_static_value(x)
    self.assertIsInstance(value, np.ndarray)
    self.assertAllEqual(np.array(1.), value)

  @test_util.run_deprecated_v1
  def testDynamicValueEndsUpBeingNonEmpty(self):
    x = array_ops.placeholder(np.float64, shape=None)
    value = du.prefer_static_value(x)
    with self.cached_session():
      self.assertAllEqual(np.zeros((2, 3)),
                          value.eval(feed_dict={x: np.zeros((2, 3))}))

  @test_util.run_deprecated_v1
  def testDynamicValueEndsUpBeingEmpty(self):
    x = array_ops.placeholder(np.int32, shape=None)
    value = du.prefer_static_value(x)
    with self.cached_session():
      self.assertAllEqual(np.array([]), value.eval(feed_dict={x: []}))

  @test_util.run_deprecated_v1
  def testDynamicValueEndsUpBeingScalar(self):
    x = array_ops.placeholder(np.int32, shape=None)
    value = du.prefer_static_value(x)
    with self.cached_session():
      self.assertAllEqual(np.array(1), value.eval(feed_dict={x: 1}))


class FillTriangularTest(test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _fill_triangular(self, x, upper=False):
    """Numpy implementation of `fill_triangular`."""
    x = np.asarray(x)
    # Formula derived by solving for n: m = n(n+1)/2.
    m = np.int32(x.shape[-1])
    n = np.sqrt(0.25 + 2. * m) - 0.5
    if n != np.floor(n):
      raise ValueError("Invalid shape.")
    n = np.int32(n)
    # We can't do: `x[..., -(n**2-m):]` because this doesn't correctly handle
    # `m == n == 1`. Hence, we do absolute indexing.
    x_tail = x[..., (m - (n * n - m)):]
    y = np.concatenate(
        [x, x_tail[..., ::-1]] if upper else [x_tail, x[..., ::-1]],
        axis=-1)
    y = y.reshape(np.concatenate([
        np.int32(x.shape[:-1]),
        np.int32([n, n]),
    ], axis=0))
    return np.triu(y) if upper else np.tril(y)

  def _run_test(self, x_, use_deferred_shape=False, **kwargs):
    x_ = np.asarray(x_)
    with self.cached_session() as sess:
      static_shape = None if use_deferred_shape else x_.shape
      x_pl = array_ops.placeholder_with_default(x_, shape=static_shape)
      # Add `zeros_like(x)` such that x's value and gradient are identical. We
      # do this so we can ensure each gradient value is mapped to the right
      # gradient location.  (Not doing this means the gradient wrt `x` is simple
      # `ones_like(x)`.)
      # Note:
      #   zeros_like_x_pl == zeros_like(x_pl)
      #   gradient(zeros_like_x_pl, x_pl) == x_pl - 1
      zeros_like_x_pl = (x_pl * array_ops.stop_gradient(x_pl - 1.)
                         - array_ops.stop_gradient(x_pl * (x_pl - 1.)))
      x = x_pl + zeros_like_x_pl
      actual = du.fill_triangular(x, **kwargs)
      grad_actual = gradients_impl.gradients(actual, x_pl)[0]
      [actual_, grad_actual_] = sess.run([actual, grad_actual],
                                         feed_dict={x_pl: x_})
    expected = self._fill_triangular(x_, **kwargs)
    if use_deferred_shape:
      self.assertEqual(None, actual.shape)
    else:
      self.assertAllEqual(expected.shape, actual.shape)
    self.assertAllClose(expected, actual_, rtol=1e-8, atol=1e-9)
    self.assertAllClose(x_, grad_actual_, rtol=1e-8, atol=1e-9)

  @test_util.run_deprecated_v1
  def testCorrectlyMakes1x1TriLower(self):
    self._run_test(self._rng.randn(3, int(1*2/2)))

  @test_util.run_deprecated_v1
  def testCorrectlyMakesNoBatchTriLower(self):
    self._run_test(self._rng.randn(int(4*5/2)))

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatchTriLower(self):
    self._run_test(self._rng.randn(2, 3, int(3*4/2)))

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatchTriLowerUnknownShape(self):
    self._run_test(self._rng.randn(2, 3, int(3*4/2)), use_deferred_shape=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatch7x7TriLowerUnknownShape(self):
    self._run_test(self._rng.randn(2, 3, int(7*8/2)), use_deferred_shape=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatch7x7TriLower(self):
    self._run_test(self._rng.randn(2, 3, int(7*8/2)))

  @test_util.run_deprecated_v1
  def testCorrectlyMakes1x1TriUpper(self):
    self._run_test(self._rng.randn(3, int(1*2/2)), upper=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesNoBatchTriUpper(self):
    self._run_test(self._rng.randn(int(4*5/2)), upper=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatchTriUpper(self):
    self._run_test(self._rng.randn(2, 2, int(3*4/2)), upper=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatchTriUpperUnknownShape(self):
    self._run_test(self._rng.randn(2, 2, int(3*4/2)),
                   use_deferred_shape=True,
                   upper=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatch7x7TriUpperUnknownShape(self):
    self._run_test(self._rng.randn(2, 3, int(7*8/2)),
                   use_deferred_shape=True,
                   upper=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatch7x7TriUpper(self):
    self._run_test(self._rng.randn(2, 3, int(7*8/2)), upper=True)


class FillTriangularInverseTest(FillTriangularTest):

  def _run_test(self, x_, use_deferred_shape=False, **kwargs):
    x_ = np.asarray(x_)
    with self.cached_session() as sess:
      static_shape = None if use_deferred_shape else x_.shape
      x_pl = array_ops.placeholder_with_default(x_, shape=static_shape)
      zeros_like_x_pl = (x_pl * array_ops.stop_gradient(x_pl - 1.)
                         - array_ops.stop_gradient(x_pl * (x_pl - 1.)))
      x = x_pl + zeros_like_x_pl
      actual = du.fill_triangular(x, **kwargs)
      inverse_actual = du.fill_triangular_inverse(actual, **kwargs)

      inverse_actual_ = sess.run(
          inverse_actual,
          feed_dict={x_pl: x_})

    if use_deferred_shape:
      self.assertEqual(None, inverse_actual.shape)
    else:
      self.assertAllEqual(x_.shape, inverse_actual.shape)
    self.assertAllEqual(x_, inverse_actual_)


class ReduceWeightedLogSumExp(test.TestCase):

  def _reduce_weighted_logsumexp(self, logx, w, axis, keep_dims=False):
    m = np.max(logx, axis=axis, keepdims=True)
    sum_ = np.sum(w * np.exp(logx - m), axis=axis, keepdims=keep_dims)
    sgn = np.sign(sum_)
    if not keep_dims:
      m = np.squeeze(m, axis=axis)
    return m + np.log(sgn * sum_), sgn

  @test_util.run_deprecated_v1
  def testNoWeights(self):
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    with self.cached_session() as sess:
      logx = constant_op.constant(logx_)
      expected = math_ops.reduce_logsumexp(logx, axis=-1)
      grad_expected = gradients_impl.gradients(expected, logx)[0]
      actual, actual_sgn = du.reduce_weighted_logsumexp(
          logx, axis=-1, return_sign=True)
      grad_actual = gradients_impl.gradients(actual, logx)[0]
      [actual_, actual_sgn_, grad_actual_,
       expected_, grad_expected_] = sess.run([
           actual, actual_sgn, grad_actual,
           expected, grad_expected])
    self.assertAllEqual(expected_, actual_)
    self.assertAllEqual(grad_expected_, grad_actual_)
    self.assertAllEqual([1., 1, 1], actual_sgn_)

  def testNegativeWeights(self):
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    w_ = np.array([[1., 1, -1],
                   [1, -2, 1],
                   [1, 0, 1]])
    expected, _ = self._reduce_weighted_logsumexp(logx_, w_, axis=-1)
    with self.cached_session() as sess:
      logx = constant_op.constant(logx_)
      w = constant_op.constant(w_)
      actual, actual_sgn = du.reduce_weighted_logsumexp(
          logx, w, axis=-1, return_sign=True)
      [actual_, actual_sgn_] = self.evaluate([actual, actual_sgn])
    self.assertAllEqual(expected, actual_)
    self.assertAllEqual([-1., -1, 1], actual_sgn_)

  def testKeepDims(self):
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    w_ = np.array([[1., 1, -1],
                   [1, -2, 1],
                   [1, 0, 1]])
    expected, _ = self._reduce_weighted_logsumexp(
        logx_, w_, axis=-1, keep_dims=True)
    with self.cached_session() as sess:
      logx = constant_op.constant(logx_)
      w = constant_op.constant(w_)
      actual, actual_sgn = du.reduce_weighted_logsumexp(
          logx, w, axis=-1, return_sign=True, keep_dims=True)
      [actual_, actual_sgn_] = self.evaluate([actual, actual_sgn])
    self.assertAllEqual(expected, actual_)
    self.assertAllEqual([[-1.], [-1], [1]], actual_sgn_)

  def testDocString(self):
    """This test verifies the correctness of the docstring examples."""

    with self.cached_session():
      x = constant_op.constant([[0., 0, 0],
                                [0, 0, 0]])

      w = constant_op.constant([[-1., 1, 1],
                                [1, 1, 1]])

      self.assertAllClose(
          np.log(4), self.evaluate(du.reduce_weighted_logsumexp(x, w)))

      with np.errstate(divide="ignore"):
        self.assertAllClose(
            np.log([0, 2, 2]),
            self.evaluate(du.reduce_weighted_logsumexp(x, w, axis=0)))

      self.assertAllClose(
          np.log([1, 3]),
          self.evaluate(du.reduce_weighted_logsumexp(x, w, axis=1)))

      self.assertAllClose(
          np.log([[1], [3]]),
          self.evaluate(
              du.reduce_weighted_logsumexp(x, w, axis=1, keep_dims=True)))

      self.assertAllClose(
          np.log(4),
          self.evaluate(du.reduce_weighted_logsumexp(x, w, axis=[0, 1])))


class GenNewSeedTest(test.TestCase):

  def testOnlyNoneReturnsNone(self):
    self.assertFalse(du.gen_new_seed(0, "salt") is None)
    self.assertTrue(du.gen_new_seed(None, "salt") is None)


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
    with self.session(use_gpu=use_gpu) as sess:
      softplus = nn_ops.softplus(np_features)
      softplus_inverse = du.softplus_inverse(softplus)
      [tf_softplus, tf_softplus_inverse] = sess.run([
          softplus, softplus_inverse])
    rtol = {"float16": 0.07, "float32": 0.003, "float64": 0.002}.get(
        str(np_features.dtype), 1e-6)
    self.assertAllCloseAccordingToType(np_softplus, tf_softplus, rtol=rtol)
    # This will test that we correctly computed the inverse by verifying we
    # recovered the original input.
    self.assertAllCloseAccordingToType(
        np_features, tf_softplus_inverse,
        atol=0., rtol=rtol)
    self.assertAllEqual(
        np.ones_like(tf_softplus).astype(np.bool_), tf_softplus > 0)

    self.assertShapeEqual(np_softplus, softplus)
    self.assertShapeEqual(np_softplus, softplus_inverse)

    self.assertAllEqual(
        np.ones_like(tf_softplus).astype(np.bool_), np.isfinite(tf_softplus))
    self.assertAllEqual(
        np.ones_like(tf_softplus_inverse).astype(np.bool_),
        np.isfinite(tf_softplus_inverse))

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
  def testGradient(self):
    with self.cached_session():
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
    tf_logging.vlog(2, "softplus (float) gradient err = ", err)
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testInverseSoftplusGradientNeverNan(self):
    with self.cached_session():
      # Note that this range contains both zero and inf.
      x = constant_op.constant(np.logspace(-8, 6).astype(np.float16))
      y = du.softplus_inverse(x)
      grads = self.evaluate(gradients_impl.gradients(y, x)[0])
      # Equivalent to `assertAllFalse` (if it existed).
      self.assertAllEqual(
          np.zeros_like(grads).astype(np.bool_), np.isnan(grads))

  @test_util.run_deprecated_v1
  def testInverseSoftplusGradientFinite(self):
    with self.cached_session():
      # This range of x is all finite, and so is 1 / x.  So the
      # gradient and its approximations should be finite as well.
      x = constant_op.constant(np.logspace(-4.8, 4.5).astype(np.float16))
      y = du.softplus_inverse(x)
      grads = self.evaluate(gradients_impl.gradients(y, x)[0])
      # Equivalent to `assertAllTrue` (if it existed).
      self.assertAllEqual(
          np.ones_like(grads).astype(np.bool_), np.isfinite(grads))


@test_util.run_all_in_graph_and_eager_modes
class ArgumentsTest(test.TestCase):

  def testNoArguments(self):
    def foo():
      return du.parent_frame_arguments()

    self.assertEqual({}, foo())

  def testPositionalArguments(self):
    def foo(a, b, c, d):  # pylint: disable=unused-argument
      return du.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3, "d": 4}, foo(1, 2, 3, 4))

    # Tests that it does not matter where this function is called, and
    # no other local variables are returned back.
    def bar(a, b, c):
      unused_x = a * b
      unused_y = c * 3
      return du.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3}, bar(1, 2, 3))

  def testOverloadedArgumentValues(self):
    def foo(a, b, c):  # pylint: disable=unused-argument
      a = 42
      b = 31
      c = 42
      return du.parent_frame_arguments()
    self.assertEqual({"a": 42, "b": 31, "c": 42}, foo(1, 2, 3))

  def testKeywordArguments(self):
    def foo(**kwargs):  # pylint: disable=unused-argument
      return du.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3, "d": 4}, foo(a=1, b=2, c=3, d=4))

  def testPositionalKeywordArgs(self):
    def foo(a, b, c, **kwargs):  # pylint: disable=unused-argument
      return du.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3}, foo(a=1, b=2, c=3))
    self.assertEqual({"a": 1, "b": 2, "c": 3, "unicorn": None},
                     foo(a=1, b=2, c=3, unicorn=None))

  def testNoVarargs(self):
    def foo(a, b, c, *varargs, **kwargs):  # pylint: disable=unused-argument
      return du.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3}, foo(a=1, b=2, c=3))
    self.assertEqual({"a": 1, "b": 2, "c": 3}, foo(1, 2, 3, *[1, 2, 3]))
    self.assertEqual({"a": 1, "b": 2, "c": 3, "unicorn": None},
                     foo(1, 2, 3, unicorn=None))
    self.assertEqual({"a": 1, "b": 2, "c": 3, "unicorn": None},
                     foo(1, 2, 3, *[1, 2, 3], unicorn=None))


if __name__ == "__main__":
  test.main()
