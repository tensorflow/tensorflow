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
"""Tests for tensorflow.ops.nn_ops.Pad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test


class PadOpTest(test.TestCase):

  def _npPad(self, inp, paddings, mode, constant_values=0):
    mode = mode.lower()
    if mode == "constant":
      return np.pad(inp, paddings, mode=mode, constant_values=constant_values)
    else:
      return np.pad(inp, paddings, mode=mode)

  def testNpPad(self):
    self.assertAllEqual(
        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 3, 3, 0, 0, 0],
                  [0, 4, 4, 0, 0, 0],
                  [0, 5, 5, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]),
        self._npPad(
            np.array([[3, 3], [4, 4], [5, 5]]),
            [[1, 2], [1, 3]],
            mode="constant"))

    self.assertAllEqual(
        np.array([[1, 1, 1, 1, 1, 1],
                  [1, 3, 3, 1, 1, 1],
                  [1, 4, 4, 1, 1, 1],
                  [1, 5, 5, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1]]),
        self._npPad(
            np.array([[3, 3], [4, 4], [5, 5]]),
            [[1, 2], [1, 3]],
            mode="constant", constant_values=1))

    self.assertAllEqual(
        np.array([[4, 3, 4, 9, 4, 3],
                  [1, 0, 1, 2, 1, 0],
                  [4, 3, 4, 9, 4, 3],
                  [1, 0, 1, 2, 1, 0]]),
        self._npPad(
            np.array([[0, 1, 2], [3, 4, 9]]),
            [[1, 1], [1, 2]],
            mode="reflect"))

    self.assertAllEqual(
        np.array([[0, 0, 1, 2, 2, 1],
                  [0, 0, 1, 2, 2, 1],
                  [3, 3, 4, 9, 9, 4],
                  [3, 3, 4, 9, 9, 4]]),
        self._npPad(
            np.array([[0, 1, 2], [3, 4, 9]]),
            [[1, 1], [1, 2]],
            mode="symmetric"))

  def _testPad(self, np_inputs, paddings, mode, constant_values):
    np_val = self._npPad(np_inputs, paddings, mode=mode,
                         constant_values=constant_values)
    with self.cached_session(use_gpu=True):
      tf_val = array_ops.pad(np_inputs, paddings, mode=mode,
                             constant_values=constant_values)
      out = self.evaluate(tf_val)
    self.assertAllEqual(np_val, out)
    self.assertShapeEqual(np_val, tf_val)

  def _testGradient(self, x, a, mode, constant_values):
    with self.cached_session(use_gpu=True):
      inx = ops.convert_to_tensor(x)
      xs = list(x.shape)
      ina = ops.convert_to_tensor(a)
      y = array_ops.pad(inx, ina, mode=mode, constant_values=constant_values)
      # Expected y's shape to be:
      ys = list(np.array(x.shape) + np.sum(np.array(a), axis=1))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, y, ys, x_init_value=x)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _testAll(self, np_inputs, paddings, constant_values):
    for mode in ("CONSTANT", "REFLECT", "SYMMETRIC", "reflect", "symmetric",
                 "constant"):
      # Zero-sized input is not allowed for REFLECT mode, but we still want
      # zero-sized input test cases for the other modes.
      if np_inputs.size or mode.upper() != "REFLECT":
        self._testPad(np_inputs, paddings, mode=mode,
                      constant_values=constant_values)
        if np_inputs.dtype == np.float32:
          self._testGradient(np_inputs, paddings, mode=mode,
                             constant_values=constant_values)

  @test_util.run_deprecated_v1
  def testInputDims(self):
    with self.session(use_gpu=True):
      with self.assertRaises(ValueError):
        array_ops.pad(array_ops.reshape(
            [1, 2], shape=[1, 2, 1, 1, 1, 1]),
                      array_ops.reshape(
                          [1, 2], shape=[1, 2]))

  @test_util.run_deprecated_v1
  def testPaddingsDim(self):
    with self.session(use_gpu=True):
      with self.assertRaises(ValueError):
        array_ops.pad(array_ops.reshape(
            [1, 2], shape=[1, 2]),
                      array_ops.reshape(
                          [1, 2], shape=[2]))

  @test_util.run_deprecated_v1
  def testPaddingsDim2(self):
    with self.session(use_gpu=True):
      with self.assertRaises(ValueError):
        array_ops.pad(array_ops.reshape(
            [1, 2], shape=[1, 2]),
                      array_ops.reshape(
                          [1, 2], shape=[2, 1]))

  @test_util.run_deprecated_v1
  def testPaddingsDim3(self):
    with self.session(use_gpu=True):
      with self.assertRaises(ValueError):
        array_ops.pad(array_ops.reshape(
            [1, 2], shape=[1, 2]),
                      array_ops.reshape(
                          [1, 2], shape=[1, 2]))

  @test_util.run_deprecated_v1
  def testPaddingsDim4(self):
    with self.session(use_gpu=True):
      with self.assertRaises(ValueError):
        array_ops.pad(array_ops.reshape(
            [1, 2], shape=[1, 2]),
                      array_ops.reshape(
                          [1, 2, 3, 4, 5, 6], shape=[3, 2]))

  @test_util.run_deprecated_v1
  def testPaddingsNonNegative(self):
    with self.session(use_gpu=True):
      with self.assertRaisesRegexp(ValueError, "must be non-negative"):
        array_ops.pad(constant_op.constant(
            [1], shape=[1]),
                      constant_op.constant(
                          [-1, 0], shape=[1, 2]))

  @test_util.run_deprecated_v1
  def testPaddingsNonNegative2(self):
    with self.session(use_gpu=True):
      with self.assertRaisesRegexp(ValueError, "must be non-negative"):
        array_ops.pad(constant_op.constant(
            [1], shape=[1]),
                      constant_op.constant(
                          [-1, 0], shape=[1, 2]))

  def testPaddingsMaximum(self):
    with self.session(use_gpu=True):
      with self.assertRaises(Exception):
        array_ops.pad(constant_op.constant(
            [1], shape=[2]),
                      constant_op.constant(
                          [2, 0], shape=[1, 2]),
                      mode="REFLECT").eval()
      with self.assertRaises(Exception):
        array_ops.pad(constant_op.constant(
            [1], shape=[2]),
                      constant_op.constant(
                          [0, 3], shape=[1, 2]),
                      mode="SYMMETRIC").eval()

  def testInvalid(self):
    with self.cached_session():
      x = [[1, 2, 3], [4, 5, 6]]
      with self.assertRaisesRegexp(ValueError, "Unknown padding mode"):
        array_ops.pad(x, [[1, 0], [2, 1]], mode="weird").eval()

  def testPaddingTypes(self):
    paddings = [[1, 0], [2, 3], [0, 2]]
    inputs = np.random.randint(-100, 100, (4, 4, 3)).astype(np.float32)
    for mode in ("CONSTANT", "REFLECT", "SYMMETRIC", "reflect", "symmetric",
                 "constant"):
      for padding_dtype in [dtypes.int32, dtypes.int64]:
        np_val = self._npPad(inputs,
                             paddings,
                             mode=mode,
                             constant_values=0)
        with self.cached_session(use_gpu=True):
          tf_val = array_ops.pad(inputs,
                                 constant_op.constant(paddings, padding_dtype),
                                 mode=mode,
                                 constant_values=0)
          out = self.evaluate(tf_val)
        self.assertAllEqual(np_val, out)
        self.assertShapeEqual(np_val, tf_val)

  def testIntTypes(self):
    # TODO(touts): Figure out why the padding tests do not work on GPU
    # for int types and rank > 2.
    for t in [np.int8, np.int32, np.int64]:
      self._testAll(
          np.random.randint(-100, 100, (4, 4, 3)).astype(t),
          [[1, 0], [2, 3], [0, 2]], 0)
      self._testAll(
          np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t),
          [[0, 0], [0, 0], [0, 0], [0, 0]], -123)

  @test_util.run_deprecated_v1
  def testFloatTypes(self):
    for t in [np.float32, np.float64]:
      self._testAll(np.random.rand(2, 5).astype(t), [[1, 0], [2, 0]], 0.0)
      self._testAll(np.random.rand(2, 3, 4).astype(t),
                    [[0, 0], [0, 0], [0, 0]], -1234.0)
      self._testAll(np.random.rand(0, 3, 4).astype(t),
                    [[0, 0], [2, 1], [2, 3]], 0.0)

  def testComplexTypes(self):
    for t in [np.complex64, np.complex128]:
      x = np.random.rand(2, 5).astype(t)
      self._testAll(x + 1j * x, [[1, 0], [2, 0]], 1234.0 - 1234.0j)
      x = np.random.rand(3, 2, 1, 1).astype(t)
      self._testAll(x + 1j * x, [[0, 0], [0, 0], [0, 0], [0, 0]], 0 + 0j)

  def testString(self):
    # Numpy does not support padding strings so we compare padding manually.
    x = ops.convert_to_tensor([["Hello", "World"],
                               ["Goodnight", "Moon"]])

    constant = array_ops.pad(x, [[1, 0], [0, 1]], mode="CONSTANT",
                             constant_values="PAD")
    reflect = array_ops.pad(x, [[1, 0], [0, 1]], mode="REFLECT",
                            constant_values="PAD")
    symmetric = array_ops.pad(x, [[1, 0], [0, 1]], mode="SYMMETRIC",
                              constant_values="PAD")
    with self.session(use_gpu=True):
      self.assertAllEqual(
          [[b"PAD", b"PAD", b"PAD"], [b"Hello", b"World", b"PAD"],
           [b"Goodnight", b"Moon", b"PAD"]], self.evaluate(constant))
      self.assertAllEqual([[b"Goodnight", b"Moon", b"Goodnight"],
                           [b"Hello", b"World", b"Hello"],
                           [b"Goodnight", b"Moon", b"Goodnight"]],
                          self.evaluate(reflect))
      self.assertAllEqual(
          [[b"Hello", b"World", b"World"], [b"Hello", b"World", b"World"],
           [b"Goodnight", b"Moon", b"Moon"]], self.evaluate(symmetric))

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # Unknown paddings shape.
    inp = constant_op.constant(0.0, shape=[4, 4, 4, 4])
    padded = array_ops.pad(inp, array_ops.placeholder(dtypes.int32))
    self.assertEqual([None, None, None, None], padded.get_shape().as_list())

    # Unknown input shape.
    inp = array_ops.placeholder(dtypes.float32)
    padded = array_ops.pad(inp, [[2, 2], [2, 2]])
    self.assertEqual([None, None], padded.get_shape().as_list())

    # Unknown input and paddings shape.
    inp = array_ops.placeholder(dtypes.float32)
    padded = array_ops.pad(inp, array_ops.placeholder(dtypes.int32))
    self.assertAllEqual(None, padded.get_shape().ndims)

  @test_util.run_deprecated_v1
  def testPartialShapeInformation(self):
    unknown = array_ops.placeholder(dtypes.int32)

    # Known input shape, partial unknown padding (one dimension).
    inp = constant_op.constant(0.0, shape=[4, 4])
    padded = array_ops.pad(inp, [[1, 2], unknown])
    self.assertEqual([7, None], padded.get_shape().as_list())

    # Known input shape, partial unknown padding (begin).
    inp = constant_op.constant(0.0, shape=[4, 4])
    padded = array_ops.pad(inp, [[unknown, 0], [1, 2]])
    self.assertEqual([None, 7], padded.get_shape().as_list())

    # Known input shape, partial unknown padding (end).
    inp = constant_op.constant(0.0, shape=[4, 4])
    padded = array_ops.pad(inp, [[1, 2], [0, unknown]])
    self.assertEqual([7, None], padded.get_shape().as_list())

    # Unknown input shape, partial unknown padding (one dimension).
    padded = array_ops.pad(unknown, [[1, 2], unknown])
    self.assertEqual([None, None], padded.get_shape().as_list())

    # Unknown input shape (rank known), partial unknown padding (one dimension).
    rank_known = array_ops.placeholder(dtypes.int32)
    rank_known.set_shape([None, None])
    padded = array_ops.pad(rank_known, [[1, 2], unknown])
    self.assertEqual([None, None], padded.get_shape().as_list())

    # Known input shape, partial unknown padding (begin), with constant begin.
    inp = constant_op.constant(0.0, shape=[4, 4])
    padded = array_ops.pad(inp, [[constant_op.constant(1, shape=[]), 2],
                                 [0, unknown]])
    self.assertEqual([7, None], padded.get_shape().as_list())

    # Known input shape, partial unknown padding (begin), with constant dim.
    inp = constant_op.constant(0.0, shape=[4, 4])
    padded = array_ops.pad(inp,
                           [constant_op.constant(1, shape=[2]), [0, unknown]])
    self.assertEqual([6, None], padded.get_shape().as_list())

    # Zero padding on a known dimension.
    inp = array_ops.placeholder(dtypes.int32, [None, None, 20])
    padded = array_ops.pad(inp, [[0, 0], [0, unknown], [0, 0]])
    self.assertEqual([None, None, 20], padded.get_shape().as_list())

  def testScalars(self):
    paddings = np.zeros((0, 2), dtype=np.int32)
    inp = np.asarray(7)
    with self.session(use_gpu=True):
      tf_val = array_ops.pad(inp, paddings)
      out = self.evaluate(tf_val)
    self.assertAllEqual(inp, out)
    self.assertShapeEqual(inp, tf_val)

  def testPadTypes(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      paddings = np.zeros((0, 2))
      inp = np.asarray(7)
      with self.cached_session(use_gpu=True):
        tf_val = array_ops.pad(inp, constant_op.constant(paddings, dtype=dtype))
        out = self.evaluate(tf_val)
      self.assertAllEqual(inp, out)
      self.assertShapeEqual(inp, tf_val)

  @test_util.run_deprecated_v1
  def testCollapseAdjacentNonPaddedDimensions(self):
    # pyformat: disable
    paddings_values = [[[0, 0], [0, 0], [0, 0], [0, 1]],
                       [[0, 0], [2, 3], [0, 0], [0, 0]],
                       [[0, 0], [0, 0], [0, 0], [0, 0]]]
    # pyformat: enable
    for paddings_value in paddings_values:
      for dtype in [dtypes.float32, dtypes.int32]:
        inp = constant_op.constant(1, shape=[8, 28, 28, 3], dtype=dtype)
        paddings = constant_op.constant(paddings_value, dtype=dtypes.int32)
        padded = array_ops.pad(inp, paddings)
        middle = array_ops.slice(padded, [row[0] for row in paddings_value],
                                 [dim.value for dim in inp.shape.dims])
        left = array_ops.slice(padded, [0, 0, 0, 0],
                               [row[0] for row in paddings_value])
        right = array_ops.slice(
            padded,
            [paddings_value[i][0] + inp.shape.dims[i].value for i in range(4)],
            [-1, -1, -1, -1])
        with self.cached_session(use_gpu=True):
          self.assertAllEqual(inp.eval(), self.evaluate(middle))
          self.assertAllEqual(
              np.zeros([row[0] for row in paddings_value]), self.evaluate(left))
          self.assertAllEqual(
              np.zeros([row[1] for row in paddings_value]),
              self.evaluate(right))


if __name__ == "__main__":
  test.main()
