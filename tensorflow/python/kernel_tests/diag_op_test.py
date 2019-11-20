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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.python.compat import compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


# LINT.IfChange
matrix_diag_v3_forward_compat_date = (2019, 12, 6)
# LINT.ThenChange(
#   //tensorflow/compiler/tests/matrix_diag_ops_test.py,
#   //tensorflow/python/ops/array_ops.py,
#   //tensorflow/python/ops/parallel_for/array_test.py
# )


default_v2_alignment = "LEFT_LEFT"
alignment_list = ["RIGHT_LEFT", "LEFT_RIGHT", "LEFT_LEFT", "RIGHT_RIGHT"]


def zip_to_first_list_length(a, b):
  if len(b) > len(a):
    return zip(a, b[:len(a)])
  return zip(a, b + [None] * (len(a) - len(b)))


def repack_diagonals(packed_diagonals,
                     diag_index,
                     num_rows,
                     num_cols,
                     align=None):
  # The original test cases are LEFT_LEFT aligned.
  if align == default_v2_alignment or align is None:
    return packed_diagonals

  align = align.split("_")
  d_lower, d_upper = diag_index
  batch_dims = packed_diagonals.ndim - (2 if d_lower < d_upper else 1)
  max_diag_len = packed_diagonals.shape[-1]
  index = (slice(None),) * batch_dims
  repacked_diagonals = np.zeros_like(packed_diagonals)

  # Aligns each diagonal row-by-row.
  for diag_index in range(d_lower, d_upper + 1):
    diag_len = min(num_rows + min(0, diag_index), num_cols - max(0, diag_index))
    row_index = d_upper - diag_index
    padding_len = max_diag_len - diag_len
    left_align = (diag_index >= 0 and
                  align[0] == "LEFT") or (diag_index <= 0 and
                                          align[1] == "LEFT")
    # Prepares index tuples.
    extra_dim = tuple() if d_lower == d_upper else (row_index,)
    packed_last_dim = (slice(None),) if left_align else (slice(0, diag_len, 1),)
    repacked_last_dim = (slice(None),) if left_align else (slice(
        padding_len, max_diag_len, 1),)
    packed_index = index + extra_dim + packed_last_dim
    repacked_index = index + extra_dim + repacked_last_dim

    # Repacks the diagonal.
    repacked_diagonals[repacked_index] = packed_diagonals[packed_index]
  return repacked_diagonals


def repack_diagonals_in_tests(tests, align=None):
  # The original test cases are LEFT_LEFT aligned.
  if align == default_v2_alignment or align is None:
    return tests

  new_tests = dict()
  # Loops through each case.
  for diag_index, (packed_diagonals, padded_diagonals) in tests.items():
    num_rows, num_cols = padded_diagonals.shape[-2:]
    repacked_diagonals = repack_diagonals(
        packed_diagonals, diag_index, num_rows, num_cols, align=align)
    new_tests[diag_index] = (repacked_diagonals, padded_diagonals)

  return new_tests


# Test cases shared by MatrixDiagV2, MatrixDiagPartV2, and MatrixSetDiagV2.
def square_cases(align=None):
  # pyformat: disable
  mat = np.array([[[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 1],
                   [3, 4, 5, 6, 7],
                   [8, 9, 1, 2, 3],
                   [4, 5, 6, 7, 8]],
                  [[9, 1, 2, 3, 4],
                   [5, 6, 7, 8, 9],
                   [1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 1],
                   [2, 3, 4, 5, 6]]])
  tests = dict()
  # tests[d_lower, d_upper] = (packed_diagonals, padded_diagonals)
  tests[-1, -1] = (np.array([[6, 4, 1, 7],
                             [5, 2, 8, 5]]),
                   np.array([[[0, 0, 0, 0, 0],
                              [6, 0, 0, 0, 0],
                              [0, 4, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 7, 0]],
                             [[0, 0, 0, 0, 0],
                              [5, 0, 0, 0, 0],
                              [0, 2, 0, 0, 0],
                              [0, 0, 8, 0, 0],
                              [0, 0, 0, 5, 0]]]))
  tests[-4, -3] = (np.array([[[8, 5],
                              [4, 0]],
                             [[6, 3],
                              [2, 0]]]),
                   np.array([[[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [8, 0, 0, 0, 0],
                              [4, 5, 0, 0, 0]],
                             [[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [6, 0, 0, 0, 0],
                              [2, 3, 0, 0, 0]]]))
  tests[-2, 1] = (np.array([[[2, 8, 6, 3, 0],
                             [1, 7, 5, 2, 8],
                             [6, 4, 1, 7, 0],
                             [3, 9, 6, 0, 0]],
                            [[1, 7, 4, 1, 0],
                             [9, 6, 3, 9, 6],
                             [5, 2, 8, 5, 0],
                             [1, 7, 4, 0, 0]]]),
                  np.array([[[1, 2, 0, 0, 0],
                             [6, 7, 8, 0, 0],
                             [3, 4, 5, 6, 0],
                             [0, 9, 1, 2, 3],
                             [0, 0, 6, 7, 8]],
                            [[9, 1, 0, 0, 0],
                             [5, 6, 7, 0, 0],
                             [1, 2, 3, 4, 0],
                             [0, 7, 8, 9, 1],
                             [0, 0, 4, 5, 6]]]))
  tests[2, 4] = (np.array([[[5, 0, 0],
                            [4, 1, 0],
                            [3, 9, 7]],
                           [[4, 0, 0],
                            [3, 9, 0],
                            [2, 8, 5]]]),
                 np.array([[[0, 0, 3, 4, 5],
                            [0, 0, 0, 9, 1],
                            [0, 0, 0, 0, 7],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                           [[0, 0, 2, 3, 4],
                            [0, 0, 0, 8, 9],
                            [0, 0, 0, 0, 5],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]]))
  # pyformat: enable
  return (mat, repack_diagonals_in_tests(tests, align))


def tall_cases(align=None):
  # pyformat: disable
  mat = np.array([[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [9, 8, 7],
                   [6, 5, 4]],
                  [[3, 2, 1],
                   [1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [9, 8, 7]]])
  tests = dict()
  tests[0, 0] = (np.array([[1, 5, 9],
                           [3, 2, 6]]),
                 np.array([[[1, 0, 0],
                            [0, 5, 0],
                            [0, 0, 9],
                            [0, 0, 0]],
                           [[3, 0, 0],
                            [0, 2, 0],
                            [0, 0, 6],
                            [0, 0, 0]]]))
  tests[-4, -3] = (np.array([[[9, 5],
                              [6, 0]],
                             [[7, 8],
                              [9, 0]]]),
                   np.array([[[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [9, 0, 0],
                              [6, 5, 0]],
                             [[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0],
                              [7, 0, 0],
                              [9, 8, 0]]]))
  tests[-2, -1] = (np.array([[[4, 8, 7],
                              [7, 8, 4]],
                             [[1, 5, 9],
                              [4, 8, 7]]]),
                   np.array([[[0, 0, 0],
                              [4, 0, 0],
                              [7, 8, 0],
                              [0, 8, 7],
                              [0, 0, 4]],
                             [[0, 0, 0],
                              [1, 0, 0],
                              [4, 5, 0],
                              [0, 8, 9],
                              [0, 0, 7]]]))
  tests[-2, 1] = (np.array([[[2, 6, 0],
                             [1, 5, 9],
                             [4, 8, 7],
                             [7, 8, 4]],
                            [[2, 3, 0],
                             [3, 2, 6],
                             [1, 5, 9],
                             [4, 8, 7]]]),
                  np.array([[[1, 2, 0],
                             [4, 5, 6],
                             [7, 8, 9],
                             [0, 8, 7],
                             [0, 0, 4]],
                            [[3, 2, 0],
                             [1, 2, 3],
                             [4, 5, 6],
                             [0, 8, 9],
                             [0, 0, 7]]]))
  tests[1, 2] = (np.array([[[3, 0],
                            [2, 6]],
                           [[1, 0],
                            [2, 3]]]),
                 np.array([[[0, 2, 3],
                            [0, 0, 6],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]],
                           [[0, 2, 1],
                            [0, 0, 3],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]]))
  # pyformat: enable
  return (mat, repack_diagonals_in_tests(tests, align))


def fat_cases(align=None):
  # pyformat: disable
  mat = np.array([[[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 1, 2, 3]],
                  [[4, 5, 6, 7],
                   [8, 9, 1, 2],
                   [3, 4, 5, 6]]])
  tests = dict()
  tests[2, 2] = (np.array([[3, 8],
                           [6, 2]]),
                 np.array([[[0, 0, 3, 0],
                            [0, 0, 0, 8],
                            [0, 0, 0, 0]],
                           [[0, 0, 6, 0],
                            [0, 0, 0, 2],
                            [0, 0, 0, 0]]]))
  tests[-2, 0] = (np.array([[[1, 6, 2],
                             [5, 1, 0],
                             [9, 0, 0]],
                            [[4, 9, 5],
                             [8, 4, 0],
                             [3, 0, 0]]]),
                  np.array([[[1, 0, 0, 0],
                             [5, 6, 0, 0],
                             [9, 1, 2, 0]],
                            [[4, 0, 0, 0],
                             [8, 9, 0, 0],
                             [3, 4, 5, 0]]]))
  tests[-1, 1] = (np.array([[[2, 7, 3],
                             [1, 6, 2],
                             [5, 1, 0]],
                            [[5, 1, 6],
                             [4, 9, 5],
                             [8, 4, 0]]]),
                  np.array([[[1, 2, 0, 0],
                             [5, 6, 7, 0],
                             [0, 1, 2, 3]],
                            [[4, 5, 0, 0],
                             [8, 9, 1, 0],
                             [0, 4, 5, 6]]]))
  tests[0, 3] = (np.array([[[4, 0, 0],
                            [3, 8, 0],
                            [2, 7, 3],
                            [1, 6, 2]],
                           [[7, 0, 0],
                            [6, 2, 0],
                            [5, 1, 6],
                            [4, 9, 5]]]),
                 np.array([[[1, 2, 3, 4],
                            [0, 6, 7, 8],
                            [0, 0, 2, 3]],
                           [[4, 5, 6, 7],
                            [0, 9, 1, 2],
                            [0, 0, 5, 6]]]))
  # pyformat: enable
  return (mat, repack_diagonals_in_tests(tests, align))


def all_tests(align=None):
  return [square_cases(align), tall_cases(align), fat_cases(align)]


class MatrixDiagTest(test.TestCase):

  def _moreCases(self, align=None):
    # Diagonal bands.
    # pyformat: disable
    vecs = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
                      [5, 6, 7, 8],
                      [9, 8, 7, 6]],
                     [[5, 4, 3, 2],
                      [1, 2, 3, 4],
                      [5, 6, 7, 8]]])
    tests = dict()
    tests[-3, -1] = (vecs,
                     np.array([[[0, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0],
                                [5, 2, 0, 0, 0],
                                [9, 6, 3, 0, 0],
                                [0, 8, 7, 4, 0]],
                               [[0, 0, 0, 0, 0],
                                [5, 0, 0, 0, 0],
                                [1, 4, 0, 0, 0],
                                [5, 2, 3, 0, 0],
                                [0, 6, 3, 2, 0]]]))
    tests[-1, 1] = (vecs,
                    np.array([[[5, 1, 0, 0],
                               [9, 6, 2, 0],
                               [0, 8, 7, 3],
                               [0, 0, 7, 8]],
                              [[1, 5, 0, 0],
                               [5, 2, 4, 0],
                               [0, 6, 3, 3],
                               [0, 0, 7, 4]]]))
    tests[2, 4] = (vecs,
                   np.array([[[0, 0, 9, 5, 1, 0],
                              [0, 0, 0, 8, 6, 2],
                              [0, 0, 0, 0, 7, 7],
                              [0, 0, 0, 0, 0, 6],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]],
                             [[0, 0, 5, 1, 5, 0],
                              [0, 0, 0, 6, 2, 4],
                              [0, 0, 0, 0, 7, 3],
                              [0, 0, 0, 0, 0, 8],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]]]))
    # pyformat: enable
    return (None, repack_diagonals_in_tests(tests, align))

  @test_util.run_deprecated_v1
  def testVector(self):
    with self.session(use_gpu=True):
      v = np.array([1.0, 2.0, 3.0])
      mat = np.diag(v)
      v_diag = array_ops.matrix_diag(v)
      self.assertEqual((3, 3), v_diag.get_shape())
      self.assertAllEqual(v_diag.eval(), mat)

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        # {Sub,Super}diagonals.
        for offset in [1, -2, 5]:
          mat = np.diag(v, offset)
          v_diag = array_ops.matrix_diag(v, k=offset)
          self.assertEqual(mat.shape, v_diag.get_shape())
          self.assertAllEqual(v_diag.eval(), mat)

        # Diagonal bands.
        for align in alignment_list:
          for _, tests in [self._moreCases(align), square_cases(align)]:
            for diags, (vecs, solution) in tests.items():
              v_diags = array_ops.matrix_diag(vecs[0], k=diags, align=align)
              self.assertEqual(v_diags.get_shape(), solution[0].shape)
              self.assertAllEqual(v_diags.eval(), solution[0])

  def _testVectorBatch(self, dtype):
    with self.cached_session(use_gpu=True):
      v_batch = np.array([[1.0, 0.0, 3.0], [4.0, 5.0, 6.0]]).astype(dtype)
      mat_batch = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
                            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0],
                             [0.0, 0.0, 6.0]]]).astype(dtype)
      v_batch_diag = array_ops.matrix_diag(v_batch)
      self.assertEqual((2, 3, 3), v_batch_diag.get_shape())
      self.assertAllEqual(v_batch_diag.eval(), mat_batch)

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        # {Sub,Super}diagonals.
        for offset in [1, -2, 5]:
          v_batch_diag = array_ops.matrix_diag(v_batch, k=offset)
          mats = [
              np.diag(v_batch[i], offset) for i in range(0, v_batch.shape[0])
          ]
          mat_batch = np.stack(mats, axis=0)
          self.assertEqual(mat_batch.shape, v_batch_diag.get_shape())
          self.assertAllEqual(v_batch_diag.eval(), mat_batch)

        # Diagonal bands with padding_value.
        for padding_value, align in zip_to_first_list_length([0, 555, -11],
                                                             alignment_list):
          for _, tests in [self._moreCases(align), square_cases(align)]:
            for diags, (vecs, solution) in tests.items():
              v_diags = array_ops.matrix_diag(
                  vecs.astype(dtype),
                  k=diags,
                  padding_value=padding_value,
                  align=align)
              mask = solution == 0
              solution = (solution + padding_value * mask).astype(dtype)
              self.assertEqual(v_diags.get_shape(), solution.shape)
              self.assertAllEqual(v_diags.eval(), solution)

  @test_util.run_deprecated_v1
  def testVectorBatch(self):
    self._testVectorBatch(np.float32)
    self._testVectorBatch(np.float64)
    self._testVectorBatch(np.int32)
    self._testVectorBatch(np.int64)
    self._testVectorBatch(np.bool)

  @test_util.run_deprecated_v1
  def testRectangularBatch(self):
    if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
      with self.cached_session(use_gpu=True):
        # Stores expected num_rows and num_cols (when the other is given).
        # expected[d_lower, d_upper] = (expected_num_rows, expected_num_cols)
        test_list = list()

        # Square cases:
        expected = {
            (-1, -1): (5, 4),
            (-4, -3): (5, 2),
            (-2, 1): (5, 5),
            (2, 4): (3, 5),
        }
        # Do not change alignment yet. Re-alignment needs to happen after the
        # solution shape is updated.
        test_list.append((expected, square_cases()))

        # More cases:
        expected = {(-3, -1): (5, 4), (-1, 1): (4, 4), (2, 4): (4, 6)}
        test_list.append((expected, self._moreCases()))

        # Tall cases
        expected = {
            (0, 0): (3, 3),
            (-4, -3): (5, 2),
            (-2, -1): (4, 3),
            (-2, 1): (3, 3),
            (1, 2): (2, 3)
        }
        test_list.append((expected, tall_cases()))

        # Fat cases
        expected = {
            (2, 2): (2, 4),
            (-2, 0): (3, 3),
            (-1, 1): (3, 3),
            (0, 3): (3, 3)
        }
        test_list.append((expected, fat_cases()))

        for padding_value, align in zip_to_first_list_length([0, 555, -11],
                                                             alignment_list):
          # Giving both num_rows and num_cols
          for _, tests in [tall_cases(align), fat_cases(align)]:
            for diags, (vecs, solution) in tests.items():
              v_diags = array_ops.matrix_diag(
                  vecs,
                  k=diags,
                  num_rows=solution.shape[-2],
                  num_cols=solution.shape[-1],
                  padding_value=padding_value,
                  align=align)
              mask = solution == 0
              solution = solution + padding_value * mask
              self.assertEqual(v_diags.get_shape(), solution.shape)
              self.assertAllEqual(v_diags.eval(), solution)

          # Giving just num_rows.
          for expected, (_, tests) in test_list:
            for diags, (_, new_num_cols) in expected.items():
              vecs, solution = tests[diags]
              solution = solution.take(indices=range(new_num_cols), axis=-1)
              # Repacks the diagonal input according to the new solution shape.
              vecs = repack_diagonals(
                  vecs, diags, solution.shape[-2], new_num_cols, align=align)
              v_diags = array_ops.matrix_diag(
                  vecs,
                  k=diags,
                  num_rows=solution.shape[-2],
                  padding_value=padding_value,
                  align=align)
              mask = solution == 0
              solution = solution + padding_value * mask
              self.assertEqual(v_diags.get_shape(), solution.shape)
              self.assertAllEqual(v_diags.eval(), solution)

          # Giving just num_cols.
          for expected, (_, tests) in test_list:
            for diags, (new_num_rows, _) in expected.items():
              vecs, solution = tests[diags]
              solution = solution.take(indices=range(new_num_rows), axis=-2)
              # Repacks the diagonal input according to the new solution shape.
              vecs = repack_diagonals(
                  vecs, diags, new_num_rows, solution.shape[-1], align=align)
              v_diags = array_ops.matrix_diag(
                  vecs,
                  k=diags,
                  num_cols=solution.shape[-1],
                  padding_value=padding_value,
                  align=align)
              mask = solution == 0
              solution = solution + padding_value * mask
              self.assertEqual(v_diags.get_shape(), solution.shape)
              self.assertAllEqual(v_diags.eval(), solution)

  @test_util.run_deprecated_v1
  def testInvalidShape(self):
    with self.assertRaisesRegexp(ValueError, "must be at least rank 1"):
      array_ops.matrix_diag(0)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123337890")  # Error messages differ
  def testInvalidShapeAtEval(self):
    with self.session(use_gpu=True):
      v = array_ops.placeholder(dtype=dtypes_lib.float32)
      with self.assertRaisesOpError("diagonal must be at least 1-dim"):
        array_ops.matrix_diag(v).eval(feed_dict={v: 0.0})

  @test_util.run_deprecated_v1
  def testGrad(self):
    shapes = ((3,), (7, 4))
    with self.session(use_gpu=True):
      for shape in shapes:
        x = constant_op.constant(np.random.rand(*shape), np.float32)
        y = array_ops.matrix_diag(x)
        error = gradient_checker.compute_gradient_error(x,
                                                        x.get_shape().as_list(),
                                                        y,
                                                        y.get_shape().as_list())
        self.assertLess(error, 1e-4)

    if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
      # {Sub,super}diagonals/band.
      tests = dict()  # tests[shape] = (d_lower, d_upper)
      tests[(3,)] = (-1, -1)
      tests[(7, 3, 4)] = (-1, 1)
      with self.session(use_gpu=True):
        for shape, diags in tests.items():
          x = constant_op.constant(np.random.rand(*shape), np.float32)
          for align in alignment_list:
            y = array_ops.matrix_diag(x, k=diags, align=align)
            error = gradient_checker.compute_gradient_error(
                x,
                x.get_shape().as_list(), y,
                y.get_shape().as_list())
            self.assertLess(error, 1e-4)


class MatrixSetDiagTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testSquare(self):
    with self.session(use_gpu=True):
      v = np.array([1.0, 2.0, 3.0])
      mat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
      mat_set_diag = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 1.0],
                               [1.0, 1.0, 3.0]])
      output = array_ops.matrix_set_diag(mat, v)
      self.assertEqual((3, 3), output.get_shape())
      self.assertAllEqual(mat_set_diag, self.evaluate(output))

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        # Diagonal bands.
        for align in alignment_list:
          _, tests = square_cases(align)
          for diags, (vecs, banded_mat) in tests.items():
            mask = banded_mat[0] == 0
            input_mat = np.random.randint(10, size=mask.shape)
            solution = input_mat * mask + banded_mat[0]
            output = array_ops.matrix_set_diag(
                input_mat, vecs[0], k=diags, align=align)
            self.assertEqual(output.get_shape(), solution.shape)
            self.assertAllEqual(output.eval(), solution)

  @test_util.run_deprecated_v1
  def testRectangular(self):
    with self.session(use_gpu=True):
      v = np.array([3.0, 4.0])
      mat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
      expected = np.array([[3.0, 1.0, 0.0], [1.0, 4.0, 1.0]])
      output = array_ops.matrix_set_diag(mat, v)
      self.assertEqual((2, 3), output.get_shape())
      self.assertAllEqual(expected, self.evaluate(output))

      v = np.array([3.0, 4.0])
      mat = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
      expected = np.array([[3.0, 1.0], [1.0, 4.0], [1.0, 1.0]])
      output = array_ops.matrix_set_diag(mat, v)
      self.assertEqual((3, 2), output.get_shape())
      self.assertAllEqual(expected, self.evaluate(output))

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        # Diagonal bands.
        for align in alignment_list:
          for _, tests in [tall_cases(align), fat_cases(align)]:
            for diags, (vecs, banded_mat) in tests.items():
              mask = banded_mat[0] == 0
              input_mat = np.random.randint(10, size=mask.shape)
              solution = input_mat * mask + banded_mat[0]
              output = array_ops.matrix_set_diag(
                  input_mat, vecs[0], k=diags, align=align)
              self.assertEqual(output.get_shape(), solution.shape)
              self.assertAllEqual(output.eval(), solution)

  def _testSquareBatch(self, dtype):
    with self.cached_session(use_gpu=True):
      v_batch = np.array([[-1.0, 0.0, -3.0], [-4.0, -5.0, -6.0]]).astype(dtype)
      mat_batch = np.array([[[1.0, 0.0, 3.0], [0.0, 2.0, 0.0], [1.0, 0.0, 3.0]],
                            [[4.0, 0.0, 4.0], [0.0, 5.0, 0.0],
                             [2.0, 0.0, 6.0]]]).astype(dtype)

      mat_set_diag_batch = np.array([[[-1.0, 0.0, 3.0], [0.0, 0.0, 0.0],
                                      [1.0, 0.0, -3.0]],
                                     [[-4.0, 0.0, 4.0], [0.0, -5.0, 0.0],
                                      [2.0, 0.0, -6.0]]]).astype(dtype)

      output = array_ops.matrix_set_diag(mat_batch, v_batch)
      self.assertEqual((2, 3, 3), output.get_shape())
      self.assertAllEqual(mat_set_diag_batch, self.evaluate(output))

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        # Diagonal bands.
        for align in alignment_list:
          _, tests = square_cases(align)
          for diags, (vecs, banded_mat) in tests.items():
            mask = banded_mat == 0
            input_mat = np.random.randint(10, size=mask.shape).astype(dtype)
            solution = (input_mat * mask + banded_mat).astype(dtype)
            output = array_ops.matrix_set_diag(
                input_mat, vecs.astype(dtype), k=diags, align=align)
            self.assertEqual(output.get_shape(), solution.shape)
            self.assertAllEqual(output.eval(), solution)

  @test_util.run_deprecated_v1
  def testSquareBatch(self):
    self._testSquareBatch(np.float32)
    self._testSquareBatch(np.float64)
    self._testSquareBatch(np.int32)
    self._testSquareBatch(np.int64)
    self._testSquareBatch(np.bool)

  @test_util.run_deprecated_v1
  def testRectangularBatch(self):
    with self.session(use_gpu=True):
      v_batch = np.array([[-1.0, -2.0], [-4.0, -5.0]])
      mat_batch = np.array([[[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]],
                            [[4.0, 0.0, 4.0], [0.0, 5.0, 0.0]]])

      mat_set_diag_batch = np.array([[[-1.0, 0.0, 3.0], [0.0, -2.0, 0.0]],
                                     [[-4.0, 0.0, 4.0], [0.0, -5.0, 0.0]]])
      output = array_ops.matrix_set_diag(mat_batch, v_batch)
      self.assertEqual((2, 2, 3), output.get_shape())
      self.assertAllEqual(mat_set_diag_batch, self.evaluate(output))

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        # Diagonal bands.
        for align in alignment_list:
          for _, tests in [tall_cases(align), fat_cases(align)]:
            for diags, pair in tests.items():
              vecs, banded_mat = pair
              mask = banded_mat == 0
              input_mat = np.random.randint(10, size=mask.shape)
              solution = input_mat * mask + banded_mat
              output = array_ops.matrix_set_diag(
                  input_mat, vecs, k=diags, align=align)
              self.assertEqual(output.get_shape(), solution.shape)
              self.assertAllEqual(output.eval(), solution)

  @test_util.run_deprecated_v1
  def testInvalidShape(self):
    with self.assertRaisesRegexp(ValueError, "must be at least rank 2"):
      array_ops.matrix_set_diag(0, [0])
    with self.assertRaisesRegexp(ValueError, "must be at least rank 1"):
      array_ops.matrix_set_diag([[0]], 0)

  @test_util.run_deprecated_v1
  def testInvalidShapeAtEval(self):
    with self.session(use_gpu=True):
      v = array_ops.placeholder(dtype=dtypes_lib.float32)
      with self.assertRaisesOpError("input must be at least 2-dim"):
        array_ops.matrix_set_diag(v, [v]).eval(feed_dict={v: 0.0})
      with self.assertRaisesOpError("diagonal must be at least 1-dim"):
        array_ops.matrix_set_diag([[v]], v).eval(feed_dict={v: 0.0})

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        d = array_ops.placeholder(dtype=dtypes_lib.float32)
        with self.assertRaisesOpError(
            "first dimensions of diagonal don't match"):
          array_ops.matrix_set_diag(v, d).eval(feed_dict={
              v: np.zeros((2, 3, 3)),
              d: np.ones((2, 4))
          })

  def _testGrad(self, input_shape, diag_shape, diags, align):
    with self.session(use_gpu=True):
      x = constant_op.constant(
          np.random.rand(*input_shape), dtype=dtypes_lib.float32)
      x_diag = constant_op.constant(
          np.random.rand(*diag_shape), dtype=dtypes_lib.float32)

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        y = array_ops.matrix_set_diag(x, x_diag, k=diags, align=align)
      else:
        y = array_ops.matrix_set_diag(x, x_diag)
      error_x = gradient_checker.compute_gradient_error(x,
                                                        x.get_shape().as_list(),
                                                        y,
                                                        y.get_shape().as_list())
      self.assertLess(error_x, 1e-4)
      error_x_diag = gradient_checker.compute_gradient_error(
          x_diag,
          x_diag.get_shape().as_list(), y,
          y.get_shape().as_list())
      self.assertLess(error_x_diag, 1e-4)

  @test_util.run_deprecated_v1
  def testGrad(self):
    input_shapes = [(3, 4, 4), (3, 3, 4), (3, 4, 3), (7, 4, 8, 8)]
    diag_bands = [(0, 0)]

    if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
      diag_bands.append((-1, 1))
    for input_shape, diags, align in itertools.product(input_shapes, diag_bands,
                                                       alignment_list):
      lower_diag_index, upper_diag_index = diags
      num_diags = upper_diag_index - lower_diag_index + 1
      num_diags_dim = () if num_diags == 1 else (num_diags,)
      diag_shape = input_shape[:-2] + num_diags_dim + (min(input_shape[-2:]),)
      self._testGrad(input_shape, diag_shape, diags, align)

  @test_util.run_deprecated_v1
  def testGradWithNoShapeInformation(self):
    with self.session(use_gpu=True) as sess:
      v = array_ops.placeholder(dtype=dtypes_lib.float32)
      mat = array_ops.placeholder(dtype=dtypes_lib.float32)
      grad_input = array_ops.placeholder(dtype=dtypes_lib.float32)
      output = array_ops.matrix_set_diag(mat, v)
      grads = gradients_impl.gradients(output, [mat, v], grad_ys=grad_input)
      grad_input_val = np.random.rand(3, 3).astype(np.float32)
      grad_vals = sess.run(
          grads,
          feed_dict={
              v: 2 * np.ones(3),
              mat: np.ones((3, 3)),
              grad_input: grad_input_val
          })
      self.assertAllEqual(np.diag(grad_input_val), grad_vals[1])
      self.assertAllEqual(grad_input_val - np.diag(np.diag(grad_input_val)),
                          grad_vals[0])


class MatrixDiagPartTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testSquare(self):
    with self.session(use_gpu=True):
      v = np.array([1.0, 2.0, 3.0])
      mat = np.diag(v)
      mat_diag = array_ops.matrix_diag_part(mat)
      self.assertEqual((3,), mat_diag.get_shape())
      self.assertAllEqual(mat_diag.eval(), v)

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        for offset in [-2, 3]:
          mat = np.diag(v, offset)
          mat_diag = array_ops.matrix_diag_part(mat, k=offset)
          self.assertEqual((3,), mat_diag.get_shape())
          self.assertAllEqual(mat_diag.eval(), v)

        # Diagonal bands.
        for align in alignment_list:
          mat, tests = square_cases(align)
          for diags, pair in tests.items():
            solution, _ = pair
            mat_diag = array_ops.matrix_diag_part(mat[0], k=diags, align=align)
            self.assertEqual(mat_diag.get_shape(), solution[0].shape)
            self.assertAllEqual(mat_diag.eval(), solution[0])

  @test_util.run_deprecated_v1
  def testRectangular(self):
    with self.session(use_gpu=True):
      mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      mat_diag = array_ops.matrix_diag_part(mat)
      self.assertAllEqual(mat_diag.eval(), np.array([1.0, 5.0]))
      mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      mat_diag = array_ops.matrix_diag_part(mat)
      self.assertAllEqual(mat_diag.eval(), np.array([1.0, 4.0]))

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        # Diagonal bands.
        for align in alignment_list:
          for mat, tests in [tall_cases(align), fat_cases(align)]:
            for diags, pair in tests.items():
              solution, _ = pair
              mat_diag = array_ops.matrix_diag_part(
                  mat[0], k=diags, align=align)
              self.assertEqual(mat_diag.get_shape(), solution[0].shape)
              self.assertAllEqual(mat_diag.eval(), solution[0])

  def _testSquareBatch(self, dtype):
    with self.cached_session(use_gpu=True):
      v_batch = np.array([[1.0, 0.0, 3.0], [4.0, 5.0, 6.0]]).astype(dtype)
      mat_batch = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
                            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0],
                             [0.0, 0.0, 6.0]]]).astype(dtype)
      self.assertEqual(mat_batch.shape, (2, 3, 3))
      mat_batch_diag = array_ops.matrix_diag_part(mat_batch)
      self.assertEqual((2, 3), mat_batch_diag.get_shape())
      self.assertAllEqual(mat_batch_diag.eval(), v_batch)

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        # Diagonal bands with padding_value.
        for padding_value, align in zip_to_first_list_length([0, 555, -11],
                                                             alignment_list):
          mat, tests = square_cases(align)
          for diags, pair in tests.items():
            solution, _ = pair
            mat_batch_diag = array_ops.matrix_diag_part(
                mat.astype(dtype),
                k=diags,
                padding_value=padding_value,
                align=align)
            mask = solution == 0
            solution = (solution + padding_value * mask).astype(dtype)
            self.assertEqual(mat_batch_diag.get_shape(), solution.shape)
            self.assertAllEqual(mat_batch_diag.eval(), solution)

  @test_util.run_deprecated_v1
  def testSquareBatch(self):
    self._testSquareBatch(np.float32)
    self._testSquareBatch(np.float64)
    self._testSquareBatch(np.int32)
    self._testSquareBatch(np.int64)
    self._testSquareBatch(np.bool)

  @test_util.run_deprecated_v1
  def testRectangularBatch(self):
    with self.session(use_gpu=True):
      v_batch = np.array([[1.0, 2.0], [4.0, 5.0]])
      mat_batch = np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]]])
      self.assertEqual(mat_batch.shape, (2, 2, 3))
      mat_batch_diag = array_ops.matrix_diag_part(mat_batch)
      self.assertEqual((2, 2), mat_batch_diag.get_shape())
      self.assertAllEqual(mat_batch_diag.eval(), v_batch)

      if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
        # Diagonal bands with padding_value and align.
        for padding_value, align in zip_to_first_list_length([0, 555, -11],
                                                             alignment_list):
          for mat, tests in [tall_cases(align), fat_cases(align)]:
            for diags, pair in tests.items():
              solution, _ = pair
              mat_batch_diag = array_ops.matrix_diag_part(
                  mat, k=diags, padding_value=padding_value, align=align)
              mask = solution == 0
              solution = solution + padding_value * mask
              self.assertEqual(mat_batch_diag.get_shape(), solution.shape)
              self.assertAllEqual(mat_batch_diag.eval(), solution)

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
      matrix = array_ops.placeholder(dtypes_lib.int32, shape=[None, None])
      result = array_ops.matrix_diag_part(matrix, k=-1)
      input_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      with self.session(use_gpu=True):
        result_eval = result.eval(feed_dict={matrix: input_matrix})
      self.assertAllEqual([4, 8], result_eval)

  @test_util.run_deprecated_v1
  def testInvalidShape(self):
    with self.assertRaisesRegexp(ValueError, "must be at least rank 2"):
      array_ops.matrix_diag_part(0)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123337890")  # Error messages differ
  def testInvalidShapeAtEval(self):
    with self.session(use_gpu=True):
      v = array_ops.placeholder(dtype=dtypes_lib.float32)
      with self.assertRaisesOpError("input must be at least 2-dim"):
        array_ops.matrix_diag_part(v).eval(feed_dict={v: 0.0})

  @test_util.run_deprecated_v1
  def testGrad(self):
    shapes = ((3, 3), (2, 3), (3, 2), (5, 3, 3))
    with self.session(use_gpu=True):
      for shape in shapes:
        x = constant_op.constant(np.random.rand(*shape), dtype=np.float32)
        y = array_ops.matrix_diag_part(x)
        error = gradient_checker.compute_gradient_error(x,
                                                        x.get_shape().as_list(),
                                                        y,
                                                        y.get_shape().as_list())
        self.assertLess(error, 1e-4)

    if compat.forward_compatible(*matrix_diag_v3_forward_compat_date):
      # {Sub,super}diagonals/band.
      tests = dict()  # tests[shape] = (d_lower, d_upper)
      tests[(3, 3)] = (-1, -1)
      tests[(7, 3, 4)] = (-1, 1)
      with self.session(use_gpu=True):
        for align in alignment_list:
          for shape, diags in tests.items():
            x = constant_op.constant(np.random.rand(*shape), np.float32)
            y = array_ops.matrix_diag_part(input=x, k=diags, align=align)
            error = gradient_checker.compute_gradient_error(
                x,
                x.get_shape().as_list(), y,
                y.get_shape().as_list())
            self.assertLess(error, 1e-4)


class DiagTest(test.TestCase):

  def _diagOp(self, diag, dtype, expected_ans, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.diag(ops.convert_to_tensor(diag.astype(dtype)))
      out = self.evaluate(tf_ans)
      tf_ans_inv = array_ops.diag_part(expected_ans)
      inv_out = self.evaluate(tf_ans_inv)
    self.assertAllClose(out, expected_ans)
    self.assertAllClose(inv_out, diag)
    self.assertShapeEqual(expected_ans, tf_ans)
    self.assertShapeEqual(diag, tf_ans_inv)

  def diagOp(self, diag, dtype, expected_ans):
    self._diagOp(diag, dtype, expected_ans, False)
    self._diagOp(diag, dtype, expected_ans, True)

  def testEmptyTensor(self):
    x = np.array([])
    expected_ans = np.empty([0, 0])
    self.diagOp(x, np.int32, expected_ans)

  def testRankOneIntTensor(self):
    x = np.array([1, 2, 3])
    expected_ans = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    self.diagOp(x, np.int32, expected_ans)
    self.diagOp(x, np.int64, expected_ans)

  def testRankOneFloatTensor(self):
    x = np.array([1.1, 2.2, 3.3])
    expected_ans = np.array([[1.1, 0, 0], [0, 2.2, 0], [0, 0, 3.3]])
    self.diagOp(x, np.float32, expected_ans)
    self.diagOp(x, np.float64, expected_ans)

  def testRankOneComplexTensor(self):
    for dtype in [np.complex64, np.complex128]:
      x = np.array([1.1 + 1.1j, 2.2 + 2.2j, 3.3 + 3.3j], dtype=dtype)
      expected_ans = np.array(
          [[1.1 + 1.1j, 0 + 0j, 0 + 0j], [0 + 0j, 2.2 + 2.2j, 0 + 0j],
           [0 + 0j, 0 + 0j, 3.3 + 3.3j]],
          dtype=dtype)
      self.diagOp(x, dtype, expected_ans)

  def testRankTwoIntTensor(self):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    expected_ans = np.array([[[[1, 0, 0], [0, 0, 0]], [[0, 2, 0], [0, 0, 0]],
                              [[0, 0, 3], [0, 0, 0]]],
                             [[[0, 0, 0], [4, 0, 0]], [[0, 0, 0], [0, 5, 0]],
                              [[0, 0, 0], [0, 0, 6]]]])
    self.diagOp(x, np.int32, expected_ans)
    self.diagOp(x, np.int64, expected_ans)

  def testRankTwoFloatTensor(self):
    x = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    expected_ans = np.array(
        [[[[1.1, 0, 0], [0, 0, 0]], [[0, 2.2, 0], [0, 0, 0]],
          [[0, 0, 3.3], [0, 0, 0]]], [[[0, 0, 0], [4.4, 0, 0]],
                                      [[0, 0, 0], [0, 5.5, 0]], [[0, 0, 0],
                                                                 [0, 0, 6.6]]]])
    self.diagOp(x, np.float32, expected_ans)
    self.diagOp(x, np.float64, expected_ans)

  def testRankTwoComplexTensor(self):
    for dtype in [np.complex64, np.complex128]:
      x = np.array(
          [[1.1 + 1.1j, 2.2 + 2.2j, 3.3 + 3.3j],
           [4.4 + 4.4j, 5.5 + 5.5j, 6.6 + 6.6j]],
          dtype=dtype)
      expected_ans = np.array(
          [[[[1.1 + 1.1j, 0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j, 0 + 0j]], [
              [0 + 0j, 2.2 + 2.2j, 0 + 0j], [0 + 0j, 0 + 0j, 0 + 0j]
          ], [[0 + 0j, 0 + 0j, 3.3 + 3.3j], [0 + 0j, 0 + 0j, 0 + 0j]]], [[
              [0 + 0j, 0 + 0j, 0 + 0j], [4.4 + 4.4j, 0 + 0j, 0 + 0j]
          ], [[0 + 0j, 0 + 0j, 0 + 0j], [0 + 0j, 5.5 + 5.5j, 0 + 0j]
             ], [[0 + 0j, 0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j, 6.6 + 6.6j]]]],
          dtype=dtype)
      self.diagOp(x, dtype, expected_ans)

  def testRankThreeFloatTensor(self):
    x = np.array([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]])
    expected_ans = np.array([[[[[[1.1, 0], [0, 0]], [[0, 0], [0, 0]]],
                               [[[0, 2.2], [0, 0]], [[0, 0], [0, 0]]]],
                              [[[[0, 0], [3.3, 0]], [[0, 0], [0, 0]]],
                               [[[0, 0], [0, 4.4]], [[0, 0], [0, 0]]]]],
                             [[[[[0, 0], [0, 0]], [[5.5, 0], [0, 0]]],
                               [[[0, 0], [0, 0]], [[0, 6.6], [0, 0]]]],
                              [[[[0, 0], [0, 0]], [[0, 0], [7.7, 0]]],
                               [[[0, 0], [0, 0]], [[0, 0], [0, 8.8]]]]]])
    self.diagOp(x, np.float32, expected_ans)
    self.diagOp(x, np.float64, expected_ans)

  def testRankThreeComplexTensor(self):
    for dtype in [np.complex64, np.complex128]:
      x = np.array(
          [[[1.1 + 1.1j, 2.2 + 2.2j], [3.3 + 3.3j, 4.4 + 4.4j]],
           [[5.5 + 5.5j, 6.6 + 6.6j], [7.7 + 7.7j, 8.8 + 8.8j]]],
          dtype=dtype)
      expected_ans = np.array(
          [[[[[[1.1 + 1.1j, 0 + 0j], [0 + 0j, 0 + 0j]], [[0 + 0j, 0 + 0j], [
              0 + 0j, 0 + 0j
          ]]], [[[0 + 0j, 2.2 + 2.2j], [0 + 0j, 0 + 0j]], [[0 + 0j, 0 + 0j], [
              0 + 0j, 0 + 0j
          ]]]], [[[[0 + 0j, 0 + 0j], [3.3 + 3.3j, 0 + 0j]], [[0 + 0j, 0 + 0j], [
              0 + 0j, 0 + 0j
          ]]], [[[0 + 0j, 0 + 0j], [0 + 0j, 4.4 + 4.4j]], [[0 + 0j, 0 + 0j], [
              0 + 0j, 0 + 0j
          ]]]]], [[[[[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]], [
              [5.5 + 5.5j, 0 + 0j], [0 + 0j, 0 + 0j]
          ]], [[[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]], [[0 + 0j, 6.6 + 6.6j], [
              0 + 0j, 0 + 0j
          ]]]], [[[[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]], [[0 + 0j, 0 + 0j], [
              7.7 + 7.7j, 0 + 0j
          ]]], [[[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]],
                [[0 + 0j, 0 + 0j], [0 + 0j, 8.8 + 8.8j]]]]]],
          dtype=dtype)
      self.diagOp(x, dtype, expected_ans)

  def testRankFourNumberTensor(self):
    for dtype in [np.float32, np.float64, np.int64, np.int32]:
      # Input with shape [2, 1, 2, 3]
      x = np.array(
          [[[[1, 2, 3], [4, 5, 6]]], [[[7, 8, 9], [10, 11, 12]]]], dtype=dtype)
      # Output with shape [2, 1, 2, 3, 2, 1, 2, 3]
      expected_ans = np.array(
          [[[[[[[[1, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]], [
              [[[0, 2, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]
          ], [[[[0, 0, 3], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]]], [[
              [[[0, 0, 0], [4, 0, 0]]], [[[0, 0, 0], [0, 0, 0]]]
          ], [[[[0, 0, 0], [0, 5, 0]]], [[[0, 0, 0], [0, 0, 0]]]], [
              [[[0, 0, 0], [0, 0, 6]]], [[[0, 0, 0], [0, 0, 0]]]
          ]]]], [[[[[[[0, 0, 0], [0, 0, 0]]], [[[7, 0, 0], [0, 0, 0]]]], [
              [[[0, 0, 0], [0, 0, 0]]], [[[0, 8, 0], [0, 0, 0]]]
          ], [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 9], [0, 0, 0]]]]], [[
              [[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [10, 0, 0]]]
          ], [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 11, 0]]]
             ], [[[[0, 0, 0], [0, 0, 0]]], [[[0, 0, 0], [0, 0, 12]]]]]]]],
          dtype=dtype)
      self.diagOp(x, dtype, expected_ans)

  @test_util.run_deprecated_v1
  def testInvalidRank(self):
    with self.assertRaisesRegexp(ValueError, "must be at least rank 1"):
      array_ops.diag(0.0)


class DiagPartOpTest(test.TestCase):

  def setUp(self):
    np.random.seed(0)

  def _diagPartOp(self, tensor, dtype, expected_ans, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      tensor = ops.convert_to_tensor(tensor.astype(dtype))
      tf_ans_inv = array_ops.diag_part(tensor)
      inv_out = self.evaluate(tf_ans_inv)
    self.assertAllClose(inv_out, expected_ans)
    self.assertShapeEqual(expected_ans, tf_ans_inv)

  def diagPartOp(self, tensor, dtype, expected_ans):
    self._diagPartOp(tensor, dtype, expected_ans, False)
    self._diagPartOp(tensor, dtype, expected_ans, True)

  def testRankTwoFloatTensor(self):
    x = np.random.rand(3, 3)
    i = np.arange(3)
    expected_ans = x[i, i]
    self.diagPartOp(x, np.float32, expected_ans)
    self.diagPartOp(x, np.float64, expected_ans)

  def testRankFourFloatTensorUnknownShape(self):
    x = np.random.rand(3, 3)
    i = np.arange(3)
    expected_ans = x[i, i]
    for shape in None, (None, 3), (3, None):
      with self.cached_session(use_gpu=False):
        t = ops.convert_to_tensor(x.astype(np.float32))
        t.set_shape(shape)
        tf_ans = array_ops.diag_part(t)
        out = self.evaluate(tf_ans)
      self.assertAllClose(out, expected_ans)
      self.assertShapeEqual(expected_ans, tf_ans)

  def testRankFourFloatTensor(self):
    x = np.random.rand(2, 3, 2, 3)
    i = np.arange(2)[:, None]
    j = np.arange(3)
    expected_ans = x[i, j, i, j]
    self.diagPartOp(x, np.float32, expected_ans)
    self.diagPartOp(x, np.float64, expected_ans)

  def testRankSixFloatTensor(self):
    x = np.random.rand(2, 2, 2, 2, 2, 2)
    i = np.arange(2)[:, None, None]
    j = np.arange(2)[:, None]
    k = np.arange(2)
    expected_ans = x[i, j, k, i, j, k]
    self.diagPartOp(x, np.float32, expected_ans)
    self.diagPartOp(x, np.float64, expected_ans)

  def testRankEightComplexTensor(self):
    x = np.random.rand(2, 2, 2, 3, 2, 2, 2, 3)
    i = np.arange(2)[:, None, None, None]
    j = np.arange(2)[:, None, None]
    k = np.arange(2)[:, None]
    l = np.arange(3)
    expected_ans = x[i, j, k, l, i, j, k, l]
    self.diagPartOp(x, np.complex64, expected_ans)
    self.diagPartOp(x, np.complex128, expected_ans)

  @test_util.run_deprecated_v1
  def testOddRank(self):
    w = np.random.rand(2)
    x = np.random.rand(2, 2, 2)
    self.assertRaises(ValueError, self.diagPartOp, w, np.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, x, np.float32, 0)
    with self.assertRaises(ValueError):
      array_ops.diag_part(0.0)

  @test_util.run_deprecated_v1
  def testUnevenDimensions(self):
    w = np.random.rand(2, 5)
    x = np.random.rand(2, 1, 2, 3)
    self.assertRaises(ValueError, self.diagPartOp, w, np.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, x, np.float32, 0)


class DiagGradOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testDiagGrad(self):
    np.random.seed(0)
    shapes = ((3,), (3, 3), (3, 3, 3))
    dtypes = (dtypes_lib.float32, dtypes_lib.float64)
    with self.session(use_gpu=False):
      errors = []
      for shape in shapes:
        for dtype in dtypes:
          x1 = constant_op.constant(np.random.rand(*shape), dtype=dtype)
          y = array_ops.diag(x1)
          error = gradient_checker.compute_gradient_error(
              x1,
              x1.get_shape().as_list(), y,
              y.get_shape().as_list())
          tf_logging.info("error = %f", error)
          self.assertLess(error, 1e-4)


class DiagGradPartOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testDiagPartGrad(self):
    np.random.seed(0)
    shapes = ((3, 3), (3, 3, 3, 3))
    dtypes = (dtypes_lib.float32, dtypes_lib.float64)
    with self.session(use_gpu=False):
      errors = []
      for shape in shapes:
        for dtype in dtypes:
          x1 = constant_op.constant(np.random.rand(*shape), dtype=dtype)
          y = array_ops.diag_part(x1)
          error = gradient_checker.compute_gradient_error(
              x1,
              x1.get_shape().as_list(), y,
              y.get_shape().as_list())
          tf_logging.info("error = %f", error)
          self.assertLess(error, 1e-4)


if __name__ == "__main__":
  test.main()
