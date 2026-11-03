# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for XLA matrix diag ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import googletest

default_v2_alignment = "LEFT_LEFT"
alignment_list = ["RIGHT_LEFT", "LEFT_RIGHT"]


def zip_to_first_list_length(a, b):
  if len(b) > len(a):
    return zip(a, b[:len(a)])
  return zip(a, b + [None] * (len(a) - len(b)))


# Routines to convert test cases to have diagonals in a specified alignment.
# Copied from //third_party/tensorflow/python/kernel_tests/array_ops/
# diag_op_test.py
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
# Copied from //third_party/tensorflow/python/kernel_tests/array_ops/
# diag_op_test.py
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
  # tests[d_lower, d_upper] = (compact_diagonals, padded_diagonals)
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
  # tests[d_lower, d_upper] = (compact_diagonals, padded_diagonals)
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
  # tests[d_lower, d_upper] = (compact_diagonals, padded_diagonals)
  tests[0, 0] = (np.array([[1, 6, 2],
                           [4, 9, 5]]),
                 np.array([[[1, 0, 0, 0],
                            [0, 6, 0, 0],
                            [0, 0, 2, 0]],
                           [[4, 0, 0, 0],
                            [0, 9, 0, 0],
                            [0, 0, 5, 0]]]))
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


class MatrixDiagTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self,
                                     params,
                                     solution,
                                     high_level=True,
                                     rtol=1e-3,
                                     atol=1e-5):
    """Verifies that matrix_diag produces `solution` when fed `params`.

    Args:
      params: dictionary containing input parameters to matrix_diag.
      solution: numpy array representing the expected output of matrix_diag.
      high_level: call high_level matrix_diag
      rtol: relative tolerance for equality test.
      atol: absolute tolerance for equality test.
    """
    diagonal = params["diagonal"]
    with self.session() as session:
      for dtype in self.numeric_types - {np.int8, np.uint8}:
        expected = solution.astype(dtype)
        with self.device_scope():
          params["diagonal"] = array_ops.placeholder(
              dtype, diagonal.shape, name="diagonal")
          if high_level:
            # wraps gen_array_ops.matrix_diag_v3
            output = array_ops.matrix_diag(**params)
          else:
            # TODO(b/201086188): Remove this case once MatrixDiag V1 is removed.
            output = gen_array_ops.matrix_diag(**params)
        result = session.run(output,
                             {params["diagonal"]: diagonal.astype(dtype)})
        self.assertEqual(output.dtype, expected.dtype)
        self.assertAllCloseAccordingToType(
            expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03)

  # Generic tests applicable to both v1 and v2 ops.
  # Originally from unary_ops_tests.py.
  def _testV1Level(self, high_level):
    # pyformat: disable
    vecs1 = np.array([[1, 2],
                      [3, 4]])
    solution1 = np.array([[[1, 0], [0, 2]],
                          [[3, 0], [0, 4]]])
    vecs2 = np.array([1, 2, 3, 4])
    solution2 = np.array([[1, 0, 0, 0],
                          [0, 2, 0, 0],
                          [0, 0, 3, 0],
                          [0, 0, 0, 4]])
    vecs3 = np.array([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7,  8,  9],  # pylint: disable=bad-whitespace
                       [10, 11, 12]]])
    solution3 = np.array([[[[1, 0, 0],
                            [0, 2, 0],
                            [0, 0, 3]],
                           [[4, 0, 0],
                            [0, 5, 0],
                            [0, 0, 6]]],
                          [[[7, 0, 0],
                            [0, 8, 0],
                            [0, 0, 9]],
                           [[10, 0, 0],
                            [0, 11, 0],
                            [0, 0, 12]]]])
    # pyformat: enable
    self._assertOpOutputMatchesExpected({"diagonal": vecs1}, solution1,
                                        high_level)
    self._assertOpOutputMatchesExpected({"diagonal": vecs2}, solution2,
                                        high_level)
    self._assertOpOutputMatchesExpected({"diagonal": vecs3}, solution3,
                                        high_level)

  def testV1(self):
    self._testV1Level(True)

  def testV1LowLevel(self):
    self._testV1Level(False)

  # From here onwards are v2-only tests.
  def testSquare(self):
    for align in alignment_list:
      for _, tests in [square_cases(align)]:
        for diag_index, (vecs, solution) in tests.items():
          params = {"diagonal": vecs[0], "k": diag_index, "align": align}
          self._assertOpOutputMatchesExpected(params, solution[0])

  def testSquareBatch(self):
    for align in alignment_list:
      for _, tests in [square_cases(align)]:
        for diag_index, (vecs, solution) in tests.items():
          params = {"diagonal": vecs, "k": diag_index, "align": align}
          self._assertOpOutputMatchesExpected(params, solution)

  def testRectangularBatch(self):
    # Stores expected num_rows and num_cols (when the other is given).
    # expected[(d_lower, d_upper)] = (expected_num_rows, expected_num_cols)
    test_list = list()

    # Do not align the test cases here. Re-alignment needs to happen after the
    # solution shape is updated.
    # Square cases:
    expected = {
        (-1, -1): (5, 4),
        (-4, -3): (5, 2),
        (-2, 1): (5, 5),
        (2, 4): (3, 5),
    }
    test_list.append((expected, square_cases()))

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

    # Giving both num_rows and num_cols
    align = alignment_list[0]
    for _, tests in [tall_cases(align), fat_cases(align)]:
      for diag_index, (vecs, solution) in tests.items():
        self._assertOpOutputMatchesExpected(
            {
                "diagonal": vecs,
                "k": diag_index,
                "num_rows": solution.shape[-2],
                "num_cols": solution.shape[-1],
                "align": align
            }, solution)

    # We go through each alignment in a round-robin manner.
    align_index = 0

    # Giving just num_rows or num_cols.
    for expected, (_, tests) in test_list:
      for diag_index, (new_num_rows, new_num_cols) in expected.items():
        align = alignment_list[align_index]
        align_index = (align_index + 1) % len(alignment_list)
        vecs, solution = tests[diag_index]
        solution_given_num_rows = solution.take(
            indices=range(new_num_cols), axis=-1)
        # Repacks the diagonal input according to the new solution shape.
        vecs_given_num_rows = repack_diagonals(
            vecs,
            diag_index,
            solution_given_num_rows.shape[-2],
            new_num_cols,
            align=align)
        self._assertOpOutputMatchesExpected(
            {
                "diagonal": vecs_given_num_rows,
                "k": diag_index,
                "num_rows": solution_given_num_rows.shape[-2],
                "align": align
            }, solution_given_num_rows)
        solution_given_num_cols = solution.take(
            indices=range(new_num_rows), axis=-2)
        # Repacks the diagonal input according to the new solution shape.
        vecs_given_num_cols = repack_diagonals(
            vecs,
            diag_index,
            new_num_rows,
            solution_given_num_cols.shape[-1],
            align=align)
        self._assertOpOutputMatchesExpected(
            {
                "diagonal": vecs_given_num_cols,
                "k": diag_index,
                "num_cols": solution_given_num_cols.shape[-1],
                "align": align
            }, solution_given_num_cols)

  def testPadding(self):
    for padding_value, align in zip_to_first_list_length([555, -11],
                                                         alignment_list):
      for _, tests in all_tests(align):
        for diag_index, (vecs, solution) in tests.items():
          mask = (solution == 0)
          solution = solution + (mask * padding_value)
          self._assertOpOutputMatchesExpected(
              {
                  "diagonal": vecs,
                  "k": diag_index,
                  "num_rows": solution.shape[-2],
                  "num_cols": solution.shape[-1],
                  "padding_value": padding_value,
                  "align": align
              }, solution)

  def _matrixDiagOfSelected(
      self, x, mask=(True, False, True, False, False), **params
  ):
    # Takes the entries of x's last dimension where mask is true, which gives
    # the diagonals a length only known at run time, and builds matrices from
    # them.
    @def_function.function(jit_compile=True)
    def f(x, mask):
      diagonal = array_ops.gather(
          x, array_ops.reshape(array_ops.where(mask), [-1]), axis=-1
      )
      return array_ops.matrix_diag(diagonal, **params)

    x = np.array(x, dtype=np.int32)
    with self.session() as session:
      with self.device_scope():
        x_ph = array_ops.placeholder(dtypes.int32, x.shape)
        mask_ph = array_ops.placeholder(dtypes.bool, [x.shape[-1]])
        output = f(x_ph, mask_ph)
      return session.run(output, {x_ph: x, mask_ph: mask})

  def testDynamicDiagonalLength(self):
    if "TPU" in self.device:
      self.skipTest("Dynamic diagonal lengths are not supported on TPU.")
    # Regression test for GitHub issue 128954: the output's matrix dimensions
    # took the bound of the diagonals' length instead of its value. The mask
    # selects 2 of 5 entries, so the diagonals have length 2 and bound 5. (XLA
    # turns a gather from 4 or fewer entries into selects, which XLA:GPU
    # doesn't support with dynamic shapes.)
    x = [1, 5, 2, 7, 3]
    self.assertAllEqual(self._matrixDiagOfSelected(x), [[1, 0], [0, 2]])
    self.assertAllEqual(
        self._matrixDiagOfSelected(x, k=1), [[0, 1, 0], [0, 0, 2], [0, 0, 0]]
    )
    # A dimension set by num_rows keeps that size.
    self.assertAllEqual(
        self._matrixDiagOfSelected(x, k=1, num_rows=5, align="LEFT_RIGHT"),
        [[0, 1, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    )
    # A size set by num_rows or num_cols that the actual length fits but the
    # bound doesn't.
    self.assertAllEqual(
        self._matrixDiagOfSelected(x, k=-1, num_rows=5, align="LEFT_RIGHT"),
        [[0, 0], [1, 0], [0, 2], [0, 0], [0, 0]],
    )
    self.assertAllEqual(
        self._matrixDiagOfSelected(x, k=1, num_cols=5),
        [[0, 1, 0, 0, 0], [0, 0, 2, 0, 0]],
    )
    self.assertAllEqual(
        self._matrixDiagOfSelected([[1, 5, 2, 7, 3], [4, 6, 8, 9, 0]]),
        [[[1, 0], [0, 2]], [[4, 0], [0, 8]]],
    )
    self.assertAllEqual(
        self._matrixDiagOfSelected(
            [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]],
            k=(-1, 1),
        ),
        [[20, 12], [30, 22]],
    )
    # A right-aligned diagonal in a band with num_rows set: the subdiagonal is
    # shorter than the bound but as long as the actual length.
    self.assertAllEqual(
        self._matrixDiagOfSelected(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            k=(-1, 0),
            num_rows=5,
            align="LEFT_RIGHT",
        ),
        [[1, 0], [6, 3], [0, 8], [0, 0], [0, 0]],
    )
    # The same with fewer rows than the bound needs: in the first, the
    # subdiagonal is padded at the bound but not at the actual length, and in
    # the second, it's padded at both.
    self.assertAllEqual(
        self._matrixDiagOfSelected(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            k=(-1, 0),
            num_rows=3,
            align="LEFT_RIGHT",
        ),
        [[1, 0], [6, 3], [0, 8]],
    )
    self.assertAllEqual(
        self._matrixDiagOfSelected(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            k=(-1, 0),
            num_rows=2,
            align="LEFT_RIGHT",
        ),
        [[1, 0], [8, 3]],
    )
    # Diagonals of length 0. In the band, diagonals 1 and -1 then come out with
    # a negative length at the actual sizes.
    self.assertAllEqual(
        self._matrixDiagOfSelected(x, mask=[False] * 5), np.zeros([0, 0])
    )
    self.assertAllEqual(
        self._matrixDiagOfSelected(
            [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]],
            mask=[False] * 5,
            k=(-1, 1),
            align="RIGHT_RIGHT",
        ),
        np.zeros([0, 0]),
    )

  def testBandFromVector(self):
    # A band of diagonals needs them stacked in a dimension that a vector
    # doesn't have, and looking for it aborted the process. array_ops'
    # matrix_diag rejects a vector of a known rank itself, so call the op.
    with self.session() as session:
      with self.device_scope():
        diagonal = array_ops.placeholder(dtypes.int32, [3])
        output = gen_array_ops.matrix_diag_v3(
            diagonal=diagonal,
            k=[-1, 1],
            num_rows=-1,
            num_cols=-1,
            padding_value=0,
            align="RIGHT_LEFT",
        )
      with self.assertRaisesOpError("The number of diagonals provided"):
        session.run(output, {diagonal: [1, 2, 3]})


class MatrixSetDiagTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self,
                                     params,
                                     solution,
                                     high_level=True,
                                     rtol=1e-3,
                                     atol=1e-5):
    """Verifies that matrix_set_diag produces `solution` when fed `params`.

    Args:
      params: dictionary containing input parameters to matrix_set_diag.
      solution: numpy array representing the expected output of matrix_set_diag.
      high_level: call high_level matrix_set_diag
      rtol: relative tolerance for equality test.
      atol: absolute tolerance for equality test.
    """
    input = params["input"]  # pylint: disable=redefined-builtin
    diagonal = params["diagonal"]
    with self.session() as session:
      for dtype in self.numeric_types - {np.int8, np.uint8}:
        expected = solution.astype(dtype)
        with self.device_scope():
          params["input"] = array_ops.placeholder(
              dtype, input.shape, name="input")
          params["diagonal"] = array_ops.placeholder(
              dtype, diagonal.shape, name="diagonal")
          if high_level:
            # wraps gen_array_ops.matrix_set_diag_v3
            output = array_ops.matrix_set_diag(**params)
          else:
            # TODO(b/201086188): Remove this case once MatrixDiag V1 is removed.
            output = gen_array_ops.matrix_set_diag(**params)
        result = session.run(
            output, {
                params["input"]: input.astype(dtype),
                params["diagonal"]: diagonal.astype(dtype)
            })
        self.assertEqual(output.dtype, expected.dtype)
        self.assertAllCloseAccordingToType(
            expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03)

  # Generic tests applicable to both v1 and v2 ops.
  # Originally from binary_ops_tests.py.
  def _testV1Level(self, high_level):
    test_cases = list()

    # pyformat: disable
    # pylint: disable=bad-whitespace
    # Square cases.
    input = np.array([[0, 1, 0],  # pylint: disable=redefined-builtin
                      [1, 0, 1],
                      [1, 1, 1]])
    diag = np.array([1, 2, 3])
    solution = np.array([[1, 1, 0],
                         [1, 2, 1],
                         [1, 1, 3]])
    test_cases.append(({"input": input, "diagonal": diag}, solution))

    input = np.array([[[1, 0, 3],
                       [0, 2, 0],
                       [1, 0, 3]],
                      [[4, 0, 4],
                       [0, 5, 0],
                       [2, 0, 6]]])
    diag = np.array([[-1,  0, -3],
                     [-4, -5, -6]])
    solution = np.array([[[-1, 0,  3],
                          [ 0, 0,  0],
                          [ 1, 0, -3]],
                         [[-4,  0,  4],
                          [ 0, -5,  0],
                          [ 2,  0, -6]]])
    test_cases.append(({"input": input, "diagonal": diag}, solution))

    # Rectangular cases.
    input = np.array([[0, 1, 0],
                      [1, 0, 1]])
    diag = np.array([3, 4])
    solution = np.array([[3, 1, 0],
                         [1, 4, 1]])
    test_cases.append(({"input": input, "diagonal": diag}, solution))

    input = np.array([[0, 1],
                      [1, 0],
                      [1, 1]])
    diag = np.array([3, 4])
    solution = np.array([[3, 1],
                         [1, 4],
                         [1, 1]])
    test_cases.append(({"input": input, "diagonal": diag}, solution))

    input = np.array([[[1, 0, 3],
                       [0, 2, 0]],
                      [[4, 0, 4],
                       [0, 5, 0]]])
    diag = np.array([[-1, -2], [-4, -5]])
    solution = np.array([[[-1,  0, 3],
                          [ 0, -2, 0]],
                         [[-4,  0, 4],
                          [ 0, -5, 0]]])
    test_cases.append(({"input": input, "diagonal": diag}, solution))
    # pylint: enable=bad-whitespace
    # pyformat: enable

    for test in test_cases:
      self._assertOpOutputMatchesExpected(test[0], test[1], high_level)

  def testV1(self):
    self._testV1Level(True)

  def testV1LowLevel(self):
    self._testV1Level(False)

  # From here onwards are v2-only tests.
  def testSingleMatrix(self):
    for align in alignment_list:
      for _, tests in all_tests(align):
        for diag_index, (vecs, banded_mat) in tests.items():
          mask = (banded_mat[0] == 0)
          input_mat = np.random.randint(10, size=mask.shape)
          solution = input_mat * mask + banded_mat[0]
          self._assertOpOutputMatchesExpected(
              {
                  "input": input_mat,
                  "diagonal": vecs[0],
                  "k": diag_index,
                  "align": align
              }, solution)

  def testBatch(self):
    for align in alignment_list:
      for _, tests in all_tests(align):
        for diag_index, (vecs, banded_mat) in tests.items():
          mask = (banded_mat == 0)
          input_mat = np.random.randint(10, size=mask.shape)
          solution = input_mat * mask + banded_mat
          self._assertOpOutputMatchesExpected(
              {
                  "input": input_mat,
                  "diagonal": vecs,
                  "k": diag_index,
                  "align": align
              }, solution)

  def _setDiagOfSelected(self, x, diagonal, masks, diag_mask, **params):
    if "TPU" in self.device:
      self.skipTest(
          "Dynamic shape execution on matrix_set_diag is not supported on TPU."
      )
    # masks maps axes of x to masks of the entries to keep along them, which
    # gives those dimensions a size only known at run time, and diag_mask does
    # the same for the diagonal's last dimension.
    axes = sorted(masks)

    @def_function.function(jit_compile=True)
    def f(x, diagonal, diag_mask, *x_masks):
      for axis, mask in zip(axes, x_masks):
        x = array_ops.gather(
            x, array_ops.reshape(array_ops.where(mask), [-1]), axis=axis
        )
      diagonal = array_ops.gather(
          diagonal, array_ops.reshape(array_ops.where(diag_mask), [-1]), axis=-1
      )
      return array_ops.matrix_set_diag(x, diagonal, **params)

    x = np.array(x, dtype=np.int32)
    diagonal = np.array(diagonal, dtype=np.int32)
    with self.session() as session:
      with self.device_scope():
        x_ph = array_ops.placeholder(dtypes.int32, x.shape)
        # MatrixSetDiagV3's shape function rejects a band of diagonals with
        # an unknown length while tracing (GitHub issue 129061), unless their
        # rank is unknown too.
        diagonal_ph = array_ops.placeholder(dtypes.int32)
        diag_mask_ph = array_ops.placeholder(dtypes.bool, [len(diag_mask)])
        mask_phs = [
            array_ops.placeholder(dtypes.bool, [x.shape[axis]]) for axis in axes
        ]
        output = f(x_ph, diagonal_ph, diag_mask_ph, *mask_phs)
      feed_dict = {x_ph: x, diagonal_ph: diagonal, diag_mask_ph: diag_mask}
      feed_dict.update({ph: masks[axis] for ph, axis in zip(mask_phs, axes)})
      return session.run(output, feed_dict)

  def testDynamicDimensions(self):
    # Regression test for GitHub issue 129054: a matrix with a dynamic
    # dimension was rejected unless the bounds of its size and its diagonal's
    # length agreed, and then its columns took the diagonal's length. The mask
    # keeps 2 of 5 entries, so the dimensions it selects have size 2 and bound
    # 5. (XLA turns a gather from 4 or fewer entries into selects, which
    # XLA:GPU doesn't support with dynamic shapes.)
    keep = [True, False, True, False, False]
    diagonal = [90, 91, 92, 93, 94]
    self.assertAllEqual(
        self._setDiagOfSelected(
            np.arange(1, 16).reshape(5, 3), diagonal, {-2: keep}, keep
        ),
        [[90, 2, 3], [7, 92, 9]],
    )
    self.assertAllEqual(
        self._setDiagOfSelected(
            np.arange(1, 26).reshape(5, 5), diagonal, {-2: keep}, keep
        ),
        [[90, 2, 3, 4, 5], [11, 92, 13, 14, 15]],
    )
    self.assertAllEqual(
        self._setDiagOfSelected(
            np.arange(1, 16).reshape(3, 5), diagonal, {-1: keep}, keep
        ),
        [[90, 3], [6, 92], [11, 13]],
    )
    self.assertAllEqual(
        self._setDiagOfSelected(
            np.arange(1, 26).reshape(5, 5), diagonal, {-2: keep, -1: keep}, keep
        ),
        [[90, 3], [11, 92]],
    )
    self.assertAllEqual(
        self._setDiagOfSelected(
            np.arange(1, 16).reshape(5, 3), diagonal, {-2: keep}, keep, k=1
        ),
        [[1, 90, 3], [7, 8, 92]],
    )
    # A diagonal whose bound is shorter than the one the matrix's bound gives.
    self.assertAllEqual(
        self._setDiagOfSelected(
            np.arange(1, 37).reshape(6, 6), diagonal, {-2: keep + [False]}, keep
        ),
        [[90, 2, 3, 4, 5, 6], [13, 92, 15, 16, 17, 18]],
    )

  def testDynamicDimensionsBatch(self):
    if "GPU" in self.device:
      self.skipTest(
          "XLA:GPU returns the batched dynamic output of this transposed."
      )
    keep = [True, False, True, False, False]
    self.assertAllEqual(
        self._setDiagOfSelected(
            np.arange(1, 31).reshape(2, 5, 3),
            [[90, 91, 92, 93, 94], [80, 81, 82, 83, 84]],
            {-2: keep},
            keep,
        ),
        [[[90, 2, 3], [7, 92, 9]], [[80, 17, 18], [22, 82, 24]]],
    )

  def testDynamicDimensionsBand(self):
    # In each case, diagonals 1 and -1 are padded differently at the actual
    # sizes than at the bound sizes, which matters when they're right-aligned.
    keep = [True, False, True, False, False]
    diagonals = [
        [70, 71, 72, 73, 74],
        [80, 81, 82, 83, 84],
        [90, 91, 92, 93, 94],
    ]
    for align, sub, sup in (
        ("RIGHT_LEFT", 90, 72),
        ("LEFT_RIGHT", 92, 70),
        ("LEFT_LEFT", 90, 70),
        ("RIGHT_RIGHT", 92, 72),
    ):
      self.assertAllEqual(
          self._setDiagOfSelected(
              np.arange(1, 16).reshape(5, 3),
              diagonals,
              {-2: keep},
              keep,
              k=(-1, 1),
              align=align,
          ),
          [[80, 70, 3], [sub, 82, 72]],
      )
      self.assertAllEqual(
          self._setDiagOfSelected(
              np.arange(1, 16).reshape(3, 5),
              diagonals,
              {-1: keep},
              keep,
              k=(-1, 1),
              align=align,
          ),
          [[80, sup], [90, 82], [11, 92]],
      )

  def testBandFromVector(self):
    # A band of diagonals needs them stacked in a dimension that a vector
    # doesn't have, and looking for it aborted the process. The ranks are left
    # unknown so that shape inference doesn't reject the vector first.
    with self.session() as session:
      with self.device_scope():
        x = array_ops.placeholder(dtypes.int32)
        diagonal = array_ops.placeholder(dtypes.int32)
        output = gen_array_ops.matrix_set_diag_v3(
            input=x, diagonal=diagonal, k=[-1, 1], align="RIGHT_LEFT"
        )
      with self.assertRaisesOpError("The number of diagonals provided"):
        session.run(
            output,
            {x: np.zeros([2, 3, 3], dtype=np.int32), diagonal: [1, 2, 3]},
        )


class MatrixDiagPartTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self,
                                     params,
                                     solution,
                                     high_level=True,
                                     rtol=1e-3,
                                     atol=1e-5):
    """Verifies that matrix_diag_part produces `solution` when fed `params`.

    Args:
      params: dictionary containing input parameters to matrix_diag_part.
      solution: numpy array representing the expected output.
      high_level: call high_level matrix_set_diag
      rtol: relative tolerance for equality test.
      atol: absolute tolerance for equality test.
    """
    input = params["input"]  # pylint: disable=redefined-builtin
    with self.session() as session:
      for dtype in self.numeric_types - {np.int8, np.uint8}:
        expected = solution.astype(dtype)
        with self.device_scope():
          params["input"] = array_ops.placeholder(
              dtype, input.shape, name="input")
          if high_level:
            # wraps gen_array_ops.matrix_diag_part_v3
            output = array_ops.matrix_diag_part(**params)
          else:
            # TODO(b/201086188): Remove this case once MatrixDiag V1 is removed.
            output = gen_array_ops.matrix_diag_part(**params)
          output = array_ops.matrix_diag_part(**params)
        result = session.run(output, {
            params["input"]: input.astype(dtype),
        })
        self.assertEqual(output.dtype, expected.dtype)
        self.assertAllCloseAccordingToType(
            expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03)

  # Generic tests applicable to both v1 and v2 ops.
  # Originally from unary_ops_tests.py.
  def _testV1Level(self, high_level):
    matrices = np.arange(3 * 2 * 4).reshape([3, 2, 4])
    solution = np.array([[0, 5], [8, 13], [16, 21]])
    self._assertOpOutputMatchesExpected({"input": matrices}, solution,
                                        high_level)

  def testV1(self):
    self._testV1Level(True)

  def testV1LowLevel(self):
    self._testV1Level(False)

  # From here onwards are v2-only tests.
  def testSingleMatrix(self):
    for align in alignment_list:
      test_list = [square_cases(align), tall_cases(align), fat_cases(align)]
      for mat, tests in test_list:
        for diag_index, (solution, _) in tests.items():
          self._assertOpOutputMatchesExpected(
              {
                  "input": mat[0],
                  "k": diag_index,
                  "align": align
              }, solution[0])

  def testBatch(self):
    for align in alignment_list:
      for mat, tests in all_tests(align):
        for diag_index, (solution, _) in tests.items():
          self._assertOpOutputMatchesExpected(
              {
                  "input": mat,
                  "k": diag_index,
                  "align": align
              }, solution)

  def testPadding(self):
    for padding_value, align in zip_to_first_list_length([555, -11],
                                                         alignment_list):
      for mat, tests in all_tests(align):
        for diag_index, (solution, _) in tests.items():
          mask = (solution == 0)
          solution = solution + (mask * padding_value)
          self._assertOpOutputMatchesExpected(
              {
                  "input": mat,
                  "k": diag_index,
                  "padding_value": padding_value,
                  "align": align
              }, solution)

  def _diagPartOfSelected(self, op, x, masks, dtype=np.int32, **params):
    # masks maps axes of x to masks of the entries to keep along them, which
    # gives those dimensions a size only known at run time.
    axes = sorted(masks)

    @def_function.function(jit_compile=True)
    def f(x, *x_masks):
      for axis, mask in zip(axes, x_masks):
        x = array_ops.gather(
            x, array_ops.reshape(array_ops.where(mask), [-1]), axis=axis
        )
      return op(x, **params)

    x = np.array(x, dtype=dtype)
    with self.session() as session:
      with self.device_scope():
        x_ph = array_ops.placeholder(dtypes.as_dtype(dtype), x.shape)
        mask_phs = [
            array_ops.placeholder(dtypes.bool, [x.shape[axis]]) for axis in axes
        ]
        output = f(x_ph, *mask_phs)
      feed_dict = {x_ph: x}
      feed_dict.update({ph: masks[axis] for ph, axis in zip(mask_phs, axes)})
      return session.run(output, feed_dict)

  def testDynamicDimensions(self):
    # Regression test for GitHub issue 129053: with a dynamic dimension, the
    # diagonals had the length they have at its bound. The mask keeps 2 of 5
    # entries, so the dimensions it selects have size 2 and bound 5.
    keep = [True, False, True, False, False]
    x = np.arange(1, 16).reshape(5, 3)  # Keeps [[1, 2, 3], [7, 8, 9]].
    for dtype in (np.int32, np.float32, np.float64):
      for op in (array_ops.matrix_diag_part, gen_array_ops.matrix_diag_part):
        self.assertAllEqual(
            self._diagPartOfSelected(op, x, {-2: keep}, dtype), [1, 8]
        )
      self.assertAllEqual(
          self._diagPartOfSelected(
              array_ops.matrix_diag_part,
              x,
              {-2: keep},
              dtype,
              k=(-1, 1),
              padding_value=9,
          ),
          [[2, 9], [1, 8], [7, 9]],
      )
      self.assertAllEqual(
          self._diagPartOfSelected(
              array_ops.matrix_diag_part,
              x,
              {-2: keep},
              dtype,
              k=(-1, 1),
              padding_value=9,
              align="LEFT_RIGHT",
          ),
          [[2, 9], [1, 8], [9, 7]],
      )
      self.assertAllEqual(
          self._diagPartOfSelected(
              gen_array_ops.matrix_diag_part_v2,
              x,
              {-2: keep},
              dtype,
              k=[-1, 1],
              padding_value=9,
          ),
          [[2, 9], [1, 8], [7, 9]],
      )
      self.assertAllEqual(
          self._diagPartOfSelected(
              array_ops.matrix_diag_part,
              np.arange(1, 16).reshape(3, 5),
              {-1: keep},
              dtype,
          ),
          [1, 8],
      )
      # With no rows kept, the diagonal is empty.
      self.assertAllEqual(
          self._diagPartOfSelected(
              array_ops.matrix_diag_part, x, {-2: [False] * 5}, dtype
          ),
          np.zeros([0]),
      )

  def testDynamicDimensionsBatch(self):
    keep = [True, False, True, False, False]
    x = np.arange(1, 31).reshape(2, 5, 3)
    self.assertAllEqual(
        self._diagPartOfSelected(array_ops.matrix_diag_part, x, {-2: keep}),
        [[1, 8], [16, 23]],
    )
    self.assertAllEqual(
        self._diagPartOfSelected(
            array_ops.matrix_diag_part,
            x,
            {-2: keep},
            k=(-1, 1),
            padding_value=9,
        ),
        [[[2, 9], [1, 8], [7, 9]], [[17, 24], [16, 23], [22, 9]]],
    )
    self.assertAllEqual(
        self._diagPartOfSelected(
            array_ops.matrix_diag_part,
            np.arange(1, 51).reshape(2, 5, 5),
            {-2: keep, -1: keep},
        ),
        [[1, 13], [26, 38]],
    )


if __name__ == "__main__":
  googletest.main()
