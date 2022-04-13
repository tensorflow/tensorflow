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
        with self.test_scope():
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
        with self.test_scope():
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
        with self.test_scope():
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


if __name__ == "__main__":
  googletest.main()
