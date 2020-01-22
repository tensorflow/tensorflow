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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.compat import compat
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


# Test cases shared by MatrixDiagV2, MatrixDiagPartV2, and MatrixSetDiagV2.
# Copied from //third_party/tensorflow/python/kernel_tests/diag_op_test.py
def square_cases():
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
  # tests[d_lower, d_upper] = (compact_diagonals, padded_diagnals)
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
  return (mat, tests)


def tall_cases():
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
  # tests[d_lower, d_upper] = (compact_diagonals, padded_diagnals)
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
  return (mat, tests)


def fat_cases():
  # pyformat: disable
  mat = np.array([[[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 1, 2, 3]],
                  [[4, 5, 6, 7],
                   [8, 9, 1, 2],
                   [3, 4, 5, 6]]])
  tests = dict()
  # tests[d_lower, d_upper] = (compact_diagonals, padded_diagnals)
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
  return (mat, tests)


class MatrixDiagTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self,
                                     params,
                                     solution,
                                     rtol=1e-3,
                                     atol=1e-5):
    """Verifies that matrix_diag produces `solution` when fed `params`.

    Args:
      params: dictionary containing input parameters to matrix_diag.
      solution: numpy array representing the expected output of matrix_diag.
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
          output = array_ops.matrix_diag(**params)
        result = session.run(output,
                             {params["diagonal"]: diagonal.astype(dtype)})
        self.assertEqual(output.dtype, expected.dtype)
        self.assertAllCloseAccordingToType(
            expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03)

  # Generic tests applicable to both v1 and v2 ops.
  # Originally from unary_ops_tests.py.
  def testV1(self):
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
    self._assertOpOutputMatchesExpected({"diagonal": vecs1}, solution1)
    self._assertOpOutputMatchesExpected({"diagonal": vecs2}, solution2)
    self._assertOpOutputMatchesExpected({"diagonal": vecs3}, solution3)

  # From here onwards are v2-only tests.
  def testSquare(self):
    # LINT.IfChange
    if compat.forward_compatible(2019, 10, 31):
    # LINT.ThenChange(//tensorflow/python/ops/array_ops.py)
      for _, tests in [square_cases()]:
        for diag_index, (vecs, solution) in tests.items():
          self._assertOpOutputMatchesExpected(
              {
                  "diagonal": vecs[0],
                  "k": diag_index
              }, solution[0])

  def testSquareBatch(self):
    # LINT.IfChange
    if compat.forward_compatible(2019, 10, 31):
    # LINT.ThenChange(//tensorflow/python/ops/array_ops.py)
      for _, tests in [square_cases()]:
        for diag_index, (vecs, solution) in tests.items():
          self._assertOpOutputMatchesExpected(
              {
                  "diagonal": vecs,
                  "k": diag_index
              }, solution)

  def testRectangularBatch(self):
    # LINT.IfChange
    if not compat.forward_compatible(2019, 10, 31):
    # LINT.ThenChange(//tensorflow/python/ops/array_ops.py)
      return

    # Stores expected num_rows and num_cols (when the other is given).
    # expected[(d_lower, d_upper)] = (expected_num_rows, expected_num_cols)
    test_list = list()

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
    for _, tests in [tall_cases(), fat_cases()]:
      for diag_index, (vecs, solution) in tests.items():
        self._assertOpOutputMatchesExpected(
            {
                "diagonal": vecs,
                "k": diag_index,
                "num_rows": solution.shape[-2],
                "num_cols": solution.shape[-1]
            }, solution)

    # Giving just num_rows or num_cols.
    for expected, (_, tests) in test_list:
      for diag_index, (new_num_rows, new_num_cols) in expected.items():
        vecs, solution = tests[diag_index]
        solution_given_num_rows = solution.take(
            indices=range(new_num_cols), axis=-1)
        self._assertOpOutputMatchesExpected(
            {
                "diagonal": vecs,
                "k": diag_index,
                "num_rows": solution_given_num_rows.shape[-2]
            }, solution_given_num_rows)
        solution_given_num_cols = solution.take(
            indices=range(new_num_rows), axis=-2)
        self._assertOpOutputMatchesExpected(
            {
                "diagonal": vecs,
                "k": diag_index,
                "num_cols": solution_given_num_cols.shape[-1]
            }, solution_given_num_cols)

  def testPadding(self):
    # LINT.IfChange
    if compat.forward_compatible(2019, 10, 31):
    # LINT.ThenChange(//tensorflow/python/ops/array_ops.py)
      for padding_value in [555, -11]:
        for _, tests in [square_cases(), tall_cases(), fat_cases()]:
          for diag_index, (vecs, solution) in tests.items():
            mask = (solution == 0)
            solution = solution + (mask * padding_value)
            self._assertOpOutputMatchesExpected(
                {
                    "diagonal": vecs,
                    "k": diag_index,
                    "num_rows": solution.shape[-2],
                    "num_cols": solution.shape[-1],
                    "padding_value": padding_value
                }, solution)


class MatrixSetDiagTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self,
                                     params,
                                     solution,
                                     rtol=1e-3,
                                     atol=1e-5):
    """Verifies that matrix_set_diag produces `solution` when fed `params`.

    Args:
      params: dictionary containing input parameters to matrix_set_diag.
      solution: numpy array representing the expected output of matrix_set_diag.
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
          output = array_ops.matrix_set_diag(**params)
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
  def testV1(self):
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
      self._assertOpOutputMatchesExpected(test[0], test[1])

  # From here onwards are v2-only tests.
  def testSingleMatrix(self):
    # LINT.IfChange
    if compat.forward_compatible(2019, 10, 31):
    # LINT.ThenChange(//tensorflow/python/ops/array_ops.py)
      for _, tests in [square_cases(), tall_cases(), fat_cases()]:
        for diag_index, (vecs, banded_mat) in tests.items():
          mask = (banded_mat[0] == 0)
          input_mat = np.random.randint(10, size=mask.shape)
          solution = input_mat * mask + banded_mat[0]
          self._assertOpOutputMatchesExpected(
              {
                  "input": input_mat,
                  "diagonal": vecs[0],
                  "k": diag_index
              }, solution)

  def testBatch(self):
    # LINT.IfChange
    if compat.forward_compatible(2019, 10, 31):
    # LINT.ThenChange(//tensorflow/python/ops/array_ops.py)
      for _, tests in [square_cases(), tall_cases(), fat_cases()]:
        for diag_index, (vecs, banded_mat) in tests.items():
          mask = (banded_mat == 0)
          input_mat = np.random.randint(10, size=mask.shape)
          solution = input_mat * mask + banded_mat
          self._assertOpOutputMatchesExpected(
              {
                  "input": input_mat,
                  "diagonal": vecs,
                  "k": diag_index
              }, solution)


class MatrixDiagPartTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self,
                                     params,
                                     solution,
                                     rtol=1e-3,
                                     atol=1e-5):
    """Verifies that matrix_diag_part produces `solution` when fed `params`.

    Args:
      params: dictionary containing input parameters to matrix_diag_part.
      solution: numpy array representing the expected output.
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
          output = array_ops.matrix_diag_part(**params)
        result = session.run(output, {
            params["input"]: input.astype(dtype),
        })
        self.assertEqual(output.dtype, expected.dtype)
        self.assertAllCloseAccordingToType(
            expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03)

  # Generic tests applicable to both v1 and v2 ops.
  # Originally from unary_ops_tests.py.
  def testV1(self):
    matrices = np.arange(3 * 2 * 4).reshape([3, 2, 4])
    solution = np.array([[0, 5], [8, 13], [16, 21]])
    self._assertOpOutputMatchesExpected({"input": matrices}, solution)

  # From here onwards are v2-only tests.
  def testSingleMatrix(self):
    # LINT.IfChange
    if compat.forward_compatible(2019, 10, 31):
    # LINT.ThenChange(//tensorflow/python/ops/array_ops.py)
      for mat, tests in [square_cases(), tall_cases(), fat_cases()]:
        for diag_index, (solution, _) in tests.items():
          self._assertOpOutputMatchesExpected({
              "input": mat[0],
              "k": diag_index
          }, solution[0])

  def testBatch(self):
    # LINT.IfChange
    if compat.forward_compatible(2019, 10, 31):
    # LINT.ThenChange(//tensorflow/python/ops/array_ops.py)
      for mat, tests in [square_cases(), tall_cases(), fat_cases()]:
        for diag_index, (solution, _) in tests.items():
          self._assertOpOutputMatchesExpected({
              "input": mat,
              "k": diag_index
          }, solution)

  def testPadding(self):
    # LINT.IfChange
    if compat.forward_compatible(2019, 10, 31):
    # LINT.ThenChange(//tensorflow/python/ops/array_ops.py)
      for padding_value in [555, -11]:
        for mat, tests in [square_cases(), tall_cases(), fat_cases()]:
          for diag_index, (solution, _) in tests.items():
            mask = (solution == 0)
            solution = solution + (mask * padding_value)
            self._assertOpOutputMatchesExpected(
                {
                    "input": mat,
                    "k": diag_index,
                    "padding_value": padding_value
                }, solution)


if __name__ == "__main__":
  googletest.main()
