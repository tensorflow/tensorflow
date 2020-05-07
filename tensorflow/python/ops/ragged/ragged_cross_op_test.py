# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.ragged.cross and tf.ragged.cross_hashed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest

ragged_const = ragged_factory_ops.constant_value
dense_const = np.array


def sparse_const(matrix):
  indices = []
  values = []
  for i, row in enumerate(matrix):
    for j, val in enumerate(row):
      indices.append([i, j])
      values.append(val)
  shape = [len(matrix), max(len(row) for row in matrix)] if matrix else [0, 0]
  if not values:
    indices = np.zeros([0, 2], dtype=np.int64)
    values = np.zeros([0], dtype=np.int64)
  return sparse_tensor.SparseTensorValue(indices, values, shape)


@test_util.run_all_in_graph_and_eager_modes
class RaggedCrossOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name='NoInputs',
          inputs=[],
          expected=ragged_const([], ragged_rank=1, dtype=dtypes.int32)),
      dict(
          testcase_name='OneInput_RaggedStr',
          inputs=[ragged_const([['a', 'b'], [], ['c']])],
          expected=ragged_const([[b'a', b'b'], [], [b'c']])),
      dict(
          testcase_name='OneInput_RaggedInt',
          inputs=[ragged_const([[1, 2, 3], [4, 5]])],
          expected=ragged_const([[b'1', b'2', b'3'], [b'4', b'5']])),
      dict(
          testcase_name='OneInput_DenseInt',
          inputs=[dense_const([[1, 2, 3], [4, 5, 6]])],
          expected=ragged_const([[b'1', b'2', b'3'], [b'4', b'5', b'6']])),
      dict(
          testcase_name='OneInput_SparseStr',
          inputs=[sparse_const([['a', 'b'], [], ['c']])],
          expected=ragged_const([[b'a', b'b'], [], [b'c']])),
      dict(
          testcase_name='TwoInputs_RaggedStr_RaggedStr',
          inputs=[
              ragged_const([['a', 'b'], [], ['c']]),
              ragged_const([['d', 'e'], ['f'], ['g']])
          ],
          expected=ragged_const([[b'a_X_d', b'a_X_e', b'b_X_d', b'b_X_e'], [],
                                 [b'c_X_g']])),
      dict(
          testcase_name='TwoInputs_RaggedInt_RaggedInt',
          inputs=[
              ragged_const([[1, 2], [], [3]]),
              ragged_const([[4, 5, 6], [], [7]])
          ],
          expected=ragged_const(
              [[b'1_X_4', b'1_X_5', b'1_X_6', b'2_X_4', b'2_X_5', b'2_X_6'], [],
               [b'3_X_7']])),
      dict(
          testcase_name='TwoInputs_RaggedStr_RaggedInt',
          inputs=[
              ragged_const([['a', 'b'], [], ['c']]),
              ragged_const([['1', '2'], ['3'], ['4']])
          ],
          expected=ragged_const([[b'a_X_1', b'a_X_2', b'b_X_1', b'b_X_2'], [],
                                 [b'c_X_4']])),
      dict(
          testcase_name='TwoInputs_SparseStr_SparseStr',
          inputs=[
              sparse_const([['a', 'b'], [], ['c']]),
              sparse_const([['d', 'e'], ['f'], ['g']])
          ],
          expected=ragged_const([[b'a_X_d', b'a_X_e', b'b_X_d', b'b_X_e'], [],
                                 [b'c_X_g']])),
      dict(
          testcase_name='TwoInputs_DenseInt_DenseInt',
          inputs=[dense_const([[1, 2], [3, 4]]),
                  dense_const([[5, 6], [7, 8]])],
          expected=ragged_const([[b'1_X_5', b'1_X_6', b'2_X_5', b'2_X_6'],
                                 [b'3_X_7', b'3_X_8', b'4_X_7', b'4_X_8']])),
      dict(
          testcase_name='TwoInputs_DenseInt_DenseStr',
          inputs=[
              dense_const([[1, 2], [3, 4]]),
              dense_const([[b'5', b'6'], [b'7', b'8']])
          ],
          expected=ragged_const([[b'1_X_5', b'1_X_6', b'2_X_5', b'2_X_6'],
                                 [b'3_X_7', b'3_X_8', b'4_X_7', b'4_X_8']])),
      dict(
          testcase_name='TwoInputs_RaggedInt_DenseInt',
          inputs=[
              ragged_const([[], [], [1, 2], [3]]),
              dense_const([[1, 2], [3, 4], [5, 6], [7, 8]])
          ],
          expected=ragged_const([[], [],
                                 [b'1_X_5', b'1_X_6', b'2_X_5', b'2_X_6'],
                                 [b'3_X_7', b'3_X_8']])),
      dict(
          # This test exercises `input_order`.
          testcase_name='TwoInputs_DenseInt_RaggedStr',
          inputs=[
              dense_const([[1, 2], [3, 4], [5, 6]]),
              ragged_const([['d', 'e'], ['f'], ['g']])
          ],
          expected=ragged_const([[b'1_X_d', b'1_X_e', b'2_X_d', b'2_X_e'],
                                 [b'3_X_f', b'4_X_f'], [b'5_X_g', b'6_X_g']]),
          matches_sparse_cross=False  # sparse doesn't preserve input order.
      ),
      dict(
          # This test exercises `input_order`.
          testcase_name='TwoInputs_SparseInt_RaggedStr',
          inputs=[
              sparse_const([[1, 2], [3, 4], [5, 6]]),
              ragged_const([['d', 'e'], ['f'], ['g']])
          ],
          expected=ragged_const([[b'1_X_d', b'1_X_e', b'2_X_d', b'2_X_e'],
                                 [b'3_X_f', b'4_X_f'], [b'5_X_g', b'6_X_g']]),
          matches_sparse_cross=False  # sparse doesn't preserve input order.
      ),
      dict(
          testcase_name='ThreeInputs_RaggedInt_RaggedInt_RaggedInt',
          inputs=[
              ragged_const([[11], [12, 13], [], [14, 15]]),
              ragged_const([[21, 22], [23], [24, 25], [26, 27]]),
              ragged_const([[31], [32, 33], [34, 35], [36, 37]])
          ],
          expected=ragged_const([[b'11_X_21_X_31', b'11_X_22_X_31'],
                                 [
                                     b'12_X_23_X_32', b'12_X_23_X_33',
                                     b'13_X_23_X_32', b'13_X_23_X_33'
                                 ], [],
                                 [
                                     b'14_X_26_X_36', b'14_X_26_X_37',
                                     b'14_X_27_X_36', b'14_X_27_X_37',
                                     b'15_X_26_X_36', b'15_X_26_X_37',
                                     b'15_X_27_X_36', b'15_X_27_X_37'
                                 ]])),
      dict(
          testcase_name='ThreeInputs_RaggedInt_SparseInt_DenseInt',
          inputs=[
              ragged_const([[11], [12, 13], [], [14, 15]]),
              sparse_const([[21, 22], [23], [24, 25], [26, 27]]),
              dense_const([[31], [32], [33], [34]])
          ],
          expected=ragged_const([[b'11_X_21_X_31', b'11_X_22_X_31'],
                                 [
                                     b'12_X_23_X_32',
                                     b'13_X_23_X_32',
                                 ], [],
                                 [
                                     b'14_X_26_X_34',
                                     b'14_X_27_X_34',
                                     b'15_X_26_X_34',
                                     b'15_X_27_X_34',
                                 ]])),
      dict(
          testcase_name='FiveInputs',
          inputs=[
              ragged_const([[1]]),
              dense_const([[2]]),
              ragged_const([[3]]),
              sparse_const([[4]]),
              ragged_const([[5]])
          ],
          expected=ragged_const([[b'1_X_2_X_3_X_4_X_5']]),
          matches_sparse_cross=False  # sparse doesn't preserve input order.
      ),
      dict(
          testcase_name='Permutation_3x3x3',
          inputs=[[['11', '12', '13']], [['21', '22', '23']],
                  [['31', '32', '33']]],
          expected=[[
              b'11_X_21_X_31', b'11_X_21_X_32', b'11_X_21_X_33',
              b'11_X_22_X_31', b'11_X_22_X_32', b'11_X_22_X_33',
              b'11_X_23_X_31', b'11_X_23_X_32', b'11_X_23_X_33',
              b'12_X_21_X_31', b'12_X_21_X_32', b'12_X_21_X_33',
              b'12_X_22_X_31', b'12_X_22_X_32', b'12_X_22_X_33',
              b'12_X_23_X_31', b'12_X_23_X_32', b'12_X_23_X_33',
              b'13_X_21_X_31', b'13_X_21_X_32', b'13_X_21_X_33',
              b'13_X_22_X_31', b'13_X_22_X_32', b'13_X_22_X_33',
              b'13_X_23_X_31', b'13_X_23_X_32', b'13_X_23_X_33'
          ]]),
      dict(
          testcase_name='BatchSizeZero',
          inputs=[
              ragged_const([], ragged_rank=1, dtype=dtypes.int32),
              sparse_const([]),
              np.zeros([0, 3], dtype=np.int32),
          ],
          expected=ragged_const([], ragged_rank=1, dtype=dtypes.int32)),
      dict(
          testcase_name='ThreeInputs_OneEmpty',
          inputs=[
              ragged_const([[1, 2]]),
              ragged_const([[]], dtype=dtypes.int32),
              ragged_const([[3, 4]])
          ],
          expected=ragged_const([[]], dtype=dtypes.string)),
      dict(
          testcase_name='ThreeInputs_AllEmpty',
          inputs=[
              ragged_const([[]], dtype=dtypes.int64),
              ragged_const([[]], dtype=dtypes.string),
              ragged_const([[]], dtype=dtypes.int32)
          ],
          expected=ragged_const([[]], ragged_rank=1, dtype=dtypes.string)),
      dict(
          testcase_name='HashedZeroBucketsDefaultKey',
          inputs=[
              ragged_const([['batch1-FC1-F1']]),
              ragged_const([['batch1-FC2-F1']]),
              ragged_const([['batch1-FC3-F1']])
          ],
          expected_hashed=ragged_const([[1971693436396284976]])),
      dict(
          testcase_name='Hashed100BucketsDefaultKey',
          inputs=[
              ragged_const([['batch1-FC1-F1']]),
              ragged_const([['batch1-FC2-F1']]),
              ragged_const([['batch1-FC3-F1']])
          ],
          num_buckets=100,
          expected_hashed=ragged_const([[83]])),
      dict(
          testcase_name='HashedZeroBucketsCustomKey',
          inputs=[
              ragged_const([['batch1-FC1-F1']]),
              ragged_const([['batch1-FC2-F1']]),
              ragged_const([['batch1-FC3-F1']])
          ],
          hash_key=ragged_array_ops._DEFAULT_CROSS_HASH_KEY + 1,
          expected_hashed=ragged_const([[4847552627144134031]])),
      dict(
          testcase_name='Hashed100BucketsCustomKey',
          inputs=[
              ragged_const([['batch1-FC1-F1']]),
              ragged_const([['batch1-FC2-F1']]),
              ragged_const([['batch1-FC3-F1']])
          ],
          num_buckets=100,
          hash_key=ragged_array_ops._DEFAULT_CROSS_HASH_KEY + 1,
          expected_hashed=ragged_const([[31]])),
      dict(
          testcase_name='HashedZeroKey',
          inputs=[
              ragged_const([['batch1-FC1-F1']]),
              ragged_const([['batch1-FC2-F1']]),
              ragged_const([['batch1-FC3-F1']])
          ],
          hash_key=0,
          expected_hashed=ragged_const([[9077905385164735582]]),
          matches_sparse_cross=False  # sparse treats hash_key=0 as None.
      ),
      dict(
          testcase_name='UInt64',
          inputs=[ragged_const([[2**64 - 1]], dtype=dtypes.uint64)],
          expected=ragged_const([[b'-1']])),
  ])
  def testRaggedCross(self,
                      inputs,
                      num_buckets=0,
                      hash_key=None,
                      expected=None,
                      expected_hashed=None,
                      matches_sparse_cross=True):
    ragged_cross = ragged_array_ops.cross(inputs)
    ragged_cross_hashed = ragged_array_ops.cross_hashed(inputs, num_buckets,
                                                        hash_key)

    if expected is not None:
      self.assertAllEqual(ragged_cross, expected)
    if expected_hashed is not None:
      self.assertAllEqual(ragged_cross_hashed, expected_hashed)

    if matches_sparse_cross:
      # Check that ragged.cross & sparse.cross match.
      sparse_inputs = [self._ragged_to_sparse(t) for t in inputs]
      sparse_cross = sparse_ops.sparse_cross(sparse_inputs)
      self.assertAllEqual(ragged_cross,
                          ragged_tensor.RaggedTensor.from_sparse(sparse_cross))

      # Check that ragged.cross_hashed & sparse.cross_hashed match.
      sparse_inputs = [self._ragged_to_sparse(t) for t in inputs]
      sparse_cross_hashed = sparse_ops.sparse_cross_hashed(
          sparse_inputs, num_buckets, hash_key)
      self.assertAllEqual(
          ragged_cross_hashed,
          ragged_tensor.RaggedTensor.from_sparse(sparse_cross_hashed))

  def testRaggedCrossLargeBatch(self):
    batch_size = 5000
    inputs = [
        ragged_const([[1, 2, 3]] * batch_size),
        ragged_const([[b'4']] * batch_size),
        dense_const([[5]] * batch_size),
        sparse_const([[6, 7]] * batch_size)
    ]

    expected = [[
        b'1_X_4_X_5_X_6', b'1_X_4_X_5_X_7', b'2_X_4_X_5_X_6', b'2_X_4_X_5_X_7',
        b'3_X_4_X_5_X_6', b'3_X_4_X_5_X_7'
    ]] * batch_size

    ragged_cross = ragged_array_ops.cross(inputs)

    # Note: we don't use assertAllEqual here because if they don't match,
    # then the code in assertAllEqual that tries to build the error message
    # is very slow, causing the test to timeout.
    # pylint: disable=g-generic-assert
    self.assertTrue(self.evaluate(ragged_cross).to_list() == expected)

  @parameterized.named_parameters([
      dict(
          testcase_name='BadDType',
          inputs=[ragged_const([[1.1], [2.2, 3.3]])],
          message=r'Unexpected dtype for inputs\[0\]'),
      dict(
          testcase_name='StaticBatchSizeMismatch1',
          inputs=[ragged_const([[1]]),
                  ragged_const([[2], [3]])],
          exception=(ValueError, errors.InvalidArgumentError),
          message='inputs must all have the same batch dimension size'),
      dict(
          testcase_name='StaticBatchSizeMismatch2',
          inputs=[ragged_const([[1]]),
                  dense_const([[2], [3]])],
          exception=(ValueError, errors.InvalidArgumentError),
          message='inputs must all have the same batch dimension size'),
  ])
  def testStaticError(self, inputs, exception=ValueError, message=None):
    with self.assertRaisesRegexp(exception, message):
      ragged_array_ops.cross(inputs)

  @parameterized.named_parameters([
      dict(
          testcase_name='3DRaggedTensor',
          inputs=[ragged_const([[[1]]], ragged_rank=1)],
          message='tf.ragged.cross only supports inputs with rank=2'),
      dict(
          testcase_name='3DDenseTensor',
          inputs=[dense_const([[[1]]])],
          message='tf.ragged.cross only supports inputs with rank=2'),
  ])
  def testRuntimeError(self,
                       inputs,
                       exception=errors.InvalidArgumentError,
                       message=None):
    with self.assertRaisesRegexp(exception, message):
      self.evaluate(ragged_array_ops.cross(inputs))

  def _ragged_to_sparse(self, t):
    if ragged_tensor.is_ragged(t):
      return ragged_tensor.convert_to_tensor_or_ragged_tensor(t).to_sparse()
    elif sparse_tensor.is_sparse(t):
      return sparse_tensor.SparseTensor.from_value(t)
    else:
      return ops.convert_to_tensor(t)


if __name__ == '__main__':
  googletest.main()
