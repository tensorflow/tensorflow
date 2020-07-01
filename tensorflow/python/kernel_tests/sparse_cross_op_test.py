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
"""Tests for sparse_cross_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class BaseSparseCrossOpTest(test.TestCase):

  def _sparse_tensor(self, data, batch_size=-1):
    """Generates a SparseTensor.

    Args:
      data: Should be a list of list of strings or int64. Each item of the outer
        list represents a batch. Each item of the batch is a feature of a
        specific feature column.
      batch_size: optional batch size, especially for cases when data has no
        entry for some batches.

    Returns:
     A SparseTensor.
    """
    indices = []
    values = []
    max_col_count = 0
    for batch, batch_ix in zip(data, range(len(data))):
      for column, column_ix in zip(batch, range(len(batch))):
        indices.append([batch_ix, column_ix])
        values.append(column)
        max_col_count = max(max_col_count, column_ix + 1)
    shape = [batch_size if batch_size != -1 else len(data), max_col_count]
    value_type = (
        dtypes.string
        if not values or isinstance(values[0], str) else dtypes.int64)
    return sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64, [len(indices), 2]),
        constant_op.constant(values, value_type, [len(indices)]),
        constant_op.constant(shape, dtypes.int64))

  def _assert_sparse_tensor_equals(self, sp1, sp2):
    self.assertAllEqual(sp1.indices.eval(), sp2.indices)
    self.assertAllEqual(sp1.values.eval(), sp2.values)
    self.assertAllEqual(sp1.dense_shape.eval(), sp2.dense_shape)

  def _assert_sparse_tensor_empty(self, sp):
    self.assertEqual(0, sp.indices.size)
    self.assertEqual(0, sp.values.size)
    # TODO(zakaria): check if we can ignore the first dim of the shape.
    self.assertEqual(0, sp.dense_shape[1])


class SparseCrossOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_simple(self):
    """Tests a simple scenario."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([['batch1-FC1-F1'],
                             ['batch2-FC1-F1', 'batch2-FC1-F2']]),
        self._sparse_tensor([['batch1-FC2-F1'],
                             ['batch2-FC2-F1', 'batch2-FC2-F2']])
    ])
    expected_out = self._sparse_tensor([['batch1-FC1-F1_X_batch1-FC2-F1'], [
        'batch2-FC1-F1_X_batch2-FC2-F1', 'batch2-FC1-F1_X_batch2-FC2-F2',
        'batch2-FC1-F2_X_batch2-FC2-F1', 'batch2-FC1-F2_X_batch2-FC2-F2'
    ]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_dense(self):
    """Tests only dense inputs."""
    op = sparse_ops.sparse_cross([
        constant_op.constant([['batch1-FC1-F1', 'batch1-FC1-F2'],
                              ['batch2-FC1-F1', 'batch2-FC1-F2']],
                             dtypes.string),
        constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                              ['batch2-FC2-F1', 'batch2-FC2-F2']],
                             dtypes.string),
    ])
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1', 'batch1-FC1-F1_X_batch1-FC2-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1', 'batch1-FC1-F2_X_batch1-FC2-F2'
    ], [
        'batch2-FC1-F1_X_batch2-FC2-F1', 'batch2-FC1-F1_X_batch2-FC2-F2',
        'batch2-FC1-F2_X_batch2-FC2-F1', 'batch2-FC1-F2_X_batch2-FC2-F2'
    ]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_integer_mixed_string_sparse(self):
    """Tests mixed type."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([[11], [333, 55555]]),
        self._sparse_tensor([['batch1-FC2-F1'],
                             ['batch2-FC2-F1', 'batch2-FC2-F2']])
    ])
    expected_out = self._sparse_tensor([['11_X_batch1-FC2-F1'], [
        '333_X_batch2-FC2-F1', '333_X_batch2-FC2-F2', '55555_X_batch2-FC2-F1',
        '55555_X_batch2-FC2-F2'
    ]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_integer_mixed_string_dense(self):
    """Tests mixed dense inputs."""
    op = sparse_ops.sparse_cross([
        constant_op.constant([[11, 333], [55555, 999999]], dtypes.int64),
        constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                              ['batch2-FC2-F1', 'batch2-FC2-F2']],
                             dtypes.string),
    ])
    expected_out = self._sparse_tensor([[
        '11_X_batch1-FC2-F1', '11_X_batch1-FC2-F2', '333_X_batch1-FC2-F1',
        '333_X_batch1-FC2-F2'
    ], [
        '55555_X_batch2-FC2-F1', '55555_X_batch2-FC2-F2',
        '999999_X_batch2-FC2-F1', '999999_X_batch2-FC2-F2'
    ]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_sparse_cross_dense(self):
    """Tests sparse and dense inputs."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([['batch1-FC1-F1'],
                             ['batch2-FC1-F1', 'batch2-FC1-F2']]),
        constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                              ['batch2-FC2-F1', 'batch2-FC2-F2']],
                             dtypes.string),
    ])
    expected_out = self._sparse_tensor(
        [['batch1-FC1-F1_X_batch1-FC2-F1', 'batch1-FC1-F1_X_batch1-FC2-F2'], [
            'batch2-FC1-F1_X_batch2-FC2-F1', 'batch2-FC1-F1_X_batch2-FC2-F2',
            'batch2-FC1-F2_X_batch2-FC2-F1', 'batch2-FC1-F2_X_batch2-FC2-F2'
        ]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_integer_sparse_input(self):
    """Tests mixed type sparse and dense inputs."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([[11], [333, 5555]]),
        constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                              ['batch2-FC2-F1', 'batch2-FC2-F2']],
                             dtypes.string),
    ])
    expected_out = self._sparse_tensor(
        [['11_X_batch1-FC2-F1', '11_X_batch1-FC2-F2'], [
            '333_X_batch2-FC2-F1', '333_X_batch2-FC2-F2',
            '5555_X_batch2-FC2-F1', '5555_X_batch2-FC2-F2'
        ]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_permutation_3x3x3(self):
    """Tests 3x3x3 permutation."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor(
            [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']]),
        self._sparse_tensor(
            [['batch1-FC2-F1', 'batch1-FC2-F2', 'batch1-FC2-F3']]),
        self._sparse_tensor(
            [['batch1-FC3-F1', 'batch1-FC3-F2', 'batch1-FC3-F3']])
    ])
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F3'
    ]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_permutation_3x1x2(self):
    """Tests 3x1x2 permutation."""
    op = sparse_ops.sparse_cross([
        self._sparse_tensor(
            [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']]),
        self._sparse_tensor([['batch1-FC2-F1']]),
        self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    ])
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F2'
    ]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_large_batch(self):
    """Tests with large batch size to force multithreading."""
    batch_size = 5000
    col1 = []
    col2 = []
    col3 = []
    for b in range(batch_size):
      col1.append(
          ['batch%d-FC1-F1' % b, 'batch%d-FC1-F2' % b, 'batch%d-FC1-F3' % b])
      col2.append(['batch%d-FC2-F1' % b])
      col3.append(['batch%d-FC3-F1' % b, 'batch%d-FC3-F2' % b])

    op = sparse_ops.sparse_cross([
        self._sparse_tensor(col1),
        self._sparse_tensor(col2),
        self._sparse_tensor(col3)
    ])

    col_out = []
    for b in range(batch_size):
      col_out.append([
          'batch%d-FC1-F1_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F1_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b),
          'batch%d-FC1-F2_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F2_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b),
          'batch%d-FC1-F3_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F3_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b)
      ])

    expected_out = self._sparse_tensor(col_out)
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_one_column_empty(self):
    """Tests when one column is empty.

    The crossed tensor should be empty.
    """
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([['batch1-FC1-F1', 'batch1-FC1-F2']]),
        self._sparse_tensor([], 1),
        self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    ])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_empty(self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_some_columns_empty(self):
    """Tests when more than one columns are empty.

    Cross for the corresponding batch should be empty.
    """
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([['batch1-FC1-F1', 'batch1-FC1-F2']], 2),
        self._sparse_tensor([['batch1-FC2-F1'], ['batch2-FC2-F1']], 2),
        self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']], 2)
    ])
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2'
    ]], 2)
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_all_columns_empty(self):
    """Tests when all columns are empty.

    The crossed tensor should be empty.
    """
    op = sparse_ops.sparse_cross([
        self._sparse_tensor([]),
        self._sparse_tensor([]),
        self._sparse_tensor([])
    ])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_empty(self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_hashed_zero_bucket_no_hash_key(self):
    op = sparse_ops.sparse_cross_hashed([
        self._sparse_tensor([['batch1-FC1-F1']]),
        self._sparse_tensor([['batch1-FC2-F1']]),
        self._sparse_tensor([['batch1-FC3-F1']])
    ])
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[1971693436396284976]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_hashed_zero_bucket(self):
    op = sparse_ops.sparse_cross_hashed(
        [
            self._sparse_tensor([['batch1-FC1-F1']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1']])
        ],
        hash_key=sparse_ops._DEFAULT_HASH_KEY + 1)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[4847552627144134031]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  # TODO(sibyl-Aix6ihai): Add benchmark to compare Hashed vs Non-hashed.
  @test_util.run_deprecated_v1
  def test_hashed_no_hash_key(self):
    op = sparse_ops.sparse_cross_hashed(
        [
            self._sparse_tensor([['batch1-FC1-F1']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1']])
        ],
        num_buckets=100)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[83]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_hashed_output(self):
    op = sparse_ops.sparse_cross_hashed(
        [
            self._sparse_tensor([['batch1-FC1-F1']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1']])
        ],
        num_buckets=100,
        hash_key=sparse_ops._DEFAULT_HASH_KEY + 1)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[31]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(op))

  @test_util.run_deprecated_v1
  def test_hashed__has_no_collision(self):
    """Tests that fingerprint concatenation has no collisions."""
    # Although the last 10 bits of 359 and 1024+359 are identical.
    # As a result, all the crosses shouldn't collide.
    t1 = constant_op.constant([[359], [359 + 1024]])
    t2 = constant_op.constant([list(range(10)), list(range(10))])
    cross = sparse_ops.sparse_cross_hashed(
        [t2, t1], num_buckets=1024, hash_key=sparse_ops._DEFAULT_HASH_KEY + 1)
    cross_dense = sparse_ops.sparse_tensor_to_dense(cross)
    with session.Session():
      values = self.evaluate(cross_dense)
      self.assertTrue(numpy.not_equal(values[0], values[1]).all())

  def test_hashed_3x1x2(self):
    """Tests 3x1x2 permutation with hashed output."""
    op = sparse_ops.sparse_cross_hashed(
        [
            self._sparse_tensor(
                [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
        ],
        num_buckets=1000)
    with self.cached_session() as sess:
      out = self.evaluate(op)
      self.assertEqual(6, len(out.values))
      self.assertAllEqual([[0, i] for i in range(6)], out.indices)
      self.assertTrue(all(x < 1000 and x >= 0 for x in out.values))
      all_values_are_different = len(out.values) == len(set(out.values))
      self.assertTrue(all_values_are_different)

  def _assert_sparse_tensor_empty(self, sp):
    self.assertEqual(0, sp.indices.size)
    self.assertEqual(0, sp.values.size)
    # TODO(zakaria): check if we can ignore the first dim of the shape.
    self.assertEqual(0, sp.dense_shape[1])

  def _assert_sparse_tensor_equals(self, sp1, sp2):
    self.assertAllEqual(sp1.indices.eval(), sp2.indices)
    self.assertAllEqual(sp1.values.eval(), sp2.values)
    self.assertAllEqual(sp1.dense_shape.eval(), sp2.dense_shape)

  def _sparse_tensor(self, data, batch_size=-1):
    """Generates a SparseTensor.

    Args:
      data: Should be a list of list of strings or int64. Each item of the outer
          list represents a batch. Each item of the batch is a feature of a
          specific feature column.
      batch_size: optional batch size, especially for cases when data has no
          entry for some batches.

    Returns:
     A SparseTensor.
    """
    indices = []
    values = []
    max_col_count = 0
    for batch, batch_ix in zip(data, range(len(data))):
      for column, column_ix in zip(batch, range(len(batch))):
        indices.append([batch_ix, column_ix])
        values.append(column)
        max_col_count = max(max_col_count, column_ix + 1)
    shape = [batch_size if batch_size != -1 else len(data), max_col_count]
    value_type = (dtypes.string if not values or isinstance(values[0], str) else
                  dtypes.int64)
    return sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64, [len(indices), 2]),
        constant_op.constant(values, value_type, [len(indices)]),
        constant_op.constant(shape, dtypes.int64))

  def test_invalid_sparse_tensors(self):
    # Test validation of invalid SparseTensors.  The SparseTensor constructor
    # prevents us from creating invalid SparseTensors (eps. in eager mode),
    # so we create valid SparseTensors and then modify them to be invalid.

    st1 = sparse_tensor.SparseTensor([[0, 0]], [0], [2, 2])
    st1._indices = array_ops.zeros([], dtypes.int64)
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Input indices should be a matrix'):
      self.evaluate(sparse_ops.sparse_cross([st1]))

    st2 = sparse_tensor.SparseTensor([[0, 0]], [0], [2, 2])
    st2._values = array_ops.zeros([], dtypes.int64)
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Input values should be a vector'):
      self.evaluate(sparse_ops.sparse_cross([st2]))

    st3 = sparse_tensor.SparseTensor([[0, 0]], [0], [2, 2])
    st3._dense_shape = array_ops.zeros([], dtypes.int64)
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Input shapes should be a vector'):
      self.evaluate(sparse_ops.sparse_cross([st3]))

  def test_bad_tensor_shapes(self):
    # All inputs must be 2D.
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Expected D2 of index to be 2'):
      st = sparse_tensor.SparseTensor([[0]], [0], [10])  # 1D SparseTensor
      self.evaluate(sparse_ops.sparse_cross([st]))

    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Dense inputs should be a matrix'):
      dt = array_ops.zeros([0])  # 1D DenseTensor.
      self.evaluate(sparse_ops.sparse_cross([dt]))

  def test_batch_size_mismatch(self):
    st1 = sparse_tensor.SparseTensor([[0, 0]], [0], [10, 10])  # batch size 10
    st2 = sparse_tensor.SparseTensor([[0, 0]], [0], [7, 10])  # batch size 7
    dt = array_ops.zeros([5, 0])  # batch size 5
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Expected batch size'):
      self.evaluate(sparse_ops.sparse_cross([st1, dt]))
    with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError),
                                'Expected batch size'):
      self.evaluate(sparse_ops.sparse_cross([st1, st2]))


class SparseCrossV2OpTest(BaseSparseCrossOpTest):

  @test_util.run_deprecated_v1
  def test_sparse(self):
    """Tests a simple scenario."""
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1'],
                                    ['batch2-FC1-F1', 'batch2-FC1-F2']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1'],
                                    ['batch2-FC2-F1', 'batch2-FC2-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices],
        values=[sp_inp_1.values, sp_inp_2.values],
        shapes=[sp_inp_1.dense_shape, sp_inp_2.dense_shape],
        dense_inputs=[],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['batch1-FC1-F1_X_batch1-FC2-F1'],
        ['batch2-FC1-F1_X_batch2-FC2-F1',
         'batch2-FC1-F1_X_batch2-FC2-F2',
         'batch2-FC1-F2_X_batch2-FC2-F1',
         'batch2-FC1-F2_X_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_sparse_sep(self):
    """Tests a simple scenario."""
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1'],
                                    ['batch2-FC1-F1', 'batch2-FC1-F2']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1'],
                                    ['batch2-FC2-F1', 'batch2-FC2-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices],
        values=[sp_inp_1.values, sp_inp_2.values],
        shapes=[sp_inp_1.dense_shape, sp_inp_2.dense_shape],
        dense_inputs=[],
        sep='_Y_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['batch1-FC1-F1_Y_batch1-FC2-F1'],
        ['batch2-FC1-F1_Y_batch2-FC2-F1',
         'batch2-FC1-F1_Y_batch2-FC2-F2',
         'batch2-FC1-F2_Y_batch2-FC2-F1',
         'batch2-FC1-F2_Y_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_dense(self):
    """Tests only dense inputs."""
    dense_inp_1 = constant_op.constant([['batch1-FC1-F1', 'batch1-FC1-F2'],
                                        ['batch2-FC1-F1', 'batch2-FC1-F2']],
                                       dtypes.string)
    dense_inp_2 = constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                                        ['batch2-FC2-F1', 'batch2-FC2-F2']],
                                       dtypes.string)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[],
        values=[],
        shapes=[],
        dense_inputs=[dense_inp_1, dense_inp_2],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['batch1-FC1-F1_X_batch1-FC2-F1', 'batch1-FC1-F1_X_batch1-FC2-F2',
         'batch1-FC1-F2_X_batch1-FC2-F1', 'batch1-FC1-F2_X_batch1-FC2-F2'
        ],
        ['batch2-FC1-F1_X_batch2-FC2-F1', 'batch2-FC1-F1_X_batch2-FC2-F2',
         'batch2-FC1-F2_X_batch2-FC2-F1', 'batch2-FC1-F2_X_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_dense_sep(self):
    """Tests only dense inputs."""
    dense_inp_1 = constant_op.constant([['batch1-FC1-F1', 'batch1-FC1-F2'],
                                        ['batch2-FC1-F1', 'batch2-FC1-F2']],
                                       dtypes.string)
    dense_inp_2 = constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                                        ['batch2-FC2-F1', 'batch2-FC2-F2']],
                                       dtypes.string)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[],
        values=[],
        shapes=[],
        dense_inputs=[dense_inp_1, dense_inp_2],
        sep='_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['batch1-FC1-F1_batch1-FC2-F1', 'batch1-FC1-F1_batch1-FC2-F2',
         'batch1-FC1-F2_batch1-FC2-F1', 'batch1-FC1-F2_batch1-FC2-F2'
        ],
        ['batch2-FC1-F1_batch2-FC2-F1', 'batch2-FC1-F1_batch2-FC2-F2',
         'batch2-FC1-F2_batch2-FC2-F1', 'batch2-FC1-F2_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_integer_mixed_string_sparse(self):
    """Tests mixed type."""
    sp_inp_1 = self._sparse_tensor([[11], [333, 55555]])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1'],
                                    ['batch2-FC2-F1', 'batch2-FC2-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices],
        values=[sp_inp_1.values, sp_inp_2.values],
        shapes=[sp_inp_1.dense_shape, sp_inp_2.dense_shape],
        dense_inputs=[],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['11_X_batch1-FC2-F1'],
        ['333_X_batch2-FC2-F1', '333_X_batch2-FC2-F2',
         '55555_X_batch2-FC2-F1', '55555_X_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_integer_mixed_string_dense(self):
    """Tests mixed dense inputs."""
    dense_inp_1 = constant_op.constant([[11, 333], [55555, 999999]],
                                       dtypes.int64)
    dense_inp_2 = constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                                        ['batch2-FC2-F1', 'batch2-FC2-F2']],
                                       dtypes.string)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[],
        values=[],
        shapes=[],
        dense_inputs=[dense_inp_1, dense_inp_2],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    # pyformat: disable
    expected_out = self._sparse_tensor([
        ['11_X_batch1-FC2-F1', '11_X_batch1-FC2-F2',
         '333_X_batch1-FC2-F1', '333_X_batch1-FC2-F2'
        ],
        ['55555_X_batch2-FC2-F1', '55555_X_batch2-FC2-F2',
         '999999_X_batch2-FC2-F1', '999999_X_batch2-FC2-F2'
        ]])
    # pyformat: enable
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_sparse_cross_dense(self):
    """Tests sparse and dense inputs."""
    sp_inp = self._sparse_tensor([['batch1-FC1-F1'],
                                  ['batch2-FC1-F1', 'batch2-FC1-F2']])
    dense_inp = constant_op.constant([['batch1-FC2-F1', 'batch1-FC2-F2'],
                                      ['batch2-FC2-F1', 'batch2-FC2-F2']],
                                     dtypes.string)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp.indices],
        values=[sp_inp.values],
        shapes=[sp_inp.dense_shape],
        dense_inputs=[dense_inp],
        sep='_X_')
    expected_out = self._sparse_tensor(
        [['batch1-FC1-F1_X_batch1-FC2-F1', 'batch1-FC1-F1_X_batch1-FC2-F2'],
         [
             'batch2-FC1-F1_X_batch2-FC2-F1', 'batch2-FC1-F1_X_batch2-FC2-F2',
             'batch2-FC1-F2_X_batch2-FC2-F1', 'batch2-FC1-F2_X_batch2-FC2-F2'
         ]])
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_permutation_3x3x3(self):
    """Tests 3x3x3 permutation."""
    sp_inp_1 = self._sparse_tensor(
        [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']])
    sp_inp_2 = self._sparse_tensor(
        [['batch1-FC2-F1', 'batch1-FC2-F2', 'batch1-FC2-F3']])
    sp_inp_3 = self._sparse_tensor(
        [['batch1-FC3-F1', 'batch1-FC3-F2', 'batch1-FC3-F3']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F1_X_batch1-FC2-F3_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F3_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F2_X_batch1-FC3-F3',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F3_X_batch1-FC3-F3'
    ]])
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_permutation_3x1x2(self):
    """Tests 3x1x2 permutation."""
    sp_inp_1 = self._sparse_tensor(
        [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F3_X_batch1-FC2-F1_X_batch1-FC3-F2'
    ]])
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_large_batch(self):
    """Tests with large batch size to force multithreading."""
    batch_size = 5000
    col1 = []
    col2 = []
    col3 = []
    for b in range(batch_size):
      col1.append(
          ['batch%d-FC1-F1' % b,
           'batch%d-FC1-F2' % b,
           'batch%d-FC1-F3' % b])
      col2.append(['batch%d-FC2-F1' % b])
      col3.append(['batch%d-FC3-F1' % b, 'batch%d-FC3-F2' % b])
    sp_inp_1 = self._sparse_tensor(col1)
    sp_inp_2 = self._sparse_tensor(col2)
    sp_inp_3 = self._sparse_tensor(col3)

    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')

    col_out = []
    for b in range(batch_size):
      col_out.append([
          'batch%d-FC1-F1_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F1_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b),
          'batch%d-FC1-F2_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F2_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b),
          'batch%d-FC1-F3_X_batch%d-FC2-F1_X_batch%d-FC3-F1' % (b, b, b),
          'batch%d-FC1-F3_X_batch%d-FC2-F1_X_batch%d-FC3-F2' % (b, b, b)
      ])

    expected_out = self._sparse_tensor(col_out)
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_one_column_empty(self):
    """Tests when one column is empty.

    The crossed tensor should be empty.
    """
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1', 'batch1-FC1-F2']])
    sp_inp_2 = self._sparse_tensor([], 1)
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_empty(self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_some_columns_empty(self):
    """Tests when more than one columns are empty.

    Cross for the corresponding batch should be empty.
    """
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1', 'batch1-FC1-F2']], 2)
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1'], ['batch2-FC2-F1']], 2)
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']], 2)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')
    expected_out = self._sparse_tensor([[
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F1_X_batch1-FC2-F1_X_batch1-FC3-F2',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F1',
        'batch1-FC1-F2_X_batch1-FC2-F1_X_batch1-FC3-F2'
    ]], 2)
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_all_columns_empty(self):
    """Tests when all columns are empty.

    The crossed tensor should be empty.
    """
    sp_inp_1 = self._sparse_tensor([])
    sp_inp_2 = self._sparse_tensor([])
    sp_inp_3 = self._sparse_tensor([])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_v2(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        sep='_X_')
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_empty(self.evaluate(out))


class SparseCrossHashedOpTest(BaseSparseCrossOpTest):

  @test_util.run_deprecated_v1
  def test_hashed_zero_bucket_no_hash_key(self):
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        num_buckets=0,
        salt=[1, 1],
        strong_hash=False)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[9186962005966787372]])
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

    # salt is not being used when `strong_hash` is False.
    inds_2, vals_2, shapes_2 = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        num_buckets=0,
        salt=[137, 173],
        strong_hash=False)
    out_2 = sparse_tensor.SparseTensor(inds_2, vals_2, shapes_2)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out_2))

  @test_util.run_deprecated_v1
  def test_hashed_output(self):
    sp_inp_1 = self._sparse_tensor([['batch1-FC1-F1']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        num_buckets=100,
        salt=[137, 173],
        strong_hash=False)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[79]])
    out = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      self._assert_sparse_tensor_equals(expected_out, self.evaluate(out))

  @test_util.run_deprecated_v1
  def test_hashed_has_no_collision(self):
    """Tests that fingerprint concatenation has no collisions."""
    # Although the last 10 bits of 359 and 1024+359 are identical.
    # As a result, all the crosses shouldn't collide.
    t1 = constant_op.constant([[359], [359 + 1024]], dtype=dtypes.int64)
    t2 = constant_op.constant(
        [list(range(10)), list(range(10))], dtype=dtypes.int64)
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[],
        values=[],
        shapes=[],
        dense_inputs=[t2, t1],
        num_buckets=1024,
        salt=[137, 173],
        strong_hash=False)
    cross = sparse_tensor.SparseTensor(inds, vals, shapes)
    cross_dense = sparse_ops.sparse_tensor_to_dense(cross)
    with session.Session():
      values = self.evaluate(cross_dense)
      self.assertTrue(numpy.not_equal(values[0], values[1]).all())

  def test_hashed_3x1x2(self):
    """Tests 3x1x2 permutation with hashed output."""
    sp_inp_1 = self._sparse_tensor(
        [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        num_buckets=1000,
        salt=[137, 173],
        strong_hash=False)
    output = sparse_tensor.SparseTensor(inds, vals, shapes)
    with self.cached_session():
      out = self.evaluate(output)
      self.assertEqual(6, len(out.values))
      self.assertAllEqual([[0, i] for i in range(6)], out.indices)
      self.assertTrue(all(x < 1000 and x >= 0 for x in out.values))
      all_values_are_different = len(out.values) == len(set(out.values))
      self.assertTrue(all_values_are_different)

  def test_hashed_different_salt(self):
    sp_inp_1 = self._sparse_tensor(
        [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        strong_hash=False,
        num_buckets=1000,
        salt=[137, 173])
    output = sparse_tensor.SparseTensor(inds, vals, shapes)
    inds_2, vals_2, shapes_2 = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        strong_hash=True,
        num_buckets=1000,
        salt=[137, 1])
    output_2 = sparse_tensor.SparseTensor(inds_2, vals_2, shapes_2)
    with self.cached_session():
      out = self.evaluate(output)
      out_2 = self.evaluate(output_2)
      self.assertAllEqual(out.indices, out_2.indices)
      self.assertNotAllEqual(out.values, out_2.values)

  def test_sep_ignored_in_hashed_out(self):
    sp_inp_1 = self._sparse_tensor(
        [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']])
    sp_inp_2 = self._sparse_tensor([['batch1-FC2-F1']])
    sp_inp_3 = self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    inds, vals, shapes = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        strong_hash=True,
        num_buckets=1000,
        salt=[137, 173])
    output = sparse_tensor.SparseTensor(inds, vals, shapes)
    inds_2, vals_2, shapes_2 = gen_sparse_ops.sparse_cross_hashed(
        indices=[sp_inp_1.indices, sp_inp_2.indices, sp_inp_3.indices],
        values=[sp_inp_1.values, sp_inp_2.values, sp_inp_3.values],
        shapes=[
            sp_inp_1.dense_shape, sp_inp_2.dense_shape, sp_inp_3.dense_shape
        ],
        dense_inputs=[],
        strong_hash=True,
        num_buckets=1000,
        salt=[137, 173])
    output_2 = sparse_tensor.SparseTensor(inds_2, vals_2, shapes_2)
    with self.cached_session():
      out = self.evaluate(output)
      out_2 = self.evaluate(output_2)
      self.assertAllEqual(out.indices, out_2.indices)
      self.assertAllEqual(out.values, out_2.values)


if __name__ == '__main__':
  test.main()
