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
"""Tests for tf.contrib.layers.sparse_feature_cross."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.ops import sparse_feature_cross_op
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class SparseCrossOpTest(test.TestCase):

  def test_simple(self):
    """Tests a simple scenario.
    """
    op = sparse_feature_cross_op.sparse_feature_cross([
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
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_dense(self):
    """Tests only dense inputs.
    """
    op = sparse_feature_cross_op.sparse_feature_cross([
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
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_integer_mixed_string_sparse(self):
    """Tests mixed type."""
    op = sparse_feature_cross_op.sparse_feature_cross([
        self._sparse_tensor([[11], [333, 55555]]),
        self._sparse_tensor([['batch1-FC2-F1'],
                             ['batch2-FC2-F1', 'batch2-FC2-F2']])
    ])
    expected_out = self._sparse_tensor([['11_X_batch1-FC2-F1'], [
        '333_X_batch2-FC2-F1', '333_X_batch2-FC2-F2', '55555_X_batch2-FC2-F1',
        '55555_X_batch2-FC2-F2'
    ]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_integer_mixed_string_dense(self):
    """Tests mixed dense inputs.
    """
    op = sparse_feature_cross_op.sparse_feature_cross([
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
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_sparse_cross_dense(self):
    """Tests sparse and dense inputs.
    """
    op = sparse_feature_cross_op.sparse_feature_cross([
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
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_integer_sparse_input(self):
    """Tests mixed type sparse and dense inputs."""
    op = sparse_feature_cross_op.sparse_feature_cross([
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
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_permutation_3x3x3(self):
    """Tests 3x3x3 permutation.
    """
    op = sparse_feature_cross_op.sparse_feature_cross([
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
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_permutation_3x1x2(self):
    """Tests 3x1x2 permutation.
    """
    op = sparse_feature_cross_op.sparse_feature_cross([
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
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_large_batch(self):
    """Tests with large batch size to force multithreading.
    """
    batch_size = 5000
    col1 = []
    col2 = []
    col3 = []
    for b in range(batch_size):
      col1.append(
          ['batch%d-FC1-F1' % b, 'batch%d-FC1-F2' % b, 'batch%d-FC1-F3' % b])
      col2.append(['batch%d-FC2-F1' % b])
      col3.append(['batch%d-FC3-F1' % b, 'batch%d-FC3-F2' % b])

    op = sparse_feature_cross_op.sparse_feature_cross([
        self._sparse_tensor(col1), self._sparse_tensor(col2),
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
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_one_column_empty(self):
    """Tests when one column is empty.

    The crossed tensor should be empty.
    """
    op = sparse_feature_cross_op.sparse_feature_cross([
        self._sparse_tensor([['batch1-FC1-F1', 'batch1-FC1-F2']]),
        self._sparse_tensor([], 1),
        self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
    ])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_empty(sess.run(op))

  def test_some_columns_empty(self):
    """Tests when more than one columns are empty.

    Cross for the corresponding batch should be empty.
    """
    op = sparse_feature_cross_op.sparse_feature_cross([
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
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_all_columns_empty(self):
    """Tests when all columns are empty.

    The crossed tensor should be empty.
    """
    op = sparse_feature_cross_op.sparse_feature_cross([
        self._sparse_tensor([]), self._sparse_tensor([]),
        self._sparse_tensor([])
    ])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_empty(sess.run(op))

  def test_hashed_output_zero_bucket(self):
    """Tests a simple scenario.
    """
    op = sparse_feature_cross_op.sparse_feature_cross(
        [
            self._sparse_tensor([['batch1-FC1-F1']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1']])
        ],
        hashed_output=True)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[3735511728867393167]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_hashed_output_zero_bucket_v2(self):
    """Tests a simple scenario.
    """
    op = sparse_feature_cross_op.sparse_feature_cross(
        [
            self._sparse_tensor([['batch1-FC1-F1']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1']])
        ],
        hashed_output=True,
        hash_key=layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[1971693436396284976]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  # TODO(sibyl-Aix6ihai): Add benchmark to compare Hashed vs Non-hashed.
  def test_hashed_output(self):
    """Tests a simple scenario.
    """
    op = sparse_feature_cross_op.sparse_feature_cross(
        [
            self._sparse_tensor([['batch1-FC1-F1']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1']])
        ],
        hashed_output=True,
        num_buckets=100)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[74]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_hashed_output_v2(self):
    """Tests a simple scenario.
    """
    op = sparse_feature_cross_op.sparse_feature_cross(
        [
            self._sparse_tensor([['batch1-FC1-F1']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1']])
        ],
        hashed_output=True,
        num_buckets=100,
        hash_key=layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY)
    # Check actual hashed output to prevent unintentional hashing changes.
    expected_out = self._sparse_tensor([[83]])
    with self.cached_session() as sess:
      self._assert_sparse_tensor_equals(expected_out, sess.run(op))

  def test_hashed_output_v1_has_collision(self):
    """Tests the old version of the fingerprint concatenation has collisions.
    """
    # The last 10 bits of 359 and 1024+359 are identical.
    # As a result, all the crosses collide.
    t1 = constant_op.constant([[359], [359 + 1024]])
    t2 = constant_op.constant([list(range(10)), list(range(10))])
    cross = sparse_feature_cross_op.sparse_feature_cross(
        [t2, t1], hashed_output=True, num_buckets=1024)
    cross_dense = sparse_ops.sparse_tensor_to_dense(cross)
    with session.Session():
      values = cross_dense.eval()
      self.assertTrue(numpy.equal(values[0], values[1]).all())

  def test_hashed_output_v2_has_no_collision(self):
    """Tests the new version of the fingerprint concatenation has no collisions.
    """
    # Although the last 10 bits of 359 and 1024+359 are identical.
    # As a result, all the crosses shouldn't collide.
    t1 = constant_op.constant([[359], [359 + 1024]])
    t2 = constant_op.constant([list(range(10)), list(range(10))])
    cross = sparse_feature_cross_op.sparse_feature_cross(
        [t2, t1],
        hashed_output=True,
        num_buckets=1024,
        hash_key=layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY)
    cross_dense = sparse_ops.sparse_tensor_to_dense(cross)
    with session.Session():
      values = cross_dense.eval()
      self.assertTrue(numpy.not_equal(values[0], values[1]).all())

  def test_hashed_3x1x2(self):
    """Tests 3x1x2 permutation with hashed output.
    """
    op = sparse_feature_cross_op.sparse_feature_cross(
        [
            self._sparse_tensor(
                [['batch1-FC1-F1', 'batch1-FC1-F2', 'batch1-FC1-F3']]),
            self._sparse_tensor([['batch1-FC2-F1']]),
            self._sparse_tensor([['batch1-FC3-F1', 'batch1-FC3-F2']])
        ],
        hashed_output=True,
        num_buckets=1000)
    with self.cached_session() as sess:
      out = sess.run(op)
      self.assertEqual(6, len(out.values))
      self.assertAllEqual([[0, i] for i in range(6)], out.indices)
      self.assertTrue(all(x < 1000 and x >= 0 for x in out.values))
      all_values_are_different = len(out.values) == len(set(out.values))
      self.assertTrue(all_values_are_different)

  def _assert_sparse_tensor_empty(self, sp):
    self.assertEquals(0, sp.indices.size)
    self.assertEquals(0, sp.values.size)
    # TODO(zakaria): check if we can ignore the first dim of the shape.
    self.assertEquals(0, sp.dense_shape[1])

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


if __name__ == '__main__':
  test.main()
