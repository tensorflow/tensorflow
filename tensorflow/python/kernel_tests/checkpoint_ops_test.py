# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for the ops to generate and execute vocab remapping."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_checkpoint_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.training import saver

FLAGS = flags.FLAGS


class GenerateVocabRemappingTest(test.TestCase):
  """Tests for the generate_vocab_remapping() method."""

  def setUp(self):
    self.new_vocab_file = os.path.join(self.get_temp_dir(),
                                       'keyword_shifted.txt')
    with open(self.new_vocab_file, 'w') as f:
      f.write('\n'.join(['MISSING', 'knitting', 'eminem']) + '\n')
    self.old_vocab_file = os.path.join(self.get_temp_dir(),
                                       'keyword.txt')
    with open(self.old_vocab_file, 'w') as f:
      f.write('\n'.join(['knitting', 'eminem', 'MISSING']) + '\n')

  @test_util.run_deprecated_v1
  def test_generate_remapping_with_no_vocab_changes(self):
    """Tests where vocab does not change at all."""
    remapping, num_present = gen_checkpoint_ops.generate_vocab_remapping(
        new_vocab_file=self.old_vocab_file,
        old_vocab_file=self.old_vocab_file,
        num_new_vocab=3,
        new_vocab_offset=0)
    expected_remapping = range(0, 3)
    expected_num_present = 3
    with self.cached_session():
      self.assertAllEqual(expected_remapping, self.evaluate(remapping))
      self.assertAllEqual(expected_num_present, self.evaluate(num_present))

  def test_generate_remapping_with_shifted_vocab(self):
    """Tests where vocab is the same, but shifted / ordered differently."""
    remapping, num_present = gen_checkpoint_ops.generate_vocab_remapping(
        new_vocab_file=self.new_vocab_file,
        old_vocab_file=self.old_vocab_file,
        num_new_vocab=3,
        new_vocab_offset=0)
    expected_remapping = [2, 0, 1]
    expected_num_present = 3
    with self.cached_session():
      self.assertAllEqual(expected_remapping, self.evaluate(remapping))
      self.assertAllEqual(expected_num_present, self.evaluate(num_present))

  def test_generate_remapping_with_offset(self):
    """Tests offset and num_new_vocab logic."""
    remapping, num_present = gen_checkpoint_ops.generate_vocab_remapping(
        new_vocab_file=self.new_vocab_file,
        old_vocab_file=self.old_vocab_file,
        num_new_vocab=1,
        new_vocab_offset=1)
    expected_remapping = [0]
    expected_num_present = 1
    with self.cached_session():
      self.assertAllEqual(expected_remapping, self.evaluate(remapping))
      self.assertAllEqual(expected_num_present, self.evaluate(num_present))

  def test_generate_remapping_with_old_vocab_size(self):
    """Tests where old_vocab_size is specified."""
    remapping, num_present = gen_checkpoint_ops.generate_vocab_remapping(
        new_vocab_file=self.new_vocab_file,
        old_vocab_file=self.old_vocab_file,
        num_new_vocab=3,
        new_vocab_offset=0,
        # Old vocabulary becomes ['knitting', 'eminem'].
        old_vocab_size=2)
    expected_remapping = [-1, 0, 1]
    expected_num_present = 2
    with self.cached_session():
      self.assertAllEqual(expected_remapping, self.evaluate(remapping))
      self.assertAllEqual(expected_num_present, self.evaluate(num_present))


class LoadAndRemapMatrixTest(test.TestCase):
  """Tests for the load_and_remap_matrix() op."""

  def setUp(self):
    ops.reset_default_graph()
    self.old_num_rows = 5
    self.old_num_cols = 16
    self.matrix_value = np.reshape(
        range(0, self.old_num_rows * self.old_num_cols), (self.old_num_rows,
                                                          self.old_num_cols))
    with variable_scope.variable_scope('some_scope'):
      matrix = variable_scope.get_variable(
          'matrix',
          dtype=dtypes.float32,
          initializer=constant_op.constant(
              self.matrix_value, dtype=dtypes.float32))
      self.old_tensor_name = 'some_scope/matrix'

    save = saver.Saver([matrix])
    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      self.bundle_file = os.path.join(test.get_temp_dir(), 'bundle_checkpoint')
      save.save(sess, self.bundle_file)

  def test_load_and_remap_no_missing(self):
    """Tests the op's load and remap where there are no missing entries."""

    # No column remapping, new weight matrix has second row, then first row.
    row_remapping = [1, 0]
    remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=row_remapping,
        col_remapping=[],
        initializing_values=[],
        num_rows=2,
        num_cols=self.old_num_cols)
    with self.cached_session():
      self.assertAllClose(self.matrix_value[row_remapping],
                          self.evaluate(remapped_matrix))

    # No row remapping, new weight matrix has third col, then first col.
    row_remapping = list(range(self.old_num_rows))
    col_remapping = [2, 0]
    remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=row_remapping,
        col_remapping=col_remapping,
        initializing_values=[],
        num_rows=len(row_remapping),
        num_cols=len(col_remapping))
    with self.cached_session():
      self.assertAllClose(self.matrix_value[row_remapping][:, col_remapping],
                          self.evaluate(remapped_matrix))

    # Both row and column remappings.
    row_remapping = [1, 0, 4]
    col_remapping = [1, 15]
    remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=row_remapping,
        col_remapping=col_remapping,
        initializing_values=[],
        num_rows=len(row_remapping),
        num_cols=len(col_remapping))
    with self.cached_session():
      self.assertAllClose(self.matrix_value[row_remapping][:, col_remapping],
                          self.evaluate(remapped_matrix))

  def test_load_and_remap_with_init(self):
    """Tests the op's load and remap where there are missing entries."""
    init_val = 42
    remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=[2, -1, 0],
        col_remapping=[1, -1],
        initializing_values=[init_val] * 4,
        num_rows=3,
        num_cols=2)

    expected_remapped_matrix = np.reshape(
        [33, init_val, init_val, init_val, 1, init_val], [3, 2])

    with self.cached_session():
      self.assertAllClose(expected_remapped_matrix,
                          self.evaluate(remapped_matrix))

  def test_load_and_remap_all_missing_rows(self):
    """Tests when all the rows are missing and need to be initialized."""
    num_rows = 7
    initializing_values = [42] * num_rows * self.old_num_cols
    remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=[-1] * num_rows,
        col_remapping=[],
        initializing_values=initializing_values,
        num_rows=num_rows,
        num_cols=self.old_num_cols)
    with self.cached_session():
      self.assertAllClose(
          np.reshape(initializing_values, (num_rows, self.old_num_cols)),
          self.evaluate(remapped_matrix))

  def test_load_and_remap_all_missing_rows_and_cols(self):
    """Tests when all the rows & cols are missing and need to be initialized."""
    num_rows = 7
    num_cols = 4
    initializing_values = [42] * num_rows * num_cols
    remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=[-1] * num_rows,
        col_remapping=[-1] * num_cols,
        initializing_values=initializing_values,
        num_rows=num_rows,
        num_cols=num_cols)
    with self.cached_session():
      self.assertAllClose(
          np.reshape(initializing_values, (num_rows, num_cols)),
          self.evaluate(remapped_matrix))

  def test_load_and_remap_invalid_dims(self):
    ckpt_path = constant_op.constant(
        '/tmp/warm_starting_util_test5kl2a3pc/tmpph76tep2/model-0',
        shape=[],
        dtype=dtypes.string)
    old_tensor_name = constant_op.constant(
        '/tmp/warm_starting_util_test5kl2a3pc/tmpph76tep2/model-0',
        shape=[],
        dtype=dtypes.string)
    row_remapping = constant_op.constant(0, shape=[], dtype=dtypes.int64)
    col_remapping = constant_op.constant(3, shape=[3], dtype=dtypes.int64)
    initializing_values = constant_op.constant([],
                                               shape=[0, 1],
                                               dtype=dtypes.float32)
    with self.cached_session(), self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError), 'tensor must be 1-D'):
      self.evaluate(
          gen_checkpoint_ops.load_and_remap_matrix(
              ckpt_path=ckpt_path,
              old_tensor_name=old_tensor_name,
              row_remapping=row_remapping,
              col_remapping=col_remapping,
              initializing_values=initializing_values,
              num_rows=1,
              num_cols=1))

  @test_util.run_deprecated_v1
  def test_load_and_remap_invalid_remapping(self):
    """Tests that errors are raised when an ID maps to multiple new IDs.

    (This should usually not happen when using public APIs).
    """
    invalid_remapping = [1, 0, 0, 0, 1, 2]

    # Invalid row remapping.
    remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=invalid_remapping,
        col_remapping=[],
        initializing_values=[],
        num_rows=len(invalid_remapping),
        num_cols=self.old_num_cols)
    with self.cached_session(), self.assertRaises(errors.UnimplementedError):
      self.evaluate(remapped_matrix)

    # Invalid column remapping.
    remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=list(range(self.old_num_rows)),
        col_remapping=invalid_remapping,
        initializing_values=[],
        num_rows=self.old_num_rows,
        num_cols=len(invalid_remapping))
    with self.cached_session(), self.assertRaises(errors.UnimplementedError):
      self.evaluate(remapped_matrix)

  @test_util.run_deprecated_v1
  def test_load_and_remap_incorrect_initializing_values(self):
    """Tests that errors are raised with incorrect number of init values."""
    remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=[2, -1, 0],
        col_remapping=[1, -1],
        # Too few initializing values - there should be 4. For some reason,
        # initializing_values must contain no element (instead of 3 or fewer) to
        # ensure that a seg fault would reliably occur if the check raising the
        # InvalidArgumentError were not present.
        initializing_values=[],
        num_rows=3,
        num_cols=2)
    with self.cached_session(), self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(remapped_matrix)

    remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=[2, -1, 0],
        col_remapping=[1, -1],
        # Too many initializing values - there should be 4.
        initializing_values=[0] * 5,
        num_rows=3,
        num_cols=2)
    with self.cached_session(), self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(remapped_matrix)


class LoadAndRemapMatrixWithMaxRowsTest(test.TestCase):
  """Tests for the load_and_remap_matrix() op.

  (Specifically focused on the max_rows_in_memory arg and its effects on
  TensorBundle's BundleReader and TensorSlice logic).
  """

  def _test_loading_variable_with_max_rows(self, np_value, partitioner,
                                           max_rows_in_memory):
    """Helper function for various tests using max_rows_in_memory."""
    ops.reset_default_graph()
    old_tensor_name = 'matrix_to_load_and_remap'
    matrix = variable_scope.get_variable(
        old_tensor_name,
        dtype=dtypes.float32,
        initializer=constant_op.constant(np_value, dtype=dtypes.float32),
        partitioner=partitioner)

    with self.cached_session() as sess:
      ckpt_path = os.path.join(test.get_temp_dir(), 'temp_ckpt')
      save = saver.Saver([matrix])
      self.evaluate(variables.global_variables_initializer())
      save.save(sess, ckpt_path)
      num_rows, num_cols = np_value.shape

      # Tests loading the entire tensor (except reversed).
      remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
          ckpt_path=ckpt_path,
          old_tensor_name=old_tensor_name,
          # Simply reverses the rows of the matrix.
          row_remapping=list(range(num_rows - 1, -1, -1)),
          col_remapping=[],
          initializing_values=[],
          num_rows=num_rows,
          num_cols=num_cols,
          max_rows_in_memory=max_rows_in_memory)
      self.assertAllClose(np_value[::-1], self.evaluate(remapped_matrix))

      # Tests loading the tensor (except for the first and last rows), with
      # uninitialized values. Requires num_rows to be at least 3 since we're
      # skipping the first and last rows.
      self.assertGreater(num_rows, 2)
      prefix_rows = 2
      suffix_rows = 3
      remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
          ckpt_path=ckpt_path,
          old_tensor_name=old_tensor_name,
          # Reverses the rows of the matrix, then prepends and appends
          # uninitialized rows.
          row_remapping=([-1] * prefix_rows + list(range(1, num_rows - 1)) +
                         [-1] * suffix_rows),
          col_remapping=[],
          initializing_values=[42] * (prefix_rows + suffix_rows) * num_cols,
          num_rows=num_rows - 2 + prefix_rows + suffix_rows,
          num_cols=num_cols,
          max_rows_in_memory=max_rows_in_memory)
      self.assertAllClose(
          np.vstack([
              np.tile(42, [prefix_rows, num_cols]), np_value[1:-1],
              np.tile(42, [suffix_rows, num_cols])
          ]), self.evaluate(remapped_matrix))

      # Tests when everything is taken from initializing_values.
      new_rows = 7
      initializing_values = [42] * new_rows * num_cols
      remapped_matrix = gen_checkpoint_ops.load_and_remap_matrix(
          ckpt_path=ckpt_path,
          old_tensor_name=old_tensor_name,
          # Nothing is loaded from the old tensor.
          row_remapping=[-1] * new_rows,
          col_remapping=[],
          initializing_values=initializing_values,
          num_rows=new_rows,
          num_cols=num_cols,
          max_rows_in_memory=max_rows_in_memory)
      self.assertAllClose(
          np.reshape(initializing_values, (new_rows, num_cols)),
          self.evaluate(remapped_matrix))

  @test_util.run_deprecated_v1
  def test_loading_rows_divisible_by_max_rows(self):
    """Tests loading normal var when rows are evenly divisible by max_rows."""
    self._test_loading_variable_with_max_rows(
        np_value=np.reshape(list(range(0, 36)), (9, 4)),
        partitioner=None,
        # 9 is evenly divisible by 3.
        max_rows_in_memory=3)

  @test_util.run_deprecated_v1
  def test_loading_rows_not_divisible_by_max_rows(self):
    """Tests loading normal var when rows aren't divisible by max_rows."""
    self._test_loading_variable_with_max_rows(
        np_value=np.reshape(list(range(0, 36)), (9, 4)),
        partitioner=None,
        # 9 is not evenly divisible by 4.
        max_rows_in_memory=4)

  @test_util.run_deprecated_v1
  def test_loading_rows_less_than_max_rows(self):
    """Tests loading normal var as a single slice.

    (When the specified max_rows_in_memory is larger than the number of rows)
    """
    self._test_loading_variable_with_max_rows(
        np_value=np.reshape(list(range(0, 36)), (9, 4)),
        partitioner=None,
        # 10 > 9.
        max_rows_in_memory=10)

  @test_util.run_deprecated_v1
  def test_loading_no_max_rows(self):
    """Tests loading normal var as a single slice with no valid max_rows."""
    self._test_loading_variable_with_max_rows(
        np_value=np.reshape(list(range(0, 18)), (6, 3)),
        partitioner=None,
        max_rows_in_memory=-1)

  @test_util.run_deprecated_v1
  def test_loading_partitions_equals_max_rows(self):
    """Tests loading partitioned var sliced on partition boundary."""
    self._test_loading_variable_with_max_rows(
        np_value=np.reshape(list(range(0, 36)), (9, 4)),
        partitioner=partitioned_variables.fixed_size_partitioner(3),
        # With a tensor of shape [9, 3] and 3 partitions, each partition has
        # exactly 3 rows.
        max_rows_in_memory=3)

  @test_util.run_deprecated_v1
  def test_loading_partitions_greater_than_max_rows(self):
    """Tests loading partitioned var with more slices than partitions."""
    self._test_loading_variable_with_max_rows(
        np_value=np.reshape(list(range(0, 36)), (9, 4)),
        partitioner=partitioned_variables.fixed_size_partitioner(3),
        # Even though each partition has 3 rows, we'll only load the tensor one
        # row at a time.
        max_rows_in_memory=1)

  @test_util.run_deprecated_v1
  def test_loading_partitions_less_than_max_rows(self):
    """Tests loading partitioned var as a single slice.

    (When the specified max_rows_in_memory is larger than the number of rows)
    """
    self._test_loading_variable_with_max_rows(
        np_value=np.reshape(list(range(0, 36)), (9, 4)),
        partitioner=partitioned_variables.fixed_size_partitioner(3),
        max_rows_in_memory=10)

  @test_util.run_deprecated_v1
  def test_loading_partitions_no_max_rows(self):
    """Tests loading partitioned var as single slice with no valid max_rows."""
    self._test_loading_variable_with_max_rows(
        np_value=np.reshape(list(range(0, 36)), (9, 4)),
        partitioner=partitioned_variables.fixed_size_partitioner(3),
        max_rows_in_memory=-1)


if __name__ == '__main__':
  test.main()
