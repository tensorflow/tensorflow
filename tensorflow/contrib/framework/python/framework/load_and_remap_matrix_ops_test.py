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
"""Functional tests for the op to generate vocab remapping."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib.framework.python.framework import load_and_remap_matrix_ops
from tensorflow.contrib.framework.python.ops import gen_checkpoint_ops
from tensorflow.contrib.framework.python.ops.gen_generate_vocab_remapping_ops import generate_vocab_remapping
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.training import saver

FLAGS = flags.FLAGS
_TESTDATA_PATH = 'contrib/framework/testdata'


class GenerateVocabRemappingTest(test.TestCase):
  """Tests for the generate_vocab_remapping() method."""

  def setUp(self):
    self.new_vocab_file = os.path.join(
        test.test_src_dir_path(_TESTDATA_PATH), 'keyword_shifted.txt')
    self.old_vocab_file = os.path.join(
        test.test_src_dir_path(_TESTDATA_PATH), 'keyword.txt')

  def test_generate_remapping_with_no_vocab_changes(self):
    """Tests where vocab does not change at all."""
    remapping, num_present = generate_vocab_remapping(
        new_vocab_file=self.old_vocab_file,
        old_vocab_file=self.old_vocab_file,
        num_new_vocab=3,
        new_vocab_offset=0)
    expected_remapping = range(0, 3)
    expected_num_present = 3
    with self.test_session():
      self.assertAllEqual(expected_remapping, remapping.eval())
      self.assertAllEqual(expected_num_present, num_present.eval())

  def test_generate_remapping_with_shifted_vocab(self):
    """Tests where vocab is the same, but shifted / ordered differently."""
    remapping, num_present = generate_vocab_remapping(
        new_vocab_file=self.new_vocab_file,
        old_vocab_file=self.old_vocab_file,
        num_new_vocab=3,
        new_vocab_offset=0)
    expected_remapping = [2, 0, 1]
    expected_num_present = 3
    with self.test_session():
      self.assertAllEqual(expected_remapping, remapping.eval())
      self.assertAllEqual(expected_num_present, num_present.eval())

  def test_generate_remapping_with_offset(self):
    """Tests offset and num_new_vocab logic."""
    remapping, num_present = generate_vocab_remapping(
        new_vocab_file=self.new_vocab_file,
        old_vocab_file=self.old_vocab_file,
        num_new_vocab=1,
        new_vocab_offset=1)
    expected_remapping = [0]
    expected_num_present = 1
    with self.test_session():
      self.assertAllEqual(expected_remapping, remapping.eval())
      self.assertAllEqual(expected_num_present, num_present.eval())


class LoadAndRemapMatrixTest(test.TestCase):
  """Tests for the load_and_remap_weight_matrix() op."""

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
    with self.test_session() as sess:
      variables.global_variables_initializer().run()
      self.bundle_file = os.path.join(test.get_temp_dir(), 'bundle_checkpoint')
      save.save(sess, self.bundle_file)

  def test_load_and_remap_no_missing(self):
    """Tests the op's load and remap where there are no missing entries."""

    # No column remapping, new weight matrix has second row, then first row.
    row_remapping = [1, 0]
    remapped_weight_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=row_remapping,
        col_remapping=[],
        initializing_values=[],
        num_rows=2,
        num_cols=self.old_num_cols)
    with self.test_session():
      self.assertAllClose(self.matrix_value[row_remapping],
                          remapped_weight_matrix.eval())

    # No row remapping, new weight matrix has third col, then first col.
    row_remapping = list(range(self.old_num_rows))
    col_remapping = [2, 0]
    remapped_weight_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=row_remapping,
        col_remapping=col_remapping,
        initializing_values=[],
        num_rows=len(row_remapping),
        num_cols=len(col_remapping))
    with self.test_session():
      self.assertAllClose(self.matrix_value[row_remapping][:, col_remapping],
                          remapped_weight_matrix.eval())

    # Both row and column remappings.
    row_remapping = [1, 0, 4]
    col_remapping = [1, 15]
    remapped_weight_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=row_remapping,
        col_remapping=col_remapping,
        initializing_values=[],
        num_rows=len(row_remapping),
        num_cols=len(col_remapping))
    with self.test_session():
      self.assertAllClose(self.matrix_value[row_remapping][:, col_remapping],
                          remapped_weight_matrix.eval())

  def test_load_and_remap_with_init(self):
    """Tests the op's load and remap where there are missing entries."""
    init_val = 42
    remapped_weight_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=[2, -1, 0],
        col_remapping=[1, -1],
        initializing_values=[init_val] * 4,
        num_rows=3,
        num_cols=2)

    expected_remapped_weight_matrix = np.reshape(
        [33, init_val, init_val, init_val, 1, init_val], [3, 2])

    with self.test_session():
      self.assertAllClose(expected_remapped_weight_matrix,
                          remapped_weight_matrix.eval())

  def test_load_and_remap_all_missing_rows(self):
    """Tests when all the rows are missing and need to be initialized."""
    num_rows = 7
    initializing_values = [42] * num_rows * self.old_num_cols
    remapped_weight_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=[-1] * num_rows,
        col_remapping=[],
        initializing_values=initializing_values,
        num_rows=num_rows,
        num_cols=self.old_num_cols)
    with self.test_session():
      self.assertAllClose(
          np.reshape(initializing_values, (num_rows, self.old_num_cols)),
          remapped_weight_matrix.eval())

  def test_load_and_remap_all_missing_rows_and_cols(self):
    """Tests when all the rows & cols are missing and need to be initialized."""
    num_rows = 7
    num_cols = 4
    initializing_values = [42] * num_rows * num_cols
    remapped_weight_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=[-1] * num_rows,
        col_remapping=[-1] * num_cols,
        initializing_values=initializing_values,
        num_rows=num_rows,
        num_cols=num_cols)
    with self.test_session():
      self.assertAllClose(
          np.reshape(initializing_values, (num_rows, num_cols)),
          remapped_weight_matrix.eval())

  def test_load_and_remap_duplicate_row_remapping(self):
    """Tests when an old row maps to multiple new rows.

    (This should usually not happen when using public APIs).
    """
    row_remapping = [1, 0, 0, 0, 1, 2]
    remapped_weight_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=row_remapping,
        col_remapping=[],
        initializing_values=[],
        num_rows=len(row_remapping),
        num_cols=self.old_num_cols)
    with self.test_session():
      self.assertAllClose(self.matrix_value[row_remapping],
                          remapped_weight_matrix.eval())

  def test_load_and_remap_invalid_col_remapping(self):
    """Tests that an error is raised when an old col maps to multiple new cols.

    (This should usually not happen when using public APIs).
    """
    col_remapping = [1, 0, 0, 0, 1, 2]
    remapped_weight_matrix = gen_checkpoint_ops.load_and_remap_matrix(
        ckpt_path=[self.bundle_file],
        old_tensor_name=self.old_tensor_name,
        row_remapping=list(range(self.old_num_rows)),
        col_remapping=col_remapping,
        initializing_values=[],
        num_rows=self.old_num_rows,
        num_cols=len(col_remapping))
    with self.test_session(), self.assertRaises(errors.UnimplementedError):
      remapped_weight_matrix.eval()


class LoadAndRemapWrappersTest(test.TestCase):
  """Tests for the functionality of the Python wrappers."""

  def setUp(self):
    self.bundle_file = os.path.join(
        test.test_src_dir_path(_TESTDATA_PATH), 'bundle_checkpoint')
    self.new_feature_vocab_file = os.path.join(
        test.test_src_dir_path(_TESTDATA_PATH), 'bundle_checkpoint_vocab.txt')
    self.old_feature_vocab_file = os.path.join(
        test.test_src_dir_path(_TESTDATA_PATH),
        'bundle_checkpoint_vocab_with_oov.txt')
    self.new_class_vocab_file = os.path.join(
        test.test_src_dir_path(_TESTDATA_PATH), 'keyword_new.txt')
    self.old_class_vocab_file = os.path.join(
        test.test_src_dir_path(_TESTDATA_PATH), 'keyword.txt')
    self.init_val = 42

    def _init_val_initializer(shape, dtype=None, partition_info=None):
      del dtype, partition_info  # Unused by this unit-testing initializer.
      return array_ops.tile(
          constant_op.constant([[self.init_val]], dtype=dtypes.float32), shape)

    self.initializer = _init_val_initializer

  def test_load_and_remap_matrix(self):
    """Tests the end-to-end loading / remapping of weights."""
    # load_and_remap_matrix() is the generalized wrapper that takes in row and
    # column vocabulary files, calls the relevant remappings, and returns the
    # weight matrix.  Take this example to be linear multi-class by providing
    # both row and column vocabularies.
    remapped_matrix = load_and_remap_matrix_ops._load_and_remap_matrix(
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_rows_to_load=4,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.bundle_file],
        new_row_vocab_offset=1,
        initializer=self.initializer,
        num_row_oov_buckets=1,
        num_col_oov_buckets=1)

    # [4 in vocab + 1 oov features, 4 in vocab + 1 oov classes].  The offset
    # means we read
    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([18, 34, 50, self.init_val, self.init_val], [5, 1]),
            np.reshape([16, 32, 48, self.init_val, self.init_val], [5, 1]),
            np.reshape([self.init_val] * 5, [5, 1]),
            np.reshape([17, 33, 49, self.init_val, self.init_val], [5, 1]),
            np.reshape([self.init_val] * 5, [5, 1])
        ],
        axis=1)

    with self.test_session():
      self.assertAllClose(expected_remapped_matrix, remapped_matrix.eval())

  def test_load_and_remap_output_layer_weight_initializer_linear(self):
    """Tests for the output layer initializer in the linear multi-class case."""
    loading_initializer = (contrib_framework.load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.bundle_file],
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_row_oov_buckets=1,
        num_col_oov_buckets=1,
        initializer=self.initializer))

    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50, self.init_val, self.init_val], [6, 1]),
            np.reshape([0, 16, 32, 48, self.init_val, self.init_val], [6, 1]),
            np.reshape([self.init_val] * 6, [6, 1]),
            np.reshape([1, 17, 33, 49, self.init_val, self.init_val], [6, 1]),
            np.reshape([self.init_val] * 6, [6, 1])
        ],
        axis=1)

    # The new weight matrix is of size
    # [5 feature vocab + 1 feature OOV, 4 class vocab + 1 class OOV].  Use a
    # partitioned variable to confirm that the offset logic works.
    remapped_matrix = variable_scope.get_variable(
        name='linear/obtained_weight_matrix',
        shape=[6, 5],
        initializer=loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.test_session():
      variables.global_variables_initializer().run()
      self.assertAllClose(expected_remapped_matrix,
                          remapped_matrix.as_tensor().eval())

  def test_load_and_remap_output_layer_weight_initializer_dnn_output(self):
    """Tests for the output layer initializer in the DNN output case."""
    loading_initializer = (contrib_framework.load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.bundle_file],
        num_col_oov_buckets=1,
        initializer=self.initializer))

    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50, 66], [5, 1]),
            np.reshape([0, 16, 32, 48, 64], [5, 1]),
            np.reshape([self.init_val] * 5, [5, 1]),
            np.reshape([1, 17, 33, 49, 65], [5, 1]),
            np.reshape([self.init_val] * 5, [5, 1])
        ],
        axis=1)

    # The new weight matrix is of size
    # [5-sized input layer, 4 class vocab + 1 class OOV].
    remapped_matrix = variable_scope.get_variable(
        name='dnn_output/obtained_weight_matrix',
        shape=[5, 5],
        initializer=loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.test_session():
      variables.global_variables_initializer().run()
      self.assertAllClose(expected_remapped_matrix,
                          remapped_matrix.as_tensor().eval())

  def test_initializer_with_oov_only_partition(self):
    """Tests for the output layer initializer where one partition is all OOV."""
    loading_initializer = (contrib_framework.load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.bundle_file],
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_row_oov_buckets=5,
        num_col_oov_buckets=1,
        initializer=self.initializer))

    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50] + [self.init_val] * 6, [10, 1]),
            np.reshape([0, 16, 32, 48] + [self.init_val] * 6, [10, 1]),
            np.reshape([self.init_val] * 10, [10, 1]),
            np.reshape([1, 17, 33, 49] + [self.init_val] * 6, [10, 1]),
            np.reshape([self.init_val] * 10, [10, 1]),
        ],
        axis=1)

    # The new weight matrix is of size
    # [5 feature vocab + 5 feature OOV, 4 class vocab + 1 class OOV].  The
    # second partition has only OOV.
    remapped_matrix = variable_scope.get_variable(
        name='linear_all_oov/obtained_weight_matrix',
        shape=[10, 5],
        initializer=loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.test_session():
      variables.global_variables_initializer().run()
      self.assertAllClose(expected_remapped_matrix,
                          remapped_matrix.as_tensor().eval())

  def test_load_and_remap_linear_multiclass_initializer_default_init(self):
    """Tests where the zeros_initializer default is used for linear."""
    loading_initializer = (contrib_framework.load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.bundle_file],
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_row_oov_buckets=1,
        num_col_oov_buckets=1))

    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50, 0, 0], [6, 1]),
            np.reshape([0, 16, 32, 48, 0, 0], [6, 1]),
            np.reshape([0] * 6, [6, 1]),
            np.reshape([1, 17, 33, 49, 0, 0], [6, 1]),
            np.reshape([0] * 6, [6, 1])
        ],
        axis=1)

    remapped_matrix = variable_scope.get_variable(
        name='linear_init_fallback/obtained_weight_matrix',
        shape=[6, 5],
        initializer=loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.test_session():
      variables.global_variables_initializer().run()
      self.assertAllClose(expected_remapped_matrix,
                          remapped_matrix.as_tensor().eval())

  def test_load_embedding_initializer(self):
    """Tests for the load_embedding initializer wrapper."""
    embedding_loading_initializer = (
        contrib_framework.load_embedding_initializer(
            new_vocab_file=self.new_feature_vocab_file,
            old_vocab_file=self.old_feature_vocab_file,
            new_vocab_size=5,
            embedding_dim=16,
            embedding_tensor_name='some_scope/embeddings',
            ckpt_path=[self.bundle_file],
            num_oov_buckets=1,
            initializer=self.initializer))

    expected_remapped_embeddings = np.concatenate(
        [
            np.reshape(range(64), [4, 16]),
            np.reshape([self.init_val] * 32, [2, 16]),
        ],
        axis=0)

    # The new weight matrix is of size
    # [5 feature vocab + 1 feature OOV, 16 (embedding dimension)], where the
    # last vocab row (2nd last row) is newly initialized (wasn't found in
    # previous vocab) and the actual last row is OOV and also newly initialized.
    # Use a partitioned variable to confirm that the offset logic works.
    remapped_embeddings = variable_scope.get_variable(
        name='embedding/obtained_embedding_matrix',
        shape=[6, 16],
        initializer=embedding_loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.test_session():
      variables.global_variables_initializer().run()
      self.assertAllClose(expected_remapped_embeddings,
                          remapped_embeddings.as_tensor().eval())


class LoadMulticlassBiasTest(test.TestCase):
  """Tests for the load_linear_multiclass_bias_initializer functionality."""

  def setUp(self):
    ops.reset_default_graph()
    dim = 1
    num = 3
    with ops.name_scope('some_scope'):
      # Basically from 0 to dim*num-1.
      flat_data = math_ops.linspace(0.0, dim * num - 1, dim * num)
      bias = variables.Variable(
          array_ops.reshape(flat_data, (num, dim)), name='bias')
    save = saver.Saver([bias])
    with self.test_session() as sess:
      variables.global_variables_initializer().run()
      self.bundle_file = os.path.join(test.get_temp_dir(), 'bias_checkpoint')
      save.save(sess, self.bundle_file)

    self.new_class_vocab_file = os.path.join(
        test.test_src_dir_path(_TESTDATA_PATH), 'keyword_new.txt')
    self.old_class_vocab_file = os.path.join(
        test.test_src_dir_path(_TESTDATA_PATH), 'keyword.txt')
    self.init_val = 42

    def _init_val_initializer(shape, dtype=None, partition_info=None):
      del dtype, partition_info  # Unused by this unit-testing initializer.
      return array_ops.tile(
          constant_op.constant([[self.init_val]], dtype=dtypes.float32), shape)

    self.initializer = _init_val_initializer

  def test_load_linear_multiclass_bias_initializer(self):
    """Tests for the bias initializer wrapper."""
    bias_loading_initializer = (
        contrib_framework.load_linear_multiclass_bias_initializer(
            new_class_vocab_file=self.new_class_vocab_file,
            old_class_vocab_file=self.old_class_vocab_file,
            new_class_vocab_size=4,
            bias_tensor_name='some_scope/bias',
            ckpt_path=[self.bundle_file],
            num_class_oov_buckets=1,
            initializer=self.initializer))

    expected_remapped_bias_vector = np.reshape(
        [2, 0, self.init_val, 1, self.init_val], [5, 1])

    # The new bias vector is of size [4 class vocab + 1 class OOV, 1].
    remapped_bias_vector = variable_scope.get_variable(
        name='bias/obtained_bias_vector',
        shape=[5, 1],
        initializer=bias_loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(3))

    with self.test_session():
      variables.global_variables_initializer().run()
      self.assertAllClose(expected_remapped_bias_vector,
                          remapped_bias_vector.as_tensor().eval())


if __name__ == '__main__':
  test.main()
