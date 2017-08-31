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
from tensorflow.contrib.framework.python.ops import checkpoint_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
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
    # _load_and_remap_matrix() is the generalized wrapper that takes in row and
    # column vocabulary files, calls the relevant remappings, and returns the
    # weight matrix.  Take this example to be linear multi-class by providing
    # both row and column vocabularies.
    remapped_matrix = checkpoint_ops._load_and_remap_matrix(
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
    """Tests for the load_embedding_initializer wrapper."""
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


class LoadVariableSlotTest(test.TestCase):
  """Tests for the load_variable_slot_initializer functionality."""

  def setUp(self):
    ops.reset_default_graph()
    dim = 1
    num = 3
    with ops.name_scope('some_scope'):
      # Basically from 0 to dim*num-1.
      flat_data = math_ops.linspace(0.0, dim * num - 1, dim * num)
      accum = variables.Variable(
          array_ops.reshape(flat_data, (num, dim)), name='accum')
    save = saver.Saver([accum])
    with self.test_session() as sess:
      variables.global_variables_initializer().run()
      self.bundle_file = os.path.join(test.get_temp_dir(), 'accum_checkpoint')
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

  def test_load_variable_slot_initializer(self):
    """Tests for the slot initializer wrapper."""
    # We have an initializer for each of two partitioned variables, which will
    # be [3, 1] and [2, 1].  The partitioning information is passed here in
    # initializer construction, as opposed to through a variable scope during
    # variable creation.
    variable_slot_initializer_part_0 = (
        contrib_framework.load_variable_slot_initializer(
            new_row_vocab_file=self.new_class_vocab_file,
            old_row_vocab_file=self.old_class_vocab_file,
            new_row_vocab_size=4,
            new_col_vocab_size=1,
            primary_partition_info=variable_scope._PartitionInfo(
                full_shape=[5, 1], var_offset=[0, 0]),
            old_tensor_name='some_scope/accum',
            ckpt_path=[self.bundle_file],
            num_row_oov_buckets=1,
            initializer=self.initializer))
    variable_slot_initializer_part_1 = (
        contrib_framework.load_variable_slot_initializer(
            new_row_vocab_file=self.new_class_vocab_file,
            old_row_vocab_file=self.old_class_vocab_file,
            new_row_vocab_size=4,
            new_col_vocab_size=1,
            primary_partition_info=variable_scope._PartitionInfo(
                full_shape=[5, 1], var_offset=[3, 0]),
            old_tensor_name='some_scope/accum',
            ckpt_path=[self.bundle_file],
            num_row_oov_buckets=1,
            initializer=self.initializer))

    expected_remapped_accum_vector_part_0 = np.reshape([2, 0, self.init_val],
                                                       [3, 1])

    expected_remapped_accum_vector_part_1 = np.reshape([1, self.init_val],
                                                       [2, 1])

    # Since there is no variable scope here, partition_info will be None, so
    # if variable_slot_initializer_part_0 and variable_slot_initializer_part_1
    # were instead instances of load_and_remap_matrix_initializer, the part_0
    # obtained vector would still be [2, 0, self.init_val], but the part_1
    # obtained vector would be [2, 0], since the partition_info would default to
    # assuming a single partition.
    remapped_accum_vector_part_0 = variable_scope.get_variable(
        name='accum/obtained_accum_vector_part_0',
        shape=[3, 1],
        initializer=variable_slot_initializer_part_0)
    remapped_accum_vector_part_1 = variable_scope.get_variable(
        name='accum/obtained_accum_vector_part_1',
        shape=[2, 1],
        initializer=variable_slot_initializer_part_1)

    with self.test_session():
      variables.global_variables_initializer().run()
      self.assertAllClose(expected_remapped_accum_vector_part_0,
                          remapped_accum_vector_part_0.eval())
      self.assertAllClose(expected_remapped_accum_vector_part_1,
                          remapped_accum_vector_part_1.eval())


if __name__ == '__main__':
  test.main()
