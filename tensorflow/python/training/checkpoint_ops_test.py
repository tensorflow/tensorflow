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
"""Functional tests for Python wrappers around warm-starting."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.training import saver as saver_lib


class LoadAndRemapWrappersTest(test.TestCase):
  """Tests for the functionality of the Python wrappers."""

  def setUp(self):
    ops.reset_default_graph()
    # Create the checkpoint file in a temporary directory.
    checkpoint_prefix = os.path.join(self.get_temp_dir(), 'model')
    # 0., 1., ..., 79. reshaped into [5, 16].
    initializer = init_ops.constant_initializer(
        np.reshape(np.linspace(0.0, 79, 5 * 16), (5, 16)))
    with self.test_session() as sess:
      with variable_scope.variable_scope('some_scope'):
        variable_scope.get_variable(name='embeddings', shape=[5, 16],
                                    initializer=initializer)
      sess.run(variables.global_variables_initializer())
      saver = saver_lib.Saver()
      saver.save(sess, checkpoint_prefix, global_step=5)
    self.checkpoint_file = '{}-5'.format(checkpoint_prefix)

    # Create the vocabulary files.
    self.new_feature_vocab_file = os.path.join(
        self.get_temp_dir(), 'new_feature_vocab.txt')
    with open(self.new_feature_vocab_file, 'w') as f:
      f.write('\n'.join(['zero', 'one', 'two', 'three', 'four']) + '\n')

    self.old_feature_vocab_file = os.path.join(
        self.get_temp_dir(), 'old_feature_vocab.txt')
    with open(self.old_feature_vocab_file, 'w') as f:
      f.write('\n'.join(['zero', 'one', 'two', 'three']) + '\n')

    self.new_class_vocab_file = os.path.join(
        self.get_temp_dir(), 'new_class_vocab.txt')
    with open(self.new_class_vocab_file, 'w') as f:
      f.write('\n'.join(['MISSING', 'knitting', 'flask', 'eminem']) + '\n')

    self.old_class_vocab_file = os.path.join(
        self.get_temp_dir(), 'old_class_vocab.txt')
    with open(self.old_class_vocab_file, 'w') as f:
      f.write('\n'.join(['knitting', 'eminem', 'MISSING']) + '\n')

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
        ckpt_path=[self.checkpoint_file],
        new_row_vocab_offset=1,
        initializer=self.initializer,
        num_row_oov_buckets=1,
        num_col_oov_buckets=1)

    # [4 in vocab + 1 oov features, 4 in vocab + 1 oov classes].  The offset
    # means we read from the first line.
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
    loading_initializer = (checkpoint_ops._load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_row_oov_buckets=1,
        num_col_oov_buckets=1,
        initializer=self.initializer))

    # The new weight matrix is of size
    # [5 feature vocab + 1 feature OOV, 4 class vocab + 1 class OOV].  Use a
    # partitioned variable to confirm that the offset logic works.
    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50, self.init_val, self.init_val], [6, 1]),
            np.reshape([0, 16, 32, 48, self.init_val, self.init_val], [6, 1]),
            np.reshape([self.init_val] * 6, [6, 1]),
            np.reshape([1, 17, 33, 49, self.init_val, self.init_val], [6, 1]),
            np.reshape([self.init_val] * 6, [6, 1])
        ],
        axis=1)
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
    loading_initializer = (checkpoint_ops._load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        num_col_oov_buckets=1,
        initializer=self.initializer))

    # The new weight matrix is of size
    # [5-sized input layer, 4 class vocab + 1 class OOV].
    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50, 66], [5, 1]),
            np.reshape([0, 16, 32, 48, 64], [5, 1]),
            np.reshape([self.init_val] * 5, [5, 1]),
            np.reshape([1, 17, 33, 49, 65], [5, 1]),
            np.reshape([self.init_val] * 5, [5, 1])
        ],
        axis=1)
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
    loading_initializer = (checkpoint_ops._load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_row_oov_buckets=5,
        num_col_oov_buckets=1,
        initializer=self.initializer))

    # The new weight matrix is of size
    # [5 feature vocab + 5 feature OOV, 4 class vocab + 1 class OOV].  The
    # second partition has only OOV.
    expected_remapped_matrix = np.concatenate(
        [
            np.reshape([2, 18, 34, 50] + [self.init_val] * 6, [10, 1]),
            np.reshape([0, 16, 32, 48] + [self.init_val] * 6, [10, 1]),
            np.reshape([self.init_val] * 10, [10, 1]),
            np.reshape([1, 17, 33, 49] + [self.init_val] * 6, [10, 1]),
            np.reshape([self.init_val] * 10, [10, 1]),
        ],
        axis=1)
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
    loading_initializer = (checkpoint_ops._load_and_remap_matrix_initializer(
        new_row_vocab_size=5,
        new_col_vocab_file=self.new_class_vocab_file,
        old_col_vocab_file=self.old_class_vocab_file,
        new_col_vocab_size=4,
        old_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        new_row_vocab_file=self.new_feature_vocab_file,
        old_row_vocab_file=self.old_feature_vocab_file,
        num_row_oov_buckets=1,
        num_col_oov_buckets=1))

    # Same as test_initializer_with_oov_only_partition, but with zero
    # initialization.
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
    embedding_loading_initializer = (checkpoint_ops._load_embedding_initializer(
        new_vocab_file=self.new_feature_vocab_file,
        old_vocab_file=self.old_feature_vocab_file,
        new_vocab_size=5,
        embedding_dim=16,
        embedding_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        num_oov_buckets=1,
        initializer=self.initializer))

    # The new weight matrix is of size
    # [5 feature vocab + 1 feature OOV, 16 (embedding dimension)], where the
    # last vocab row (2nd last row) is newly initialized (wasn't found in
    # previous vocab) and the actual last row is OOV and also newly initialized.
    # Use a partitioned variable to confirm that the offset logic works.
    expected_remapped_embeddings = np.concatenate(
        [
            np.reshape(range(64), [4, 16]),
            np.reshape([self.init_val] * 32, [2, 16]),
        ],
        axis=0)
    remapped_embeddings = variable_scope.get_variable(
        name='embedding/obtained_embedding_matrix',
        shape=[6, 16],
        initializer=embedding_loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.test_session():
      variables.global_variables_initializer().run()
      self.assertAllClose(expected_remapped_embeddings,
                          remapped_embeddings.as_tensor().eval())

  def test_load_embedding_initializer_large_oov(self):
    """Tests for the large OOV case for load_embedding_initializer wrapper."""
    self.new_feature_vocab_file = os.path.join(
        self.get_temp_dir(), 'new_feature_vocab.txt')
    with open(self.new_feature_vocab_file, 'w') as f:
      f.write('\n'.join(['one', 'zero', 'two', 'four']) + '\n')

    # Checkpoint has 5 entries, 3 of which correspond to OOV.
    self.old_feature_vocab_file = os.path.join(
        self.get_temp_dir(), 'old_feature_vocab.txt')
    with open(self.old_feature_vocab_file, 'w') as f:
      f.write('\n'.join(['zero', 'one']) + '\n')

    embedding_loading_initializer = (checkpoint_ops._load_embedding_initializer(
        new_vocab_file=self.new_feature_vocab_file,
        old_vocab_file=self.old_feature_vocab_file,
        new_vocab_size=4,
        embedding_dim=16,
        embedding_tensor_name='some_scope/embeddings',
        ckpt_path=[self.checkpoint_file],
        num_oov_buckets=5,
        initializer=self.initializer))

    # The new weight matrix is of size
    # [4 feature vocab + 5 feature OOV, 16 (embedding dimension)], where the
    # 3rd and 4th rows are not found in the old vocabulary and therefore newly
    # initialized.  The last five rows are OOV and also newly initialized.
    # Use a partitioned variable to confirm that the offset logic works.
    expected_remapped_embeddings = np.concatenate(
        [
            np.reshape(range(16, 32), [1, 16]),
            np.reshape(range(16), [1, 16]),
            np.reshape([self.init_val] * 112, [7, 16]),
        ],
        axis=0)
    remapped_embeddings = variable_scope.get_variable(
        name='embedding/obtained_embedding_matrix',
        shape=[9, 16],
        initializer=embedding_loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.test_session():
      variables.global_variables_initializer().run()
      self.assertAllClose(expected_remapped_embeddings,
                          remapped_embeddings.as_tensor().eval())

  def test_load_embedding_initializer_old_row_vocab(self):
    """Tests for load_embedding_initializer where we constrain old vocab."""
    embedding_loading_initializer = (
        checkpoint_ops._load_embedding_initializer(
            new_vocab_file=self.new_feature_vocab_file,
            old_vocab_file=self.old_feature_vocab_file,
            # Considered old vocabulary becomes ['zero', 'one', 'two'].  This
            # means 'three' in the new vocabulary is newly initialized.
            old_vocab_size=3,
            new_vocab_size=5,
            embedding_dim=16,
            embedding_tensor_name='some_scope/embeddings',
            ckpt_path=[self.checkpoint_file],
            num_oov_buckets=1,
            initializer=self.initializer))

    # The new weight matrix is of size
    # [5 feature vocab + 1 feature OOV, 16 (embedding dimension)], where the
    # last vocab row (2nd last row) is newly initialized (wasn't found in
    # previous vocab) and the actual last row is OOV and also newly initialized.
    # Use a partitioned variable to confirm that the offset logic works.
    expected_remapped_embeddings = np.concatenate(
        [
            np.reshape(range(48), [3, 16]),
            np.reshape([self.init_val] * 48, [3, 16]),
        ],
        axis=0)
    remapped_embeddings = variable_scope.get_variable(
        name='embedding/obtained_embedding_matrix',
        shape=[6, 16],
        initializer=embedding_loading_initializer,
        partitioner=partitioned_variables.fixed_size_partitioner(2))

    with self.test_session():
      variables.global_variables_initializer().run()
      self.assertAllClose(expected_remapped_embeddings,
                          remapped_embeddings.as_tensor().eval())

if __name__ == '__main__':
  test.main()
