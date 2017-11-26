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
"""Tests for warm_starting_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import six

from tensorflow.python.estimator import warm_starting_util as ws_util
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib

ones = init_ops.ones_initializer
norms = init_ops.truncated_normal_initializer
rand = init_ops.random_uniform_initializer


class WarmStartingUtilTest(test.TestCase):

  def _write_vocab(self, string_values, file_name):
    vocab_file = os.path.join(self.get_temp_dir(), file_name)
    with open(vocab_file, "w") as f:
      f.write("\n".join(string_values))
    return vocab_file

  def _write_checkpoint(self, sess):
    sess.run(variables.global_variables_initializer())
    saver = saver_lib.Saver()
    ckpt_prefix = os.path.join(self.get_temp_dir(), "model")
    ckpt_state_name = "checkpoint"
    saver.save(
        sess, ckpt_prefix, global_step=0, latest_filename=ckpt_state_name)

  def _create_prev_run_var(self,
                           var_name,
                           shape=None,
                           initializer=None,
                           partitioner=None):
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        var = variable_scope.get_variable(
            var_name,
            shape=shape,
            initializer=initializer,
            partitioner=partitioner)
        self._write_checkpoint(sess)
        if partitioner:
          self.assertTrue(isinstance(var, variables.PartitionedVariable))
          var = var._get_variable_list()
        return var, sess.run(var)

  def _create_prev_run_multiple_vars(self,
                                     var_names,
                                     initializers,
                                     shapes=None,
                                     partitioners=None):
    if not shapes:
      shapes = [None] * len(var_names)
    if not partitioners:
      partitioners = [None] * len(var_names)
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        var_list = []
        for var_name, shape, initializer, partitioner in zip(
            var_names, shapes, initializers, partitioners):
          var_list.append(
              variable_scope.get_variable(
                  var_name,
                  shape=shape,
                  initializer=initializer,
                  partitioner=partitioner))
        self._write_checkpoint(sess)
        run_vars = []
        for var, partitioner in zip(var_list, partitioners):
          if partitioner:
            self.assertTrue(isinstance(var, variables.PartitionedVariable))
            run_vars.append(sess.run(var._get_variable_list()))
          else:
            run_vars.append(sess.run(var))
        return var_list, run_vars

  def _create_dummy_inputs(self):
    return {
        "sc_int": array_ops.sparse_placeholder(dtypes.int32),
        "sc_hash": array_ops.sparse_placeholder(dtypes.string),
        "sc_keys": array_ops.sparse_placeholder(dtypes.string),
        "sc_vocab": array_ops.sparse_placeholder(dtypes.string),
        "real": array_ops.placeholder(dtypes.float32)
    }

  def _create_linear_model(self, feature_cols, partitioner):
    cols_to_vars = {}
    with variable_scope.variable_scope("", partitioner=partitioner):
      # Create the variables.
      fc.linear_model(
          features=self._create_dummy_inputs(),
          feature_columns=feature_cols,
          units=1,
          cols_to_vars=cols_to_vars)
    # Return a dictionary mapping each column to its variable, dropping the
    # 'bias' key that's also filled.
    cols_to_vars.pop("bias")
    return cols_to_vars

  def _assert_cols_to_vars(self, cols_to_vars, cols_to_expected_values, sess):
    for col, expected_values in six.iteritems(cols_to_expected_values):
      for i, var in enumerate(cols_to_vars[col]):
        self.assertAllClose(expected_values[i], var.eval(sess))

  def testWarmStartVar(self):
    _, prev_val = self._create_prev_run_var(
        "fruit_weights", initializer=[[0.5], [1.], [1.5], [2.]])

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.]])
        ws_util._warmstart_var(fruit_weights, self.get_temp_dir())
        sess.run(variables.global_variables_initializer())
        self.assertAllEqual(prev_val, fruit_weights.eval(sess))

  def testWarmStartVarPrevVarPartitioned(self):
    _, weights = self._create_prev_run_var(
        "fruit_weights",
        shape=[4, 1],
        initializer=[[0.5], [1.], [1.5], [2.]],
        partitioner=lambda shape, dtype: [2, 1])
    prev_val = np.concatenate([weights[0], weights[1]], axis=0)

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.]])
        ws_util._warmstart_var(fruit_weights, self.get_temp_dir())
        sess.run(variables.global_variables_initializer())
        self.assertAllEqual(prev_val, fruit_weights.eval(sess))

  def testWarmStartVarCurrentVarPartitioned(self):
    _, prev_val = self._create_prev_run_var(
        "fruit_weights", initializer=[[0.5], [1.], [1.5], [2.]])

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights",
            shape=[4, 1],
            initializer=[[0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        self.assertTrue(
            isinstance(fruit_weights, variables.PartitionedVariable))
        ws_util._warmstart_var(fruit_weights, self.get_temp_dir())
        sess.run(variables.global_variables_initializer())
        fruit_weights = fruit_weights._get_variable_list()
        new_val = np.concatenate(
            [fruit_weights[0].eval(sess), fruit_weights[1].eval(sess)], axis=0)
        self.assertAllEqual(prev_val, new_val)

  def testWarmStartVarBothVarsPartitioned(self):
    _, weights = self._create_prev_run_var(
        "old_scope/fruit_weights",
        shape=[4, 1],
        initializer=[[0.5], [1.], [1.5], [2.]],
        partitioner=lambda shape, dtype: [2, 1])
    prev_val = np.concatenate([weights[0], weights[1]], axis=0)
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "new_scope/fruit_weights",
            shape=[4, 1],
            initializer=[[0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        self.assertTrue(
            isinstance(fruit_weights, variables.PartitionedVariable))
        ws_util._warmstart_var(
            fruit_weights,
            self.get_temp_dir(),
            prev_tensor_name="old_scope/fruit_weights")
        sess.run(variables.global_variables_initializer())
        fruit_weights = fruit_weights._get_variable_list()
        new_val = np.concatenate(
            [fruit_weights[0].eval(sess), fruit_weights[1].eval(sess)], axis=0)
        self.assertAllEqual(prev_val, new_val)

  def testWarmStartVarMultipleVars(self):
    _, prev_vals = self._create_prev_run_multiple_vars(
        var_names=["fruit_weights", "other_weights"],
        initializers=[[[0.5], [1.], [1.5], [2.]], [[.05], [.1], [.15], [.2]]])

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.]])
        other_weights = variable_scope.get_variable(
            "other_weights", initializer=[[0.], [0.], [0.], [0.]])
        ws_util._warmstart_var([fruit_weights, other_weights],
                               self.get_temp_dir())
        sess.run(variables.global_variables_initializer())
        self.assertAllEqual(prev_vals[0], fruit_weights.eval(sess))
        self.assertAllEqual(prev_vals[1], other_weights.eval(sess))

  def testWarmStartVarMultipleVarsBothPartitioned(self):
    _, prev_vals = self._create_prev_run_multiple_vars(
        var_names=["fruit_weights", "other_weights"],
        shapes=[[4, 1], [4, 1]],
        initializers=[[[0.5], [1.], [1.5], [2.]], [[.05], [.1], [.15], [.2]]],
        partitioners=[lambda shape, dtype: [2, 1], lambda shape, dtype: [2, 1]])

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights",
            shape=[4, 1],
            initializer=[[0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        other_weights = variable_scope.get_variable(
            "other_weights",
            shape=[4, 1],
            initializer=[[0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        ws_util._warmstart_var([fruit_weights, other_weights],
                               self.get_temp_dir())
        sess.run(variables.global_variables_initializer())
        fruit_weights = fruit_weights._get_variable_list()
        new_fruit_weights_val = np.concatenate(
            [fruit_weights[0].eval(sess), fruit_weights[1].eval(sess)], axis=0)
        other_weights = other_weights._get_variable_list()
        new_other_weights_val = np.concatenate(
            [other_weights[0].eval(sess), other_weights[1].eval(sess)], axis=0)
        self.assertAllEqual(
            np.concatenate(prev_vals[0], axis=0), new_fruit_weights_val)
        self.assertAllEqual(
            np.concatenate(prev_vals[1], axis=0), new_other_weights_val)

  def testWarmStartVarMultipleVarsMixOfPartitions(self):
    # First is not partitioned, but the second two are.
    _, prev_vals = self._create_prev_run_multiple_vars(
        var_names=["fruit_weights", "other_weights", "veggie_weights"],
        shapes=[None, [4, 1], [4, 1]],
        initializers=[[[0.5], [1.], [1.5], [2.]], [[.05], [.1], [.15], [.2]],
                      [[5.], [10.], [15.], [20.]]],
        partitioners=[
            None, lambda shape, dtype: [2, 1], lambda shape, dtype: [2, 1]
        ])

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.]])
        other_weights = variable_scope.get_variable(
            "other_weights",
            shape=[4, 1],
            initializer=[[0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        veggie_weights = variable_scope.get_variable(
            "veggie_weights",
            shape=[4, 1],
            initializer=[[0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        # Flatten one of the partitioned variables.
        ws_util._warmstart_var([fruit_weights, other_weights] +
                               veggie_weights._get_variable_list(),
                               self.get_temp_dir())
        sess.run(variables.global_variables_initializer())
        veggie_weights = veggie_weights._get_variable_list()
        new_veggie_weights_val = np.concatenate(
            [veggie_weights[0].eval(sess), veggie_weights[1].eval(sess)],
            axis=0)
        other_weights = other_weights._get_variable_list()
        new_other_weights_val = np.concatenate(
            [other_weights[0].eval(sess), other_weights[1].eval(sess)], axis=0)
        self.assertAllEqual(prev_vals[0], fruit_weights.eval(sess))
        self.assertAllEqual(
            np.concatenate(prev_vals[1], axis=0), new_other_weights_val)
        self.assertAllEqual(
            np.concatenate(prev_vals[2], axis=0), new_veggie_weights_val)

  def testWarmStartVarWithVocab(self):
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    _, _ = self._create_prev_run_var(
        "fruit_weights", initializer=[[0.5], [1.], [1.5], [2.]])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry"], "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.], [0.]])
        ws_util._warmstart_var_with_vocab(fruit_weights, new_vocab_path, 5,
                                          self.get_temp_dir(), prev_vocab_path)
        sess.run(variables.global_variables_initializer())
        self.assertAllEqual([[2.], [1.5], [1.], [0.5], [0.]],
                            fruit_weights.eval(sess))

  def testWarmStartVarWithVocabConstrainedOldVocabSize(self):
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    _, _ = self._create_prev_run_var(
        "fruit_weights", initializer=[[0.5], [1.], [1.5], [2.]])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry"], "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.], [0.]])
        ws_util._warmstart_var_with_vocab(
            fruit_weights,
            new_vocab_path,
            5,
            self.get_temp_dir(),
            prev_vocab_path,
            previous_vocab_size=2)
        sess.run(variables.global_variables_initializer())
        # Old vocabulary limited to ['apple', 'banana'].
        self.assertAllEqual([[0.], [0.], [1.], [0.5], [0.]],
                            fruit_weights.eval(sess))

  def testWarmStartVarWithVocabPrevVarPartitioned(self):
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    _, _ = self._create_prev_run_var(
        "fruit_weights",
        shape=[4, 1],
        initializer=[[0.5], [1.], [1.5], [2.]],
        partitioner=lambda shape, dtype: [2, 1])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry"], "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights", initializer=[[0.], [0.], [0.], [0.], [0.]])
        ws_util._warmstart_var_with_vocab(fruit_weights, new_vocab_path, 5,
                                          self.get_temp_dir(), prev_vocab_path)
        sess.run(variables.global_variables_initializer())
        self.assertAllEqual([[2.], [1.5], [1.], [0.5], [0.]],
                            fruit_weights.eval(sess))

  def testWarmStartVarWithVocabCurrentVarPartitioned(self):
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    _, _ = self._create_prev_run_var(
        "fruit_weights", initializer=[[0.5], [1.], [1.5], [2.]])

    # New vocab with elements in reverse order and one new element.
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry"], "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights",
            shape=[6, 1],
            initializer=[[0.], [0.], [0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        ws_util._warmstart_var_with_vocab(
            fruit_weights,
            new_vocab_path,
            5,
            self.get_temp_dir(),
            prev_vocab_path,
            current_oov_buckets=1)
        sess.run(variables.global_variables_initializer())
        self.assertTrue(
            isinstance(fruit_weights, variables.PartitionedVariable))
        fruit_weights_vars = fruit_weights._get_variable_list()
        self.assertAllEqual([[2.], [1.5], [1.]],
                            fruit_weights_vars[0].eval(sess))
        self.assertAllEqual([[0.5], [0.], [0.]],
                            fruit_weights_vars[1].eval(sess))

  def testWarmStartVarWithVocabBothVarsPartitioned(self):
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    _, _ = self._create_prev_run_var(
        "fruit_weights",
        shape=[4, 1],
        initializer=[[0.5], [1.], [1.5], [2.]],
        partitioner=lambda shape, dtype: [2, 1])

    # New vocab with elements in reverse order and two new elements.
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry",
         "blueberry"], "new_vocab")
    # New session and new graph.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        fruit_weights = variable_scope.get_variable(
            "fruit_weights",
            shape=[6, 1],
            initializer=[[0.], [0.], [0.], [0.], [0.], [0.]],
            partitioner=lambda shape, dtype: [2, 1])
        ws_util._warmstart_var_with_vocab(fruit_weights, new_vocab_path, 6,
                                          self.get_temp_dir(), prev_vocab_path)
        sess.run(variables.global_variables_initializer())
        self.assertTrue(
            isinstance(fruit_weights, variables.PartitionedVariable))
        fruit_weights_vars = fruit_weights._get_variable_list()
        self.assertAllEqual([[2.], [1.5], [1.]],
                            fruit_weights_vars[0].eval(sess))
        self.assertAllEqual([[0.5], [0.], [0.]],
                            fruit_weights_vars[1].eval(sess))

  def testWarmStartInputLayer_SparseColumnIntegerized(self):
    # Create feature column.
    sc_int = fc.categorical_column_with_identity("sc_int", num_buckets=10)

    # Save checkpoint from which to warm-start.
    _, prev_int_val = self._create_prev_run_var(
        "linear_model/sc_int/weights", shape=[10, 1], initializer=ones())
    # Verify we initialized the values correctly.
    self.assertAllEqual(np.ones([10, 1]), prev_int_val)

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_int], partitioner)
        sess.run(variables.global_variables_initializer())
        # Without warmstarting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {sc_int: [np.zeros([10, 1])]},
                                  sess)

    # New graph, new session with warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_int], partitioner)
        ws_util._warmstart_input_layer(cols_to_vars,
                                       ws_util._WarmStartSettings(
                                           self.get_temp_dir()))
        sess.run(variables.global_variables_initializer())
        # Verify weights were correctly warmstarted.
        self._assert_cols_to_vars(cols_to_vars, {sc_int: [prev_int_val]}, sess)

  def testWarmStartInputLayer_SparseColumnHashed(self):
    # Create feature column.
    sc_hash = fc.categorical_column_with_hash_bucket(
        "sc_hash", hash_bucket_size=15)

    # Save checkpoint from which to warm-start.
    _, prev_hash_val = self._create_prev_run_var(
        "linear_model/sc_hash/weights", shape=[15, 1], initializer=norms())

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_hash], partitioner)
        sess.run(variables.global_variables_initializer())
        # Without warmstarting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {sc_hash: [np.zeros([15, 1])]},
                                  sess)

    # New graph, new session with warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_hash], partitioner)
        ws_util._warmstart_input_layer(cols_to_vars,
                                       ws_util._WarmStartSettings(
                                           self.get_temp_dir()))
        sess.run(variables.global_variables_initializer())
        # Verify weights were correctly warmstarted.
        self._assert_cols_to_vars(cols_to_vars, {sc_hash: [prev_hash_val]},
                                  sess)

  def testWarmStartInputLayer_SparseColumnVocabulary(self):
    # Create vocab for sparse column "sc_vocab".
    vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                   "vocab")
    # Create feature column.
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=vocab_path, vocabulary_size=4)

    # Save checkpoint from which to warm-start.
    _, prev_vocab_val = self._create_prev_run_var(
        "linear_model/sc_vocab/weights", shape=[4, 1], initializer=ones())

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
        sess.run(variables.global_variables_initializer())
        # Without warmstarting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [np.zeros([4, 1])]},
                                  sess)

    # New graph, new session with warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
        # Since old vocab is not explicitly set in WarmStartSettings, the old
        # vocab is assumed to be same as new vocab.
        ws_util._warmstart_input_layer(cols_to_vars,
                                       ws_util._WarmStartSettings(
                                           self.get_temp_dir()))
        sess.run(variables.global_variables_initializer())
        # Verify weights were correctly warmstarted.
        self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [prev_vocab_val]},
                                  sess)

  def testWarmStartInputLayer_SparseColumnVocabularyConstrainedVocabSizes(self):
    # Create old vocabulary, and use a size smaller than the total number of
    # entries.
    old_vocab_path = self._write_vocab(["apple", "guava", "banana"],
                                       "old_vocab")
    old_vocab_size = 2  # ['apple', 'guava']

    # Create new vocab for sparse column "sc_vocab".
    current_vocab_path = self._write_vocab(
        ["apple", "banana", "guava", "orange"], "current_vocab")
    # Create feature column.  Only use 2 of the actual entries, resulting in
    # ['apple', 'banana'] for the new vocabulary.
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=current_vocab_path, vocabulary_size=2)

    # Save checkpoint from which to warm-start.
    self._create_prev_run_var(
        "linear_model/sc_vocab/weights", shape=[2, 1], initializer=ones())

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
        sess.run(variables.global_variables_initializer())
        # Without warmstarting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [np.zeros([2, 1])]},
                                  sess)

    # New graph, new session with warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([sc_vocab], partitioner)
        warmstart_settings = ws_util._WarmStartSettings(
            ckpt_to_initialize_from=self.get_temp_dir(),
            col_to_prev_vocab={
                sc_vocab: (old_vocab_path, old_vocab_size)
            })
        ws_util._warmstart_input_layer(cols_to_vars, warmstart_settings)
        sess.run(variables.global_variables_initializer())
        # Verify weights were correctly warmstarted.  'banana' isn't in the
        # first two entries of the old vocabulary, so it's newly initialized.
        self._assert_cols_to_vars(cols_to_vars, {sc_vocab: [[[1], [0]]]}, sess)

  def testWarmStartInputLayer_BucketizedColumn(self):
    # Create feature column.
    real = fc.numeric_column("real")
    real_bucket = fc.bucketized_column(real, boundaries=[0., 1., 2., 3.])

    # Save checkpoint from which to warm-start.
    _, prev_bucket_val = self._create_prev_run_var(
        "linear_model/real_bucketized/weights",
        shape=[5, 1],
        initializer=norms())

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([real_bucket], partitioner)
        sess.run(variables.global_variables_initializer())
        # Without warmstarting, the weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars,
                                  {real_bucket: [np.zeros([5, 1])]}, sess)

    # New graph, new session with warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model([real_bucket], partitioner)
        ws_util._warmstart_input_layer(cols_to_vars,
                                       ws_util._WarmStartSettings(
                                           self.get_temp_dir()))
        sess.run(variables.global_variables_initializer())
        # Verify weights were correctly warmstarted.
        self._assert_cols_to_vars(cols_to_vars,
                                  {real_bucket: [prev_bucket_val]}, sess)

  def testWarmStartInputLayer_MultipleCols(self):
    # Create vocab for sparse column "sc_vocab".
    vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                   "vocab")

    # Create feature columns.
    sc_int = fc.categorical_column_with_identity("sc_int", num_buckets=10)
    sc_hash = fc.categorical_column_with_hash_bucket(
        "sc_hash", hash_bucket_size=15)
    sc_keys = fc.categorical_column_with_vocabulary_list(
        "sc_keys", vocabulary_list=["a", "b", "c", "e"])
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=vocab_path, vocabulary_size=4)
    real = fc.numeric_column("real")
    real_bucket = fc.bucketized_column(real, boundaries=[0., 1., 2., 3.])
    cross = fc.crossed_column([sc_keys, sc_vocab], hash_bucket_size=20)
    all_linear_cols = [sc_int, sc_hash, sc_keys, sc_vocab, real_bucket, cross]

    # Save checkpoint from which to warm-start.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        sc_int_weights = variable_scope.get_variable(
            "linear_model/sc_int/weights", shape=[10, 1], initializer=ones())
        sc_hash_weights = variable_scope.get_variable(
            "linear_model/sc_hash/weights", shape=[15, 1], initializer=norms())
        sc_keys_weights = variable_scope.get_variable(
            "linear_model/sc_keys/weights", shape=[4, 1], initializer=rand())
        sc_vocab_weights = variable_scope.get_variable(
            "linear_model/sc_vocab/weights", shape=[4, 1], initializer=ones())
        real_bucket_weights = variable_scope.get_variable(
            "linear_model/real_bucketized/weights",
            shape=[5, 1],
            initializer=norms())
        cross_weights = variable_scope.get_variable(
            "linear_model/sc_keys_X_sc_vocab/weights",
            shape=[20, 1],
            initializer=rand())
        self._write_checkpoint(sess)
        (prev_int_val, prev_hash_val, prev_keys_val, prev_vocab_val,
         prev_bucket_val, prev_cross_val) = sess.run([
             sc_int_weights, sc_hash_weights, sc_keys_weights, sc_vocab_weights,
             real_bucket_weights, cross_weights
         ])
        # Verify we initialized the values correctly.
        self.assertAllEqual(np.ones([10, 1]), prev_int_val)

    partitioner = lambda shape, dtype: [1] * len(shape)
    # New graph, new session WITHOUT warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model(all_linear_cols, partitioner)
        sess.run(variables.global_variables_initializer())
        # Without warmstarting, all weights should be initialized using default
        # initializer (which is init_ops.zeros_initializer).
        self._assert_cols_to_vars(cols_to_vars, {
            sc_int: [np.zeros([10, 1])],
            sc_hash: [np.zeros([15, 1])],
            sc_keys: [np.zeros([4, 1])],
            sc_vocab: [np.zeros([4, 1])],
            real_bucket: [np.zeros([5, 1])],
            cross: [np.zeros([20, 1])],
        }, sess)

    # New graph, new session with warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model(all_linear_cols, partitioner)
        ws_util._warmstart_input_layer(cols_to_vars,
                                       ws_util._WarmStartSettings(
                                           self.get_temp_dir()))
        sess.run(variables.global_variables_initializer())
        # Verify weights were correctly warmstarted.
        self._assert_cols_to_vars(cols_to_vars, {
            sc_int: [prev_int_val],
            sc_hash: [prev_hash_val],
            sc_keys: [prev_keys_val],
            sc_vocab: [prev_vocab_val],
            real_bucket: [prev_bucket_val],
            cross: [prev_cross_val],
        }, sess)

  def testWarmStartInputLayerMoreSettings(self):
    # Create old and new vocabs for sparse column "sc_vocab".
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry",
         "blueberry"], "new_vocab")
    # Create feature columns.
    sc_hash = fc.categorical_column_with_hash_bucket(
        "sc_hash", hash_bucket_size=15)
    sc_keys = fc.categorical_column_with_vocabulary_list(
        "sc_keys", vocabulary_list=["a", "b", "c", "e"])
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=new_vocab_path, vocabulary_size=6)
    all_linear_cols = [sc_hash, sc_keys, sc_vocab]

    # Save checkpoint from which to warm-start.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        _ = variable_scope.get_variable(
            "linear_model/sc_hash/weights", shape=[15, 1], initializer=norms())
        sc_keys_weights = variable_scope.get_variable(
            "some_other_name", shape=[4, 1], initializer=rand())
        _ = variable_scope.get_variable(
            "linear_model/sc_vocab/weights",
            initializer=[[0.5], [1.], [2.], [3.]])
        self._write_checkpoint(sess)
        prev_keys_val = sess.run(sc_keys_weights)

    def _partitioner(shape, dtype):  # pylint:disable=unused-argument
      # Partition each var into 2 equal slices.
      partitions = [1] * len(shape)
      partitions[0] = min(2, shape[0].value)
      return partitions

    # New graph, new session with warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = self._create_linear_model(all_linear_cols, _partitioner)
        ws_settings = ws_util._WarmStartSettings(
            self.get_temp_dir(),
            col_to_prev_vocab={sc_vocab: prev_vocab_path},
            col_to_prev_tensor={sc_keys: "some_other_name"},
            exclude_columns=[sc_hash])
        ws_util._warmstart_input_layer(cols_to_vars, ws_settings)
        sess.run(variables.global_variables_initializer())
        # Verify weights were correctly warmstarted.  Var corresponding to
        # sc_hash should not be warm-started.  Var corresponding to sc_vocab
        # should be correctly warmstarted after vocab remapping.
        self._assert_cols_to_vars(cols_to_vars, {
            sc_keys:
                np.split(prev_keys_val, 2),
            sc_hash: [np.zeros([8, 1]), np.zeros([7, 1])],
            sc_vocab: [
                np.array([[3.], [2.], [1.]]),
                np.array([[0.5], [0.], [0.]])
            ]
        }, sess)

  def testWarmStartInputLayerEmbeddingColumn(self):
    # Create old and new vocabs for embedding column "sc_vocab".
    prev_vocab_path = self._write_vocab(["apple", "banana", "guava", "orange"],
                                        "old_vocab")
    new_vocab_path = self._write_vocab(
        ["orange", "guava", "banana", "apple", "raspberry", "blueberry"],
        "new_vocab")

    # Save checkpoint from which to warm-start.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        _ = variable_scope.get_variable(
            "input_layer/sc_vocab_embedding/embedding_weights",
            initializer=[[0.5, 0.4], [1., 1.1], [2., 2.2], [3., 3.3]])
        self._write_checkpoint(sess)

    def _partitioner(shape, dtype):  # pylint:disable=unused-argument
      # Partition each var into 2 equal slices.
      partitions = [1] * len(shape)
      partitions[0] = min(2, shape[0].value)
      return partitions

    # Create feature columns.
    sc_vocab = fc.categorical_column_with_vocabulary_file(
        "sc_vocab", vocabulary_file=new_vocab_path, vocabulary_size=6)
    emb_vocab = fc.embedding_column(
        categorical_column=sc_vocab,
        dimension=2,
        # Can't use constant_initializer with load_and_remap.  In practice,
        # use a truncated normal initializer.
        initializer=init_ops.random_uniform_initializer(
            minval=0.42, maxval=0.42))
    all_deep_cols = [emb_vocab]
    # New graph, new session with warmstarting.
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        cols_to_vars = {}
        with variable_scope.variable_scope("", partitioner=_partitioner):
          # Create the variables.
          fc.input_layer(
              features=self._create_dummy_inputs(),
              feature_columns=all_deep_cols,
              cols_to_vars=cols_to_vars)
        ws_settings = ws_util._WarmStartSettings(
            self.get_temp_dir(), col_to_prev_vocab={
                emb_vocab: prev_vocab_path
            })
        ws_util._warmstart_input_layer(cols_to_vars, ws_settings)
        sess.run(variables.global_variables_initializer())
        # Verify weights were correctly warmstarted. Var corresponding to
        # emb_vocab should be correctly warmstarted after vocab remapping.
        # Missing values are filled in with the EmbeddingColumn's initializer.
        self._assert_cols_to_vars(
            cols_to_vars, {
                emb_vocab: [
                    np.array([[3., 3.3], [2., 2.2], [1., 1.1]]),
                    np.array([[0.5, 0.4], [0.42, 0.42], [0.42, 0.42]])
                ]
            }, sess)

  def testErrorConditions(self):
    self.assertRaises(ValueError, ws_util._WarmStartSettings, None)
    x = variable_scope.get_variable(
        "x",
        shape=[4, 1],
        initializer=ones(),
        partitioner=lambda shape, dtype: [2, 1])

    # List of PartitionedVariable is invalid type when warmstarting with vocab.
    self.assertRaises(TypeError, ws_util._warmstart_var_with_vocab, [x], "/tmp",
                      5, "/tmp", "/tmp")
    # Keys of type other than FeatureColumn.
    self.assertRaises(TypeError, ws_util._warmstart_input_layer,
                      {"StringType": x}, ws_util._WarmStartSettings("/tmp"))


if __name__ == "__main__":
  test.main()
