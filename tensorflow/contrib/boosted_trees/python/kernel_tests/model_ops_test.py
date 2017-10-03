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
"""Tests for the GTFlow model ops.

The tests cover:
- Loading a model from protobufs.
- Running Predictions using an existing model.
- Serializing the model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.boosted_trees.proto import tree_config_pb2
from tensorflow.contrib.boosted_trees.python.ops import model_ops
from tensorflow.contrib.boosted_trees.python.ops import prediction_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import saver


def _append_to_leaf(leaf, c_id, w):
  """Helper method for building tree leaves.

  Appends weight contributions for the given class index to a leaf node.

  Args:
    leaf: leaf node to append to.
    c_id: class Id for the weight update.
    w: weight contribution value.
  """
  leaf.sparse_vector.index.append(c_id)
  leaf.sparse_vector.value.append(w)


def _set_float_split(split, feat_col, thresh, l_id, r_id):
  """Helper method for building tree float splits.

  Sets split feature column, threshold and children.

  Args:
    split: split node to update.
    feat_col: feature column for the split.
    thresh: threshold to split on forming rule x <= thresh.
    l_id: left child Id.
    r_id: right child Id.
  """
  split.feature_column = feat_col
  split.threshold = thresh
  split.left_id = l_id
  split.right_id = r_id


class ModelOpsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    """Sets up test for model_ops.

    Create a batch of two examples having one dense float, two sparse float and
    one sparse int features.
    The data looks like the following:
    | Instance | Dense0 | SparseF0 | SparseF1 | SparseI0 |
    | 0        |  7     |    -3    |          |          |
    | 1        | -2     |          | 4        |   9,1    |
    """
    super(ModelOpsTest, self).setUp()
    self._dense_float_tensor = np.array([[7.0], [-2.0]])
    self._sparse_float_indices1 = np.array([[0, 0]])
    self._sparse_float_values1 = np.array([-3.0])
    self._sparse_float_shape1 = np.array([2, 1])
    self._sparse_float_indices2 = np.array([[1, 0]])
    self._sparse_float_values2 = np.array([4.0])
    self._sparse_float_shape2 = np.array([2, 1])
    self._sparse_int_indices1 = np.array([[1, 0], [1, 1]])
    self._sparse_int_values1 = np.array([9, 1])
    self._sparse_int_shape1 = np.array([2, 2])
    self._seed = 123

  def testCreate(self):
    with self.test_session():
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree = tree_ensemble_config.trees.add()
      _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)
      tree_ensemble_config.tree_weights.append(1.0)

      # Prepare learner config.
      learner_config = learner_pb2.LearnerConfig()
      learner_config.num_classes = 2

      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=3,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="create_tree")
      resources.initialize_resources(resources.shared_resources()).run()

      result, _, _ = prediction_ops.gradient_trees_prediction(
          tree_ensemble_handle,
          self._seed, [self._dense_float_tensor], [
              self._sparse_float_indices1, self._sparse_float_indices2
          ], [self._sparse_float_values1, self._sparse_float_values2],
          [self._sparse_float_shape1,
           self._sparse_float_shape2], [self._sparse_int_indices1],
          [self._sparse_int_values1], [self._sparse_int_shape1],
          learner_config=learner_config.SerializeToString(),
          apply_dropout=False,
          apply_averaging=False,
          center_bias=False,
          reduce_dim=True)
      self.assertAllClose(result.eval(), [[-0.4], [-0.4]])
      stamp_token = model_ops.tree_ensemble_stamp_token(tree_ensemble_handle)
      self.assertEqual(stamp_token.eval(), 3)

  def testSerialization(self):
    with ops.Graph().as_default() as graph:
      with self.test_session(graph):
        tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
        # Bias tree only for second class.
        tree1 = tree_ensemble_config.trees.add()
        _append_to_leaf(tree1.nodes.add().leaf, 1, -0.2)

        tree_ensemble_config.tree_weights.append(1.0)

        # Depth 2 tree.
        tree2 = tree_ensemble_config.trees.add()
        tree_ensemble_config.tree_weights.append(1.0)
        _set_float_split(tree2.nodes.add()
                         .sparse_float_binary_split_default_right.split, 1, 4.0,
                         1, 2)
        _set_float_split(tree2.nodes.add().dense_float_binary_split, 0, 9.0, 3,
                         4)
        _append_to_leaf(tree2.nodes.add().leaf, 0, 0.5)
        _append_to_leaf(tree2.nodes.add().leaf, 1, 1.2)
        _append_to_leaf(tree2.nodes.add().leaf, 0, -0.9)

        tree_ensemble_handle = model_ops.tree_ensemble_variable(
            stamp_token=7,
            tree_ensemble_config=tree_ensemble_config.SerializeToString(),
            name="saver_tree")
        stamp_token, serialized_config = model_ops.tree_ensemble_serialize(
            tree_ensemble_handle)
        resources.initialize_resources(resources.shared_resources()).run()
        self.assertEqual(stamp_token.eval(), 7)
        serialized_config = serialized_config.eval()

    with ops.Graph().as_default() as graph:
      with self.test_session(graph):
        tree_ensemble_handle2 = model_ops.tree_ensemble_variable(
            stamp_token=9,
            tree_ensemble_config=serialized_config,
            name="saver_tree2")
        resources.initialize_resources(resources.shared_resources()).run()

        # Prepare learner config.
        learner_config = learner_pb2.LearnerConfig()
        learner_config.num_classes = 3

        result, _, _ = prediction_ops.gradient_trees_prediction(
            tree_ensemble_handle2,
            self._seed, [self._dense_float_tensor], [
                self._sparse_float_indices1, self._sparse_float_indices2
            ], [self._sparse_float_values1, self._sparse_float_values2],
            [self._sparse_float_shape1,
             self._sparse_float_shape2], [self._sparse_int_indices1],
            [self._sparse_int_values1], [self._sparse_int_shape1],
            learner_config=learner_config.SerializeToString(),
            apply_dropout=False,
            apply_averaging=False,
            center_bias=False,
            reduce_dim=True)

        # Re-serialize tree.
        stamp_token2, serialized_config2 = model_ops.tree_ensemble_serialize(
            tree_ensemble_handle2)

        # The first example will get bias class 1 -0.2 from first tree and
        # leaf 2 payload (sparse feature missing) of 0.5 hence [0.5, -0.2],
        # the second example will get the same bias class 1 -0.2 and leaf 3
        # payload of class 1 1.2 hence [0.0, 1.0].
        self.assertEqual(stamp_token2.eval(), 9)

        # Class 2 does have scores in the leaf => it gets score 0.
        self.assertEqual(serialized_config2.eval(), serialized_config)
        self.assertAllClose(result.eval(), [[0.5, -0.2], [0, 1.0]])

  def testRestore(self):
    # Calling self.test_session() without a graph specified results in
    # TensorFlowTestCase caching the session and returning the same one
    # every time. In this test, we need to create two different sessions
    # which is why we also create a graph and pass it to self.test_session()
    # to ensure no caching occurs under the hood.
    save_path = os.path.join(self.get_temp_dir(), "restore-test")
    with ops.Graph().as_default() as graph:
      with self.test_session(graph) as sess:
        # Prepare learner config.
        learner_config = learner_pb2.LearnerConfig()
        learner_config.num_classes = 2

        # Add the first tree and save.
        tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
        tree = tree_ensemble_config.trees.add()
        tree_ensemble_config.tree_metadata.add().is_finalized = True
        tree_ensemble_config.tree_weights.append(1.0)
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.1)
        tree_ensemble_handle = model_ops.tree_ensemble_variable(
            stamp_token=3,
            tree_ensemble_config=tree_ensemble_config.SerializeToString(),
            name="restore_tree")
        resources.initialize_resources(resources.shared_resources()).run()
        variables.initialize_all_variables().run()
        my_saver = saver.Saver()

        # Add the second tree and replace the ensemble of the handle.
        tree2 = tree_ensemble_config.trees.add()
        tree_ensemble_config.tree_weights.append(1.0)
        _append_to_leaf(tree2.nodes.add().leaf, 0, -1.0)
        # Predict to confirm.
        with ops.control_dependencies([
            model_ops.tree_ensemble_deserialize(
                tree_ensemble_handle,
                stamp_token=3,
                tree_ensemble_config=tree_ensemble_config.SerializeToString())
        ]):
          result, _, _ = prediction_ops.gradient_trees_prediction(
              tree_ensemble_handle,
              self._seed, [self._dense_float_tensor], [
                  self._sparse_float_indices1, self._sparse_float_indices2
              ], [self._sparse_float_values1, self._sparse_float_values2],
              [self._sparse_float_shape1,
               self._sparse_float_shape2], [self._sparse_int_indices1],
              [self._sparse_int_values1], [self._sparse_int_shape1],
              learner_config=learner_config.SerializeToString(),
              apply_dropout=False,
              apply_averaging=False,
              center_bias=False,
              reduce_dim=True)
        self.assertAllClose([[-1.1], [-1.1]], result.eval())
        # Save before adding other trees.
        val = my_saver.save(sess, save_path)
        self.assertEqual(save_path, val)

        # Add more trees after saving.
        tree3 = tree_ensemble_config.trees.add()
        tree_ensemble_config.tree_weights.append(1.0)
        _append_to_leaf(tree3.nodes.add().leaf, 0, -10.0)
        # Predict to confirm.
        with ops.control_dependencies([
            model_ops.tree_ensemble_deserialize(
                tree_ensemble_handle,
                stamp_token=3,
                tree_ensemble_config=tree_ensemble_config.SerializeToString())
        ]):
          result, _, _ = prediction_ops.gradient_trees_prediction(
              tree_ensemble_handle,
              self._seed, [self._dense_float_tensor], [
                  self._sparse_float_indices1, self._sparse_float_indices2
              ], [self._sparse_float_values1, self._sparse_float_values2],
              [self._sparse_float_shape1,
               self._sparse_float_shape2], [self._sparse_int_indices1],
              [self._sparse_int_values1], [self._sparse_int_shape1],
              learner_config=learner_config.SerializeToString(),
              apply_dropout=False,
              apply_averaging=False,
              center_bias=False,
              reduce_dim=True)
        self.assertAllClose(result.eval(), [[-11.1], [-11.1]])

    # Start a second session.  In that session the parameter nodes
    # have not been initialized either.
    with ops.Graph().as_default() as graph:
      with self.test_session(graph) as sess:
        tree_ensemble_handle = model_ops.tree_ensemble_variable(
            stamp_token=0, tree_ensemble_config="", name="restore_tree")
        my_saver = saver.Saver()
        my_saver.restore(sess, save_path)
        result, _, _ = prediction_ops.gradient_trees_prediction(
            tree_ensemble_handle,
            self._seed, [self._dense_float_tensor], [
                self._sparse_float_indices1, self._sparse_float_indices2
            ], [self._sparse_float_values1, self._sparse_float_values2],
            [self._sparse_float_shape1,
             self._sparse_float_shape2], [self._sparse_int_indices1],
            [self._sparse_int_values1], [self._sparse_int_shape1],
            learner_config=learner_config.SerializeToString(),
            apply_dropout=False,
            apply_averaging=False,
            center_bias=False,
            reduce_dim=True)
        # Make sure we only have the first and second tree.
        # The third tree was added after the save.
        self.assertAllClose(result.eval(), [[-1.1], [-1.1]])


if __name__ == "__main__":
  googletest.main()
