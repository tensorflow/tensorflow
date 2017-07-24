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
"""Tests for the GTFlow ensemble optimization ops.

The tests cover:
- Adding a newly built tree to an existing ensemble
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.boosted_trees.proto import tree_config_pb2
from tensorflow.contrib.boosted_trees.python.ops import ensemble_optimizer_ops
from tensorflow.contrib.boosted_trees.python.ops import model_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


def _append_to_leaf(leaf, class_id, weight):
  """Helper method for building tree leaves.

  Appends weight contributions for the given class index to a leaf node.

  Args:
    leaf: leaf node to append to, int
    class_id: class Id for the weight update, int
    weight: weight contribution value, float
  """
  leaf.sparse_vector.index.append(class_id)
  leaf.sparse_vector.value.append(weight)


class EnsembleOptimizerOpsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    """Create an ensemble of 2 trees."""
    super(EnsembleOptimizerOpsTest, self).setUp()
    self._tree_ensemble = tree_config_pb2.DecisionTreeEnsembleConfig()
    # First tree.
    tree_1 = self._tree_ensemble.trees.add()
    _append_to_leaf(tree_1.nodes.add().leaf, 0, 0.4)
    _append_to_leaf(tree_1.nodes.add().leaf, 1, 0.6)
    # Second tree.
    tree_2 = self._tree_ensemble.trees.add()
    _append_to_leaf(tree_2.nodes.add().leaf, 0, 1)
    _append_to_leaf(tree_2.nodes.add().leaf, 1, 0)

    self._tree_ensemble.tree_weights.append(1.0)
    self._tree_ensemble.tree_weights.append(1.0)

    meta_1 = self._tree_ensemble.tree_metadata.add()
    meta_1.num_tree_weight_updates = 2
    meta_2 = self._tree_ensemble.tree_metadata.add()
    meta_2.num_tree_weight_updates = 3

    # Ensemble to be added.
    self._ensemble_to_add = tree_config_pb2.DecisionTreeEnsembleConfig()

    self._tree_to_add = self._ensemble_to_add.trees.add()
    _append_to_leaf(self._tree_to_add.nodes.add().leaf, 0, 0.3)
    _append_to_leaf(self._tree_to_add.nodes.add().leaf, 1, 0.7)

  def testWithEmptyEnsemble(self):
    with self.test_session():
      # Create an empty ensemble.
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="empty")

      # Create zero feature importance.
      feature_usage_counts = variables.Variable(
          initial_value=array_ops.zeros([1], dtypes.int64),
          name="feature_usage_counts",
          trainable=False)
      feature_gains = variables.Variable(
          initial_value=array_ops.zeros([1], dtypes.float32),
          name="feature_gains",
          trainable=False)

      resources.initialize_resources(resources.shared_resources()).run()
      variables.initialize_all_variables().run()

      with ops.control_dependencies([
          ensemble_optimizer_ops.add_trees_to_ensemble(
              tree_ensemble_handle,
              self._ensemble_to_add.SerializeToString(),
              feature_usage_counts, [2],
              feature_gains, [0.4], [[]],
              learning_rate=1.0)
      ]):
        result = model_ops.tree_ensemble_serialize(tree_ensemble_handle)[1]

      # Output.
      output_ensemble = tree_config_pb2.DecisionTreeEnsembleConfig()
      output_ensemble.ParseFromString(result.eval())
      self.assertProtoEquals(self._tree_to_add, output_ensemble.trees[0])
      self.assertEqual(1, len(output_ensemble.trees))

      self.assertAllEqual([1.0], output_ensemble.tree_weights)

      self.assertEqual(1,
                       output_ensemble.tree_metadata[0].num_tree_weight_updates)

      self.assertAllEqual([2], feature_usage_counts.eval())
      self.assertArrayNear([0.4], feature_gains.eval(), 1e-6)

  def testWithExistingEnsemble(self):
    with self.test_session():
      # Create existing tree ensemble.
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=self._tree_ensemble.SerializeToString(),
          name="existing")
      # Create non-zero feature importance.
      feature_usage_counts = variables.Variable(
          initial_value=np.array([0, 4, 1], np.int64),
          name="feature_usage_counts",
          trainable=False)
      feature_gains = variables.Variable(
          initial_value=np.array([0.0, 0.3, 0.05], np.float32),
          name="feature_gains",
          trainable=False)

      resources.initialize_resources(resources.shared_resources()).run()
      variables.initialize_all_variables().run()
      output_ensemble = tree_config_pb2.DecisionTreeEnsembleConfig()
      with ops.control_dependencies([
          ensemble_optimizer_ops.add_trees_to_ensemble(
              tree_ensemble_handle,
              self._ensemble_to_add.SerializeToString(),
              feature_usage_counts, [1, 2, 0],
              feature_gains, [0.02, 0.1, 0.0], [[], []],
              learning_rate=1)
      ]):
        output_ensemble.ParseFromString(
            model_ops.tree_ensemble_serialize(tree_ensemble_handle)[1].eval())

      # Output.
      self.assertEqual(3, len(output_ensemble.trees))
      self.assertProtoEquals(self._tree_to_add, output_ensemble.trees[2])

      self.assertAllEqual([1.0, 1.0, 1.0], output_ensemble.tree_weights)

      self.assertEqual(2,
                       output_ensemble.tree_metadata[0].num_tree_weight_updates)
      self.assertEqual(3,
                       output_ensemble.tree_metadata[1].num_tree_weight_updates)
      self.assertEqual(1,
                       output_ensemble.tree_metadata[2].num_tree_weight_updates)
      self.assertAllEqual([1, 6, 1], feature_usage_counts.eval())
      self.assertArrayNear([0.02, 0.4, 0.05], feature_gains.eval(), 1e-6)

  def testWithExistingEnsembleAndDropout(self):
    with self.test_session():
      tree_ensemble = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Add 10 trees with some weights.
      for i in range(0, 10):
        tree = tree_ensemble.trees.add()
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)
        tree_ensemble.tree_weights.append(i + 1)
        meta = tree_ensemble.tree_metadata.add()
        meta.num_tree_weight_updates = 1
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble.SerializeToString(),
          name="existing")
      # Create non-zero feature importance.
      feature_usage_counts = variables.Variable(
          initial_value=np.array([2, 3], np.int64),
          name="feature_usage_counts",
          trainable=False)
      feature_gains = variables.Variable(
          initial_value=np.array([0.0, 0.3], np.float32),
          name="feature_gains",
          trainable=False)

      resources.initialize_resources(resources.shared_resources()).run()
      variables.initialize_all_variables().run()

      dropped = [1, 6, 8]
      dropped_original_weights = [2.0, 7.0, 9.0]

      output_ensemble = tree_config_pb2.DecisionTreeEnsembleConfig()
      with ops.control_dependencies([
          ensemble_optimizer_ops.add_trees_to_ensemble(
              tree_ensemble_handle,
              self._ensemble_to_add.SerializeToString(),
              feature_usage_counts, [1, 2],
              feature_gains, [0.5, 0.3], [dropped, dropped_original_weights],
              learning_rate=0.1)
      ]):
        output_ensemble.ParseFromString(
            model_ops.tree_ensemble_serialize(tree_ensemble_handle)[1].eval())

      # Output.
      self.assertEqual(11, len(output_ensemble.trees))
      self.assertProtoEquals(self._tree_to_add, output_ensemble.trees[10])
      self.assertAllClose(4.5, output_ensemble.tree_weights[10])

      self.assertAllClose([1., 1.5, 3., 4., 5., 6., 5.25, 8., 6.75, 10., 4.5],
                          output_ensemble.tree_weights)

      self.assertEqual(1,
                       output_ensemble.tree_metadata[0].num_tree_weight_updates)
      self.assertEqual(2,
                       output_ensemble.tree_metadata[1].num_tree_weight_updates)
      self.assertEqual(1,
                       output_ensemble.tree_metadata[2].num_tree_weight_updates)

      self.assertEqual(1,
                       output_ensemble.tree_metadata[3].num_tree_weight_updates)
      self.assertEqual(1,
                       output_ensemble.tree_metadata[4].num_tree_weight_updates)
      self.assertEqual(1,
                       output_ensemble.tree_metadata[5].num_tree_weight_updates)
      self.assertEqual(2,
                       output_ensemble.tree_metadata[6].num_tree_weight_updates)
      self.assertEqual(1,
                       output_ensemble.tree_metadata[7].num_tree_weight_updates)
      self.assertEqual(2,
                       output_ensemble.tree_metadata[8].num_tree_weight_updates)
      self.assertEqual(1,
                       output_ensemble.tree_metadata[9].num_tree_weight_updates)
      self.assertEqual(
          1, output_ensemble.tree_metadata[10].num_tree_weight_updates)
      self.assertAllEqual([3, 5], feature_usage_counts.eval())
      self.assertArrayNear([0.05, 0.33], feature_gains.eval(), 1e-6)

  def testWithEmptyEnsembleAndShrinkage(self):
    with self.test_session():
      # Add shrinkage config.
      learning_rate = 0.0001
      tree_ensemble = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble.SerializeToString(),
          name="existing")

      # Create zero feature importance.
      feature_usage_counts = variables.Variable(
          initial_value=np.array([0, 0], np.int64),
          name="feature_usage_counts",
          trainable=False)
      feature_gains = variables.Variable(
          initial_value=np.array([0.0, 0.0], np.float32),
          name="feature_gains",
          trainable=False)

      resources.initialize_resources(resources.shared_resources()).run()
      variables.initialize_all_variables().run()

      output_ensemble = tree_config_pb2.DecisionTreeEnsembleConfig()
      with ops.control_dependencies([
          ensemble_optimizer_ops.add_trees_to_ensemble(
              tree_ensemble_handle,
              self._ensemble_to_add.SerializeToString(),
              feature_usage_counts, [1, 2],
              feature_gains, [0.5, 0.3], [[], []],
              learning_rate=learning_rate)
      ]):
        output_ensemble.ParseFromString(
            model_ops.tree_ensemble_serialize(tree_ensemble_handle)[1].eval())

      # New tree is added with shrinkage weight.
      self.assertAllClose([learning_rate], output_ensemble.tree_weights)
      self.assertEqual(1,
                       output_ensemble.tree_metadata[0].num_tree_weight_updates)
      self.assertAllEqual([1, 2], feature_usage_counts.eval())
      self.assertArrayNear([0.5 * learning_rate, 0.3 * learning_rate],
                           feature_gains.eval(), 1e-6)

  def testWithExistingEnsembleAndShrinkage(self):
    with self.test_session():
      # Add shrinkage config.
      learning_rate = 0.0001
      tree_ensemble = tree_config_pb2.DecisionTreeEnsembleConfig()
      # Add 10 trees with some weights.
      for i in range(0, 5):
        tree = tree_ensemble.trees.add()
        _append_to_leaf(tree.nodes.add().leaf, 0, -0.4)
        tree_ensemble.tree_weights.append(i + 1)
        meta = tree_ensemble.tree_metadata.add()
        meta.num_tree_weight_updates = 1
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble.SerializeToString(),
          name="existing")

      # Create non-zero feature importance.
      feature_usage_counts = variables.Variable(
          initial_value=np.array([4, 7], np.int64),
          name="feature_usage_counts",
          trainable=False)
      feature_gains = variables.Variable(
          initial_value=np.array([0.2, 0.8], np.float32),
          name="feature_gains",
          trainable=False)

      resources.initialize_resources(resources.shared_resources()).run()
      variables.initialize_all_variables().run()

      output_ensemble = tree_config_pb2.DecisionTreeEnsembleConfig()
      with ops.control_dependencies([
          ensemble_optimizer_ops.add_trees_to_ensemble(
              tree_ensemble_handle,
              self._ensemble_to_add.SerializeToString(),
              feature_usage_counts, [1, 2],
              feature_gains, [0.5, 0.3], [[], []],
              learning_rate=learning_rate)
      ]):
        output_ensemble.ParseFromString(
            model_ops.tree_ensemble_serialize(tree_ensemble_handle)[1].eval())

      # The weights of previous trees stayed the same, new tree (LAST) is added
      # with shrinkage weight.
      self.assertAllClose([1.0, 2.0, 3.0, 4.0, 5.0, learning_rate],
                          output_ensemble.tree_weights)

      # Check that all number of updates are equal to 1 (e,g, no old tree weight
      # got adjusted.
      for i in range(0, 6):
        self.assertEqual(
            1, output_ensemble.tree_metadata[i].num_tree_weight_updates)

      # Ensure feature importance was aggregated correctly.
      self.assertAllEqual([5, 9], feature_usage_counts.eval())
      self.assertArrayNear(
          [0.2 + 0.5 * learning_rate, 0.8 + 0.3 * learning_rate],
          feature_gains.eval(), 1e-6)

if __name__ == "__main__":
  googletest.main()
