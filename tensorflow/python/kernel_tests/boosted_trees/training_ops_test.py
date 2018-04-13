# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for boosted_trees training kernels."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from google.protobuf import text_format
from tensorflow.core.kernels.boosted_trees import boosted_trees_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.ops import boosted_trees_ops
from tensorflow.python.ops import resources
from tensorflow.python.platform import googletest


class UpdateTreeEnsembleOpTest(test_util.TensorFlowTestCase):
  """Tests for growing tree ensemble from split candidates."""

  def testGrowWithEmptyEnsemble(self):
    """Test growing an empty ensemble."""
    with self.test_session() as session:
      # Create empty ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble('ensemble')
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      feature_ids = [0, 2, 6]

      # Prepare feature inputs.
      # Note that features 1 & 3 have the same gain but different splits.
      feature1_nodes = np.array([0], dtype=np.int32)
      feature1_gains = np.array([7.62], dtype=np.float32)
      feature1_thresholds = np.array([52], dtype=np.int32)
      feature1_left_node_contribs = np.array([[-4.375]], dtype=np.float32)
      feature1_right_node_contribs = np.array([[7.143]], dtype=np.float32)

      feature2_nodes = np.array([0], dtype=np.int32)
      feature2_gains = np.array([0.63], dtype=np.float32)
      feature2_thresholds = np.array([23], dtype=np.int32)
      feature2_left_node_contribs = np.array([[-0.6]], dtype=np.float32)
      feature2_right_node_contribs = np.array([[0.24]], dtype=np.float32)

      # Feature split with the highest gain.
      feature3_nodes = np.array([0], dtype=np.int32)
      feature3_gains = np.array([7.65], dtype=np.float32)
      feature3_thresholds = np.array([7], dtype=np.int32)
      feature3_left_node_contribs = np.array([[-4.89]], dtype=np.float32)
      feature3_right_node_contribs = np.array([[5.3]], dtype=np.float32)

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=0.1,
          pruning_mode=boosted_trees_ops.PruningMode.NO_PRUNING,
          # Tree will be finalized now, since we will reach depth 1.
          max_depth=1,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes, feature2_nodes, feature3_nodes],
          gains=[feature1_gains, feature2_gains, feature3_gains],
          thresholds=[
              feature1_thresholds, feature2_thresholds, feature3_thresholds
          ],
          left_node_contribs=[
              feature1_left_node_contribs, feature2_left_node_contribs,
              feature3_left_node_contribs
          ],
          right_node_contribs=[
              feature1_right_node_contribs, feature2_right_node_contribs,
              feature3_right_node_contribs
          ])
      session.run(grow_op)

      new_stamp, serialized = session.run(tree_ensemble.serialize())

      tree_ensemble = boosted_trees_pb2.TreeEnsemble()
      tree_ensemble.ParseFromString(serialized)

      # Note that since the tree is finalized, we added a new dummy tree.
      expected_result = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 6
              threshold: 7
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.65
            }
          }
          nodes {
            leaf {
              scalar: -0.489
            }
          }
          nodes {
            leaf {
              scalar: 0.53
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: true
        }
        tree_metadata {
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertProtoEquals(expected_result, tree_ensemble)

  def testGrowExistingEnsembleTreeNotFinalized(self):
    """Test growing an existing ensemble with the last tree not finalized."""
    with self.test_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge("""
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 0.714
            }
          }
          nodes {
            leaf {
              scalar: -0.4375
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare feature inputs.
      # feature 1 only has a candidate for node 1, feature 2 has candidates
      # for both nodes and feature 3 only has a candidate for node 2.

      feature_ids = [0, 1, 0]

      feature1_nodes = np.array([1], dtype=np.int32)
      feature1_gains = np.array([1.4], dtype=np.float32)
      feature1_thresholds = np.array([21], dtype=np.int32)
      feature1_left_node_contribs = np.array([[-6.0]], dtype=np.float32)
      feature1_right_node_contribs = np.array([[1.65]], dtype=np.float32)

      feature2_nodes = np.array([1, 2], dtype=np.int32)
      feature2_gains = np.array([0.63, 2.7], dtype=np.float32)
      feature2_thresholds = np.array([23, 7], dtype=np.int32)
      feature2_left_node_contribs = np.array([[-0.6], [-1.5]], dtype=np.float32)
      feature2_right_node_contribs = np.array([[0.24], [2.3]], dtype=np.float32)

      feature3_nodes = np.array([2], dtype=np.int32)
      feature3_gains = np.array([1.7], dtype=np.float32)
      feature3_thresholds = np.array([3], dtype=np.int32)
      feature3_left_node_contribs = np.array([[-0.75]], dtype=np.float32)
      feature3_right_node_contribs = np.array([[1.93]], dtype=np.float32)

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=0.1,
          pruning_mode=boosted_trees_ops.PruningMode.NO_PRUNING,
          # tree is going to be finalized now, since we reach depth 2.
          max_depth=2,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes, feature2_nodes, feature3_nodes],
          gains=[feature1_gains, feature2_gains, feature3_gains],
          thresholds=[
              feature1_thresholds, feature2_thresholds, feature3_thresholds
          ],
          left_node_contribs=[
              feature1_left_node_contribs, feature2_left_node_contribs,
              feature3_left_node_contribs
          ],
          right_node_contribs=[
              feature1_right_node_contribs, feature2_right_node_contribs,
              feature3_right_node_contribs
          ])
      session.run(grow_op)

      # Expect the split for node 1 to be chosen from feature 1 and
      # the split for node 2 to be chosen from feature 2.
      # The grown tree should be finalized as max tree depth is 2 and we have
      # grown 2 layers.
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      tree_ensemble = boosted_trees_pb2.TreeEnsemble()
      tree_ensemble.ParseFromString(serialized)

      expected_result = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            bucketized_split {
              threshold: 21
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 1.4
              original_leaf {
                scalar: 0.714
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 7
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 2.7
              original_leaf {
                scalar: -0.4375
              }
            }
          }
          nodes {
            leaf {
              scalar: 0.114
            }
          }
          nodes {
            leaf {
              scalar: 0.879
            }
          }
          nodes {
            leaf {
              scalar: -0.5875
            }
          }
          nodes {
            leaf {
              scalar: -0.2075
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          is_finalized: true
          num_layers_grown: 2
        }
        tree_metadata {
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertProtoEquals(expected_result, tree_ensemble)

  def testGrowExistingEnsembleTreeFinalized(self):
    """Test growing an existing ensemble with the last tree finalized."""
    with self.test_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge("""
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 7.14
            }
          }
          nodes {
            leaf {
              scalar: -4.375
            }
          }
        }
        trees {
          nodes {
            leaf {
              scalar: 0.0
            }
          }
        }
        tree_weights: 0.15
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: true
        }
        tree_metadata {
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare feature inputs.

      feature_ids = [75]

      feature1_nodes = np.array([0], dtype=np.int32)
      feature1_gains = np.array([-1.4], dtype=np.float32)
      feature1_thresholds = np.array([21], dtype=np.int32)
      feature1_left_node_contribs = np.array([[-6.0]], dtype=np.float32)
      feature1_right_node_contribs = np.array([[1.65]], dtype=np.float32)

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          pruning_mode=boosted_trees_ops.PruningMode.NO_PRUNING,
          learning_rate=0.1,
          max_depth=2,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes],
          gains=[feature1_gains],
          thresholds=[feature1_thresholds],
          left_node_contribs=[feature1_left_node_contribs],
          right_node_contribs=[feature1_right_node_contribs])
      session.run(grow_op)

      # Expect a new tree added, with a split on feature 75
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      tree_ensemble = boosted_trees_pb2.TreeEnsemble()
      tree_ensemble.ParseFromString(serialized)

      expected_result = """
       trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 7.14
            }
          }
          nodes {
            leaf {
              scalar: -4.375
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 75
              threshold: 21
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: -1.4
            }
          }
          nodes {
            leaf {
              scalar: -0.6
            }
          }
          nodes {
            leaf {
              scalar: 0.165
            }
          }
        }
        tree_weights: 0.15
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 2
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertProtoEquals(expected_result, tree_ensemble)

  def testPrePruning(self):
    """Test growing an existing ensemble with pre-pruning."""
    with self.test_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge("""
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 7.14
            }
          }
          nodes {
            leaf {
              scalar: -4.375
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare feature inputs.
      # For node 1, the best split is on feature 2 (gain -0.63), but the gain
      # is negative so node 1 will not be split.
      # For node 2, the best split is on feature 3, gain is positive.

      feature_ids = [0, 1, 0]

      feature1_nodes = np.array([1], dtype=np.int32)
      feature1_gains = np.array([-1.4], dtype=np.float32)
      feature1_thresholds = np.array([21], dtype=np.int32)
      feature1_left_node_contribs = np.array([[-6.0]], dtype=np.float32)
      feature1_right_node_contribs = np.array([[1.65]], dtype=np.float32)

      feature2_nodes = np.array([1, 2], dtype=np.int32)
      feature2_gains = np.array([-0.63, 2.7], dtype=np.float32)
      feature2_thresholds = np.array([23, 7], dtype=np.int32)
      feature2_left_node_contribs = np.array([[-0.6], [-1.5]], dtype=np.float32)
      feature2_right_node_contribs = np.array([[0.24], [2.3]], dtype=np.float32)

      feature3_nodes = np.array([2], dtype=np.int32)
      feature3_gains = np.array([2.8], dtype=np.float32)
      feature3_thresholds = np.array([3], dtype=np.int32)
      feature3_left_node_contribs = np.array([[-0.75]], dtype=np.float32)
      feature3_right_node_contribs = np.array([[1.93]], dtype=np.float32)

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=0.1,
          pruning_mode=boosted_trees_ops.PruningMode.PRE_PRUNING,
          max_depth=3,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes, feature2_nodes, feature3_nodes],
          gains=[feature1_gains, feature2_gains, feature3_gains],
          thresholds=[
              feature1_thresholds, feature2_thresholds, feature3_thresholds
          ],
          left_node_contribs=[
              feature1_left_node_contribs, feature2_left_node_contribs,
              feature3_left_node_contribs
          ],
          right_node_contribs=[
              feature1_right_node_contribs, feature2_right_node_contribs,
              feature3_right_node_contribs
          ])
      session.run(grow_op)

      # Expect the split for node 1 to be chosen from feature 1 and
      # the split for node 2 to be chosen from feature 2.
      # The grown tree should not be finalized as max tree depth is 3 and
      # it's only grown 2 layers.
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      tree_ensemble = boosted_trees_pb2.TreeEnsemble()
      tree_ensemble.ParseFromString(serialized)

      expected_result = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 7.14
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 3
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 2.8
              original_leaf {
                scalar: -4.375
              }
            }
          }
          nodes {
            leaf {
              scalar: -4.45
            }
          }
          nodes {
            leaf {
              scalar: -4.182
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          is_finalized: false
          num_layers_grown: 2
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 3
          last_layer_node_end: 5
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertProtoEquals(expected_result, tree_ensemble)

  def testMetadataWhenCantSplitDueToEmptySplits(self):
    """Test that the metadata is updated even though we can't split."""
    with self.test_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 0.714
            }
          }
          nodes {
            leaf {
              scalar: -0.4375
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3

        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare feature inputs.
      # feature 1 only has a candidate for node 1, feature 2 has candidates
      # for both nodes and feature 3 only has a candidate for node 2.

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=0.1,
          pruning_mode=boosted_trees_ops.PruningMode.NO_PRUNING,
          max_depth=2,
          # No splits are available.
          feature_ids=[],
          node_ids=[],
          gains=[],
          thresholds=[],
          left_node_contribs=[],
          right_node_contribs=[])
      session.run(grow_op)

      # Expect no new splits created, but attempted (global) stats updated. Meta
      # data for this tree should not be updated (we didn't succeed building a
      # layer. Node ranges don't change.
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      tree_ensemble = boosted_trees_pb2.TreeEnsemble()
      tree_ensemble.ParseFromString(serialized)

      expected_result = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 0.714
            }
          }
          nodes {
            leaf {
              scalar: -0.4375
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertProtoEquals(expected_result, tree_ensemble)

  def testMetadataWhenCantSplitDuePrePruning(self):
    """Test metadata is updated correctly when no split due to prepruning."""
    with self.test_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 7.14
            }
          }
          nodes {
            leaf {
              scalar: -4.375
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare feature inputs.
      feature_ids = [0, 1, 0]

      # All the gains are negative.
      feature1_nodes = np.array([1], dtype=np.int32)
      feature1_gains = np.array([-1.4], dtype=np.float32)
      feature1_thresholds = np.array([21], dtype=np.int32)
      feature1_left_node_contribs = np.array([[-6.0]], dtype=np.float32)
      feature1_right_node_contribs = np.array([[1.65]], dtype=np.float32)

      feature2_nodes = np.array([1, 2], dtype=np.int32)
      feature2_gains = np.array([-0.63, -2.7], dtype=np.float32)
      feature2_thresholds = np.array([23, 7], dtype=np.int32)
      feature2_left_node_contribs = np.array([[-0.6], [-1.5]], dtype=np.float32)
      feature2_right_node_contribs = np.array([[0.24], [2.3]], dtype=np.float32)

      feature3_nodes = np.array([2], dtype=np.int32)
      feature3_gains = np.array([-2.8], dtype=np.float32)
      feature3_thresholds = np.array([3], dtype=np.int32)
      feature3_left_node_contribs = np.array([[-0.75]], dtype=np.float32)
      feature3_right_node_contribs = np.array([[1.93]], dtype=np.float32)

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=0.1,
          pruning_mode=boosted_trees_ops.PruningMode.PRE_PRUNING,
          max_depth=3,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes, feature2_nodes, feature3_nodes],
          gains=[feature1_gains, feature2_gains, feature3_gains],
          thresholds=[
              feature1_thresholds, feature2_thresholds, feature3_thresholds
          ],
          left_node_contribs=[
              feature1_left_node_contribs, feature2_left_node_contribs,
              feature3_left_node_contribs
          ],
          right_node_contribs=[
              feature1_right_node_contribs, feature2_right_node_contribs,
              feature3_right_node_contribs
          ])
      session.run(grow_op)

      # Expect that no new split was created because all the gains were negative
      # Global metadata should be updated, tree metadata should not be updated.
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      tree_ensemble = boosted_trees_pb2.TreeEnsemble()
      tree_ensemble.ParseFromString(serialized)

      expected_result = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 7.14
            }
          }
          nodes {
            leaf {
              scalar: -4.375
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertProtoEquals(expected_result, tree_ensemble)

  def testPostPruningOfSomeNodes(self):
    """Test growing an ensemble with post-pruning."""
    with self.test_session() as session:
      # Create empty ensemble.
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle

      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare inputs.
      # Second feature has larger (but still negative gain).
      feature_ids = [0, 1]

      feature1_nodes = np.array([0], dtype=np.int32)
      feature1_gains = np.array([-1.3], dtype=np.float32)
      feature1_thresholds = np.array([7], dtype=np.int32)
      feature1_left_node_contribs = np.array([[0.013]], dtype=np.float32)
      feature1_right_node_contribs = np.array([[0.0143]], dtype=np.float32)

      feature2_nodes = np.array([0], dtype=np.int32)
      feature2_gains = np.array([-0.2], dtype=np.float32)
      feature2_thresholds = np.array([33], dtype=np.int32)
      feature2_left_node_contribs = np.array([[0.01]], dtype=np.float32)
      feature2_right_node_contribs = np.array([[0.0143]], dtype=np.float32)

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=1.0,
          pruning_mode=boosted_trees_ops.PruningMode.POST_PRUNING,
          max_depth=3,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes, feature2_nodes],
          gains=[feature1_gains, feature2_gains],
          thresholds=[feature1_thresholds, feature2_thresholds],
          left_node_contribs=[
              feature1_left_node_contribs, feature2_left_node_contribs
          ],
          right_node_contribs=[
              feature1_right_node_contribs, feature2_right_node_contribs
          ])

      session.run(grow_op)

      # Expect the split from second features to be chosen despite the negative
      # gain.
      # No pruning happened just yet.
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      res_ensemble = boosted_trees_pb2.TreeEnsemble()
      res_ensemble.ParseFromString(serialized)

      expected_result = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 33
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: -0.2
            }
          }
          nodes {
            leaf {
              scalar: 0.01
            }
          }
          nodes {
            leaf {
              scalar: 0.0143
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertProtoEquals(expected_result, res_ensemble)

      # Prepare the second layer.
      # Note that node 1 gain is negative and node 2 gain is positive.
      feature_ids = [3]
      feature1_nodes = np.array([1, 2], dtype=np.int32)
      feature1_gains = np.array([-0.2, 0.5], dtype=np.float32)
      feature1_thresholds = np.array([7, 5], dtype=np.int32)
      feature1_left_node_contribs = np.array(
          [[0.07], [0.041]], dtype=np.float32)
      feature1_right_node_contribs = np.array(
          [[0.083], [0.064]], dtype=np.float32)

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=1.0,
          pruning_mode=boosted_trees_ops.PruningMode.POST_PRUNING,
          max_depth=3,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes],
          gains=[feature1_gains],
          thresholds=[feature1_thresholds],
          left_node_contribs=[feature1_left_node_contribs],
          right_node_contribs=[feature1_right_node_contribs])

      session.run(grow_op)

      # After adding this layer, the tree will not be finalized
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      res_ensemble = boosted_trees_pb2.TreeEnsemble()
      res_ensemble.ParseFromString(serialized)
      expected_result = """
        trees {
          nodes {
            bucketized_split {
              feature_id:1
              threshold: 33
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: -0.2
            }
          }
          nodes {
            bucketized_split {
              feature_id: 3
              threshold: 7
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: -0.2
              original_leaf {
                scalar: 0.01
               }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 3
              threshold: 5
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 0.5
              original_leaf {
                scalar: 0.0143
               }
            }
          }
          nodes {
            leaf {
              scalar: 0.08
            }
          }
          nodes {
            leaf {
              scalar: 0.093
            }
          }
          nodes {
            leaf {
              scalar: 0.0553
            }
          }
          nodes {
            leaf {
                scalar: 0.0783
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
          last_layer_node_start: 3
          last_layer_node_end: 7
        }
       """
      self.assertEqual(new_stamp, 2)

      self.assertProtoEquals(expected_result, res_ensemble)
      # Now split the leaf 3, again with negative gain. After this layer, the
      # tree will be finalized, and post-pruning happens. The leafs 3,4,7,8 will
      # be pruned out.

      # Prepare the third layer.
      feature_ids = [92]
      feature1_nodes = np.array([3], dtype=np.int32)
      feature1_gains = np.array([-0.45], dtype=np.float32)
      feature1_thresholds = np.array([11], dtype=np.int32)
      feature1_left_node_contribs = np.array([[0.15]], dtype=np.float32)
      feature1_right_node_contribs = np.array([[0.5]], dtype=np.float32)

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=1.0,
          pruning_mode=boosted_trees_ops.PruningMode.POST_PRUNING,
          max_depth=3,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes],
          gains=[feature1_gains],
          thresholds=[feature1_thresholds],
          left_node_contribs=[feature1_left_node_contribs],
          right_node_contribs=[feature1_right_node_contribs])

      session.run(grow_op)
      # After adding this layer, the tree will be finalized
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      res_ensemble = boosted_trees_pb2.TreeEnsemble()
      res_ensemble.ParseFromString(serialized)
      # Node that nodes 3, 4, 7 and 8 got deleted, so metadata stores has ids
      # mapped to their parent node 1, with the respective change in logits.
      expected_result = """
        trees {
          nodes {
            bucketized_split {
              feature_id:1
              threshold: 33
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: -0.2
            }
          }
          nodes {
            leaf {
              scalar: 0.01
            }
          }
          nodes {
            bucketized_split {
              feature_id: 3
              threshold: 5
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 0.5
              original_leaf {
                scalar: 0.0143
               }
            }
          }
          nodes {
            leaf {
              scalar: 0.0553
            }
          }
          nodes {
            leaf {
              scalar: 0.0783
            }
          }
        }
        trees {
          nodes {
            leaf {
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 3
          is_finalized: true
          post_pruned_nodes_meta {
            new_node_id: 0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 2
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.07
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.083
          }
          post_pruned_nodes_meta {
            new_node_id: 3
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 4
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.22
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.57
          }
        }
        tree_metadata {
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 3
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
       """
      self.assertEqual(new_stamp, 3)
      self.assertProtoEquals(expected_result, res_ensemble)

  def testPostPruningOfAllNodes(self):
    """Test growing an ensemble with post-pruning, with all nodes are pruned."""
    with self.test_session() as session:
      # Create empty ensemble.
      # Create empty ensemble.
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle

      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare inputs. All have negative gains.
      feature_ids = [0, 1]

      feature1_nodes = np.array([0], dtype=np.int32)
      feature1_gains = np.array([-1.3], dtype=np.float32)
      feature1_thresholds = np.array([7], dtype=np.int32)
      feature1_left_node_contribs = np.array([[0.013]], dtype=np.float32)
      feature1_right_node_contribs = np.array([[0.0143]], dtype=np.float32)

      feature2_nodes = np.array([0], dtype=np.int32)
      feature2_gains = np.array([-0.62], dtype=np.float32)
      feature2_thresholds = np.array([33], dtype=np.int32)
      feature2_left_node_contribs = np.array([[0.01]], dtype=np.float32)
      feature2_right_node_contribs = np.array([[0.0143]], dtype=np.float32)

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=1.0,
          pruning_mode=boosted_trees_ops.PruningMode.POST_PRUNING,
          max_depth=2,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes, feature2_nodes],
          gains=[feature1_gains, feature2_gains],
          thresholds=[feature1_thresholds, feature2_thresholds],
          left_node_contribs=[
              feature1_left_node_contribs, feature2_left_node_contribs
          ],
          right_node_contribs=[
              feature1_right_node_contribs, feature2_right_node_contribs
          ])

      session.run(grow_op)

      # Expect the split from feature 2 to be chosen despite the negative gain.
      # The grown tree should not be finalized as max tree depth is 2 so no
      # pruning occurs.
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      res_ensemble = boosted_trees_pb2.TreeEnsemble()
      res_ensemble.ParseFromString(serialized)

      expected_result = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 33
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: -0.62
            }
          }
          nodes {
            leaf {
              scalar: 0.01
            }
          }
          nodes {
            leaf {
              scalar: 0.0143
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 1
          last_layer_node_end: 3
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertProtoEquals(expected_result, res_ensemble)

      # Prepare inputs.
      # All have negative gain.
      feature_ids = [3]
      feature1_nodes = np.array([1, 2], dtype=np.int32)
      feature1_gains = np.array([-0.2, -0.5], dtype=np.float32)
      feature1_thresholds = np.array([77, 79], dtype=np.int32)
      feature1_left_node_contribs = np.array([[0.023], [0.3]], dtype=np.float32)
      feature1_right_node_contribs = np.array(
          [[0.012343], [24]], dtype=np.float32)

      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=1.0,
          pruning_mode=boosted_trees_ops.PruningMode.POST_PRUNING,
          max_depth=2,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes],
          gains=[feature1_gains],
          thresholds=[feature1_thresholds],
          left_node_contribs=[feature1_left_node_contribs],
          right_node_contribs=[feature1_right_node_contribs])

      session.run(grow_op)

      # Expect the split from feature 1 to be chosen despite the negative gain.
      # The grown tree should be finalized. Since all nodes have negative gain,
      # the whole tree is pruned.
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      res_ensemble = boosted_trees_pb2.TreeEnsemble()
      res_ensemble.ParseFromString(serialized)

      # Expect the ensemble to be empty as post-pruning will prune
      # the entire finalized tree.
      self.assertEqual(new_stamp, 2)
      self.assertProtoEquals(
          """
      trees {
        nodes {
          leaf {
          }
        }
      }
      trees {
        nodes {
          leaf {
          }
        }
      }
      tree_weights: 1.0
      tree_weights: 1.0
      tree_metadata{
        num_layers_grown: 2
        is_finalized: true
        post_pruned_nodes_meta {
          new_node_id: 0
          logit_change: 0.0
        }
        post_pruned_nodes_meta {
          new_node_id: 0
          logit_change: -0.01
        }
        post_pruned_nodes_meta {
          new_node_id: 0
          logit_change: -0.0143
        }
        post_pruned_nodes_meta {
          new_node_id: 0
          logit_change: -0.033
        }
        post_pruned_nodes_meta {
          new_node_id: 0
          logit_change: -0.022343
        }
        post_pruned_nodes_meta {
          new_node_id: 0
          logit_change: -0.3143
        }
        post_pruned_nodes_meta {
          new_node_id: 0
          logit_change: -24.0143
        }
      }
      tree_metadata {
      }
      growing_metadata {
        num_trees_attempted: 1
        num_layers_attempted: 2
        last_layer_node_start: 0
        last_layer_node_end: 1
      }
      """, res_ensemble)

  def testPostPruningChangesNothing(self):
    """Test growing an ensemble with post-pruning with all gains >0."""
    with self.test_session() as session:
      # Create empty ensemble.
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle

      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare inputs.
      # Second feature has larger (but still negative gain).
      feature_ids = [3, 4]

      feature1_nodes = np.array([0], dtype=np.int32)
      feature1_gains = np.array([7.62], dtype=np.float32)
      feature1_thresholds = np.array([52], dtype=np.int32)
      feature1_left_node_contribs = np.array([[-4.375]], dtype=np.float32)
      feature1_right_node_contribs = np.array([[7.143]], dtype=np.float32)

      feature2_nodes = np.array([0], dtype=np.int32)
      feature2_gains = np.array([0.63], dtype=np.float32)
      feature2_thresholds = np.array([23], dtype=np.int32)
      feature2_left_node_contribs = np.array([[-0.6]], dtype=np.float32)
      feature2_right_node_contribs = np.array([[0.24]], dtype=np.float32)

      # Grow tree ensemble.
      grow_op = boosted_trees_ops.update_ensemble(
          tree_ensemble_handle,
          learning_rate=1.0,
          pruning_mode=boosted_trees_ops.PruningMode.POST_PRUNING,
          max_depth=1,
          feature_ids=feature_ids,
          node_ids=[feature1_nodes, feature2_nodes],
          gains=[feature1_gains, feature2_gains],
          thresholds=[feature1_thresholds, feature2_thresholds],
          left_node_contribs=[
              feature1_left_node_contribs, feature2_left_node_contribs
          ],
          right_node_contribs=[
              feature1_right_node_contribs, feature2_right_node_contribs
          ])

      session.run(grow_op)

      # Expect the split from the first feature to be chosen.
      # Pruning got triggered but changed nothing.
      new_stamp, serialized = session.run(tree_ensemble.serialize())
      res_ensemble = boosted_trees_pb2.TreeEnsemble()
      res_ensemble.ParseFromString(serialized)

      expected_result = """
        trees {
          nodes {
            bucketized_split {
              feature_id: 3
              threshold: 52
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: -4.375
            }
          }
          nodes {
            leaf {
              scalar: 7.143
            }
          }
        }
        trees {
          nodes {
            leaf {
            }
          }
        }
        tree_weights: 1.0
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: true
        }
        tree_metadata {
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
          last_layer_node_start: 0
          last_layer_node_end: 1
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertProtoEquals(expected_result, res_ensemble)


if __name__ == '__main__':
  googletest.main()
