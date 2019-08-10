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
"""Tests boosted_trees prediction kernels."""
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


class TrainingPredictionOpsTest(test_util.TensorFlowTestCase):
  """Tests prediction ops for training."""

  @test_util.run_deprecated_v1
  def testCachedPredictionOnEmptyEnsemble(self):
    """Tests that prediction on a dummy ensemble does not fail."""
    with self.cached_session() as session:
      # Create a dummy ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto='')
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # No previous cached values.
      cached_tree_ids = [0, 0]
      cached_node_ids = [0, 0]

      # We have two features: 0 and 1. Values don't matter here on a dummy
      # ensemble.
      feature_0_values = [67, 5]
      feature_1_values = [9, 17]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # Nothing changed.
      self.assertAllClose(cached_tree_ids, new_tree_ids)
      self.assertAllClose(cached_node_ids, new_node_ids)
      self.assertAllClose([[0], [0]], logits_updates)

  @test_util.run_deprecated_v1
  def testCachedPredictionOnEmptyEnsembleMultiClass(self):
    """Tests that prediction on dummy ensemble does not fail for multi class."""
    with self.cached_session() as session:
      # Create a dummy ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto='')
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # No previous cached values.
      cached_tree_ids = [0, 0]
      cached_node_ids = [0, 0]

      # We have two features: 0 and 1. Values don't matter here on a dummy
      # ensemble.
      feature_0_values = [67, 5]
      feature_1_values = [9, 17]

      # Multi class.
      logits_dimension = 2

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=logits_dimension)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # Nothing changed.
      self.assertAllClose(cached_tree_ids, new_tree_ids)
      self.assertAllClose(cached_node_ids, new_node_ids)
      self.assertAllClose([[0, 0], [0, 0]], logits_updates)

  @test_util.run_deprecated_v1
  def testNoCachedPredictionButTreeExists(self):
    """Tests that predictions are updated once trees are added."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 15
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 1.14
            }
          }
          nodes {
            leaf {
              scalar: 8.79
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          is_finalized: true
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Two examples, none were cached before.
      cached_tree_ids = [0, 0]
      cached_node_ids = [0, 0]

      feature_0_values = [67, 5]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values],
          logits_dimension=1)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # We are in the first tree.
      self.assertAllClose([0, 0], new_tree_ids)
      self.assertAllClose([2, 1], new_node_ids)
      self.assertAllClose([[0.1 * 8.79], [0.1 * 1.14]], logits_updates)

  @test_util.run_deprecated_v1
  def testNoCachedPredictionButTreeExistsMultiClass(self):
    """Tests predictions are updated once trees are added for multi class."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 15
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              vector: {
                value: 1.14
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 8.79
              }
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          is_finalized: true
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Two examples, none were cached before.
      cached_tree_ids = [0, 0]
      cached_node_ids = [0, 0]

      feature_0_values = [67, 5]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values],
          logits_dimension=1)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # We are in the first tree.
      self.assertAllClose([0, 0], new_tree_ids)
      self.assertAllClose([2, 1], new_node_ids)
      expected_logit_updates = 0.1 * np.array([[8.79], [1.14]])
      self.assertAllClose(expected_logit_updates, logits_updates)

  @test_util.run_deprecated_v1
  def testCachedPredictionIsCurrent(self):
    """Tests that prediction based on previous node in the tree works."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge("""
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 15
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
              original_leaf {
                scalar: -2
              }
            }
          }
          nodes {
            leaf {
              scalar: 1.14
            }
          }
          nodes {
            leaf {
              scalar: 8.79
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          is_finalized: true
          num_layers_grown: 2
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Two examples, one was cached in node 1 first, another in node 2.
      cached_tree_ids = [0, 0]
      cached_node_ids = [1, 2]

      # We have two features: 0 and 1. Values don't matter because trees didn't
      # change.
      feature_0_values = [67, 5]
      feature_1_values = [9, 17]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # Nothing changed.
      self.assertAllClose(cached_tree_ids, new_tree_ids)
      self.assertAllClose(cached_node_ids, new_node_ids)
      self.assertAllClose([[0], [0]], logits_updates)

  @test_util.run_deprecated_v1
  def testCachedPredictionIsCurrentMultiClass(self):
    """Tests that cached prediction is current for multi class."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 15
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
              original_leaf {
                vector: {
                  value: -2
                }
                vector: {
                  value: 1.2
                }
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 1.14
              }
              vector: {
                value: -0.5
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 8.79
              }
              vector: {
                value: 0.2
              }
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          is_finalized: true
          num_layers_grown: 2
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Two examples, one was cached in node 1 first, another in node 2.
      cached_tree_ids = [0, 0]
      cached_node_ids = [1, 2]

      # We have two features: 0 and 1. Values don't matter because trees didn't
      # change.
      feature_0_values = [67, 5]
      feature_1_values = [9, 17]

      # Multi-class.
      logits_dimension = 2

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=logits_dimension)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # Nothing changed.
      self.assertAllClose(cached_tree_ids, new_tree_ids)
      self.assertAllClose(cached_node_ids, new_node_ids)
      self.assertAllClose([[0, 0], [0, 0]], logits_updates)

  @test_util.run_deprecated_v1
  def testCachedPredictionFromTheSameTree(self):
    """Tests that prediction based on previous node in the tree works."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge("""
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 15
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
              original_leaf {
                scalar: -2
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 7
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 1.4
              original_leaf {
                scalar: 7.14
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 7
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 2.7
              original_leaf {
                scalar: -4.375
              }
            }
          }
          nodes {
            leaf {
              scalar: 1.14
            }
          }
          nodes {
            leaf {
              scalar: 8.79
            }
          }
          nodes {
            leaf {
              scalar: -5.875
            }
          }
          nodes {
            leaf {
              scalar: -2.075
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          is_finalized: true
          num_layers_grown: 2
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Two examples, one was cached in node 1 first, another in node 0.
      cached_tree_ids = [0, 0]
      cached_node_ids = [1, 0]

      # We have two features: 0 and 1.
      feature_0_values = [67, 5]
      feature_1_values = [9, 17]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # We are still in the same tree.
      self.assertAllClose([0, 0], new_tree_ids)
      # When using the full tree, the first example will end up in node 4,
      # the second in node 5.
      self.assertAllClose([4, 5], new_node_ids)
      # Full predictions for each instance would be 8.79 and -5.875,
      # so an update from the previous cached values lr*(7.14 and -2) would be
      # 1.65 and -3.875, and then multiply them by 0.1 (lr)
      self.assertAllClose([[0.1 * 1.65], [0.1 * -3.875]], logits_updates)

  @test_util.run_deprecated_v1
  def testCachedPredictionFromTheSameTreeMultiClass(self):
    """Tests that cache prediction works within a tree for multi-class."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 15
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
              original_leaf {
                vector: {
                  value: -2
                }
                vector: {
                  value: 1.2
                }
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 7
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 1.4
              original_leaf {
                vector: {
                  value: 7.14
                }
                vector: {
                  value: -3.2
                }
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 7
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 2.7
              original_leaf {
                vector: {
                  value: -4.375
                }
                vector: {
                  value: 0.9
                }
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 1.14
              }
              vector: {
                value: 0.27
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 8.79
              }
              vector: {
                value: -3.4
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: -5.875
              }
              vector: {
                value: 1.61
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: -2.075
              }
              vector: {
                value: 3.48
              }
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          is_finalized: true
          num_layers_grown: 2
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Two examples, one was cached in node 1 first, another in node 0.
      cached_tree_ids = [0, 0]
      cached_node_ids = [1, 0]

      # We have two features: 0 and 1.
      feature_0_values = [67, 5]
      feature_1_values = [9, 17]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=2)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # We are still in the same tree.
      self.assertAllClose([0, 0], new_tree_ids)
      # When using the full tree, the first example will end up in node 4,
      # the second in node 5.
      self.assertAllClose([4, 5], new_node_ids)
      # Full predictions for example 1: [8.79, -3.4], example 2: [-5.875, 1.61].
      # So an update from the previous cached values lr*([7.14, -3.2] and [-2,
      # 1.2]) would be [1.65, -0.2] for example1 and [-3.875, 0.41] for
      # example2; and then multiply them by 0.1 (lr).
      self.assertAllClose(0.1 * np.array([[1.65, -0.2], [-3.875, 0.41]]),
                          logits_updates)

  @test_util.run_deprecated_v1
  def testCachedPredictionFromPreviousTree(self):
    """Tests the predictions work when we have cache from previous trees."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge("""
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 28
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 1.14
            }
          }
          nodes {
            leaf {
              scalar: 8.79
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 26
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 50
              left_id: 3
              right_id: 4
            }
          }
          nodes {
            leaf {
              scalar: 7
            }
          }
          nodes {
            leaf {
              scalar: 5
            }
          }
          nodes {
            leaf {
              scalar: 6
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 34
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            leaf {
              scalar: -7.0
            }
          }
          nodes {
            leaf {
              scalar: 5.0
            }
          }
        }
        tree_metadata {
          is_finalized: true
        }
        tree_metadata {
          is_finalized: true
        }
        tree_metadata {
          is_finalized: false
        }
        tree_weights: 0.1
        tree_weights: 0.1
        tree_weights: 0.1
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Two examples, one was cached in node 1 first, another in node 0.
      cached_tree_ids = [0, 0]
      cached_node_ids = [1, 0]

      # We have two features: 0 and 1.
      feature_0_values = [36, 32]
      feature_1_values = [11, 27]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)
      # Example 1 will get to node 3 in tree 1 and node 2 of tree 2
      # Example 2 will get to node 2 in tree 1 and node 1 of tree 2

      # We are in the last tree.
      self.assertAllClose([2, 2], new_tree_ids)
      # When using the full tree, the first example will end up in node 4,
      # the second in node 5.
      self.assertAllClose([2, 1], new_node_ids)
      # Example 1: tree 0: 8.79, tree 1: 5.0, tree 2: 5.0 = >
      #            change = 0.1*(5.0+5.0)
      # Example 2: tree 0: 1.14, tree 1: 7.0, tree 2: -7 = >
      #            change= 0.1(1.14+7.0-7.0)
      self.assertAllClose([[1], [0.114]], logits_updates)

  @test_util.run_deprecated_v1
  def testCachedPredictionFromPreviousTreeMultiClass(self):
    """Tests predictions when we have cache from previous trees multi-class."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 28
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
              original_leaf {
                vector: {
                  value: 0
                }
                vector: {
                  value: 0
                }
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 1.14
              }
              vector: {
                value: 0.27
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 8.79
              }
              vector: {
                value: -3.4
              }
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 26
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 50
              left_id: 3
              right_id: 4
            }
          }
          nodes {
            leaf {
              vector: {
                value: 7
              }
              vector: {
                value: 1.12
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 5
              }
              vector: {
                value: -0.4
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 6
              }
              vector: {
                value: 1.4
              }
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 34
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            leaf {
              vector: {
                value: -7
              }
              vector: {
                value: 3.4
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 5.0
              }
              vector: {
                value: 1.24
              }
            }
          }
        }
        tree_metadata {
          is_finalized: true
        }
        tree_metadata {
          is_finalized: true
        }
        tree_metadata {
          is_finalized: false
        }
        tree_weights: 0.1
        tree_weights: 0.1
        tree_weights: 0.1
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # Two examples, one was cached in node 1 first, another in node 0.
      cached_tree_ids = [0, 0]
      cached_node_ids = [1, 0]

      # We have two features: 0 and 1.
      feature_0_values = [36, 32]
      feature_1_values = [11, 27]

      # Multi-class.
      logits_dimension = 2

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=logits_dimension)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)
      # Example 1 will get to node 3 in tree 1 and node 2 of tree 2
      # Example 2 will get to node 2 in tree 1 and node 1 of tree 2

      # We are in the last tree.
      self.assertAllClose([2, 2], new_tree_ids)
      self.assertAllClose([2, 1], new_node_ids)
      # Example 1 was cached at tree 0, node 1.
      # Example 1: tree 0: [8.79, -3.4], tree 1: [5, -0.4], tree 2: [5, 1.24]
      #            change = 0.1*(5.0+5.0, -0.4+1.24)
      # Example 2 was cached at tree 0, node 0.
      # Example 2: tree 0: [1.14, 0.27], tree 1: [7.0, 1.12], tree 2: [-7, 3.4]
      #            change= 0.1(1.14+7.0-7.0, 0.27+1.12+3.4)
      self.assertAllClose(0.1 * np.array([[10, 0.84], [1.14, 4.79]]),
                          logits_updates)

  @test_util.run_deprecated_v1
  def testCategoricalSplits(self):
    """Tests the training prediction work for categorical splits."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            categorical_split {
              feature_id: 1
              value: 2
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            categorical_split {
              feature_id: 0
              value: 13
              left_id: 3
              right_id: 4
            }
          }
          nodes {
            leaf {
              scalar: 7.0
            }
          }
          nodes {
            leaf {
              scalar: 5.0
            }
          }
          nodes {
            leaf {
              scalar: 6.0
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          is_finalized: true
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      feature_0_values = [13, 1, 3]
      feature_1_values = [2, 2, 1]

      # No previous cached values.
      cached_tree_ids = [0, 0, 0]
      cached_node_ids = [0, 0, 0]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      self.assertAllClose([0, 0, 0], new_tree_ids)
      self.assertAllClose([3, 4, 2], new_node_ids)
      self.assertAllClose([[5.], [6.], [7.]], logits_updates)

  @test_util.run_deprecated_v1
  def testCategoricalSplitsMultiClass(self):
    """Tests the training prediction work for categorical splits."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            categorical_split {
              feature_id: 1
              value: 2
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            categorical_split {
              feature_id: 0
              value: 13
              left_id: 3
              right_id: 4
            }
          }
          nodes {
          leaf {
              vector: {
                value: 7
              }
              vector: {
                value: 1.12
              }
            }
          }
          nodes {
          leaf {
              vector: {
                value: 5
              }
              vector: {
                value: 1.24
              }
            }
          }
          nodes {
          leaf {
              vector: {
                value: 6
              }
              vector: {
                value: 1.4
              }
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          is_finalized: true
        }
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      feature_0_values = [13, 1, 3]
      feature_1_values = [2, 2, 1]

      # No previous cached values.
      cached_tree_ids = [0, 0, 0]
      cached_node_ids = [0, 0, 0]

      # Multi-class.
      logits_dimension = 2

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=logits_dimension)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      self.assertAllClose([0, 0, 0], new_tree_ids)
      self.assertAllClose([3, 4, 2], new_node_ids)
      self.assertAllClose([[5., 1.24], [6., 1.4], [7., 1.12]], logits_updates)

  @test_util.run_deprecated_v1
  def testCachedPredictionFromTheSameTreeWithPostPrunedNodes(self):
    """Tests that prediction based on previous node in the tree works."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id:0
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
              feature_id: 1
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
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 3
        }
      """, tree_ensemble_config)

      # Create existing ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      cached_tree_ids = [0, 0, 0, 0, 0, 0]
      # Leaves 3,4, 7 and 8 got deleted during post-pruning, leaves 5 and 6
      # changed the ids to 3 and 4 respectively.
      cached_node_ids = [3, 4, 5, 6, 7, 8]

      # We have two features: 0 and 1.
      feature_0_values = [12, 17, 35, 36, 23, 11]
      feature_1_values = [12, 12, 17, 18, 123, 24]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # We are still in the same tree.
      self.assertAllClose([0, 0, 0, 0, 0, 0], new_tree_ids)
      # Examples from leaves 3,4,7,8 should be in leaf 1, examples from leaf 5
      # and 6 in leaf 3 and 4.
      self.assertAllClose([1, 1, 3, 4, 1, 1], new_node_ids)

      cached_values = [[0.08], [0.093], [0.0553], [0.0783], [0.15 + 0.08],
                       [0.5 + 0.08]]
      self.assertAllClose([[0.01], [0.01], [0.0553], [0.0783], [0.01], [0.01]],
                          logits_updates + cached_values)

  @test_util.run_deprecated_v1
  def testCachedPredictionFromTheSameTreeWithPostPrunedNodesMultiClass(self):
    """Tests that prediction based on previous node in tree works multiclass."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id:0
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
              vector: {
                value: 0.01
              }
              vector: {
                value: 0.032
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 5
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 0.5
              original_leaf {
                vector: {
                  value: 0.0143
                }
                vector: {
                  value: 0.022
                }
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 0.0553
              }
              vector: {
                value: -0.02
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 0.0783
              }
              vector: {
                value: 0.012
              }
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 3
          is_finalized: true
          post_pruned_nodes_meta {
            new_node_id: 0
            logit_change: 0.0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: 0.0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 2
            logit_change: 0.0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.07
            logit_change: -0.02
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.083
            logit_change: -0.42
          }
          post_pruned_nodes_meta {
            new_node_id: 3
            logit_change: 0.0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 4
            logit_change: 0.0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.22
            logit_change: -0.05
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.57
            logit_change: -0.11
          }
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 3
        }
      """, tree_ensemble_config)

      # Create existing ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      cached_tree_ids = [0, 0, 0, 0, 0, 0]
      # Leaves 3,4, 7 and 8 got deleted during post-pruning, leaves 5 and 6
      # changed the ids to 3 and 4 respectively.
      cached_node_ids = [3, 4, 5, 6, 7, 8]

      # We have two features: 0 and 1.
      feature_0_values = [12, 17, 35, 36, 23, 11]
      feature_1_values = [12, 12, 17, 18, 123, 24]

      # Multi-class.
      logits_dimension = 2

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=logits_dimension)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # We are still in the same tree.
      self.assertAllClose([0, 0, 0, 0, 0, 0], new_tree_ids)
      # Examples from leaves 3,4,7,8 should be in leaf 1, examples from leaf 5
      # and 6 in leaf 3 and 4.
      self.assertAllClose([1, 1, 3, 4, 1, 1], new_node_ids)
      cached_values = np.array([[0.01 + 0.07, 0.032 + 0.02],
                                [0.01 + 0.083, 0.032 + 0.42], [0.0553, -0.02],
                                [0.0783, 0.012],
                                [0.08 + (-0.07 + 0.22), 0.052 + (-0.02 + 0.05)],
                                [0.08 + (-0.07 + 0.57),
                                 0.052 + (-0.02 + 0.11)]])
      self.assertAllClose([[0.01, 0.032], [0.01, 0.032], [0.0553, -0.02],
                           [0.0783, 0.012], [0.01, 0.032], [0.01, 0.032]],
                          np.array(logits_updates) + cached_values)

  @test_util.run_deprecated_v1
  def testCachedPredictionFromThePreviousTreeWithPostPrunedNodes(self):
    """Tests that prediction based on previous node in the tree works."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id:0
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
              feature_id: 1
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
              scalar: 0.55
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
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 4
        }
      """, tree_ensemble_config)

      # Create existing ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      cached_tree_ids = [0, 0, 0, 0, 0, 0]
      # Leaves 3,4, 7 and 8 got deleted during post-pruning, leaves 5 and 6
      # changed the ids to 3 and 4 respectively.
      cached_node_ids = [3, 4, 5, 6, 7, 8]

      # We have two features: 0 and 1.
      feature_0_values = [12, 17, 35, 36, 23, 11]
      feature_1_values = [12, 12, 17, 18, 123, 24]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # We are in the last tree.
      self.assertAllClose([1, 1, 1, 1, 1, 1], new_tree_ids)
      # Examples from leaves 3,4,7,8 should be in leaf 1, examples from leaf 5
      # and 6 in leaf 3 and 4 in tree 0. For tree 1, all of the examples are in
      # the root node.
      self.assertAllClose([0, 0, 0, 0, 0, 0], new_node_ids)

      cached_values = [[0.08], [0.093], [0.0553], [0.0783], [0.15 + 0.08],
                       [0.5 + 0.08]]

      root = 0.55
      self.assertAllClose([[root + 0.01], [root + 0.01], [root + 0.0553],
                           [root + 0.0783], [root + 0.01], [root + 0.01]],
                          logits_updates + cached_values)

  @test_util.run_deprecated_v1
  def testCachedPredictionFromThePreviousTreeWithPostPrunedNodesMultiClass(
      self):
    """Tests that prediction from pruned previous tree works multi class."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id:0
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
              vector: {
                value: 0.01
              }
              vector: {
                value: 0.032
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 5
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 0.5
              original_leaf {
                vector: {
                  value: 0.0143
                }
                vector: {
                  value: 0.022
                }
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 0.0553
              }
              vector: {
                value: -0.02
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 0.0783
              }
              vector: {
                value: 0.012
              }
            }
          }
        }
        trees {
          nodes {
            leaf {
              vector: {
                value: 0.55
              }
              vector: {
                value: 0.03
              }
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
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: 0.0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 2
            logit_change: 0.0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.07
            logit_change: -0.02
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.083
            logit_change: -0.42
          }
          post_pruned_nodes_meta {
            new_node_id: 3
            logit_change: 0.0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 4
            logit_change: 0.0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.22
            logit_change: -0.05
          }
          post_pruned_nodes_meta {
            new_node_id: 1
            logit_change: -0.57
            logit_change: -0.11
          }
        }
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 4
        }
      """, tree_ensemble_config)

      # Create existing ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      cached_tree_ids = [0, 0, 0, 0, 0, 0]
      # Leaves 3,4, 7 and 8 got deleted during post-pruning, leaves 5 and 6
      # changed the ids to 3 and 4 respectively.
      cached_node_ids = [3, 4, 5, 6, 7, 8]

      # We have two features: 0 and 1.
      feature_0_values = [12, 17, 35, 36, 23, 11]
      feature_1_values = [12, 12, 17, 18, 123, 24]

      # Multi class.
      logits_dimension = 2

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=logits_dimension)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # We are in the last tree.
      self.assertAllClose([1, 1, 1, 1, 1, 1], new_tree_ids)
      # Examples from leaves 3,4,7,8 should be in leaf 1, examples from leaf 5
      # and 6 in leaf 3 and 4 in tree 0. For tree 1, all of the examples are in
      # the root node.
      self.assertAllClose([0, 0, 0, 0, 0, 0], new_node_ids)
      tree1_logits = np.array([[0.01, 0.032], [0.01, 0.032], [0.0553, -0.02],
                               [0.0783, 0.012], [0.01, 0.032], [0.01, 0.032]])
      tree2_root_weights = [0.55, 0.03]
      expected_logits = tree1_logits
      expected_logits[:, 0] += tree2_root_weights[0]
      expected_logits[:, 1] += tree2_root_weights[1]
      cached_values = np.array([[0.01 + 0.07, 0.032 + 0.02],
                                [0.01 + 0.083, 0.032 + 0.42], [0.0553, -0.02],
                                [0.0783, 0.012],
                                [0.08 + (-0.07 + 0.22), 0.052 + (-0.02 + 0.05)],
                                [0.08 + (-0.07 + 0.57),
                                 0.052 + (-0.02 + 0.11)]])
      self.assertAllClose(expected_logits,
                          np.array(logits_updates) + cached_values)

  @test_util.run_deprecated_v1
  def testCachedPredictionTheWholeTreeWasPruned(self):
    """Tests that prediction based on previous node in the tree works."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            leaf {
              scalar: 0.00
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: true
          post_pruned_nodes_meta {
            new_node_id: 0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 0
            logit_change: -6.0
          }
          post_pruned_nodes_meta {
            new_node_id: 0
            logit_change: 5.0
          }
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """, tree_ensemble_config)

      # Create existing ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      cached_tree_ids = [
          0,
          0,
      ]
      # The predictions were cached in 1 and 2, both were pruned to the root.
      cached_node_ids = [1, 2]

      # We have two features: 0 and 1.These are not going to be used anywhere.
      feature_0_values = [12, 17]
      feature_1_values = [12, 12]

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # We are in the last tree.
      self.assertAllClose([0, 0], new_tree_ids)
      self.assertAllClose([0, 0], new_node_ids)

      self.assertAllClose([[-6.0], [5.0]], logits_updates)

  @test_util.run_deprecated_v1
  def testCachedPredictionTheWholeTreeWasPrunedMultiClass(self):
    """Tests that prediction works when whole tree was pruned multi class."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            leaf {
              vector: {
                value: 0.00
              }
              vector: {
                value: 0.00
              }
            }
          }
        }
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 1
          is_finalized: true
          post_pruned_nodes_meta {
            new_node_id: 0
            logit_change: 0.0
            logit_change: 0.0
          }
          post_pruned_nodes_meta {
            new_node_id: 0
            logit_change: -6.0
            logit_change: -2.0
          }
          post_pruned_nodes_meta {
            new_node_id: 0
            logit_change: 5.0
            logit_change: -0.4
          }
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """, tree_ensemble_config)

      # Create existing ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()
      cached_tree_ids = [0, 0]
      # The predictions were cached in 1 and 2, both were pruned to the root.
      cached_node_ids = [1, 2]

      # We have two features: 0 and 1.These are not going to be used anywhere.
      feature_0_values = [12, 17]
      feature_1_values = [12, 12]

      # Multi class.
      logits_dimension = 2

      # Grow tree ensemble.
      predict_op = boosted_trees_ops.training_predict(
          tree_ensemble_handle,
          cached_tree_ids=cached_tree_ids,
          cached_node_ids=cached_node_ids,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=logits_dimension)
      logits_updates, new_tree_ids, new_node_ids = session.run(predict_op)

      # We are in the last tree.
      self.assertAllClose([0, 0], new_tree_ids)
      self.assertAllClose([0, 0], new_node_ids)
      self.assertAllClose([[-6.0, -2.0], [5.0, -0.4]], logits_updates)


class PredictionOpsTest(test_util.TensorFlowTestCase):
  """Tests prediction ops for inference."""

  @test_util.run_deprecated_v1
  def testPredictionOnEmptyEnsemble(self):
    """Tests that prediction on a empty ensemble does not fail."""
    with self.cached_session() as session:
      # Create an empty ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto='')
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      feature_0_values = [36, 32]
      feature_1_values = [11, 27]
      expected_logits = [[0.0], [0.0]]

      # Prediction should work fine.
      predict_op = boosted_trees_ops.predict(
          tree_ensemble_handle,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits = session.run(predict_op)
      self.assertAllClose(expected_logits, logits)

  @test_util.run_deprecated_v1
  def testPredictionOnEmptyEnsembleMultiClass(self):
    """Tests that prediction on empty ensemble does not fail for multiclass."""
    with self.cached_session() as session:
      # Create an empty ensemble.
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto='')
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      feature_0_values = [36, 32]
      feature_1_values = [11, 27]
      logits_dimension = 2
      expected_logits = [[0.0, 0.0], [0.0, 0.0]]

      # Prediction should work fine.
      predict_op = boosted_trees_ops.predict(
          tree_ensemble_handle,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=logits_dimension)

      logits = session.run(predict_op)
      self.assertAllClose(expected_logits, logits)

  @test_util.run_deprecated_v1
  def testPredictionMultipleTree(self):
    """Tests the predictions work when we have multiple trees."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 28
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              scalar: 1.14
            }
          }
          nodes {
            leaf {
              scalar: 8.79
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 26
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 50
              left_id: 3
              right_id: 4
            }
          }
          nodes {
            leaf {
              scalar: 7.0
            }
          }
          nodes {
            leaf {
              scalar: 5.0
            }
          }
          nodes {
            leaf {
              scalar: 6.0
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 34
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            leaf {
              scalar: -7.0
            }
          }
          nodes {
            leaf {
              scalar: 5.0
            }
          }
        }
        tree_weights: 0.1
        tree_weights: 0.2
        tree_weights: 1.0
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      feature_0_values = [36, 32]
      feature_1_values = [11, 27]

      # Example 1: tree 0: 1.14, tree 1: 5.0, tree 2: 5.0 = >
      #            logit = 0.1*1.14+0.2*5.0+1*5
      # Example 2: tree 0: 1.14, tree 1: 7.0, tree 2: -7 = >
      #            logit= 0.1*1.14+0.2*7.0-1*7.0
      expected_logits = [[6.114], [-5.486]]

      # Prediction should work fine.
      predict_op = boosted_trees_ops.predict(
          tree_ensemble_handle,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits = session.run(predict_op)
      self.assertAllClose(expected_logits, logits)

  @test_util.run_deprecated_v1
  def testPredictionMultipleTreeMultiClass(self):
    """Tests the predictions work when we have multiple trees."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 28
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              vector: {
                value: 0.51
              }
              vector: {
                value: 1.14
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 1.29
              }
              vector: {
                value: 8.79
              }
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 26
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 50
              left_id: 3
              right_id: 4
            }
          }
          nodes {
            leaf {
              vector: {
                value: -4.33
              }
              vector: {
                value: 7.0
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 0.2
              }
              vector: {
                value: 5.0
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: -4.1
              }
              vector: {
                value: 6.0
              }
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 34
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            leaf {
              vector: {
                value: 2.0
              }
              vector: {
                value: -7.0
              }
            }
          }
          nodes {
            leaf {
              vector: {
                value: 6.3
              }
              vector: {
                value: 5.0
              }
            }
          }
        }
        tree_weights: 0.1
        tree_weights: 0.2
        tree_weights: 1.0
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      feature_0_values = [36, 32]
      feature_1_values = [11, 27]

      # Example 1: tree 0: (0.51, 1.14), tree 1: (0.2, 5.0), tree 2: (6.3, 5.0)
      #
      #            logits = (0.1*0.51+0.2*0.2+1*6.3,
      #                      0.1*1.14+0.2*5.0+1*5)
      # Example 2: tree 0: (0.51, 1.14), tree 1: (-4.33, 7.0), tree 2: (2.0, -7)
      #
      #            logits = (0.1*0.51+0.2*-4.33+1*2.0,
      #                      0.1*1.14+0.2*7.0+1*-7)
      logits_dimension = 2
      expected_logits = [[6.391, 6.114], [1.185, -5.486]]

      # Prediction should work fine.
      predict_op = boosted_trees_ops.predict(
          tree_ensemble_handle,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=logits_dimension)

      logits = session.run(predict_op)
      self.assertAllClose(expected_logits, logits)

  @test_util.run_deprecated_v1
  def testCategoricalSplits(self):
    """Tests the predictions work for categorical splits."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            categorical_split {
              feature_id: 1
              value: 2
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            categorical_split {
              feature_id: 0
              value: 13
              left_id: 3
              right_id: 4
            }
          }
          nodes {
            leaf {
              scalar: 7.0
            }
          }
          nodes {
            leaf {
              scalar: 5.0
            }
          }
          nodes {
            leaf {
              scalar: 6.0
            }
          }
        }
        tree_weights: 1.0
      """, tree_ensemble_config)

      # Create existing ensemble with one root split
      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      feature_0_values = [13, 1, 3]
      feature_1_values = [2, 2, 1]

      expected_logits = [[5.], [6.], [7.]]

      # Prediction should work fine.
      predict_op = boosted_trees_ops.predict(
          tree_ensemble_handle,
          bucketized_features=[feature_0_values, feature_1_values],
          logits_dimension=1)

      logits = session.run(predict_op)
      self.assertAllClose(expected_logits, logits)


class FeatureContribsOpsTest(test_util.TensorFlowTestCase):
  """Tests feature contribs ops for model understanding."""

  @test_util.run_deprecated_v1
  def testContribsForOnlyABiasNode(self):
    """Tests case when, after training, only left with a bias node.

    For example, this could happen if the final ensemble contains one tree that
    got pruned up to the root.
    """
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            leaf {
              scalar: 1.72
            }
          }
        }
        tree_weights: 0.1
        tree_metadata: {
          num_layers_grown: 0
        }
      """, tree_ensemble_config)

      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      # All features are unused.
      feature_0_values = [36, 32]
      feature_1_values = [13, -29]
      feature_2_values = [11, 27]

      # Expected logits are computed by traversing the logit path and
      # subtracting child logits from parent logits.
      bias = 1.72 * 0.1  # Root node of tree_0.
      expected_feature_ids = ((), ())
      expected_logits_paths = ((bias,), (bias,))

      bucketized_features = [
          feature_0_values, feature_1_values, feature_2_values
      ]

      debug_op = boosted_trees_ops.example_debug_outputs(
          tree_ensemble_handle,
          bucketized_features=bucketized_features,
          logits_dimension=1)

      serialized_examples_debug_outputs = session.run(debug_op)
      feature_ids = []
      logits_paths = []
      for example in serialized_examples_debug_outputs:
        example_debug_outputs = boosted_trees_pb2.DebugOutput()
        example_debug_outputs.ParseFromString(example)
        feature_ids.append(example_debug_outputs.feature_ids)
        logits_paths.append(example_debug_outputs.logits_path)

      self.assertAllClose(feature_ids, expected_feature_ids)
      self.assertAllClose(logits_paths, expected_logits_paths)

  @test_util.run_deprecated_v1
  def testContribsMultipleTreeWhenFirstTreeIsABiasNode(self):
    """Tests case when, after training, first tree contains only a bias node."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            leaf {
              scalar: 1.72
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 26
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 50
              left_id: 3
              right_id: 4
            }
            metadata {
              original_leaf: {scalar: 5.5}
            }
          }
          nodes {
            leaf {
              scalar: 7.0
            }
          }
          nodes {
            leaf {
              scalar: 5.0
            }
          }
          nodes {
            leaf {
              scalar: 6.0
            }
          }
        }
        tree_weights: 1.
        tree_weights: 0.1
        tree_metadata: {
          num_layers_grown: 0
        }
        tree_metadata: {
          num_layers_grown: 1
        }
      """, tree_ensemble_config)

      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      feature_0_values = [36, 32]
      feature_1_values = [13, -29]  # Unused feature.
      feature_2_values = [11, 27]

      # Expected logits are computed by traversing the logit path and
      # subtracting child logits from parent logits.
      expected_feature_ids = ((2, 0), (2,))
      # bias = 1.72 * 1.  # Root node of tree_0.
      # example_0 :  (bias, 0.1 * 5.5 + bias, 0.1 * 5. + bias)
      # example_1 :  (bias, 0.1 * 7. + bias )
      expected_logits_paths = ((1.72, 2.27, 2.22), (1.72, 2.42))

      bucketized_features = [
          feature_0_values, feature_1_values, feature_2_values
      ]

      debug_op = boosted_trees_ops.example_debug_outputs(
          tree_ensemble_handle,
          bucketized_features=bucketized_features,
          logits_dimension=1)

      serialized_examples_debug_outputs = session.run(debug_op)
      feature_ids = []
      logits_paths = []
      for example in serialized_examples_debug_outputs:
        example_debug_outputs = boosted_trees_pb2.DebugOutput()
        example_debug_outputs.ParseFromString(example)
        feature_ids.append(example_debug_outputs.feature_ids)
        logits_paths.append(example_debug_outputs.logits_path)

      self.assertAllClose(feature_ids, expected_feature_ids)
      self.assertAllClose(logits_paths, expected_logits_paths)

  @test_util.run_deprecated_v1
  def testContribsMultipleTree(self):
    """Tests that the contribs work when we have multiple trees."""
    with self.cached_session() as session:
      tree_ensemble_config = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge(
          """
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 28
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
              original_leaf: {scalar: 2.1}
            }
          }
          nodes {
            leaf {
              scalar: 1.14
            }
          }
          nodes {
            leaf {
              scalar: 8.79
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 2
              threshold: 26
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 50
              left_id: 3
              right_id: 4
            }
            metadata {
              original_leaf: {scalar: 5.5}
            }
          }
          nodes {
            leaf {
              scalar: 7.0
            }
          }
          nodes {
            leaf {
              scalar: 5.0
            }
          }
          nodes {
            leaf {
              scalar: 6.0
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 0
              threshold: 34
              left_id: 1
              right_id: 2
            }
          }
          nodes {
            leaf {
              scalar: -7.0
            }
          }
          nodes {
            leaf {
              scalar: 5.0
            }
          }
        }
        tree_weights: 0.1
        tree_weights: 0.2
        tree_weights: 1.0
        tree_metadata: {
          num_layers_grown: 1
        }
        tree_metadata: {
          num_layers_grown: 2
        }
        tree_metadata: {
          num_layers_grown: 1
        }
      """, tree_ensemble_config)

      tree_ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble', serialized_proto=tree_ensemble_config.SerializeToString())
      tree_ensemble_handle = tree_ensemble.resource_handle
      resources.initialize_resources(resources.shared_resources()).run()

      feature_0_values = [36, 32]
      feature_1_values = [13, -29]  # Unused. Feature is not in above ensemble.
      feature_2_values = [11, 27]

      # Expected logits are computed by traversing the logit path and
      # subtracting child logits from parent logits.
      bias = 2.1 * 0.1  # Root node of tree_0.
      expected_feature_ids = ((2, 2, 0, 0), (2, 2, 0))
      # example_0 :  (bias, 0.1 * 1.14, 0.2 * 5.5 + .114, 0.2 * 5. + .114,
      # 1.0 * 5.0 + 0.2 * 5. + .114)
      # example_1 :  (bias, 0.1 * 1.14, 0.2 * 7 + .114,
      # 1.0 * -7. + 0.2 * 7 + .114)
      expected_logits_paths = ((bias, 0.114, 1.214, 1.114, 6.114),
                               (bias, 0.114, 1.514, -5.486))

      bucketized_features = [
          feature_0_values, feature_1_values, feature_2_values
      ]

      debug_op = boosted_trees_ops.example_debug_outputs(
          tree_ensemble_handle,
          bucketized_features=bucketized_features,
          logits_dimension=1)

      serialized_examples_debug_outputs = session.run(debug_op)
      feature_ids = []
      logits_paths = []
      for example in serialized_examples_debug_outputs:
        example_debug_outputs = boosted_trees_pb2.DebugOutput()
        example_debug_outputs.ParseFromString(example)
        feature_ids.append(example_debug_outputs.feature_ids)
        logits_paths.append(example_debug_outputs.logits_path)

      self.assertAllClose(feature_ids, expected_feature_ids)
      self.assertAllClose(logits_paths, expected_logits_paths)


if __name__ == '__main__':
  googletest.main()
