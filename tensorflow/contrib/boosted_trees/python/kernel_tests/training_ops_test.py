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
"""Tests for the GTFlow training Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from google.protobuf import text_format

from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.boosted_trees.proto import split_info_pb2
from tensorflow.contrib.boosted_trees.proto import tree_config_pb2
from tensorflow.contrib.boosted_trees.python.ops import model_ops
from tensorflow.contrib.boosted_trees.python.ops import training_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resources
from tensorflow.python.platform import googletest


def _gen_learner_config(num_classes,
                        l1_reg,
                        l2_reg,
                        tree_complexity,
                        max_depth,
                        min_node_weight,
                        pruning_mode,
                        growing_mode,
                        dropout_probability=None,
                        dropout_learning_rate=None,
                        dropout_prob_of_skipping=None):
  """Create a serialized learner config with the desired settings."""
  config = learner_pb2.LearnerConfig()
  config.num_classes = num_classes
  config.regularization.l1 = l1_reg
  config.regularization.l2 = l2_reg
  config.regularization.tree_complexity = tree_complexity
  config.constraints.max_tree_depth = max_depth
  config.constraints.min_node_weight = min_node_weight
  config.pruning_mode = pruning_mode
  config.growing_mode = growing_mode

  if dropout_probability is not None:
    config.learning_rate_tuner.dropout.dropout_probability = dropout_probability
  if dropout_learning_rate is not None:
    config.learning_rate_tuner.dropout.learning_rate = dropout_learning_rate
  if dropout_prob_of_skipping is not None:
    config.learning_rate_tuner.dropout.dropout_prob_of_skipping = (
        dropout_prob_of_skipping)
  return config.SerializeToString()


def _gen_dense_split_info(fc, threshold, left_weight, right_weight):
  split_str = """
    split_node {
      dense_float_binary_split {
        feature_column: %d
        threshold: %f
      }
    }
    left_child {
      sparse_vector {
        index: 0
        value: %f
      }
    }
    right_child {
      sparse_vector {
        index: 0
        value: %f
      }
    }""" % (fc, threshold, left_weight, right_weight)
  split = split_info_pb2.SplitInfo()
  text_format.Merge(split_str, split)
  return split.SerializeToString()


def _gen_categorical_split_info(fc, feat_id, left_weight, right_weight):
  split_str = """
    split_node {
      categorical_id_binary_split {
        feature_column: %d
        feature_id: %d
      }
    }
    left_child {
      sparse_vector {
        index: 0
        value: %f
      }
    }
    right_child {
      sparse_vector {
        index: 0
        value: %f
      }
    }""" % (fc, feat_id, left_weight, right_weight)
  split = split_info_pb2.SplitInfo()
  text_format.Merge(split_str, split)
  return split.SerializeToString()


def _get_bias_update(grads, hess):
  return array_ops.where(hess > 0, -grads / hess, array_ops.zeros_like(grads))


class CenterTreeEnsembleBiasOpTest(test_util.TensorFlowTestCase):
  """Tests for centering tree ensemble bias."""

  def testCenterBias(self):
    """Tests bias centering for multiple iterations."""
    with self.test_session() as session:
      # Create empty ensemble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = _gen_learner_config(
          num_classes=3,
          l1_reg=0,
          l2_reg=0,
          tree_complexity=0,
          max_depth=4,
          min_node_weight=0,
          pruning_mode=learner_pb2.LearnerConfig.PRE_PRUNE,
          growing_mode=learner_pb2.LearnerConfig.WHOLE_TREE,
          # Dropout does not change anything here.
          dropout_probability=0.5)

      # Center bias for the initial step.
      grads = constant_op.constant([0.4, -0.3])
      hess = constant_op.constant([2.0, 1.0])
      continue_centering1 = training_ops.center_tree_ensemble_bias(
          tree_ensemble_handle,
          stamp_token=0,
          next_stamp_token=1,
          delta_updates=_get_bias_update(grads, hess),
          learner_config=learner_config)
      continue_centering = session.run(continue_centering1)
      self.assertEqual(continue_centering, True)

      # Validate ensemble state.
      # dim 0 update: -0.4/2.0 = -0.2
      # dim 1 update: +0.3/1.0 = +0.3
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=1))
      tree_ensemble_config.ParseFromString(serialized)
      expected_result = """
        trees {
          nodes {
            leaf {
              vector {
                value: -0.2
                value: 0.3
              }
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
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertEqual(stats.num_trees, 0)
      self.assertEqual(stats.num_layers, 1)
      self.assertEqual(stats.active_tree, 1)
      self.assertEqual(stats.active_layer, 1)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 1)
      self.assertProtoEquals(expected_result, tree_ensemble_config)

      # Center bias for another step.
      # dim 0 update: -0.06/0.5 = -0.12
      # dim 1 update: -0.01/0.5 = -0.02
      grads = constant_op.constant([0.06, 0.01])
      hess = constant_op.constant([0.5, 0.5])
      continue_centering2 = training_ops.center_tree_ensemble_bias(
          tree_ensemble_handle,
          stamp_token=1,
          next_stamp_token=2,
          delta_updates=_get_bias_update(grads, hess),
          learner_config=learner_config)
      continue_centering = session.run(continue_centering2)
      self.assertEqual(continue_centering, True)

      # Validate ensemble state.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=2))
      tree_ensemble_config.ParseFromString(serialized)
      expected_result = """
        trees {
          nodes {
            leaf {
              vector {
                value: -0.32
                value: 0.28
              }
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
        }
      """
      self.assertEqual(new_stamp, 2)
      self.assertEqual(stats.num_trees, 0)
      self.assertEqual(stats.num_layers, 1)
      self.assertEqual(stats.active_tree, 1)
      self.assertEqual(stats.active_layer, 1)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 1)
      self.assertProtoEquals(expected_result, tree_ensemble_config)

      # Center bias for another step, but this time updates are negligible.
      grads = constant_op.constant([0.0000001, -0.00003])
      hess = constant_op.constant([0.5, 0.0])
      continue_centering3 = training_ops.center_tree_ensemble_bias(
          tree_ensemble_handle,
          stamp_token=2,
          next_stamp_token=3,
          delta_updates=_get_bias_update(grads, hess),
          learner_config=learner_config)
      continue_centering = session.run(continue_centering3)
      self.assertEqual(continue_centering, False)

      # Validate ensemble stamp.
      new_stamp, _ = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=3))
      self.assertEqual(new_stamp, 3)
      self.assertEqual(stats.num_trees, 1)
      self.assertEqual(stats.num_layers, 1)
      self.assertEqual(stats.active_tree, 1)
      self.assertEqual(stats.active_layer, 1)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 1)


class GrowTreeEnsembleOpTest(test_util.TensorFlowTestCase):
  """Tests for growing tree ensemble from split candidates."""

  def testGrowEmptyEnsemble(self):
    """Test growing an empty ensemble."""
    with self.test_session() as session:
      # Create empty ensemble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = _gen_learner_config(
          num_classes=2,
          l1_reg=0,
          l2_reg=0,
          tree_complexity=0,
          max_depth=1,
          min_node_weight=0,
          pruning_mode=learner_pb2.LearnerConfig.PRE_PRUNE,
          growing_mode=learner_pb2.LearnerConfig.WHOLE_TREE,
          # Dropout does not change anything here, tree is not finalized.
          dropout_probability=0.5)

      # Prepare handler inputs.
      # Note that handlers 1 & 3 have the same gain but different splits.
      handler1_partitions = np.array([0], dtype=np.int32)
      handler1_gains = np.array([7.62], dtype=np.float32)
      handler1_split = [_gen_dense_split_info(0, 0.52, -4.375, 7.143)]
      handler2_partitions = np.array([0], dtype=np.int32)
      handler2_gains = np.array([0.63], dtype=np.float32)
      handler2_split = [_gen_dense_split_info(0, 0.23, -0.6, 0.24)]
      handler3_partitions = np.array([0], dtype=np.int32)
      handler3_gains = np.array([7.62], dtype=np.float32)
      handler3_split = [_gen_categorical_split_info(0, 7, -4.375, 7.143)]

      # Grow tree ensemble.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=0,
          next_stamp_token=1,
          learning_rate=0.1,
          partition_ids=[
              handler1_partitions, handler2_partitions, handler3_partitions
          ],
          gains=[handler1_gains, handler2_gains, handler3_gains],
          splits=[handler1_split, handler2_split, handler3_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect the simpler split from handler 1 to be chosen.
      # The grown tree should be finalized as max tree depth is 1.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=1))
      tree_ensemble_config.ParseFromString(serialized)
      expected_result = """
        trees {
          nodes {
            dense_float_binary_split {
              threshold: 0.52
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -4.375
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 7.143
              }
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 1
          is_finalized: true
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertEqual(stats.num_trees, 1)
      self.assertEqual(stats.num_layers, 1)
      self.assertEqual(stats.active_tree, 1)
      self.assertEqual(stats.active_layer, 1)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 1)
      self.assertProtoEquals(expected_result, tree_ensemble_config)

  def testGrowExistingEnsembleTreeNotFinalized(self):
    """Test growing an existing ensemble with the last tree not finalized."""
    with self.test_session() as session:
      # Create existing ensemble with one root split
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      text_format.Merge("""
        trees {
          nodes {
            categorical_id_binary_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 7.61999988556
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 7.14300012589
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -4.375
              }
            }
          }
        }
        tree_weights: 0.10000000149
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """, tree_ensemble_config)
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = _gen_learner_config(
          num_classes=2,
          l1_reg=0,
          l2_reg=0,
          tree_complexity=0,
          max_depth=3,
          min_node_weight=0,
          pruning_mode=learner_pb2.LearnerConfig.PRE_PRUNE,
          growing_mode=learner_pb2.LearnerConfig.WHOLE_TREE,
          # Dropout does not change anything here - tree is not finalized.
          dropout_probability=0.5)

      # Prepare handler inputs.
      # Handler 1 only has a candidate for partition 1, handler 2 has candidates
      # for both partitions and handler 3 only has a candidate for partition 2.
      handler1_partitions = np.array([1], dtype=np.int32)
      handler1_gains = np.array([1.4], dtype=np.float32)
      handler1_split = [_gen_dense_split_info(0, 0.21, -6.0, 1.65)]
      handler2_partitions = np.array([1, 2], dtype=np.int32)
      handler2_gains = np.array([0.63, 2.7], dtype=np.float32)
      handler2_split = [
          _gen_dense_split_info(0, 0.23, -0.6, 0.24),
          _gen_categorical_split_info(1, 7, -1.5, 2.3)
      ]
      handler3_partitions = np.array([2], dtype=np.int32)
      handler3_gains = np.array([1.7], dtype=np.float32)
      handler3_split = [_gen_categorical_split_info(0, 3, -0.75, 1.93)]

      # Grow tree ensemble.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=0,
          next_stamp_token=1,
          learning_rate=0.1,
          partition_ids=[
              handler1_partitions, handler2_partitions, handler3_partitions
          ],
          gains=[handler1_gains, handler2_gains, handler3_gains],
          splits=[handler1_split, handler2_split, handler3_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect the split for partition 1 to be chosen from handler 1 and
      # the split for partition 2 to be chosen from handler 2.
      # The grown tree should not be finalized as max tree depth is 3 and
      # it's only grown 2 layers.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=1))
      tree_ensemble_config.ParseFromString(serialized)
      expected_result = """
        trees {
          nodes {
            categorical_id_binary_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 7.61999988556
            }
          }
          nodes {
            dense_float_binary_split {
              threshold: 0.21
              left_id: 3
              right_id: 4
            }
            node_metadata {
              gain: 1.4
            }
          }
          nodes {
            categorical_id_binary_split {
              feature_column: 1
              feature_id: 7
              left_id: 5
              right_id: 6
            }
            node_metadata {
              gain: 2.7
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -6.0
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 1.65
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -1.5
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 2.3
              }
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 2
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertEqual(stats.num_trees, 0)
      self.assertEqual(stats.num_layers, 2)
      self.assertEqual(stats.active_tree, 1)
      self.assertEqual(stats.active_layer, 2)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 2)
      self.assertProtoEquals(expected_result, tree_ensemble_config)

  def testGrowExistingEnsembleTreeFinalized(self):
    """Test growing an existing ensemble with the last tree finalized."""
    with self.test_session() as session:
      # Create existing ensemble with one root split
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      text_format.Merge("""
        trees {
          nodes {
            categorical_id_binary_split {
              feature_column: 3
              feature_id: 7
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 1.3
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 2.3
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -0.9
              }
            }
          }
        }
        tree_weights: 0.10000000149
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 1
          is_finalized: true
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """, tree_ensemble_config)
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = _gen_learner_config(
          num_classes=2,
          l1_reg=0,
          l2_reg=0,
          tree_complexity=0,
          max_depth=1,
          min_node_weight=0,
          pruning_mode=learner_pb2.LearnerConfig.PRE_PRUNE,
          growing_mode=learner_pb2.LearnerConfig.WHOLE_TREE)

      # Prepare handler inputs.
      handler1_partitions = np.array([0], dtype=np.int32)
      handler1_gains = np.array([7.62], dtype=np.float32)
      handler1_split = [_gen_dense_split_info(5, 0.52, -4.375, 7.143)]
      handler2_partitions = np.array([0], dtype=np.int32)
      handler2_gains = np.array([0.63], dtype=np.float32)
      handler2_split = [_gen_dense_split_info(2, 0.23, -0.6, 0.24)]
      handler3_partitions = np.array([0], dtype=np.int32)
      handler3_gains = np.array([7.62], dtype=np.float32)
      handler3_split = [_gen_categorical_split_info(8, 7, -4.375, 7.143)]

      # Grow tree ensemble.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=0,
          next_stamp_token=1,
          learning_rate=0.2,
          partition_ids=[
              handler1_partitions, handler2_partitions, handler3_partitions
          ],
          gains=[handler1_gains, handler2_gains, handler3_gains],
          splits=[handler1_split, handler2_split, handler3_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect a new tree to be added with the split from handler 1.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=1))
      tree_ensemble_config.ParseFromString(serialized)
      expected_result = """
        trees {
          nodes {
            categorical_id_binary_split {
              feature_column: 3
              feature_id: 7
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 1.3
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 2.3
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -0.9
              }
            }
          }
        }
        trees {
          nodes {
            dense_float_binary_split {
              feature_column: 5
              threshold: 0.52
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -4.375
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 7.143
              }
            }
          }
        }
        tree_weights: 0.1
        tree_weights: 0.2
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 1
          is_finalized: true
        }
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 1
          is_finalized: true
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 2
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertEqual(stats.num_trees, 2)
      self.assertEqual(stats.num_layers, 2)
      self.assertEqual(stats.active_tree, 2)
      self.assertEqual(stats.active_layer, 1)
      self.assertEqual(stats.attempted_trees, 2)
      self.assertEqual(stats.attempted_layers, 2)
      self.assertProtoEquals(expected_result, tree_ensemble_config)

  def testGrowEnsemblePrePrune(self):
    """Test growing an ensemble with pre-pruning."""
    with self.test_session() as session:
      # Create empty ensemble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = _gen_learner_config(
          num_classes=2,
          l1_reg=0,
          l2_reg=0,
          tree_complexity=0,
          max_depth=1,
          min_node_weight=0,
          pruning_mode=learner_pb2.LearnerConfig.PRE_PRUNE,
          growing_mode=learner_pb2.LearnerConfig.WHOLE_TREE)

      # Prepare handler inputs.
      # All handlers have negative gain.
      handler1_partitions = np.array([0], dtype=np.int32)
      handler1_gains = np.array([-0.62], dtype=np.float32)
      handler1_split = [_gen_dense_split_info(0, 0.52, 0.01, 0.0143)]
      handler2_partitions = np.array([0], dtype=np.int32)
      handler2_gains = np.array([-1.3], dtype=np.float32)
      handler2_split = [_gen_categorical_split_info(0, 7, 0.013, 0.0143)]

      # Grow tree ensemble.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=0,
          next_stamp_token=1,
          learning_rate=0.1,
          partition_ids=[handler1_partitions, handler2_partitions],
          gains=[handler1_gains, handler2_gains],
          splits=[handler1_split, handler2_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect the ensemble to be empty.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=1))
      tree_ensemble_config.ParseFromString(serialized)
      self.assertEqual(new_stamp, 1)
      self.assertEqual(stats.num_trees, 0)
      self.assertEqual(stats.num_layers, 0)
      self.assertEqual(stats.active_tree, 0)
      self.assertEqual(stats.active_layer, 0)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 1)
      self.assertProtoEquals("""
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """, tree_ensemble_config)

  def testGrowEnsemblePostPruneNone(self):
    """Test growing an empty ensemble."""
    with self.test_session() as session:
      # Create empty ensemble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = _gen_learner_config(
          num_classes=2,
          l1_reg=0,
          l2_reg=0,
          tree_complexity=0,
          max_depth=1,
          min_node_weight=0,
          pruning_mode=learner_pb2.LearnerConfig.POST_PRUNE,
          growing_mode=learner_pb2.LearnerConfig.WHOLE_TREE)

      # Prepare handler inputs.
      # Note that handlers 1 & 3 have the same gain but different splits.
      handler1_partitions = np.array([0], dtype=np.int32)
      handler1_gains = np.array([7.62], dtype=np.float32)
      handler1_split = [_gen_dense_split_info(0, 0.52, -4.375, 7.143)]
      handler2_partitions = np.array([0], dtype=np.int32)
      handler2_gains = np.array([0.63], dtype=np.float32)
      handler2_split = [_gen_dense_split_info(0, 0.23, -0.6, 0.24)]
      handler3_partitions = np.array([0], dtype=np.int32)
      handler3_gains = np.array([7.62], dtype=np.float32)
      handler3_split = [_gen_categorical_split_info(0, 7, -4.375, 7.143)]

      # Grow tree ensemble.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=0,
          next_stamp_token=1,
          learning_rate=0.1,
          partition_ids=[
              handler1_partitions, handler2_partitions, handler3_partitions
          ],
          gains=[handler1_gains, handler2_gains, handler3_gains],
          splits=[handler1_split, handler2_split, handler3_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect the simpler split from handler 1 to be chosen.
      # The grown tree should be finalized as max tree depth is 1.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=1))
      tree_ensemble_config.ParseFromString(serialized)
      expected_result = """
        trees {
          nodes {
            dense_float_binary_split {
              threshold: 0.52
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -4.375
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 7.143
              }
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 1
          is_finalized: true
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertEqual(stats.num_trees, 1)
      self.assertEqual(stats.num_layers, 1)
      self.assertEqual(stats.active_tree, 1)
      self.assertEqual(stats.active_layer, 1)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 1)
      self.assertProtoEquals(expected_result, tree_ensemble_config)

  def testGrowEnsemblePostPruneAll(self):
    """Test growing an ensemble with post-pruning."""
    with self.test_session() as session:
      # Create empty ensemble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = _gen_learner_config(
          num_classes=2,
          l1_reg=0,
          l2_reg=0,
          tree_complexity=0,
          max_depth=2,
          min_node_weight=0,
          pruning_mode=learner_pb2.LearnerConfig.POST_PRUNE,
          growing_mode=learner_pb2.LearnerConfig.WHOLE_TREE)

      # Prepare handler inputs.
      # All handlers have negative gain.
      handler1_partitions = np.array([0], dtype=np.int32)
      handler1_gains = np.array([-1.3], dtype=np.float32)
      handler1_split = [_gen_categorical_split_info(0, 7, 0.013, 0.0143)]
      handler2_partitions = np.array([0], dtype=np.int32)
      handler2_gains = np.array([-0.62], dtype=np.float32)
      handler2_split = [_gen_dense_split_info(0, 0.33, 0.01, 0.0143)]

      # Grow tree ensemble.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=0,
          next_stamp_token=1,
          learning_rate=0.1,
          partition_ids=[handler1_partitions, handler2_partitions],
          gains=[handler1_gains, handler2_gains],
          splits=[handler1_split, handler2_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect the split from handler 2 to be chosen despite the negative gain.
      # The grown tree should not be finalized as max tree depth is 2 so no
      # pruning occurs.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      tree_ensemble_config.ParseFromString(serialized)
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=1))
      expected_result = """
        trees {
          nodes {
            dense_float_binary_split {
              threshold: 0.33
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: -0.62
              original_leaf {
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 0.01
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 0.0143
              }
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertEqual(stats.num_trees, 0)
      self.assertEqual(stats.num_layers, 1)
      self.assertEqual(stats.active_tree, 1)
      self.assertEqual(stats.active_layer, 1)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 1)
      self.assertProtoEquals(expected_result, tree_ensemble_config)

      # Prepare handler inputs.
      # All handlers have negative gain.
      handler1_partitions = np.array([1, 2], dtype=np.int32)
      handler1_gains = np.array([-0.2, -0.5], dtype=np.float32)
      handler1_split = [
          _gen_categorical_split_info(3, 7, 0.07, 0.083),
          _gen_categorical_split_info(3, 5, 0.041, 0.064)
      ]

      # Grow tree ensemble.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=1,
          next_stamp_token=2,
          learning_rate=0.1,
          partition_ids=[handler1_partitions],
          gains=[handler1_gains],
          splits=[handler1_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect the ensemble to be empty as post-pruning will prune
      # the entire finalized tree.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=2))
      tree_ensemble_config.ParseFromString(serialized)
      self.assertEqual(new_stamp, 2)
      self.assertEqual(stats.num_trees, 0)
      self.assertEqual(stats.num_layers, 0)
      self.assertEqual(stats.active_tree, 0)
      self.assertEqual(stats.active_layer, 0)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 2)
      self.assertProtoEquals("""
      growing_metadata {
        num_trees_attempted: 1
        num_layers_attempted: 2
      }
      """, tree_ensemble_config)

  def testGrowEnsemblePostPrunePartial(self):
    """Test growing an ensemble with post-pruning."""
    with self.test_session() as session:
      # Create empty ensemble.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = _gen_learner_config(
          num_classes=2,
          l1_reg=0,
          l2_reg=0,
          tree_complexity=0,
          max_depth=2,
          min_node_weight=0,
          pruning_mode=learner_pb2.LearnerConfig.POST_PRUNE,
          growing_mode=learner_pb2.LearnerConfig.WHOLE_TREE)

      # Prepare handler inputs.
      # Second handler has positive gain.
      handler1_partitions = np.array([0], dtype=np.int32)
      handler1_gains = np.array([-1.3], dtype=np.float32)
      handler1_split = [_gen_categorical_split_info(0, 7, 0.013, 0.0143)]
      handler2_partitions = np.array([0], dtype=np.int32)
      handler2_gains = np.array([-0.2], dtype=np.float32)
      handler2_split = [_gen_dense_split_info(0, 0.33, 0.01, 0.0143)]

      # Grow tree ensemble.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=0,
          next_stamp_token=1,
          learning_rate=0.1,
          partition_ids=[handler1_partitions, handler2_partitions],
          gains=[handler1_gains, handler2_gains],
          splits=[handler1_split, handler2_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect the split from handler 2 to be chosen despite the negative gain.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=1))
      tree_ensemble_config.ParseFromString(serialized)
      expected_result = """
        trees {
          nodes {
            dense_float_binary_split {
              threshold: 0.33
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: -0.2
              original_leaf {
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 0.01
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 0.0143
              }
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertEqual(stats.num_trees, 0)
      self.assertEqual(stats.num_layers, 1)
      self.assertEqual(stats.active_tree, 1)
      self.assertEqual(stats.active_layer, 1)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 1)
      self.assertProtoEquals(expected_result, tree_ensemble_config)

      # Prepare handler inputs for second layer.
      # Note that partition 1 gain is negative and partition 2 gain is positive.
      handler1_partitions = np.array([1, 2], dtype=np.int32)
      handler1_gains = np.array([-0.2, 0.5], dtype=np.float32)
      handler1_split = [
          _gen_categorical_split_info(3, 7, 0.07, 0.083),
          _gen_categorical_split_info(3, 5, 0.041, 0.064)
      ]

      # Grow tree ensemble.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=1,
          next_stamp_token=2,
          learning_rate=0.1,
          partition_ids=[handler1_partitions],
          gains=[handler1_gains],
          splits=[handler1_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect the negative gain split of partition 1 to be pruned and the
      # positive gain split of partition 2 to be retained.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=2))
      tree_ensemble_config.ParseFromString(serialized)
      expected_result = """
        trees {
          nodes {
            dense_float_binary_split {
              threshold: 0.33
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: -0.2
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 0.01
              }
            }
          }
          nodes {
            categorical_id_binary_split {
              feature_column: 3
              feature_id: 5
              left_id: 3
              right_id: 4
            }
            node_metadata {
              gain: 0.5
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 0.041
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 0.064
              }
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 2
          is_finalized: true
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
        }
      """
      self.assertEqual(new_stamp, 2)
      self.assertEqual(stats.num_trees, 1)
      self.assertEqual(stats.num_layers, 2)
      self.assertEqual(stats.active_tree, 1)
      self.assertEqual(stats.active_layer, 2)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 2)
      self.assertProtoEquals(expected_result, tree_ensemble_config)

  def testGrowEnsembleTreeLayerByLayer(self):
    """Test growing an existing ensemble with the last tree not finalized."""
    with self.test_session() as session:
      # Create existing ensemble with one root split
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      text_format.Merge("""
        trees {
          nodes {
            categorical_id_binary_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 7.62
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 7.143
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -4.375
              }
            }
          }
        }
        tree_weights: 0.10000000149
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 1
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 1
        }
      """, tree_ensemble_config)
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = _gen_learner_config(
          num_classes=2,
          l1_reg=0,
          l2_reg=0,
          tree_complexity=0,
          max_depth=3,
          min_node_weight=0,
          pruning_mode=learner_pb2.LearnerConfig.PRE_PRUNE,
          growing_mode=learner_pb2.LearnerConfig.LAYER_BY_LAYER,
          # Dropout will have no effect, since the tree will not be fully grown.
          dropout_probability=1.0)

      # Prepare handler inputs.
      # Handler 1 only has a candidate for partition 1, handler 2 has candidates
      # for both partitions and handler 3 only has a candidate for partition 2.
      handler1_partitions = np.array([1], dtype=np.int32)
      handler1_gains = np.array([1.4], dtype=np.float32)
      handler1_split = [_gen_dense_split_info(0, 0.21, -6.0, 1.65)]
      handler2_partitions = np.array([1, 2], dtype=np.int32)
      handler2_gains = np.array([0.63, 2.7], dtype=np.float32)
      handler2_split = [
          _gen_dense_split_info(0, 0.23, -0.6, 0.24),
          _gen_categorical_split_info(1, 7, -1.5, 2.3)
      ]
      handler3_partitions = np.array([2], dtype=np.int32)
      handler3_gains = np.array([1.7], dtype=np.float32)
      handler3_split = [_gen_categorical_split_info(0, 3, -0.75, 1.93)]

      # Grow tree ensemble layer by layer.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=0,
          next_stamp_token=1,
          learning_rate=0.1,
          partition_ids=[
              handler1_partitions, handler2_partitions, handler3_partitions
          ],
          gains=[handler1_gains, handler2_gains, handler3_gains],
          splits=[handler1_split, handler2_split, handler3_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect the split for partition 1 to be chosen from handler 1 and
      # the split for partition 2 to be chosen from handler 2.
      # The grown tree should not be finalized as max tree depth is 3 and
      # it's only grown 2 layers.
      # The partition 1 split weights get added to original leaf weight 7.143.
      # The partition 2 split weights get added to original leaf weight -4.375.
      new_stamp, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      stats = session.run(
          training_ops.tree_ensemble_stats(tree_ensemble_handle, stamp_token=1))
      tree_ensemble_config.ParseFromString(serialized)
      expected_result = """
        trees {
          nodes {
            categorical_id_binary_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 7.62
            }
          }
          nodes {
            dense_float_binary_split {
              threshold: 0.21
              left_id: 3
              right_id: 4
            }
            node_metadata {
              gain: 1.4
            }
          }
          nodes {
            categorical_id_binary_split {
              feature_column: 1
              feature_id: 7
              left_id: 5
              right_id: 6
            }
            node_metadata {
              gain: 2.7
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 1.143
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 8.793
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -5.875
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -2.075
              }
            }
          }
        }
        tree_weights: 0.1
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 2
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 2
        }
      """
      self.assertEqual(new_stamp, 1)
      self.assertEqual(stats.num_trees, 0)
      self.assertEqual(stats.num_layers, 2)
      self.assertEqual(stats.active_tree, 1)
      self.assertEqual(stats.active_layer, 2)
      self.assertEqual(stats.attempted_trees, 1)
      self.assertEqual(stats.attempted_layers, 2)
      self.assertProtoEquals(expected_result, tree_ensemble_config)

  def testGrowExistingEnsembleTreeFinalizedWithDropout(self):
    """Test growing an existing ensemble with the last tree finalized."""
    with self.test_session() as session:
      # Create existing ensemble with one root split and one bias tree.
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      text_format.Merge("""
        trees {
          nodes {
            leaf {
              vector {
                value: -0.32
                value: 0.28
              }
            }
          }
        }
        trees {
          nodes {
            categorical_id_binary_split {
              feature_column: 3
              feature_id: 7
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 1.3
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: 2.3
              }
            }
          }
          nodes {
            leaf {
              sparse_vector {
                index: 0
                value: -0.9
              }
            }
          }
        }
        tree_weights: 0.7
        tree_weights: 1
        tree_metadata {
          num_tree_weight_updates: 1
          num_layers_grown: 1
          is_finalized: true
        }
        tree_metadata {
          num_tree_weight_updates: 5
          num_layers_grown: 1
          is_finalized: true
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 2
        }
      """, tree_ensemble_config)
      tree_ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0,
          tree_ensemble_config=tree_ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()

      # Prepare learner config.
      learner_config = _gen_learner_config(
          num_classes=2,
          l1_reg=0,
          l2_reg=0,
          tree_complexity=0,
          max_depth=1,
          min_node_weight=0,
          pruning_mode=learner_pb2.LearnerConfig.PRE_PRUNE,
          growing_mode=learner_pb2.LearnerConfig.WHOLE_TREE,
          dropout_probability=1.0)

      # Prepare handler inputs.
      handler1_partitions = np.array([0], dtype=np.int32)
      handler1_gains = np.array([7.62], dtype=np.float32)
      handler1_split = [_gen_dense_split_info(5, 0.52, -4.375, 7.143)]
      handler2_partitions = np.array([0], dtype=np.int32)
      handler2_gains = np.array([0.63], dtype=np.float32)
      handler2_split = [_gen_dense_split_info(2, 0.23, -0.6, 0.24)]
      handler3_partitions = np.array([0], dtype=np.int32)
      handler3_gains = np.array([7.62], dtype=np.float32)
      handler3_split = [_gen_categorical_split_info(8, 7, -4.375, 7.143)]

      # Grow tree ensemble.
      grow_op = training_ops.grow_tree_ensemble(
          tree_ensemble_handle,
          stamp_token=0,
          next_stamp_token=1,
          learning_rate=1,
          partition_ids=[
              handler1_partitions, handler2_partitions, handler3_partitions
          ],
          gains=[handler1_gains, handler2_gains, handler3_gains],
          splits=[handler1_split, handler2_split, handler3_split],
          learner_config=learner_config,
          dropout_seed=123,
          center_bias=True)
      session.run(grow_op)

      # Expect a new tree to be added with the split from handler 1.
      _, serialized = session.run(
          model_ops.tree_ensemble_serialize(tree_ensemble_handle))
      tree_ensemble_config.ParseFromString(serialized)

      self.assertEqual(3, len(tree_ensemble_config.trees))
      # Both trees got 0.5 as weights, bias tree is untouched.
      self.assertAllClose([0.7, 0.5, 0.5], tree_ensemble_config.tree_weights)

      self.assertEqual(
          1, tree_ensemble_config.tree_metadata[0].num_tree_weight_updates)
      self.assertEqual(
          6, tree_ensemble_config.tree_metadata[1].num_tree_weight_updates)
      self.assertEqual(
          2, tree_ensemble_config.tree_metadata[2].num_tree_weight_updates)


if __name__ == "__main__":
  googletest.main()
