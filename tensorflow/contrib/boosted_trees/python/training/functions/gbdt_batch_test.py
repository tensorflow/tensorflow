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
"""Tests for GBDT train function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf import text_format

from tensorflow.contrib import layers
from tensorflow.contrib.boosted_trees.proto import learner_pb2
from tensorflow.contrib.boosted_trees.proto import tree_config_pb2
from tensorflow.contrib.boosted_trees.python.ops import model_ops
from tensorflow.contrib.boosted_trees.python.training.functions import gbdt_batch
from tensorflow.contrib.boosted_trees.python.utils import losses

from tensorflow.python.feature_column import feature_column_lib as core_feature_column
from tensorflow.contrib.layers.python.layers import feature_column as feature_column_lib
from tensorflow.contrib.learn.python.learn.estimators import model_fn


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


def _squared_loss(label, unused_weights, predictions):
  """Unweighted loss implementation."""
  loss = math_ops.reduce_sum(
      math_ops.square(predictions - label), 1, keep_dims=True)
  return loss


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


class GbdtTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(GbdtTest, self).setUp()

  def testExtractFeatures(self):
    """Tests feature extraction."""
    with self.test_session():
      features = {}
      features["dense_float"] = array_ops.zeros([2, 1], dtypes.float32)
      features["sparse_float"] = sparse_tensor.SparseTensor(
          array_ops.zeros([2, 2], dtypes.int64),
          array_ops.zeros([2], dtypes.float32),
          array_ops.zeros([2], dtypes.int64))
      features["sparse_int"] = sparse_tensor.SparseTensor(
          array_ops.zeros([2, 2], dtypes.int64),
          array_ops.zeros([2], dtypes.int64),
          array_ops.zeros([2], dtypes.int64))
      (fc_names, dense_floats, sparse_float_indices, sparse_float_values,
       sparse_float_shapes, sparse_int_indices, sparse_int_values,
       sparse_int_shapes) = (
           gbdt_batch.extract_features(features, None, use_core_columns=False))
      self.assertEqual(len(fc_names), 3)
      self.assertAllEqual(fc_names,
                          ["dense_float", "sparse_float", "sparse_int"])
      self.assertEqual(len(dense_floats), 1)
      self.assertEqual(len(sparse_float_indices), 1)
      self.assertEqual(len(sparse_float_values), 1)
      self.assertEqual(len(sparse_float_shapes), 1)
      self.assertEqual(len(sparse_int_indices), 1)
      self.assertEqual(len(sparse_int_values), 1)
      self.assertEqual(len(sparse_int_shapes), 1)
      self.assertAllEqual(dense_floats[0].eval(),
                          features["dense_float"].eval())
      self.assertAllEqual(sparse_float_indices[0].eval(),
                          features["sparse_float"].indices.eval())
      self.assertAllEqual(sparse_float_values[0].eval(),
                          features["sparse_float"].values.eval())
      self.assertAllEqual(sparse_float_shapes[0].eval(),
                          features["sparse_float"].dense_shape.eval())
      self.assertAllEqual(sparse_int_indices[0].eval(),
                          features["sparse_int"].indices.eval())
      self.assertAllEqual(sparse_int_values[0].eval(),
                          features["sparse_int"].values.eval())
      self.assertAllEqual(sparse_int_shapes[0].eval(),
                          features["sparse_int"].dense_shape.eval())

  def testExtractFeaturesWithTransformation(self):
    """Tests feature extraction."""
    with self.test_session():
      features = {}
      features["dense_float"] = array_ops.zeros([2, 1], dtypes.float32)
      features["sparse_float"] = sparse_tensor.SparseTensor(
          array_ops.zeros([2, 2], dtypes.int64),
          array_ops.zeros([2], dtypes.float32),
          array_ops.zeros([2], dtypes.int64))
      features["sparse_categorical"] = sparse_tensor.SparseTensor(
          array_ops.zeros([2, 2], dtypes.int64),
          array_ops.zeros(
              [2], dtypes.string), array_ops.zeros([2], dtypes.int64))
      feature_columns = set()
      feature_columns.add(layers.real_valued_column("dense_float"))
      feature_columns.add(
          layers.feature_column._real_valued_var_len_column(
              "sparse_float", is_sparse=True))
      feature_columns.add(
          feature_column_lib.sparse_column_with_hash_bucket(
              "sparse_categorical", hash_bucket_size=1000000))
      (fc_names, dense_floats, sparse_float_indices, sparse_float_values,
       sparse_float_shapes, sparse_int_indices, sparse_int_values,
       sparse_int_shapes) = (
           gbdt_batch.extract_features(
               features, feature_columns, use_core_columns=False))
      self.assertEqual(len(fc_names), 3)
      self.assertAllEqual(fc_names,
                          ["dense_float", "sparse_float", "sparse_categorical"])
      self.assertEqual(len(dense_floats), 1)
      self.assertEqual(len(sparse_float_indices), 1)
      self.assertEqual(len(sparse_float_values), 1)
      self.assertEqual(len(sparse_float_shapes), 1)
      self.assertEqual(len(sparse_int_indices), 1)
      self.assertEqual(len(sparse_int_values), 1)
      self.assertEqual(len(sparse_int_shapes), 1)
      self.assertAllEqual(dense_floats[0].eval(),
                          features["dense_float"].eval())
      self.assertAllEqual(sparse_float_indices[0].eval(),
                          features["sparse_float"].indices.eval())
      self.assertAllEqual(sparse_float_values[0].eval(),
                          features["sparse_float"].values.eval())
      self.assertAllEqual(sparse_float_shapes[0].eval(),
                          features["sparse_float"].dense_shape.eval())
      self.assertAllEqual(sparse_int_indices[0].eval(),
                          features["sparse_categorical"].indices.eval())
      self.assertAllEqual(sparse_int_values[0].eval(), [397263, 397263])
      self.assertAllEqual(sparse_int_shapes[0].eval(),
                          features["sparse_categorical"].dense_shape.eval())

  def testExtractFeaturesFromCoreFeatureColumns(self):
    """Tests feature extraction when using core columns."""
    with self.test_session():
      features = {}
      # Sparse float column does not exist in core, so only dense numeric and
      # categorical.
      features["dense_float"] = array_ops.zeros([2, 1], dtypes.float32)
      features["sparse_categorical"] = sparse_tensor.SparseTensor(
          array_ops.zeros([2, 2], dtypes.int64),
          array_ops.zeros([2], dtypes.string), array_ops.zeros([2],
                                                               dtypes.int64))

      feature_columns = set()
      feature_columns.add(core_feature_column.numeric_column("dense_float"))
      feature_columns.add(
          core_feature_column.categorical_column_with_hash_bucket(
              "sparse_categorical", hash_bucket_size=1000000))
      (fc_names, dense_floats, _, _, _, sparse_int_indices, sparse_int_values,
       sparse_int_shapes) = (
           gbdt_batch.extract_features(
               features, feature_columns, use_core_columns=True))
      self.assertEqual(len(fc_names), 2)
      self.assertAllEqual(fc_names, ["dense_float", "sparse_categorical"])
      self.assertEqual(len(dense_floats), 1)
      self.assertEqual(len(sparse_int_indices), 1)
      self.assertEqual(len(sparse_int_values), 1)
      self.assertEqual(len(sparse_int_shapes), 1)
      self.assertAllEqual(dense_floats[0].eval(),
                          features["dense_float"].eval())
      self.assertAllEqual(sparse_int_indices[0].eval(),
                          features["sparse_categorical"].indices.eval())
      self.assertAllEqual(sparse_int_values[0].eval(), [397263, 397263])
      self.assertAllEqual(sparse_int_shapes[0].eval(),
                          features["sparse_categorical"].dense_shape.eval())

  def testTrainFnChiefNoBiasCentering(self):
    """Tests the train function running on chief without bias centering."""
    with self.test_session() as sess:
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="tree_ensemble")
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 0.1
      learner_config.num_classes = 2
      learner_config.regularization.l1 = 0
      learner_config.regularization.l2 = 0
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.min_node_weight = 0
      features = {}
      features["dense_float"] = array_ops.ones([4, 1], dtypes.float32)

      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=True,
          num_ps_replicas=0,
          center_bias=False,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=1, features=features)

      predictions = array_ops.constant(
          [[0.0], [1.0], [0.0], [2.0]], dtype=dtypes.float32)
      partition_ids = array_ops.zeros([4], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions": predictions,
          "predictions_no_dropout": predictions,
          "partition_ids": partition_ids,
          "ensemble_stamp": ensemble_stamp,
          "num_trees": 12,
      }

      labels = array_ops.ones([4, 1], dtypes.float32)
      weights = array_ops.ones([4, 1], dtypes.float32)
      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              _squared_loss(labels, weights, predictions)),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # On first run, expect no splits to be chosen because the quantile
      # buckets will not be ready.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 0)
      self.assertEquals(len(output.tree_weights), 0)
      self.assertEquals(stamp_token.eval(), 1)

      # Update the stamp to be able to run a second time.
      sess.run([ensemble_stamp.assign_add(1)])

      # On second run, expect a trivial split to be chosen to basically
      # predict the average.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 1)
      self.assertAllClose(output.tree_weights, [0.1])
      self.assertEquals(stamp_token.eval(), 2)
      expected_tree = """
          nodes {
            dense_float_binary_split {
              threshold: 1.0
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 0
            }
          }
          nodes {
            leaf {
              vector {
                value: 0.25
              }
            }
          }
          nodes {
            leaf {
              vector {
                value: 0.0
              }
            }
          }"""
      self.assertProtoEquals(expected_tree, output.trees[0])

  def testTrainFnChiefScalingNumberOfExamples(self):
    """Tests the train function running on chief without bias centering."""
    with self.test_session() as sess:
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="tree_ensemble")
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 0.1
      learner_config.num_classes = 2
      learner_config.regularization.l1 = 0
      learner_config.regularization.l2 = 0
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.min_node_weight = 0
      num_examples_fn = (
          lambda layer: math_ops.pow(math_ops.cast(2, dtypes.int64), layer) * 1)
      features = {}
      features["dense_float"] = array_ops.ones([4, 1], dtypes.float32)
      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=True,
          num_ps_replicas=0,
          center_bias=False,
          ensemble_handle=ensemble_handle,
          examples_per_layer=num_examples_fn,
          learner_config=learner_config,
          logits_dimension=1, features=features)

      predictions = array_ops.constant(
          [[0.0], [1.0], [0.0], [2.0]], dtype=dtypes.float32)
      partition_ids = array_ops.zeros([4], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions": predictions,
          "predictions_no_dropout": predictions,
          "partition_ids": partition_ids,
          "ensemble_stamp": ensemble_stamp,
          "num_trees": 12,
      }

      labels = array_ops.ones([4, 1], dtypes.float32)
      weights = array_ops.ones([4, 1], dtypes.float32)
      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              _squared_loss(labels, weights, predictions)),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # On first run, expect no splits to be chosen because the quantile
      # buckets will not be ready.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 0)
      self.assertEquals(len(output.tree_weights), 0)
      self.assertEquals(stamp_token.eval(), 1)

      # Update the stamp to be able to run a second time.
      sess.run([ensemble_stamp.assign_add(1)])

      # On second run, expect a trivial split to be chosen to basically
      # predict the average.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 1)
      self.assertAllClose(output.tree_weights, [0.1])
      self.assertEquals(stamp_token.eval(), 2)
      expected_tree = """
          nodes {
            dense_float_binary_split {
              threshold: 1.0
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 0
            }
          }
          nodes {
            leaf {
              vector {
                value: 0.25
              }
            }
          }
          nodes {
            leaf {
              vector {
                value: 0.0
              }
            }
          }"""
      self.assertProtoEquals(expected_tree, output.trees[0])

  def testTrainFnChiefWithBiasCentering(self):
    """Tests the train function running on chief with bias centering."""
    with self.test_session():
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="tree_ensemble")
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 0.1
      learner_config.num_classes = 2
      learner_config.regularization.l1 = 0
      learner_config.regularization.l2 = 0
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.min_node_weight = 0
      features = {}
      features["dense_float"] = array_ops.ones([4, 1], dtypes.float32)

      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=True,
          num_ps_replicas=0,
          center_bias=True,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=1, features=features)

      predictions = array_ops.constant(
          [[0.0], [1.0], [0.0], [2.0]], dtype=dtypes.float32)
      partition_ids = array_ops.zeros([4], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions": predictions,
          "predictions_no_dropout": predictions,
          "partition_ids": partition_ids,
          "ensemble_stamp": ensemble_stamp,
          "num_trees": 12,
      }

      labels = array_ops.ones([4, 1], dtypes.float32)
      weights = array_ops.ones([4, 1], dtypes.float32)
      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              _squared_loss(labels, weights, predictions)),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # On first run, expect bias to be centered.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      expected_tree = """
          nodes {
            leaf {
              vector {
                value: 0.25
              }
            }
          }"""
      self.assertEquals(len(output.trees), 1)
      self.assertAllEqual(output.tree_weights, [1.0])
      self.assertProtoEquals(expected_tree, output.trees[0])
      self.assertEquals(stamp_token.eval(), 1)

  def testTrainFnNonChiefNoBiasCentering(self):
    """Tests the train function running on worker without bias centering."""
    with self.test_session():
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="tree_ensemble")
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 0.1
      learner_config.num_classes = 2
      learner_config.regularization.l1 = 0
      learner_config.regularization.l2 = 0
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.min_node_weight = 0
      features = {}
      features["dense_float"] = array_ops.ones([4, 1], dtypes.float32)

      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=False,
          num_ps_replicas=0,
          center_bias=False,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=1, features=features)

      predictions = array_ops.constant(
          [[0.0], [1.0], [0.0], [2.0]], dtype=dtypes.float32)
      partition_ids = array_ops.zeros([4], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions": predictions,
          "predictions_no_dropout": predictions,
          "partition_ids": partition_ids,
          "ensemble_stamp": ensemble_stamp
      }

      labels = array_ops.ones([4, 1], dtypes.float32)
      weights = array_ops.ones([4, 1], dtypes.float32)
      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              _squared_loss(labels, weights, predictions)),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # Regardless of how many times the train op is run, a non-chief worker
      # can only accumulate stats so the tree ensemble never changes.
      for _ in range(5):
        train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 0)
      self.assertEquals(len(output.tree_weights), 0)
      self.assertEquals(stamp_token.eval(), 0)

  def testTrainFnNonChiefWithCentering(self):
    """Tests the train function running on worker with bias centering."""
    with self.test_session():
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="tree_ensemble")
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 0.1
      learner_config.num_classes = 2
      learner_config.regularization.l1 = 0
      learner_config.regularization.l2 = 0
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.min_node_weight = 0
      features = {}
      features["dense_float"] = array_ops.ones([4, 1], dtypes.float32)

      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=False,
          num_ps_replicas=0,
          center_bias=True,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=1, features=features)

      predictions = array_ops.constant(
          [[0.0], [1.0], [0.0], [2.0]], dtype=dtypes.float32)
      partition_ids = array_ops.zeros([4], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions": predictions,
          "predictions_no_dropout": predictions,
          "partition_ids": partition_ids,
          "ensemble_stamp": ensemble_stamp
      }

      labels = array_ops.ones([4, 1], dtypes.float32)
      weights = array_ops.ones([4, 1], dtypes.float32)
      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              _squared_loss(labels, weights, predictions)),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # Regardless of how many times the train op is run, a non-chief worker
      # can only accumulate stats so the tree ensemble never changes.
      for _ in range(5):
        train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 0)
      self.assertEquals(len(output.tree_weights), 0)
      self.assertEquals(stamp_token.eval(), 0)

  def testPredictFn(self):
    """Tests the predict function."""
    with self.test_session() as sess:
      # Create ensemble with one bias node.
      ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      text_format.Merge("""
          trees {
            nodes {
              leaf {
                vector {
                  value: 0.25
                }
              }
            }
          }
          tree_weights: 1.0
          tree_metadata {
            num_tree_weight_updates: 1
            num_layers_grown: 1
            is_finalized: true
          }""", ensemble_config)
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=3,
          tree_ensemble_config=ensemble_config.SerializeToString(),
          name="tree_ensemble")
      resources.initialize_resources(resources.shared_resources()).run()
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 0.1
      learner_config.num_classes = 2
      learner_config.regularization.l1 = 0
      learner_config.regularization.l2 = 0
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.min_node_weight = 0
      features = {}
      features["dense_float"] = array_ops.ones([4, 1], dtypes.float32)
      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=False,
          num_ps_replicas=0,
          center_bias=True,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=1, features=features)

      # Create predict op.
      mode = model_fn.ModeKeys.EVAL
      predictions_dict = sess.run(gbdt_model.predict(mode))
      self.assertEquals(predictions_dict["ensemble_stamp"], 3)
      self.assertAllClose(predictions_dict["predictions"], [[0.25], [0.25],
                                                            [0.25], [0.25]])
      self.assertAllClose(predictions_dict["partition_ids"], [0, 0, 0, 0])

  def testTrainFnMulticlassFullHessian(self):
    """Tests the GBDT train for multiclass full hessian."""
    with self.test_session() as sess:
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="tree_ensemble")

      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 1
      # Use full hessian multiclass strategy.
      learner_config.multi_class_strategy = (
          learner_pb2.LearnerConfig.FULL_HESSIAN)
      learner_config.num_classes = 5
      learner_config.regularization.l1 = 0
      # To make matrix inversible.
      learner_config.regularization.l2 = 1e-5
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.min_node_weight = 0
      features = {}
      batch_size = 3
      features["dense_float"] = array_ops.constant(
          [0.3, 1.5, 1.1], dtype=dtypes.float32)

      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=True,
          num_ps_replicas=0,
          center_bias=False,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=5, features=features)

      predictions = array_ops.constant(
          [[0.0, -1.0, 0.5, 1.2, 3.1], [1.0, 0.0, 0.8, 0.3, 1.0],
           [0.0, 0.0, 0.0, 0.0, 1.2]],
          dtype=dtypes.float32)

      labels = array_ops.constant([[2], [2], [3]], dtype=dtypes.float32)
      weights = array_ops.ones([batch_size, 1], dtypes.float32)

      partition_ids = array_ops.zeros([batch_size], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions": predictions,
          "predictions_no_dropout": predictions,
          "partition_ids": partition_ids,
          "ensemble_stamp": ensemble_stamp,
          "num_trees": 0,
      }

      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              losses.per_example_maxent_loss(
                  labels,
                  weights,
                  predictions,
                  num_classes=learner_config.num_classes)[0]),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # On first run, expect no splits to be chosen because the quantile
      # buckets will not be ready.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 0)
      self.assertEquals(len(output.tree_weights), 0)
      self.assertEquals(stamp_token.eval(), 1)

      # Update the stamp to be able to run a second time.
      sess.run([ensemble_stamp.assign_add(1)])
      # On second run, expect a trivial split to be chosen to basically
      # predict the average.
      train_op.run()
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())

      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output.ParseFromString(serialized.eval())
      self.assertEqual(len(output.trees), 1)
      # We got 3 nodes: one parent and 2 leafs.
      self.assertEqual(len(output.trees[0].nodes), 3)
      self.assertAllClose(output.tree_weights, [1])
      self.assertEquals(stamp_token.eval(), 2)

      # Leafs should have a dense vector of size 5.
      expected_leaf_1 = [-3.4480, -3.4429, 13.8490, -3.45, -3.4508]
      expected_leaf_2 = [-1.2547, -1.3145, 1.52, 2.3875, -1.3264]
      self.assertArrayNear(expected_leaf_1,
                           output.trees[0].nodes[1].leaf.vector.value, 1e-3)
      self.assertArrayNear(expected_leaf_2,
                           output.trees[0].nodes[2].leaf.vector.value, 1e-3)

  def testTrainFnMulticlassDiagonalHessian(self):
    """Tests the GBDT train for multiclass diagonal hessian."""
    with self.test_session() as sess:
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="tree_ensemble")

      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 1
      # Use full hessian multiclass strategy.
      learner_config.multi_class_strategy = (
          learner_pb2.LearnerConfig.DIAGONAL_HESSIAN)
      learner_config.num_classes = 5
      learner_config.regularization.l1 = 0
      # To make matrix inversible.
      learner_config.regularization.l2 = 1e-5
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.min_node_weight = 0
      batch_size = 3
      features = {}
      features["dense_float"] = array_ops.constant(
          [0.3, 1.5, 1.1], dtype=dtypes.float32)

      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=True,
          num_ps_replicas=0,
          center_bias=False,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=5, features=features)

      predictions = array_ops.constant(
          [[0.0, -1.0, 0.5, 1.2, 3.1], [1.0, 0.0, 0.8, 0.3, 1.0],
           [0.0, 0.0, 0.0, 0.0, 1.2]],
          dtype=dtypes.float32)

      labels = array_ops.constant([[2], [2], [3]], dtype=dtypes.float32)
      weights = array_ops.ones([batch_size, 1], dtypes.float32)

      partition_ids = array_ops.zeros([batch_size], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions": predictions,
          "predictions_no_dropout": predictions,
          "partition_ids": partition_ids,
          "ensemble_stamp": ensemble_stamp,
          "num_trees": 0,
      }

      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              losses.per_example_maxent_loss(
                  labels,
                  weights,
                  predictions,
                  num_classes=learner_config.num_classes)[0]),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # On first run, expect no splits to be chosen because the quantile
      # buckets will not be ready.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEqual(len(output.trees), 0)
      self.assertEqual(len(output.tree_weights), 0)
      self.assertEqual(stamp_token.eval(), 1)

      # Update the stamp to be able to run a second time.
      sess.run([ensemble_stamp.assign_add(1)])
      # On second run, expect a trivial split to be chosen to basically
      # predict the average.
      train_op.run()
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())

      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output.ParseFromString(serialized.eval())
      self.assertEqual(len(output.trees), 1)
      # We got 3 nodes: one parent and 2 leafs.
      self.assertEqual(len(output.trees[0].nodes), 3)
      self.assertAllClose(output.tree_weights, [1])
      self.assertEqual(stamp_token.eval(), 2)

      # Leafs should have a dense vector of size 5.
      expected_leaf_1 = [-1.0354, -1.0107, 17.2976, -1.1313, -4.5023]
      expected_leaf_2 = [-1.2924, -1.1376, 2.2042, 3.1052, -1.6269]
      self.assertArrayNear(expected_leaf_1,
                           output.trees[0].nodes[1].leaf.vector.value, 1e-3)
      self.assertArrayNear(expected_leaf_2,
                           output.trees[0].nodes[2].leaf.vector.value, 1e-3)

  def testTrainFnMulticlassTreePerClass(self):
    """Tests the GBDT train for multiclass tree per class strategy."""
    with self.test_session() as sess:
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="tree_ensemble")

      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 1
      # Use full hessian multiclass strategy.
      learner_config.multi_class_strategy = (
          learner_pb2.LearnerConfig.TREE_PER_CLASS)
      learner_config.num_classes = 5
      learner_config.regularization.l1 = 0
      # To make matrix inversible.
      learner_config.regularization.l2 = 1e-5
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.min_node_weight = 0
      features = {
          "dense_float": array_ops.constant(
              [[1.0], [1.5], [2.0]], dtypes.float32),
      }

      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=True,
          num_ps_replicas=0,
          center_bias=False,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=5, features=features)

      batch_size = 3
      predictions = array_ops.constant(
          [[0.0, -1.0, 0.5, 1.2, 3.1], [1.0, 0.0, 0.8, 0.3, 1.0],
           [0.0, 0.0, 0.0, 2.0, 1.2]],
          dtype=dtypes.float32)

      labels = array_ops.constant([[2], [2], [3]], dtype=dtypes.float32)
      weights = array_ops.ones([batch_size, 1], dtypes.float32)

      partition_ids = array_ops.zeros([batch_size], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions": predictions,
          "predictions_no_dropout": predictions,
          "partition_ids": partition_ids,
          "ensemble_stamp": ensemble_stamp,
          # This should result in a tree built for a class 2.
          "num_trees": 13,
      }

      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              losses.per_example_maxent_loss(
                  labels,
                  weights,
                  predictions,
                  num_classes=learner_config.num_classes)[0]),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # On first run, expect no splits to be chosen because the quantile
      # buckets will not be ready.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEqual(len(output.trees), 0)
      self.assertEqual(len(output.tree_weights), 0)
      self.assertEqual(stamp_token.eval(), 1)

      # Update the stamp to be able to run a second time.
      sess.run([ensemble_stamp.assign_add(1)])
      # On second run, expect a trivial split to be chosen to basically
      # predict the average.
      train_op.run()
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())

      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output.ParseFromString(serialized.eval())
      self.assertEqual(len(output.trees), 1)
      self.assertAllClose(output.tree_weights, [1])
      self.assertEqual(stamp_token.eval(), 2)

      # One node for a split, two children nodes.
      self.assertEqual(3, len(output.trees[0].nodes))

      # Leafs will have a sparse vector for class 3.
      self.assertEqual(1,
                       len(output.trees[0].nodes[1].leaf.sparse_vector.index))
      self.assertEqual(3, output.trees[0].nodes[1].leaf.sparse_vector.index[0])
      self.assertAlmostEqual(
          -1.13134455681, output.trees[0].nodes[1].leaf.sparse_vector.value[0])

      self.assertEqual(1,
                       len(output.trees[0].nodes[2].leaf.sparse_vector.index))
      self.assertEqual(3, output.trees[0].nodes[2].leaf.sparse_vector.index[0])
      self.assertAllClose(
          0.893284678459,
          output.trees[0].nodes[2].leaf.sparse_vector.value[0],
          atol=1e-4, rtol=1e-4)

  def testTrainFnChiefFeatureSelectionReachedLimitNoGoodSplit(self):
    """Tests the train function running on chief with feature selection."""
    with self.test_session() as sess:
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="tree_ensemble")
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 0.1
      learner_config.num_classes = 2
      learner_config.regularization.l1 = 0
      learner_config.regularization.l2 = 0
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.max_number_of_unique_feature_columns = 1
      learner_config.constraints.min_node_weight = 0
      features = {}
      features["dense_float_0"] = array_ops.ones([4, 1], dtypes.float32)
      # Feature 1 is predictive but it won't be used because we have reached the
      # limit of num_used_handlers >= max_number_of_unique_feature_columns
      features["dense_float_1"] = array_ops.constant([0, 0, 1, 1],
                                                     dtypes.float32)

      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=True,
          num_ps_replicas=0,
          center_bias=False,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=1,
          features=features)

      predictions = array_ops.constant(
          [[0.0], [1.0], [0.0], [2.0]], dtype=dtypes.float32)
      partition_ids = array_ops.zeros([4], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions":
              predictions,
          "predictions_no_dropout":
              predictions,
          "partition_ids":
              partition_ids,
          "ensemble_stamp":
              ensemble_stamp,
          "num_trees":
              12,
          "num_used_handlers":
              array_ops.constant(1, dtype=dtypes.int64),
          "used_handlers_mask":
              array_ops.constant([True, False], dtype=dtypes.bool),
      }

      labels = array_ops.constant([0, 0, 1, 1], dtypes.float32)
      weights = array_ops.ones([4, 1], dtypes.float32)
      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              _squared_loss(labels, weights, predictions)),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # On first run, expect no splits to be chosen because the quantile
      # buckets will not be ready.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 0)
      self.assertEquals(len(output.tree_weights), 0)
      self.assertEquals(stamp_token.eval(), 1)

      # Update the stamp to be able to run a second time.
      sess.run([ensemble_stamp.assign_add(1)])

      # On second run, expect a trivial split to be chosen to basically
      # predict the average.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 1)
      self.assertAllClose(output.tree_weights, [0.1])
      self.assertEquals(stamp_token.eval(), 2)
      expected_tree = """
          nodes {
            dense_float_binary_split {
              feature_column: 0
              threshold: 1.0
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 0
            }
          }
          nodes {
            leaf {
              vector {
                value: -0.25
              }
            }
          }
          nodes {
            leaf {
              vector {
                value: 0.0
              }
            }
          }"""
      self.assertProtoEquals(expected_tree, output.trees[0])

  def testTrainFnChiefFeatureSelectionWithGoodSplits(self):
    """Tests the train function running on chief with feature selection."""
    with self.test_session() as sess:
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config="", name="tree_ensemble")
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 0.1
      learner_config.num_classes = 2
      learner_config.regularization.l1 = 0
      learner_config.regularization.l2 = 0
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.max_number_of_unique_feature_columns = 1
      learner_config.constraints.min_node_weight = 0
      features = {}
      features["dense_float_0"] = array_ops.ones([4, 1], dtypes.float32)
      # Feature 1 is predictive and is in our selected features so it will be
      # used even when we're at the limit.
      features["dense_float_1"] = array_ops.constant([0, 0, 1, 1],
                                                     dtypes.float32)

      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=True,
          num_ps_replicas=0,
          center_bias=False,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=1,
          features=features)

      predictions = array_ops.constant(
          [[0.0], [1.0], [0.0], [2.0]], dtype=dtypes.float32)
      partition_ids = array_ops.zeros([4], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions":
              predictions,
          "predictions_no_dropout":
              predictions,
          "partition_ids":
              partition_ids,
          "ensemble_stamp":
              ensemble_stamp,
          "num_trees":
              12,
          "num_used_handlers":
              array_ops.constant(1, dtype=dtypes.int64),
          "used_handlers_mask":
              array_ops.constant([False, True], dtype=dtypes.bool),
      }

      labels = array_ops.constant([0, 0, 1, 1], dtypes.float32)
      weights = array_ops.ones([4, 1], dtypes.float32)
      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              _squared_loss(labels, weights, predictions)),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # On first run, expect no splits to be chosen because the quantile
      # buckets will not be ready.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 0)
      self.assertEquals(len(output.tree_weights), 0)
      self.assertEquals(stamp_token.eval(), 1)

      # Update the stamp to be able to run a second time.
      sess.run([ensemble_stamp.assign_add(1)])

      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())

      self.assertEquals(len(output.trees), 1)
      self.assertAllClose(output.tree_weights, [0.1])
      self.assertEquals(stamp_token.eval(), 2)
      expected_tree = """
          nodes {
            dense_float_binary_split {
              feature_column: 1
              left_id: 1
              right_id: 2
            }
            node_metadata {
              gain: 0.5
            }
          }
          nodes {
            leaf {
              vector {
                value: 0.0
              }
            }
          }
          nodes {
            leaf {
              vector {
                value: -0.5
              }
            }
          }"""
      self.assertProtoEquals(expected_tree, output.trees[0])

  def testTrainFnChiefFeatureSelectionReachedLimitIncrementAttemptedLayer(self):
    """Tests the train function running on chief with feature selection."""
    with self.test_session() as sess:
      tree_ensemble_config = tree_config_pb2.DecisionTreeEnsembleConfig()
      tree = tree_ensemble_config.trees.add()

      _set_float_split(tree.nodes.add()
                       .sparse_float_binary_split_default_right.split, 2, 4.0,
                       1, 2)
      _append_to_leaf(tree.nodes.add().leaf, 0, 0.5)
      _append_to_leaf(tree.nodes.add().leaf, 1, 1.2)
      tree_ensemble_config.tree_weights.append(1.0)
      metadata = tree_ensemble_config.tree_metadata.add()
      metadata.is_finalized = False
      metadata.num_layers_grown = 1
      tree_ensemble_config = tree_ensemble_config.SerializeToString()
      ensemble_handle = model_ops.tree_ensemble_variable(
          stamp_token=0, tree_ensemble_config=tree_ensemble_config,
          name="tree_ensemble")
      learner_config = learner_pb2.LearnerConfig()
      learner_config.learning_rate_tuner.fixed.learning_rate = 0.1
      learner_config.num_classes = 2
      learner_config.regularization.l1 = 0
      learner_config.regularization.l2 = 0
      learner_config.constraints.max_tree_depth = 1
      learner_config.constraints.max_number_of_unique_feature_columns = 1
      learner_config.constraints.min_node_weight = 0
      features = {}
      # Both features will be disabled since the feature selection limit is
      # already reached.
      features["dense_float_0"] = array_ops.ones([4, 1], dtypes.float32)
      features["dense_float_1"] = array_ops.constant([0, 0, 1, 1],
                                                     dtypes.float32)

      gbdt_model = gbdt_batch.GradientBoostedDecisionTreeModel(
          is_chief=True,
          num_ps_replicas=0,
          center_bias=False,
          ensemble_handle=ensemble_handle,
          examples_per_layer=1,
          learner_config=learner_config,
          logits_dimension=1,
          features=features)

      predictions = array_ops.constant(
          [[0.0], [1.0], [0.0], [2.0]], dtype=dtypes.float32)
      partition_ids = array_ops.zeros([4], dtypes.int32)
      ensemble_stamp = variables.Variable(
          initial_value=0,
          name="ensemble_stamp",
          trainable=False,
          dtype=dtypes.int64)

      predictions_dict = {
          "predictions":
              predictions,
          "predictions_no_dropout":
              predictions,
          "partition_ids":
              partition_ids,
          "ensemble_stamp":
              ensemble_stamp,
          "num_trees":
              12,
          # We have somehow reached our limit 1. Both of the handlers will be
          # disabled.
          "num_used_handlers":
              array_ops.constant(1, dtype=dtypes.int64),
          "used_handlers_mask":
              array_ops.constant([False, False], dtype=dtypes.bool),
      }

      labels = array_ops.constant([0, 0, 1, 1], dtypes.float32)
      weights = array_ops.ones([4, 1], dtypes.float32)
      # Create train op.
      train_op = gbdt_model.train(
          loss=math_ops.reduce_mean(
              _squared_loss(labels, weights, predictions)),
          predictions_dict=predictions_dict,
          labels=labels)
      variables.global_variables_initializer().run()
      resources.initialize_resources(resources.shared_resources()).run()

      # On first run, expect no splits to be chosen because the quantile
      # buckets will not be ready.
      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      self.assertEquals(len(output.trees), 1)
      self.assertEquals(output.growing_metadata.num_layers_attempted, 1)
      self.assertEquals(stamp_token.eval(), 1)

      # Update the stamp to be able to run a second time.
      sess.run([ensemble_stamp.assign_add(1)])

      train_op.run()
      stamp_token, serialized = model_ops.tree_ensemble_serialize(
          ensemble_handle)
      output = tree_config_pb2.DecisionTreeEnsembleConfig()
      output.ParseFromString(serialized.eval())
      # Make sure the trees are not modified, but the num_layers_attempted is
      # incremented so that eventually the training stops.
      self.assertEquals(len(output.trees), 1)
      self.assertEquals(len(output.trees[0].nodes), 3)

      self.assertEquals(output.growing_metadata.num_layers_attempted, 2)

if __name__ == "__main__":
  googletest.main()
