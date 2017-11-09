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
"""Strategy to export custom proto formats."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from tensorflow.contrib.boosted_trees.proto import tree_config_pb2
from tensorflow.contrib.boosted_trees.python.training.functions import gbdt_batch
from tensorflow.contrib.decision_trees.proto import generic_tree_model_extensions_pb2
from tensorflow.contrib.decision_trees.proto import generic_tree_model_pb2
from tensorflow.contrib.learn.python.learn import export_strategy
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader as saved_model_loader
from tensorflow.python.saved_model import tag_constants


def make_custom_export_strategy(name,
                                convert_fn,
                                feature_columns,
                                export_input_fn):
  """Makes custom exporter of GTFlow tree format.

  Args:
    name: A string, for the name of the export strategy.
    convert_fn: A function that converts the tree proto to desired format and
      saves it to the desired location. Can be None to skip conversion.
    feature_columns: A list of feature columns.
    export_input_fn: A function that takes no arguments and returns an
      `InputFnOps`.

  Returns:
    An `ExportStrategy`.
  """
  base_strategy = saved_model_export_utils.make_export_strategy(
      serving_input_fn=export_input_fn)
  input_fn = export_input_fn()
  (sorted_feature_names, dense_floats, sparse_float_indices, _, _,
   sparse_int_indices, _, _) = gbdt_batch.extract_features(
       input_fn.features, feature_columns)

  def export_fn(estimator, export_dir, checkpoint_path=None, eval_result=None):
    """A wrapper to export to SavedModel, and convert it to other formats."""
    result_dir = base_strategy.export(estimator, export_dir,
                                      checkpoint_path,
                                      eval_result)
    with ops.Graph().as_default() as graph:
      with tf_session.Session(graph=graph) as sess:
        saved_model_loader.load(
            sess, [tag_constants.SERVING], result_dir)
        # Note: This is GTFlow internal API and might change.
        ensemble_model = graph.get_operation_by_name(
            "ensemble_model/TreeEnsembleSerialize")
        _, dfec_str = sess.run(ensemble_model.outputs)
        dtec = tree_config_pb2.DecisionTreeEnsembleConfig()
        dtec.ParseFromString(dfec_str)
        # Export the result in the same folder as the saved model.
        if convert_fn:
          convert_fn(dtec, sorted_feature_names,
                     len(dense_floats),
                     len(sparse_float_indices),
                     len(sparse_int_indices), result_dir, eval_result)
        feature_importances = _get_feature_importances(
            dtec, sorted_feature_names,
            len(dense_floats),
            len(sparse_float_indices), len(sparse_int_indices))
        sorted_by_importance = sorted(
            feature_importances.items(), key=lambda x: -x[1])
        assets_dir = os.path.join(result_dir, "assets.extra")
        gfile.MakeDirs(assets_dir)
        with gfile.GFile(os.path.join(assets_dir, "feature_importances"),
                         "w") as f:
          f.write("\n".join("%s, %f" % (k, v) for k, v in sorted_by_importance))
    return result_dir
  return export_strategy.ExportStrategy(name, export_fn)


def convert_to_universal_format(dtec, sorted_feature_names,
                                num_dense, num_sparse_float,
                                num_sparse_int,
                                feature_name_to_proto=None):
  """Convert GTFlow trees to universal format."""
  del num_sparse_int  # unused.
  model_and_features = generic_tree_model_pb2.ModelAndFeatures()
  # TODO(jonasz): Feature descriptions should contain information about how each
  # feature is processed before it's fed to the model (e.g. bucketing
  # information). As of now, this serves as a list of features the model uses.
  for feature_name in sorted_feature_names:
    if not feature_name_to_proto:
      model_and_features.features[feature_name].SetInParent()
    else:
      model_and_features.features[feature_name].CopyFrom(
          feature_name_to_proto[feature_name])
  model = model_and_features.model
  model.ensemble.summation_combination_technique.SetInParent()
  for tree_idx in range(len(dtec.trees)):
    gtflow_tree = dtec.trees[tree_idx]
    tree_weight = dtec.tree_weights[tree_idx]
    member = model.ensemble.members.add()
    member.submodel_id.value = tree_idx
    tree = member.submodel.decision_tree
    for node_idx in range(len(gtflow_tree.nodes)):
      gtflow_node = gtflow_tree.nodes[node_idx]
      node = tree.nodes.add()
      node_type = gtflow_node.WhichOneof("node")
      node.node_id.value = node_idx
      if node_type == "leaf":
        leaf = gtflow_node.leaf
        if leaf.HasField("vector"):
          for weight in leaf.vector.value:
            new_value = node.leaf.vector.value.add()
            new_value.float_value = weight * tree_weight
        else:
          for index, weight in zip(
              leaf.sparse_vector.index, leaf.sparse_vector.value):
            new_value = node.leaf.sparse_vector.sparse_value[index]
            new_value.float_value = weight * tree_weight
      else:
        node = node.binary_node
        # Binary nodes here.
        if node_type == "dense_float_binary_split":
          split = gtflow_node.dense_float_binary_split
          feature_id = split.feature_column
          inequality_test = node.inequality_left_child_test
          inequality_test.feature_id.id.value = sorted_feature_names[feature_id]
          inequality_test.type = (
              generic_tree_model_pb2.InequalityTest.LESS_OR_EQUAL)
          inequality_test.threshold.float_value = split.threshold
        elif node_type == "sparse_float_binary_split_default_left":
          split = gtflow_node.sparse_float_binary_split_default_left.split
          node.default_direction = (
              generic_tree_model_pb2.BinaryNode.LEFT)
          # TODO(nponomareva): adjust this id assignement when we allow multi-
          # column sparse tensors.
          feature_id = split.feature_column + num_dense
          inequality_test = node.inequality_left_child_test
          inequality_test.feature_id.id.value = sorted_feature_names[feature_id]
          inequality_test.type = (
              generic_tree_model_pb2.InequalityTest.LESS_OR_EQUAL)
          inequality_test.threshold.float_value = split.threshold
        elif node_type == "sparse_float_binary_split_default_right":
          split = gtflow_node.sparse_float_binary_split_default_right.split
          node.default_direction = (
              generic_tree_model_pb2.BinaryNode.RIGHT)
          # TODO(nponomareva): adjust this id assignement when we allow multi-
          # column sparse tensors.
          feature_id = split.feature_column + num_dense
          inequality_test = node.inequality_left_child_test
          inequality_test.feature_id.id.value = sorted_feature_names[feature_id]
          inequality_test.type = (
              generic_tree_model_pb2.InequalityTest.LESS_OR_EQUAL)
          inequality_test.threshold.float_value = split.threshold
        elif node_type == "categorical_id_binary_split":
          split = gtflow_node.categorical_id_binary_split
          node.default_direction = generic_tree_model_pb2.BinaryNode.RIGHT
          feature_id = split.feature_column + num_dense + num_sparse_float
          categorical_test = (
              generic_tree_model_extensions_pb2.MatchingValuesTest())
          categorical_test.feature_id.id.value = sorted_feature_names[
              feature_id]
          matching_id = categorical_test.value.add()
          matching_id.int64_value = split.feature_id
          node.custom_left_child_test.Pack(categorical_test)
        else:
          raise ValueError("Unexpected node type %s", node_type)
        node.left_child_id.value = split.left_id
        node.right_child_id.value = split.right_id
  return model_and_features


def _get_feature_importances(dtec, feature_names, num_dense_floats,
                             num_sparse_float, num_sparse_int):
  """Export the feature importance per feature column."""
  del num_sparse_int    # Unused.
  sums = collections.defaultdict(lambda: 0)
  for tree_idx in range(len(dtec.trees)):
    tree = dtec.trees[tree_idx]
    for tree_node in tree.nodes:
      node_type = tree_node.WhichOneof("node")
      if node_type == "dense_float_binary_split":
        split = tree_node.dense_float_binary_split
        split_column = feature_names[split.feature_column]
      elif node_type == "sparse_float_binary_split_default_left":
        split = tree_node.sparse_float_binary_split_default_left.split
        split_column = feature_names[split.feature_column + num_dense_floats]
      elif node_type == "sparse_float_binary_split_default_right":
        split = tree_node.sparse_float_binary_split_default_right.split
        split_column = feature_names[split.feature_column + num_dense_floats]
      elif node_type == "categorical_id_binary_split":
        split = tree_node.categorical_id_binary_split
        split_column = feature_names[split.feature_column + num_dense_floats +
                                     num_sparse_float]
      elif node_type == "categorical_id_set_membership_binary_split":
        split = tree_node.categorical_id_set_membership_binary_split
        split_column = feature_names[split.feature_column + num_dense_floats +
                                     num_sparse_float]
      elif node_type == "leaf":
        assert tree_node.node_metadata.gain == 0
        continue
      else:
        raise ValueError("Unexpected split type %s", node_type)
      # Apply shrinkage factor. It is important since it is not always uniform
      # across different trees.
      sums[split_column] += (
          tree_node.node_metadata.gain * dtec.tree_weights[tree_idx])
  return dict(sums)
