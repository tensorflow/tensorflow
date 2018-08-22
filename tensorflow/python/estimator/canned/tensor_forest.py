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
"""Estimator classes for TensorForest."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.ops import tensor_forest_ops, math_ops, array_ops

from tensorflow.python.util.tf_export import estimator_export

_ForestHParams = collections.namedtuple('TreeHParams', [
    'num_output',
    'n_trees', 'max_nodes', 'num_splits_to_consider',
    'split_after_samples', 'is_regression',
])

TREE_PATHS_PREDICTION_KEY = 'tree_paths'
VARIANCE_PREDICTION_KEY = 'prediction_variance'
EPSILON = 0.000001


def _ensure_logits(head, logits):
  # For binary classification problems, convert probabilities to logits.
  # Includes hack to get around the fact that a probability might be 0 or
  # 1.
  if head.logits_dimension == 2:
    class_1_probs = array_ops.slice(logits, [0, 1], [-1, 1])
  return math_ops.log(
      math_ops.maximum(class_1_probs / math_ops.maximum(
          1.0 - class_1_probs, EPSILON), EPSILON))


def _bt_model_fn(features, labels, mode, head, sorted_feature_columns, forest_hparams, config, name='tensor_forest'):
  graph_builder = RandomForestGraphs(
      forest_hparams, config)

  transformed_features = feature_column_lib._transform_features(
      features, sorted_feature_columns)

  stacked_feature = array_ops.concat(transformed_features.values(), axis=1)

  logits, tree_paths, regression_variance = graph_builder.inference_graph(
      stacked_feature)

  logits = _ensure_logits(head, logits)

  summary.scalar('average_tree_size', graph_builder.average_size())

  training_graph = None
  # if labels is not None and mode == model_fn_lib.ModeKeys.TRAIN:
  #   with ops.control_dependencies([logits.op]):
  #     training_graph = control_flow_ops.group(
  #         graph_builder.training_graph(
  #             transformed_features, labels),
  #         state_ops.assign_add(training_util.get_global_step(), 1),
  #         name=name)

  def _train_op_fn(unused_loss):
    return training_graph

  estimator_spec = head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_train_op_fn,
      logits=logits)

  extra_predictions = {
      TREE_PATHS_PREDICTION_KEY: tree_paths
  }

  if not forest_hparams.is_classification:
      extra_predictions[VARIANCE_PREDICTION_KEY] = regression_variance,

  estimator_spec = estimator_spec._replace(
      predictions=estimator_spec.predictions.update(extra_predictions))

  return estimator_spec


class RandomForestGraphs(object):
  """Builds TF graphs for random forest training and inference."""

  def __init__(self,
               params,
               configs,
               tree_configs=None,
               tree_stats=None):
    self.params = params
    self.configs = configs
    self.variables = tensor_forest_ops.ForestVariables(
        self.params,
        tree_configs=tree_configs,
        tree_stats=tree_stats)
    self.decision_trees = [
        RandomDecisionTreeGraphs(self.variables[i], self.params, i)
        for i in range(self.params.n_trees)
    ]

  def inference_graph(self, input_data, **inference_args):
    probabilities = []
    paths = []
    for decision_tree in self.decision_trees:
      probs, path = decision_tree.inference_graph(
          input_data,
          **inference_args)
      probabilities.append(probs)
      paths.append(path)
    # shape of all_predict should be [batch_size, n_trees, num_outputs]
    all_predict = array_ops.stack(probabilities, axis=1)
    average_values = math_ops.div(
        math_ops.reduce_sum(all_predict, 1),
        self.params.n_trees,
        name='probabilities')
    tree_paths = array_ops.stack(paths, axis=1)

    expected_squares = math_ops.div(
        math_ops.reduce_sum(all_predict * all_predict, 1),
        self.params.n_trees)
    regression_variance = math_ops.maximum(
        0., expected_squares - average_values * average_values)

    return average_values, tree_paths, regression_variance

  def average_size(self):
    sizes = []
    for decision_tree in self.decision_trees:
      sizes.append(decision_tree.size())
    return math_ops.reduce_mean(math_ops.to_float(array_ops.stack(sizes)))


class RandomDecisionTreeGraphs(object):
  """Builds TF graphs for random tree training and inference."""

  def __init__(self, variables, params, tree_num):
    self.variables = variables
    self.params = params
    self.tree_num = tree_num

  def inference_graph(self, input_data):
    return tensor_forest_ops.predict(
        self.variables.tree,
        input_data,
        params=self.params)

  def size(self):
    return tensor_forest_ops.tree_size(self.variables.tree)


@estimator_export('estimator.TensorForestClassifier')
class TensorForestClassifier(estimator.Estimator):

  def __init__(self,
               feature_columns,
               model_dir=None,
               n_classes=2,
               label_vocabulary=None,
               head=None,
               n_trees=100,
               max_nodes=1000,
               num_splits_to_consider=10,
               split_after_samples=250,
               config=None):

    if head is None:
      head = head_lib._binary_logistic_or_multi_class_head(
          n_classes=n_classes,
          weight_column=None,
          label_vocabulary=label_vocabulary,
          loss_reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)

    forest_hparams = _ForestHParams(
        n_classes,
        n_trees,
        max_nodes,
        num_splits_to_consider,
        split_after_samples,
        is_regression=False)

    assert all(map(lambda fc: isinstance(fc, feature_column_lib.DenseColumn),
                   feature_columns)), 'Only Dense Column supported'
    sorted_feature_columns = sorted(feature_columns, key=lambda fc: fc.name)

    def _model_fn(features, labels, mode, config):
      return _bt_model_fn(
        features, labels, mode, head, sorted_feature_columns, forest_hparams, config)

    super(TensorForestClassifier, self).__init__(
      model_fn=_model_fn, model_dir=model_dir, config=config)
