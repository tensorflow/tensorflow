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

from tensorflow import logging

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
from tensorflow.python.training import training_util
from tensorflow.python.summary import summary
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.ops import tensor_forest_ops


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


def _bt_model_fn(features, labels, mode, head, feature_columns, forest_hparams, config, name='tensor_forest'):
  graph_builder = RandomForestGraphs(
      forest_hparams, config)

  transformed_features = feature_column_lib._transform_features(
      features, sorted_feature_columns)

  logits, tree_paths, regression_variance = graph_builder.inference_graph(
      transformed_features)

  logits = _ensure_logits(head, logits)

  summary.scalar('average_tree_size', graph_builder.average_size())

  training_graph = None
  if labels is not None and mode == model_fn_lib.ModeKeys.TRAIN:
    with ops.control_dependencies([logits.op]):
      training_graph = control_flow_ops.group(
          graph_builder.training_graph(
              transformed_features, labels),
          state_ops.assign_add(training_util.get_global_step(), 1),
          name=name)

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
        training=training,
        tree_configs=tree_configs,
        tree_stats=tree_stats)
    tree_params_proto =
    self.decision_trees = [
        RandomDecisionTreeGraphs(self.variables[i], self.params, i)
        for i in range(self.params.num_trees)
    ]

  def training_graph(self,
                     input_data,
                     input_labels,
                     **tree_kwargs):
    tree_graphs = []
    seed = self.configs.tf_random_seed

    for n, decision_tree in enumerate(self.decision_trees):
      if seed is not None:
        seed += n
      tree_graphs.append(decision_tree.training_graph(
          input_data,
          input_labels,
          seed,
          **tree_kwargs))

    return control_flow_ops.group(*tree_graphs, name='train')

  def inference_graph(self, input_data, **inference_args):
    probabilities = []
    paths = []
    for decision_tree in self.decision_trees:
      probs, path = decision_tree.inference_graph(
          input_data,
          **inference_args)
      probabilities.append(probs)
      paths.append(path)
    # shape of all_predict should be [batch_size, num_trees, num_outputs]
    all_predict = array_ops.stack(probabilities, axis=1)
    average_values = math_ops.div(
        math_ops.reduce_sum(all_predict, 1),
        self.params.num_trees,
        name='probabilities')
    tree_paths = array_ops.stack(paths, axis=1)

    expected_squares = math_ops.div(
        math_ops.reduce_sum(all_predict * all_predict, 1),
        self.params.num_trees)
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

  def __init__(self, variables, serialized_params_proto, tree_num):
    self.variables = variables
    self.serialized_params_proto = serialized_params_proto
    self.tree_num = tree_num

  def training_graph(self,
                     input_data,
                     input_labels,
                     random_seed,
                     ):
    leaf_ids = tensor_forest_ops.traverse_tree(
        self.variables.tree,
        input_data,
        params=self.serialized_params_proto)

    update_model = tensor_forest_ops.update_model(
        self.variables.tree,
        leaf_ids,
        input_labels,
        params=self.serialized_params_proto)

    finished_nodes = tensor_forest_ops.process_input(
        self.variables.tree,
        self.variables.stats,
        input_data,
        input_labels,
        leaf_ids,
        random_seed=random_seed,
        params=self.serialized_params_proto)

    with ops.control_dependencies([update_model]):
      return tensor_forest_ops.grow_tree(
          self.variables.tree,
          self.variables.stats,
          finished_nodes,
          params=self.serialized_params_proto)

  def inference_graph(self, input_data):
    return tensor_forest_ops.predict(
        self.variables.tree,
        input_data,
        params=self.param)

  def size(self):
    return tensor_forest_ops.tree_size(self.variables.tree)


@estimator_export('estimator.TensorForestClassifier')
class TensorForestClassifier(estimator.Estimator):


"""A Classifier for Tensorflow Tensor Forest models.

"""

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
  """Initializes a `TensorForestClassifier` instance.

Example:

```python
feature_1 = numeric_column('feature_1')
feature_2 = numeric_column('feature_2')

classifier = estimator.TensorForestClassifier(
    feature_columns=[feature_1, feature_2],
    n_trees=100,
    ... <some other params>
)

def input_fn_train():
  ...
  return dataset

classifier.train(input_fn=input_fn_train)

def input_fn_predict():
  ...
  return dataset

classifier.predict(input_fn=input_fn_predict)

def input_fn_eval():
  ...
  return dataset

metrics = classifier.evaluate(input_fn=input_fn_eval)
```

Args:
  feature_columns: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from FeatureColumn.
  n_classes: Defaults to 2. The number of classes in a classification problem.
  model_dir: Directory to save model parameters, graph and etc.
    This can also be used to load checkpoints from the directory
    into an estimator to continue training a previously saved model.
  label_vocabulary: A list of strings representing all possible label values.
    If provided, labels must be of string type and their values must be present
    in label_vocabulary list. If label_vocabulary is omitted, it is assumed that
    the labels are already encoded as integer values within {0, 1} for n_classes=2,
    or encoded as integer values in {0, 1,..., n_classes-1} for n_classes>2.
    If vocabulary is not provided and labels are of string,
    an error will be generated.
  head: A head_lib._Head instance, the loss would be calculated for metrics
    purpose and not being used for training. If not provided,
    one will be automatically created based on params
  n_trees: The number of trees to create. Defaults to 100.
    There usually isn't any accuracy gain from using higher
    values (assuming deep enough trees are built).
  max_nodes: Defaults to 10k. No tree is allowed to grow beyond max_nodes
    nodes, and training stops when all trees in the forest are this large.
  num_splits_to_consider: Defaults to sqrt(num_features). In the extremely
    randomized tree training algorithm, only this many potential splits
    are evaluated for each tree node.
  split_after_samples: Defaults to 250. In our online version of extremely
    randomized tree training, we pick a split for a node after it has
    accumulated this many training samples.
  config: RunConfig object to configure the runtime settings.

Raises:
  ValueError: when wrong arguments are given or unsupported functionalities
     are requested.
"""
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
        features, labels, mode, head, feature_columns, forest_hparams, config)

  super(TensorForestClassifier, self).__init__(
      model_fn=_model_fn, model_dir=model_dir, config=config)
