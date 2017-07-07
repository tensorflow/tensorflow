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
"""Extremely random forest graph builder using TF resources handles."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from google.protobuf import text_format

from tensorflow.contrib.decision_trees.proto import generic_tree_model_pb2 as _tree_proto
from tensorflow.contrib.tensor_forest.proto import tensor_forest_params_pb2 as _params_proto
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.tensor_forest.python.ops import model_ops
from tensorflow.contrib.tensor_forest.python.ops import stats_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging


# Stores tuples of (leaf model type, stats model type)
CLASSIFICATION_LEAF_MODEL_TYPES = {
    'all_dense': (_params_proto.MODEL_DENSE_CLASSIFICATION,
                  _params_proto.STATS_DENSE_GINI),
    'all_sparse': (_params_proto.MODEL_SPARSE_CLASSIFICATION,
                   _params_proto.STATS_SPARSE_GINI),
    'sparse_then_dense':
        (_params_proto.MODEL_SPARSE_OR_DENSE_CLASSIFICATION,
         _params_proto.STATS_SPARSE_THEN_DENSE_GINI),
}
REGRESSION_MODEL_TYPE = (
    _params_proto.MODEL_REGRESSION,
    _params_proto.STATS_LEAST_SQUARES_REGRESSION,
    _params_proto.COLLECTION_BASIC)

COLLECTION_TYPES = {
    'basic': _params_proto.COLLECTION_BASIC,
    'graph_runner': _params_proto.GRAPH_RUNNER_COLLECTION
}

FINISH_TYPES = {
    'basic': _params_proto.SPLIT_FINISH_BASIC,
    'hoeffding': _params_proto.SPLIT_FINISH_DOMINATE_HOEFFDING,
    'bootstrap': _params_proto.SPLIT_FINISH_DOMINATE_BOOTSTRAP
}
PRUNING_TYPES = {
    'none': _params_proto.SPLIT_PRUNE_NONE,
    'half': _params_proto.SPLIT_PRUNE_HALF,
    'quarter': _params_proto.SPLIT_PRUNE_QUARTER,
    '10_percent': _params_proto.SPLIT_PRUNE_10_PERCENT,
    'hoeffding': _params_proto.SPLIT_PRUNE_HOEFFDING,
}
SPLIT_TYPES = {
    'less_or_equal': _tree_proto.InequalityTest.LESS_OR_EQUAL,
    'less': _tree_proto.InequalityTest.LESS_THAN
}


def build_params_proto(params):
  """Build a TensorForestParams proto out of the V4ForestHParams object."""
  proto = _params_proto.TensorForestParams()
  proto.num_trees = params.num_trees
  proto.max_nodes = params.max_nodes
  proto.is_regression = params.regression
  proto.num_outputs = params.num_classes
  proto.num_features = params.num_features

  proto.leaf_type = params.v4_leaf_model_type
  proto.stats_type = params.v4_stats_model_type
  proto.collection_type = params.v4_split_collection_type
  proto.pruning_type.type = params.v4_pruning_type
  proto.finish_type.type = params.v4_finish_type

  proto.inequality_test_type = params.v4_split_type

  proto.drop_final_class = False
  proto.collate_examples = params.v4_collate_examples
  proto.checkpoint_stats = params.v4_checkpoint_stats
  proto.use_running_stats_method = params.v4_use_running_stats_method
  proto.initialize_average_splits = params.v4_initialize_average_splits

  if params.v4_prune_every_samples:
    text_format.Merge(params.v4_prune_every_samples,
                      proto.pruning_type.prune_every_samples)
  else:
    # Pruning half-way through split_after_samples seems like a decent default,
    # making it easy to select the number being pruned with v4_pruning_type
    # while not paying the cost of pruning too often.  Note that this only holds
    # if not using a depth-dependent split_after_samples.
    if params.v4_split_after_samples:
      logging.error(
          'If using depth-dependent split_after_samples and also pruning, '
          'need to set v4_prune_every_samples')
    proto.pruning_type.prune_every_samples.constant_value = (
        params.split_after_samples / 2)

  if params.v4_finish_check_every_samples:
    text_format.Merge(params.v4_finish_check_every_samples,
                      proto.finish_type.check_every_steps)
  else:
    # Checking for finish every quarter through split_after_samples seems
    # like a decent default. We don't want to incur the checking cost too often,
    # but (at least for hoeffding) it's lower than the cost of pruning so
    # we can do it a little more frequently.
    proto.finish_type.check_every_steps.constant_value = int(
        params.split_after_samples / 4)

  if params.v4_split_after_samples:
    text_format.Merge(params.v4_split_after_samples, proto.split_after_samples)
  else:
    proto.split_after_samples.constant_value = params.split_after_samples

  if params.v4_num_splits_to_consider:
    text_format.Merge(params.v4_num_splits_to_consider,
                      proto.num_splits_to_consider)
  else:
    proto.num_splits_to_consider.constant_value = params.num_splits_to_consider

  proto.dominate_fraction.constant_value = params.dominate_fraction
  proto.min_split_samples.constant_value = params.split_after_samples

  if params.v4_param_file:
    with open(params.v4_param_file) as f:
      text_format.Merge(f.read(), proto)

  return proto


class V4ForestHParams(object):

  def __init__(self, hparams):
    for k, v in six.iteritems(hparams.__dict__):
      setattr(self, k, v)

    # How to store leaf models.
    model_name = getattr(self, 'v4_model_name', 'all_dense')
    self.v4_leaf_model_type = (
        REGRESSION_MODEL_TYPE[0] if self.regression else
        CLASSIFICATION_LEAF_MODEL_TYPES[model_name][0])

    # How to store stats objects.
    self.v4_stats_model_type = (
        REGRESSION_MODEL_TYPE[1] if self.regression else
        CLASSIFICATION_LEAF_MODEL_TYPES[model_name][1])

    split_collection_name = getattr(self, 'v4_split_collection_name',
                                    'basic')
    self.v4_split_collection_type = (
        REGRESSION_MODEL_TYPE[2] if self.regression else
        COLLECTION_TYPES[split_collection_name])

    finish_name = getattr(self, 'v4_finish_name', 'basic')
    self.v4_finish_type = (
        _params_proto.SPLIT_FINISH_BASIC if self.regression else
        FINISH_TYPES[finish_name])

    pruning_name = getattr(self, 'v4_pruning_name', 'none')
    self.v4_pruning_type = PRUNING_TYPES[pruning_name]

    self.v4_collate_examples = getattr(self, 'v4_collate_examples', False)

    self.v4_checkpoint_stats = getattr(self, 'v4_checkpoint_stats', False)
    self.v4_use_running_stats_method = getattr(
        self, 'v4_use_running_stats_method', False)
    self.v4_initialize_average_splits = getattr(
        self, 'v4_initialize_average_splits', False)

    self.v4_param_file = getattr(self, 'v4_param_file', None)

    self.v4_split_type = getattr(self, 'v4_split_type',
                                 SPLIT_TYPES['less_or_equal'])

    # Special versions of the normal parameters, that support depth-dependence
    self.v4_num_splits_to_consider = getattr(self, 'v4_num_splits_to_consider',
                                             None)
    self.v4_split_after_samples = getattr(self, 'v4_split_after_samples',
                                          None)
    self.v4_finish_check_every_samples = getattr(
        self, 'v4_finish_check_every_samples', None)
    self.v4_prune_every_samples = getattr(
        self, 'v4_prune_every_samples', None)


class TreeTrainingVariablesV4(tensor_forest.TreeTrainingVariables):
  """Stores tf.Variables for training a single random tree."""

  def __init__(self, params, tree_num, training):
    if (not hasattr(params, 'params_proto') or
        not isinstance(params.params_proto,
                       _params_proto.TensorForestParams)):
      params.params_proto = build_params_proto(params)

    params.serialized_params_proto = params.params_proto.SerializeToString()
    self.stats = None
    if training:
      # TODO(gilberth): Manually shard this to be able to fit it on
      # multiple machines.
      self.stats = stats_ops.fertile_stats_variable(
          params, '', self.get_tree_name('stats', tree_num))
    self.tree = model_ops.tree_variable(
        params, '', self.stats, self.get_tree_name('tree', tree_num))


class RandomTreeGraphsV4(tensor_forest.RandomTreeGraphs):
  """Builds TF graphs for random tree training and inference."""

  def tree_initialization(self):
    return control_flow_ops.no_op()

  def training_graph(self, input_data,
                     input_labels,
                     random_seed,
                     data_spec,
                     sparse_features=None,
                     input_weights=None):
    if input_weights is None:
      input_weights = []

    sparse_indices = []
    sparse_values = []
    sparse_shape = []
    if sparse_features is not None:
      sparse_indices = sparse_features.indices
      sparse_values = sparse_features.values
      sparse_shape = sparse_features.dense_shape

    if input_data is None:
      input_data = []

    finished_nodes = stats_ops.process_input_v4(
        self.variables.tree,
        self.variables.stats,
        input_data,
        sparse_indices,
        sparse_values,
        sparse_shape,
        input_labels,
        input_weights,
        input_spec=data_spec.SerializeToString(),
        random_seed=random_seed,
        params=self.params.serialized_params_proto)

    return stats_ops.grow_tree_v4(self.variables.tree, self.variables.stats,
                                  finished_nodes,
                                  params=self.params.serialized_params_proto)

  def inference_graph(self, input_data, data_spec, sparse_features=None):
    sparse_indices = []
    sparse_values = []
    sparse_shape = []
    if sparse_features is not None:
      sparse_indices = sparse_features.indices
      sparse_values = sparse_features.values
      sparse_shape = sparse_features.dense_shape
    if input_data is None:
      input_data = []

    return model_ops.tree_predictions_v4(
        self.variables.tree,
        input_data,
        sparse_indices,
        sparse_values,
        sparse_shape,
        input_spec=data_spec.SerializeToString(),
        params=self.params.serialized_params_proto)

  def average_impurity(self):
    return constant_op.constant(0)

  def size(self):
    """Constructs a TF graph for evaluating the current number of nodes.

    Returns:
      The current number of nodes in the tree.
    """
    return model_ops.tree_size(self.variables.tree)

  def feature_usage_counts(self):
    return model_ops.feature_usage_counts(
        self.variables.tree, params=self.params.serialized_params_proto)


class RandomForestGraphsV4(tensor_forest.RandomForestGraphs):

  def __init__(self, params, tree_graphs=None, tree_variables_class=None,
               **kwargs):
    if not isinstance(params, V4ForestHParams):
      params = V4ForestHParams(params)
    super(RandomForestGraphsV4, self).__init__(
        params, tree_graphs=tree_graphs or RandomTreeGraphsV4,
        tree_variables_class=(tree_variables_class or TreeTrainingVariablesV4),
        **kwargs)
