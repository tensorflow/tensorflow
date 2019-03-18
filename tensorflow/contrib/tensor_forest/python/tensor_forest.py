# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Extremely random forest graph builder. go/brain-tree."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numbers
import random

from google.protobuf import text_format

from tensorflow.contrib.decision_trees.proto import generic_tree_model_pb2 as _tree_proto
from tensorflow.contrib.framework.python.ops import variables as framework_variables
from tensorflow.contrib.tensor_forest.proto import tensor_forest_params_pb2 as _params_proto
from tensorflow.contrib.tensor_forest.python.ops import data_ops
from tensorflow.contrib.tensor_forest.python.ops import model_ops
from tensorflow.contrib.tensor_forest.python.ops import stats_ops

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
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


def parse_number_or_string_to_proto(proto, param):
  if isinstance(param, numbers.Number):
    proto.constant_value = param
  else:  # assume it's a string
    if param.isdigit():
      proto.constant_value = int(param)
    else:
      text_format.Merge(param, proto)


def build_params_proto(params):
  """Build a TensorForestParams proto out of the V4ForestHParams object."""
  proto = _params_proto.TensorForestParams()
  proto.num_trees = params.num_trees
  proto.max_nodes = params.max_nodes
  proto.is_regression = params.regression
  proto.num_outputs = params.num_classes
  proto.num_features = params.num_features

  proto.leaf_type = params.leaf_model_type
  proto.stats_type = params.stats_model_type
  proto.collection_type = _params_proto.COLLECTION_BASIC
  proto.pruning_type.type = params.pruning_type
  proto.finish_type.type = params.finish_type

  proto.inequality_test_type = params.split_type

  proto.drop_final_class = False
  proto.collate_examples = params.collate_examples
  proto.checkpoint_stats = params.checkpoint_stats
  proto.use_running_stats_method = params.use_running_stats_method
  proto.initialize_average_splits = params.initialize_average_splits
  proto.inference_tree_paths = params.inference_tree_paths

  parse_number_or_string_to_proto(proto.pruning_type.prune_every_samples,
                                  params.prune_every_samples)
  parse_number_or_string_to_proto(proto.finish_type.check_every_steps,
                                  params.early_finish_check_every_samples)
  parse_number_or_string_to_proto(proto.split_after_samples,
                                  params.split_after_samples)
  parse_number_or_string_to_proto(proto.num_splits_to_consider,
                                  params.num_splits_to_consider)

  proto.dominate_fraction.constant_value = params.dominate_fraction

  if params.param_file:
    with open(params.param_file) as f:
      text_format.Merge(f.read(), proto)

  return proto


# A convenience class for holding random forest hyperparameters.
#
# To just get some good default parameters, use:
#   hparams = ForestHParams(num_classes=2, num_features=40).fill()
#
# Note that num_classes can not be inferred and so must always be specified.
# Also, either num_splits_to_consider or num_features should be set.
#
# To override specific values, pass them to the constructor:
#   hparams = ForestHParams(num_classes=5, num_trees=10, num_features=5).fill()
#
# TODO(thomaswc): Inherit from tf.HParams when that is publicly available.
class ForestHParams(object):
  """A base class for holding hyperparameters and calculating good defaults."""

  def __init__(
      self,
      num_trees=100,
      max_nodes=10000,
      bagging_fraction=1.0,
      num_splits_to_consider=0,
      feature_bagging_fraction=1.0,
      max_fertile_nodes=0,  # deprecated, unused.
      split_after_samples=250,
      valid_leaf_threshold=1,
      dominate_method='bootstrap',
      dominate_fraction=0.99,
      model_name='all_dense',
      split_finish_name='basic',
      split_pruning_name='none',
      prune_every_samples=0,
      early_finish_check_every_samples=0,
      collate_examples=False,
      checkpoint_stats=False,
      use_running_stats_method=False,
      initialize_average_splits=False,
      inference_tree_paths=False,
      param_file=None,
      split_name='less_or_equal',
      **kwargs):
    self.num_trees = num_trees
    self.max_nodes = max_nodes
    self.bagging_fraction = bagging_fraction
    self.feature_bagging_fraction = feature_bagging_fraction
    self.num_splits_to_consider = num_splits_to_consider
    self.max_fertile_nodes = max_fertile_nodes
    self.split_after_samples = split_after_samples
    self.valid_leaf_threshold = valid_leaf_threshold
    self.dominate_method = dominate_method
    self.dominate_fraction = dominate_fraction
    self.model_name = model_name
    self.split_finish_name = split_finish_name
    self.split_pruning_name = split_pruning_name
    self.collate_examples = collate_examples
    self.checkpoint_stats = checkpoint_stats
    self.use_running_stats_method = use_running_stats_method
    self.initialize_average_splits = initialize_average_splits
    self.inference_tree_paths = inference_tree_paths
    self.param_file = param_file
    self.split_name = split_name
    self.early_finish_check_every_samples = early_finish_check_every_samples
    self.prune_every_samples = prune_every_samples

    for name, value in kwargs.items():
      setattr(self, name, value)

  def values(self):
    return self.__dict__

  def fill(self):
    """Intelligently sets any non-specific parameters."""
    # Fail fast if num_classes or num_features isn't set.
    _ = getattr(self, 'num_classes')
    _ = getattr(self, 'num_features')

    self.bagged_num_features = int(self.feature_bagging_fraction *
                                   self.num_features)

    self.bagged_features = None
    if self.feature_bagging_fraction < 1.0:
      self.bagged_features = [random.sample(
          range(self.num_features),
          self.bagged_num_features) for _ in range(self.num_trees)]

    self.regression = getattr(self, 'regression', False)

    # Num_outputs is the actual number of outputs (a single prediction for
    # classification, a N-dimensional point for regression).
    self.num_outputs = self.num_classes if self.regression else 1

    # Add an extra column to classes for storing counts, which is needed for
    # regression and avoids having to recompute sums for classification.
    self.num_output_columns = self.num_classes + 1

    # Our experiments have found that num_splits_to_consider = num_features
    # gives good accuracy.
    self.num_splits_to_consider = self.num_splits_to_consider or min(
        max(10, math.floor(math.sqrt(self.num_features))), 1000)

    # If base_random_seed is 0, the current time will be used to seed the
    # random number generators for each tree.  If non-zero, the i-th tree
    # will be seeded with base_random_seed + i.
    self.base_random_seed = getattr(self, 'base_random_seed', 0)

    # How to store leaf models.
    self.leaf_model_type = (
        REGRESSION_MODEL_TYPE[0] if self.regression else
        CLASSIFICATION_LEAF_MODEL_TYPES[self.model_name][0])

    # How to store stats objects.
    self.stats_model_type = (
        REGRESSION_MODEL_TYPE[1] if self.regression else
        CLASSIFICATION_LEAF_MODEL_TYPES[self.model_name][1])

    self.finish_type = (
        _params_proto.SPLIT_FINISH_BASIC if self.regression else
        FINISH_TYPES[self.split_finish_name])

    self.pruning_type = PRUNING_TYPES[self.split_pruning_name]

    if self.pruning_type == _params_proto.SPLIT_PRUNE_NONE:
      self.prune_every_samples = 0
    else:
      if (not self.prune_every_samples and
          not (isinstance(numbers.Number) or
               self.split_after_samples.isdigit())):
        logging.error(
            'Must specify prune_every_samples if using a depth-dependent '
            'split_after_samples')
      # Pruning half-way through split_after_samples seems like a decent
      # default, making it easy to select the number being pruned with
      # pruning_type while not paying the cost of pruning too often.  Note that
      # this only holds if not using a depth-dependent split_after_samples.
      self.prune_every_samples = (self.prune_every_samples or
                                  int(self.split_after_samples) / 2)

    if self.finish_type == _params_proto.SPLIT_FINISH_BASIC:
      self.early_finish_check_every_samples = 0
    else:
      if (not self.early_finish_check_every_samples and
          not (isinstance(numbers.Number) or
               self.split_after_samples.isdigit())):
        logging.error(
            'Must specify prune_every_samples if using a depth-dependent '
            'split_after_samples')
      # Checking for early finish every quarter through split_after_samples
      # seems like a decent default. We don't want to incur the checking cost
      # too often, but (at least for hoeffding) it's lower than the cost of
      # pruning so we can do it a little more frequently.
      self.early_finish_check_every_samples = (
          self.early_finish_check_every_samples or
          int(self.split_after_samples) / 4)

    self.split_type = SPLIT_TYPES[self.split_name]

    return self


def get_epoch_variable():
  """Returns the epoch variable, or [0] if not defined."""
  # Grab epoch variable defined in
  # //third_party/tensorflow/python/training/input.py::limit_epochs
  for v in tf_variables.local_variables():
    if 'limit_epochs/epoch' in v.op.name:
      return array_ops.reshape(v, [1])
  # TODO(thomaswc): Access epoch from the data feeder.
  return [0]


# A simple container to hold the training variables for a single tree.
class TreeVariables(object):
  """Stores tf.Variables for training a single random tree.

  Uses tf.get_variable to get tree-specific names so that this can be used
  with a tf.learn-style implementation (one that trains a model, saves it,
  then relies on restoring that model to evaluate).
  """

  def __init__(self, params, tree_num, training, tree_config='', tree_stat=''):
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
          params, tree_stat, self.get_tree_name('stats', tree_num))
    self.tree = model_ops.tree_variable(
        params, tree_config, self.stats, self.get_tree_name('tree', tree_num))

  def get_tree_name(self, name, num):
    return '{0}-{1}'.format(name, num)


class ForestVariables(object):
  """A container for a forests training data, consisting of multiple trees.

  Instantiates a TreeVariables object for each tree. We override the
  __getitem__ and __setitem__ function so that usage looks like this:

    forest_variables = ForestVariables(params)

    ... forest_variables.tree ...
  """

  def __init__(self, params, device_assigner, training=True,
               tree_variables_class=TreeVariables,
               tree_configs=None, tree_stats=None):
    self.variables = []
    # Set up some scalar variables to run through the device assigner, then
    # we can use those to colocate everything related to a tree.
    self.device_dummies = []
    with ops.device(device_assigner):
      for i in range(params.num_trees):
        self.device_dummies.append(variable_scope.get_variable(
            name='device_dummy_%d' % i, shape=0))

    for i in range(params.num_trees):
      with ops.device(self.device_dummies[i].device):
        kwargs = {}
        if tree_configs is not None:
          kwargs.update(dict(tree_config=tree_configs[i]))
        if tree_stats is not None:
          kwargs.update(dict(tree_stat=tree_stats[i]))
        self.variables.append(tree_variables_class(
            params, i, training, **kwargs))

  def __setitem__(self, t, val):
    self.variables[t] = val

  def __getitem__(self, t):
    return self.variables[t]


class RandomForestGraphs(object):
  """Builds TF graphs for random forest training and inference."""

  def __init__(self,
               params,
               tree_configs=None,
               tree_stats=None,
               device_assigner=None,
               variables=None,
               tree_variables_class=TreeVariables,
               tree_graphs=None,
               training=True):
    self.params = params
    self.device_assigner = (
        device_assigner or framework_variables.VariableDeviceChooser())
    logging.info('Constructing forest with params = ')
    logging.info(self.params.__dict__)
    self.variables = variables or ForestVariables(
        self.params, device_assigner=self.device_assigner, training=training,
        tree_variables_class=tree_variables_class,
        tree_configs=tree_configs, tree_stats=tree_stats)
    tree_graph_class = tree_graphs or RandomTreeGraphs
    self.trees = [
        tree_graph_class(self.variables[i], self.params, i)
        for i in range(self.params.num_trees)
    ]

  def _bag_features(self, tree_num, input_data):
    split_data = array_ops.split(
        value=input_data, num_or_size_splits=self.params.num_features, axis=1)
    return array_ops.concat(
        [split_data[ind] for ind in self.params.bagged_features[tree_num]], 1)

  def get_all_resource_handles(self):
    return ([self.variables[i].tree for i in range(len(self.trees))] +
            [self.variables[i].stats for i in range(len(self.trees))])

  def training_graph(self,
                     input_data,
                     input_labels,
                     num_trainers=1,
                     trainer_id=0,
                     **tree_kwargs):
    """Constructs a TF graph for training a random forest.

    Args:
      input_data: A tensor or dict of string->Tensor for input data.
      input_labels: A tensor or placeholder for labels associated with
        input_data.
      num_trainers: Number of parallel trainers to split trees among.
      trainer_id: Which trainer this instance is.
      **tree_kwargs: Keyword arguments passed to each tree's training_graph.

    Returns:
      The last op in the random forest training graph.

    Raises:
      NotImplementedError: If trying to use bagging with sparse features.
    """
    processed_dense_features, processed_sparse_features, data_spec = (
        data_ops.ParseDataTensorOrDict(input_data))

    if input_labels is not None:
      labels = data_ops.ParseLabelTensorOrDict(input_labels)

    data_spec = data_spec or self.get_default_data_spec(input_data)

    tree_graphs = []
    trees_per_trainer = self.params.num_trees / num_trainers
    tree_start = int(trainer_id * trees_per_trainer)
    tree_end = int((trainer_id + 1) * trees_per_trainer)
    for i in range(tree_start, tree_end):
      with ops.device(self.variables.device_dummies[i].device):
        seed = self.params.base_random_seed
        if seed != 0:
          seed += i
        # If using bagging, randomly select some of the input.
        tree_data = processed_dense_features
        tree_labels = labels
        if self.params.bagging_fraction < 1.0:
          # TODO(gilberth): Support bagging for sparse features.
          if processed_sparse_features is not None:
            raise NotImplementedError(
                'Bagging not supported with sparse features.')
          # TODO(thomaswc): This does sampling without replacement.  Consider
          # also allowing sampling with replacement as an option.
          batch_size = array_ops.strided_slice(
              array_ops.shape(processed_dense_features), [0], [1])
          r = random_ops.random_uniform(batch_size, seed=seed)
          mask = math_ops.less(
              r, array_ops.ones_like(r) * self.params.bagging_fraction)
          gather_indices = array_ops.squeeze(
              array_ops.where(mask), axis=[1])
          # TODO(thomaswc): Calculate out-of-bag data and labels, and store
          # them for use in calculating statistics later.
          tree_data = array_ops.gather(processed_dense_features, gather_indices)
          tree_labels = array_ops.gather(labels, gather_indices)
        if self.params.bagged_features:
          if processed_sparse_features is not None:
            raise NotImplementedError(
                'Feature bagging not supported with sparse features.')
          tree_data = self._bag_features(i, tree_data)

        tree_graphs.append(self.trees[i].training_graph(
            tree_data,
            tree_labels,
            seed,
            data_spec=data_spec,
            sparse_features=processed_sparse_features,
            **tree_kwargs))

    return control_flow_ops.group(*tree_graphs, name='train')

  def inference_graph(self, input_data, **inference_args):
    """Constructs a TF graph for evaluating a random forest.

    Args:
      input_data: A tensor or dict of string->Tensor for the input data.
                  This input_data must generate the same spec as the
                  input_data used in training_graph:  the dict must have
                  the same keys, for example, and all tensors must have
                  the same size in their first dimension.
      **inference_args: Keyword arguments to pass through to each tree.

    Returns:
      A tuple of (probabilities, tree_paths, variance).

    Raises:
      NotImplementedError: If trying to use feature bagging with sparse
        features.
    """
    processed_dense_features, processed_sparse_features, data_spec = (
        data_ops.ParseDataTensorOrDict(input_data))

    probabilities = []
    paths = []
    for i in range(self.params.num_trees):
      with ops.device(self.variables.device_dummies[i].device):
        tree_data = processed_dense_features
        if self.params.bagged_features:
          if processed_sparse_features is not None:
            raise NotImplementedError(
                'Feature bagging not supported with sparse features.')
          tree_data = self._bag_features(i, tree_data)
        probs, path = self.trees[i].inference_graph(
            tree_data,
            data_spec,
            sparse_features=processed_sparse_features,
            **inference_args)
        probabilities.append(probs)
        paths.append(path)
    with ops.device(self.variables.device_dummies[0].device):
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
    """Constructs a TF graph for evaluating the average size of a forest.

    Returns:
      The average number of nodes over the trees.
    """
    sizes = []
    for i in range(self.params.num_trees):
      with ops.device(self.variables.device_dummies[i].device):
        sizes.append(self.trees[i].size())
    return math_ops.reduce_mean(
        math_ops.cast(array_ops.stack(sizes), dtypes.float32))

  # pylint: disable=unused-argument
  def training_loss(self, features, labels, name='training_loss'):
    return math_ops.negative(self.average_size(), name=name)

  # pylint: disable=unused-argument
  def validation_loss(self, features, labels):
    return math_ops.negative(self.average_size())

  def average_impurity(self):
    """Constructs a TF graph for evaluating the leaf impurity of a forest.

    Returns:
      The last op in the graph.
    """
    impurities = []
    for i in range(self.params.num_trees):
      with ops.device(self.variables.device_dummies[i].device):
        impurities.append(self.trees[i].average_impurity())
    return math_ops.reduce_mean(array_ops.stack(impurities))

  def feature_importances(self):
    tree_counts = [self.trees[i].feature_usage_counts()
                   for i in range(self.params.num_trees)]
    total_counts = math_ops.reduce_sum(array_ops.stack(tree_counts, 0), 0)
    return total_counts / math_ops.reduce_sum(total_counts)


class RandomTreeGraphs(object):
  """Builds TF graphs for random tree training and inference."""

  def __init__(self, variables, params, tree_num):
    self.variables = variables
    self.params = params
    self.tree_num = tree_num

  def training_graph(self,
                     input_data,
                     input_labels,
                     random_seed,
                     data_spec,
                     sparse_features=None,
                     input_weights=None):

    """Constructs a TF graph for training a random tree.

    Args:
      input_data: A tensor or placeholder for input data.
      input_labels: A tensor or placeholder for labels associated with
        input_data.
      random_seed: The random number generator seed to use for this tree.  0
        means use the current time as the seed.
      data_spec: A data_ops.TensorForestDataSpec object specifying the
        original feature/columns of the data.
      sparse_features: A tf.SparseTensor for sparse input data.
      input_weights: A float tensor or placeholder holding per-input weights,
        or None if all inputs are to be weighted equally.

    Returns:
      The last op in the random tree training graph.
    """
    # TODO(gilberth): Use this.
    unused_epoch = math_ops.cast(get_epoch_variable(), dtypes.int32)

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

    leaf_ids = model_ops.traverse_tree_v4(
        self.variables.tree,
        input_data,
        sparse_indices,
        sparse_values,
        sparse_shape,
        input_spec=data_spec.SerializeToString(),
        params=self.params.serialized_params_proto)

    update_model = model_ops.update_model_v4(
        self.variables.tree,
        leaf_ids,
        input_labels,
        input_weights,
        params=self.params.serialized_params_proto)

    finished_nodes = stats_ops.process_input_v4(
        self.variables.tree,
        self.variables.stats,
        input_data,
        sparse_indices,
        sparse_values,
        sparse_shape,
        input_labels,
        input_weights,
        leaf_ids,
        input_spec=data_spec.SerializeToString(),
        random_seed=random_seed,
        params=self.params.serialized_params_proto)

    with ops.control_dependencies([update_model]):
      return stats_ops.grow_tree_v4(
          self.variables.tree,
          self.variables.stats,
          finished_nodes,
          params=self.params.serialized_params_proto)

  def inference_graph(self, input_data, data_spec, sparse_features=None):
    """Constructs a TF graph for evaluating a random tree.

    Args:
      input_data: A tensor or placeholder for input data.
      data_spec: A TensorForestDataSpec proto specifying the original
        input columns.
      sparse_features: A tf.SparseTensor for sparse input data.

    Returns:
      A tuple of (probabilities, tree_paths).
    """
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

  def size(self):
    """Constructs a TF graph for evaluating the current number of nodes.

    Returns:
      The current number of nodes in the tree.
    """
    return model_ops.tree_size(self.variables.tree)

  def feature_usage_counts(self):
    return model_ops.feature_usage_counts(
        self.variables.tree, params=self.params.serialized_params_proto)
