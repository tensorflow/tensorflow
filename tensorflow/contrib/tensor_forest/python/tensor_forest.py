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
import random
import sys

from tensorflow.contrib.losses.python.losses import loss_ops
from tensorflow.contrib.tensor_forest.python import constants
from tensorflow.contrib.tensor_forest.python.ops import tensor_forest_ops

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging


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

  def __init__(self,
               num_trees=100,
               max_nodes=10000,
               bagging_fraction=1.0,
               num_splits_to_consider=0,
               feature_bagging_fraction=1.0,
               max_fertile_nodes=0,
               split_after_samples=250,
               min_split_samples=5,
               valid_leaf_threshold=1,
               dominate_method='bootstrap',
               dominate_fraction=0.99,
               **kwargs):
    self.num_trees = num_trees
    self.max_nodes = max_nodes
    self.bagging_fraction = bagging_fraction
    self.feature_bagging_fraction = feature_bagging_fraction
    self.num_splits_to_consider = num_splits_to_consider
    self.max_fertile_nodes = max_fertile_nodes
    self.split_after_samples = split_after_samples
    self.min_split_samples = min_split_samples
    self.valid_leaf_threshold = valid_leaf_threshold
    self.dominate_method = dominate_method
    self.dominate_fraction = dominate_fraction

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
    # classification, a N-dimenensional point for regression).
    self.num_outputs = self.num_classes if self.regression else 1

    # Add an extra column to classes for storing counts, which is needed for
    # regression and avoids having to recompute sums for classification.
    self.num_output_columns = self.num_classes + 1

    # Our experiments have found that num_splits_to_consider = num_features
    # gives good accuracy.
    self.num_splits_to_consider = self.num_splits_to_consider or min(
        self.num_features, 1000)

    self.max_fertile_nodes = (self.max_fertile_nodes or
                              int(math.ceil(self.max_nodes / 2.0)))

    # We have num_splits_to_consider slots to fill, and we want to spend
    # approximately split_after_samples samples initializing them.
    num_split_initializiations_per_input = max(1, int(math.floor(
        self.num_splits_to_consider / self.split_after_samples)))
    self.split_initializations_per_input = getattr(
        self, 'split_initializations_per_input',
        num_split_initializiations_per_input)

    # If base_random_seed is 0, the current time will be used to seed the
    # random number generators for each tree.  If non-zero, the i-th tree
    # will be seeded with base_random_seed + i.
    self.base_random_seed = getattr(self, 'base_random_seed', 0)

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
class TreeTrainingVariables(object):
  """Stores tf.Variables for training a single random tree.

  Uses tf.get_variable to get tree-specific names so that this can be used
  with a tf.learn-style implementation (one that trains a model, saves it,
  then relies on restoring that model to evaluate).
  """

  def __init__(self, params, tree_num, training):
    self.tree = variable_scope.get_variable(
        name=self.get_tree_name('tree', tree_num), dtype=dtypes.int32,
        shape=[params.max_nodes, 2],
        initializer=init_ops.constant_initializer(-2))
    self.tree_thresholds = variable_scope.get_variable(
        name=self.get_tree_name('tree_thresholds', tree_num),
        shape=[params.max_nodes],
        initializer=init_ops.constant_initializer(-1.0))
    self.end_of_tree = variable_scope.get_variable(
        name=self.get_tree_name('end_of_tree', tree_num),
        dtype=dtypes.int32,
        initializer=constant_op.constant([1]))
    self.start_epoch = variable_scope.get_variable(
        name=self.get_tree_name('start_epoch', tree_num),
        dtype=dtypes.int32, shape=[params.max_nodes],
        initializer=init_ops.constant_initializer(0))

    if training:
      self.node_to_accumulator_map = variable_scope.get_variable(
          name=self.get_tree_name('node_to_accumulator_map', tree_num),
          shape=[params.max_nodes],
          dtype=dtypes.int32,
          initializer=init_ops.constant_initializer(-1))
      self.accumulator_to_node_map = variable_scope.get_variable(
          name=self.get_tree_name('accumulator_to_node_map', tree_num),
          shape=[params.max_fertile_nodes],
          dtype=dtypes.int32,
          initializer=init_ops.constant_initializer(-1))

      self.candidate_split_features = variable_scope.get_variable(
          name=self.get_tree_name('candidate_split_features', tree_num),
          shape=[params.max_fertile_nodes, params.num_splits_to_consider],
          dtype=dtypes.int32,
          initializer=init_ops.constant_initializer(-1))
      self.candidate_split_thresholds = variable_scope.get_variable(
          name=self.get_tree_name('candidate_split_thresholds', tree_num),
          shape=[params.max_fertile_nodes, params.num_splits_to_consider],
          initializer=init_ops.constant_initializer(0.0))

    # Statistics shared by classification and regression.
    self.node_sums = variable_scope.get_variable(
        name=self.get_tree_name('node_sums', tree_num),
        shape=[params.max_nodes, params.num_output_columns],
        initializer=init_ops.constant_initializer(0.0))

    if training:
      self.candidate_split_sums = variable_scope.get_variable(
          name=self.get_tree_name('candidate_split_sums', tree_num),
          shape=[params.max_fertile_nodes, params.num_splits_to_consider,
                 params.num_output_columns],
          initializer=init_ops.constant_initializer(0.0))
      self.accumulator_sums = variable_scope.get_variable(
          name=self.get_tree_name('accumulator_sums', tree_num),
          shape=[params.max_fertile_nodes, params.num_output_columns],
          initializer=init_ops.constant_initializer(-1.0))

      # Regression also tracks second order stats.
      if params.regression:
        self.node_squares = variable_scope.get_variable(
            name=self.get_tree_name('node_squares', tree_num),
            shape=[params.max_nodes, params.num_output_columns],
            initializer=init_ops.constant_initializer(0.0))

        self.candidate_split_squares = variable_scope.get_variable(
            name=self.get_tree_name('candidate_split_squares', tree_num),
            shape=[params.max_fertile_nodes, params.num_splits_to_consider,
                   params.num_output_columns],
            initializer=init_ops.constant_initializer(0.0))

        self.accumulator_squares = variable_scope.get_variable(
            name=self.get_tree_name('accumulator_squares', tree_num),
            shape=[params.max_fertile_nodes, params.num_output_columns],
            initializer=init_ops.constant_initializer(-1.0))

      else:
        self.node_squares = constant_op.constant(
            0.0, name=self.get_tree_name('node_squares', tree_num))

        self.candidate_split_squares = constant_op.constant(
            0.0, name=self.get_tree_name('candidate_split_squares', tree_num))

        self.accumulator_squares = constant_op.constant(
            0.0, name=self.get_tree_name('accumulator_squares', tree_num))

  def get_tree_name(self, name, num):
    return '{0}-{1}'.format(name, num)


class ForestStats(object):

  def __init__(self, tree_stats, params):
    """A simple container for stats about a forest."""
    self.tree_stats = tree_stats
    self.params = params

  def get_average(self, thing):
    val = 0.0
    for i in range(self.params.num_trees):
      val += getattr(self.tree_stats[i], thing)

    return val / self.params.num_trees


class TreeStats(object):

  def __init__(self, num_nodes, num_leaves):
    self.num_nodes = num_nodes
    self.num_leaves = num_leaves


class ForestTrainingVariables(object):
  """A container for a forests training data, consisting of multiple trees.

  Instantiates a TreeTrainingVariables object for each tree. We override the
  __getitem__ and __setitem__ function so that usage looks like this:

    forest_variables = ForestTrainingVariables(params)

    ... forest_variables.tree ...
  """

  def __init__(self, params, device_assigner, training=True,
               tree_variables_class=TreeTrainingVariables):
    self.variables = []
    for i in range(params.num_trees):
      with ops.device(device_assigner.get_device(i)):
        self.variables.append(tree_variables_class(params, i, training))

  def __setitem__(self, t, val):
    self.variables[t] = val

  def __getitem__(self, t):
    return self.variables[t]


class RandomForestDeviceAssigner(object):
  """A device assigner that uses the default device.

  Write subclasses that implement get_device for control over how trees
  get assigned to devices.  This assumes that whole trees are assigned
  to a device.
  """

  def __init__(self):
    self.cached = None

  def get_device(self, unused_tree_num):
    if not self.cached:
      dummy = constant_op.constant(0)
      self.cached = dummy.device

    return self.cached


class RandomForestGraphs(object):
  """Builds TF graphs for random forest training and inference."""

  def __init__(self,
               params,
               device_assigner=None,
               variables=None,
               tree_variables_class=TreeTrainingVariables,
               tree_graphs=None,
               training=True):
    self.params = params
    self.device_assigner = device_assigner or RandomForestDeviceAssigner()
    logging.info('Constructing forest with params = ')
    logging.info(self.params.__dict__)
    self.variables = variables or ForestTrainingVariables(
        self.params, device_assigner=self.device_assigner, training=training,
        tree_variables_class=tree_variables_class)
    tree_graph_class = tree_graphs or RandomTreeGraphs
    self.trees = [
        tree_graph_class(self.variables[i], self.params, i)
        for i in range(self.params.num_trees)
    ]

  def _bag_features(self, tree_num, input_data):
    split_data = array_ops.split(
        value=input_data, num_or_size_splits=self.params.num_features, axis=1)
    return array_ops.concat_v2(
        [split_data[ind] for ind in self.params.bagged_features[tree_num]], 1)

  def training_graph(self,
                     input_data,
                     input_labels,
                     data_spec=None,
                     **tree_kwargs):
    """Constructs a TF graph for training a random forest.

    Args:
      input_data: A tensor or SparseTensor or placeholder for input data.
      input_labels: A tensor or placeholder for labels associated with
        input_data.
      data_spec: A list of tf.dtype values specifying the original types of
        each column.
      **tree_kwargs: Keyword arguments passed to each tree's training_graph.

    Returns:
      The last op in the random forest training graph.
    """
    data_spec = [constants.DATA_FLOAT] if data_spec is None else data_spec
    tree_graphs = []
    for i in range(self.params.num_trees):
      with ops.device(self.device_assigner.get_device(i)):
        seed = self.params.base_random_seed
        if seed != 0:
          seed += i
        # If using bagging, randomly select some of the input.
        tree_data = input_data
        tree_labels = input_labels
        if self.params.bagging_fraction < 1.0:
          # TODO(thomaswc): This does sampling without replacment.  Consider
          # also allowing sampling with replacement as an option.
          batch_size = array_ops.strided_slice(
              array_ops.shape(input_data), [0], [1])
          r = random_ops.random_uniform(batch_size, seed=seed)
          mask = math_ops.less(
              r, array_ops.ones_like(r) * self.params.bagging_fraction)
          gather_indices = array_ops.squeeze(
              array_ops.where(mask), squeeze_dims=[1])
          # TODO(thomaswc): Calculate out-of-bag data and labels, and store
          # them for use in calculating statistics later.
          tree_data = array_ops.gather(input_data, gather_indices)
          tree_labels = array_ops.gather(input_labels, gather_indices)
        if self.params.bagged_features:
          tree_data = self._bag_features(i, tree_data)

        initialization = self.trees[i].tree_initialization()

        with ops.control_dependencies([initialization]):
          tree_graphs.append(
              self.trees[i].training_graph(
                  tree_data, tree_labels, seed, data_spec=data_spec,
                  **tree_kwargs))

    return control_flow_ops.group(*tree_graphs, name='train')

  def inference_graph(self, input_data, data_spec=None, **inference_args):
    """Constructs a TF graph for evaluating a random forest.

    Args:
      input_data: A tensor or SparseTensor or placeholder for input data.
      data_spec: A list of tf.dtype values specifying the original types of
        each column.
      **inference_args: Keyword arguments to pass through to each tree.

    Returns:
      The last op in the random forest inference graph.
    """
    data_spec = [constants.DATA_FLOAT] if data_spec is None else data_spec
    probabilities = []
    for i in range(self.params.num_trees):
      with ops.device(self.device_assigner.get_device(i)):
        tree_data = input_data
        if self.params.bagged_features:
          tree_data = self._bag_features(i, input_data)
        probabilities.append(self.trees[i].inference_graph(
            tree_data, data_spec, **inference_args))
    with ops.device(self.device_assigner.get_device(0)):
      all_predict = array_ops.pack(probabilities)
      return math_ops.div(
          math_ops.reduce_sum(all_predict, 0), self.params.num_trees,
          name='probabilities')

  def average_size(self):
    """Constructs a TF graph for evaluating the average size of a forest.

    Returns:
      The average number of nodes over the trees.
    """
    sizes = []
    for i in range(self.params.num_trees):
      with ops.device(self.device_assigner.get_device(i)):
        sizes.append(self.trees[i].size())
    return math_ops.reduce_mean(math_ops.to_float(array_ops.pack(sizes)))

  # pylint: disable=unused-argument
  def training_loss(self, features, labels, data_spec=None,
                    name='training_loss'):
    return math_ops.neg(self.average_size(), name=name)

  # pylint: disable=unused-argument
  def validation_loss(self, features, labels):
    return math_ops.neg(self.average_size())

  def average_impurity(self):
    """Constructs a TF graph for evaluating the leaf impurity of a forest.

    Returns:
      The last op in the graph.
    """
    impurities = []
    for i in range(self.params.num_trees):
      with ops.device(self.device_assigner.get_device(i)):
        impurities.append(self.trees[i].average_impurity())
    return math_ops.reduce_mean(array_ops.pack(impurities))

  def get_stats(self, session):
    tree_stats = []
    for i in range(self.params.num_trees):
      with ops.device(self.device_assigner.get_device(i)):
        tree_stats.append(self.trees[i].get_stats(session))
    return ForestStats(tree_stats, self.params)


def one_hot_wrapper(num_classes, loss_fn):
  """Some loss functions take one-hot labels."""
  def _loss(probs, targets):
    one_hot_labels = array_ops.one_hot(
        math_ops.to_int32(targets), num_classes,
        on_value=1., off_value=0., dtype=dtypes.float32)
    return loss_fn(probs, one_hot_labels)
  return _loss


class TrainingLossForest(RandomForestGraphs):
  """Random Forest that uses training loss as the termination criteria."""

  def __init__(self, params, loss_fn=None, **kwargs):
    """Initialize.

    Args:
      params: Like RandomForestGraphs, a ForestHParams object.
      loss_fn: A function that takes probabilities and targets and returns
        a loss for each example.
      **kwargs: Keyword args to pass to superclass (RandomForestGraphs).
    """
    self.loss_fn = loss_fn or one_hot_wrapper(params.num_classes,
                                              loss_ops.log_loss)
    self._loss = None
    super(TrainingLossForest, self).__init__(params, **kwargs)

  def _get_loss(self, features, labels, data_spec=None):
    """Constructs, caches, and returns the inference-based loss."""
    if self._loss is not None:
      return self._loss

    def _average_loss():
      probs = self.inference_graph(features, data_spec=data_spec)
      return math_ops.reduce_sum(self.loss_fn(
          probs, labels)) / math_ops.to_float(
              array_ops.shape(features)[0])

    self._loss = control_flow_ops.cond(
        self.average_size() > 0, _average_loss,
        lambda: constant_op.constant(sys.maxsize, dtype=dtypes.float32))

    return self._loss

  def training_graph(self, input_data, input_labels, data_spec=None,
                     **kwargs):
    loss = self._get_loss(input_data, input_labels, data_spec=data_spec)
    with ops.control_dependencies([loss.op]):
      return super(TrainingLossForest, self).training_graph(
          input_data, input_labels, **kwargs)

  def training_loss(self, features, labels, data_spec=None,
                    name='training_loss'):
    return array_ops.identity(
        self._get_loss(features, labels, data_spec=data_spec), name=name)


class RandomTreeGraphs(object):
  """Builds TF graphs for random tree training and inference."""

  def __init__(self, variables, params, tree_num):
    self.variables = variables
    self.params = params
    self.tree_num = tree_num

  def tree_initialization(self):
    def _init_tree():
      return state_ops.scatter_update(self.variables.tree, [0], [[-1, -1]]).op

    def _nothing():
      return control_flow_ops.no_op()

    return control_flow_ops.cond(
        math_ops.equal(
            array_ops.squeeze(
                array_ops.strided_slice(self.variables.tree, [0, 0], [1, 1])),
            -2), _init_tree, _nothing)

  def _gini(self, class_counts):
    """Calculate the Gini impurity.

    If c(i) denotes the i-th class count and c = sum_i c(i) then
      score = 1 - sum_i ( c(i) / c )^2

    Args:
      class_counts: A 2-D tensor of per-class counts, usually a slice or
        gather from variables.node_sums.

    Returns:
      A 1-D tensor of the Gini impurities for each row in the input.
    """
    smoothed = 1.0 + array_ops.slice(class_counts, [0, 1], [-1, -1])
    sums = math_ops.reduce_sum(smoothed, 1)
    sum_squares = math_ops.reduce_sum(math_ops.square(smoothed), 1)

    return 1.0 - sum_squares / (sums * sums)

  def _weighted_gini(self, class_counts):
    """Our split score is the Gini impurity times the number of examples.

    If c(i) denotes the i-th class count and c = sum_i c(i) then
      score = c * (1 - sum_i ( c(i) / c )^2 )
            = c - sum_i c(i)^2 / c
    Args:
      class_counts: A 2-D tensor of per-class counts, usually a slice or
        gather from variables.node_sums.

    Returns:
      A 1-D tensor of the Gini impurities for each row in the input.
    """
    smoothed = 1.0 + array_ops.slice(class_counts, [0, 1], [-1, -1])
    sums = math_ops.reduce_sum(smoothed, 1)
    sum_squares = math_ops.reduce_sum(math_ops.square(smoothed), 1)

    return sums - sum_squares / sums

  def _variance(self, sums, squares):
    """Calculate the variance for each row of the input tensors.

    Variance is V = E[x^2] - (E[x])^2.

    Args:
      sums: A tensor containing output sums, usually a slice from
        variables.node_sums.  Should contain the number of examples seen
        in index 0 so we can calculate expected value.
      squares: Same as sums, but sums of squares.

    Returns:
      A 1-D tensor of the variances for each row in the input.
    """
    total_count = array_ops.slice(sums, [0, 0], [-1, 1])
    e_x = sums / total_count
    e_x2 = squares / total_count

    return math_ops.reduce_sum(e_x2 - math_ops.square(e_x), 1)

  def training_graph(self,
                     input_data,
                     input_labels,
                     random_seed,
                     data_spec,
                     input_weights=None):

    """Constructs a TF graph for training a random tree.

    Args:
      input_data: A tensor or SparseTensor or placeholder for input data.
      input_labels: A tensor or placeholder for labels associated with
        input_data.
      random_seed: The random number generator seed to use for this tree.  0
        means use the current time as the seed.
      data_spec: A list of tf.dtype values specifying the original types of
        each column.
      input_weights: A float tensor or placeholder holding per-input weights,
        or None if all inputs are to be weighted equally.

    Returns:
      The last op in the random tree training graph.
    """
    epoch = math_ops.to_int32(get_epoch_variable())

    if input_weights is None:
      input_weights = []

    sparse_indices = []
    sparse_values = []
    sparse_shape = []
    if isinstance(input_data, sparse_tensor.SparseTensor):
      sparse_indices = input_data.indices
      sparse_values = input_data.values
      sparse_shape = input_data.dense_shape
      input_data = []

    # Count extremely random stats.
    (node_sums, node_squares, splits_indices, splits_sums, splits_squares,
     totals_indices, totals_sums, totals_squares,
     input_leaves) = (tensor_forest_ops.count_extremely_random_stats(
         input_data,
         sparse_indices,
         sparse_values,
         sparse_shape,
         data_spec,
         input_labels,
         input_weights,
         self.variables.tree,
         self.variables.tree_thresholds,
         self.variables.node_to_accumulator_map,
         self.variables.candidate_split_features,
         self.variables.candidate_split_thresholds,
         self.variables.start_epoch,
         epoch,
         num_classes=self.params.num_output_columns,
         regression=self.params.regression))
    node_update_ops = []
    node_update_ops.append(
        state_ops.assign_add(self.variables.node_sums, node_sums))

    splits_update_ops = []
    splits_update_ops.append(
        tensor_forest_ops.scatter_add_ndim(self.variables.candidate_split_sums,
                                           splits_indices, splits_sums))
    splits_update_ops.append(
        tensor_forest_ops.scatter_add_ndim(self.variables.accumulator_sums,
                                           totals_indices, totals_sums))

    if self.params.regression:
      node_update_ops.append(state_ops.assign_add(self.variables.node_squares,
                                                  node_squares))
      splits_update_ops.append(
          tensor_forest_ops.scatter_add_ndim(
              self.variables.candidate_split_squares, splits_indices,
              splits_squares))
      splits_update_ops.append(
          tensor_forest_ops.scatter_add_ndim(self.variables.accumulator_squares,
                                             totals_indices, totals_squares))

    # Sample inputs.
    update_indices, feature_updates, threshold_updates = (
        tensor_forest_ops.sample_inputs(
            input_data,
            sparse_indices,
            sparse_values,
            sparse_shape,
            input_weights,
            self.variables.node_to_accumulator_map,
            input_leaves,
            self.variables.candidate_split_features,
            self.variables.candidate_split_thresholds,
            split_initializations_per_input=(
                self.params.split_initializations_per_input),
            split_sampling_random_seed=random_seed))
    update_features_op = state_ops.scatter_update(
        self.variables.candidate_split_features, update_indices,
        feature_updates)
    update_thresholds_op = state_ops.scatter_update(
        self.variables.candidate_split_thresholds, update_indices,
        threshold_updates)

    # Calculate finished nodes.
    with ops.control_dependencies(splits_update_ops):
      # Passing input_leaves to finished nodes here means that nodes that
      # have become stale won't be deallocated until an input reaches them,
      # because we're trying to avoid considering every fertile node for
      # performance reasons.
      finished, stale = tensor_forest_ops.finished_nodes(
          input_leaves,
          self.variables.node_to_accumulator_map,
          self.variables.candidate_split_sums,
          self.variables.candidate_split_squares,
          self.variables.accumulator_sums,
          self.variables.accumulator_squares,
          self.variables.start_epoch,
          epoch,
          num_split_after_samples=self.params.split_after_samples,
          min_split_samples=self.params.min_split_samples,
          dominate_method=self.params.dominate_method,
          dominate_fraction=self.params.dominate_fraction)

    # Update leaf scores.
    # TODO(thomaswc): Store the leaf scores in a TopN and only update the
    # scores of the leaves that were touched by this batch of input.
    children = array_ops.squeeze(
        array_ops.slice(self.variables.tree, [0, 0], [-1, 1]), squeeze_dims=[1])
    is_leaf = math_ops.equal(constants.LEAF_NODE, children)
    leaves = math_ops.to_int32(
        array_ops.squeeze(
            array_ops.where(is_leaf), squeeze_dims=[1]))
    non_fertile_leaves = array_ops.boolean_mask(
        leaves, math_ops.less(array_ops.gather(
            self.variables.node_to_accumulator_map, leaves), 0))

    # TODO(gilberth): It should be possible to limit the number of non
    # fertile leaves we calculate scores for, especially since we can only take
    # at most array_ops.shape(finished)[0] of them.
    with ops.control_dependencies(node_update_ops):
      sums = array_ops.gather(self.variables.node_sums, non_fertile_leaves)
      if self.params.regression:
        squares = array_ops.gather(self.variables.node_squares,
                                   non_fertile_leaves)
        non_fertile_leaf_scores = self._variance(sums, squares)
      else:
        non_fertile_leaf_scores = self._weighted_gini(sums)

    # Calculate best splits.
    with ops.control_dependencies(splits_update_ops):
      split_indices = tensor_forest_ops.best_splits(
          finished,
          self.variables.node_to_accumulator_map,
          self.variables.candidate_split_sums,
          self.variables.candidate_split_squares,
          self.variables.accumulator_sums,
          self.variables.accumulator_squares,
          regression=self.params.regression)

    # Grow tree.
    with ops.control_dependencies([update_features_op, update_thresholds_op]):
      (tree_update_indices, tree_children_updates, tree_threshold_updates,
       new_eot) = (tensor_forest_ops.grow_tree(
           self.variables.end_of_tree, self.variables.node_to_accumulator_map,
           finished, split_indices, self.variables.candidate_split_features,
           self.variables.candidate_split_thresholds))
      tree_update_op = state_ops.scatter_update(
          self.variables.tree, tree_update_indices, tree_children_updates)
      thresholds_update_op = state_ops.scatter_update(
          self.variables.tree_thresholds, tree_update_indices,
          tree_threshold_updates)
      # TODO(thomaswc): Only update the epoch on the new leaves.
      new_epoch_updates = epoch * array_ops.ones_like(tree_threshold_updates,
                                                      dtype=dtypes.int32)
      epoch_update_op = state_ops.scatter_update(
          self.variables.start_epoch, tree_update_indices,
          new_epoch_updates)

    # Update fertile slots.
    with ops.control_dependencies([tree_update_op]):
      (n2a_map_updates, a2n_map_updates, accumulators_cleared,
       accumulators_allocated) = (tensor_forest_ops.update_fertile_slots(
           finished,
           non_fertile_leaves,
           non_fertile_leaf_scores,
           self.variables.end_of_tree,
           self.variables.accumulator_sums,
           self.variables.node_to_accumulator_map,
           stale,
           self.variables.node_sums,
           regression=self.params.regression))

    # Ensure end_of_tree doesn't get updated until UpdateFertileSlots has
    # used it to calculate new leaves.
    gated_new_eot, = control_flow_ops.tuple(
        [new_eot], control_inputs=[n2a_map_updates])
    eot_update_op = state_ops.assign(self.variables.end_of_tree, gated_new_eot)

    updates = []
    updates.append(eot_update_op)
    updates.append(tree_update_op)
    updates.append(thresholds_update_op)
    updates.append(epoch_update_op)

    updates.append(
        state_ops.scatter_update(self.variables.node_to_accumulator_map,
                                 n2a_map_updates[0], n2a_map_updates[1]))

    updates.append(
        state_ops.scatter_update(self.variables.accumulator_to_node_map,
                                 a2n_map_updates[0], a2n_map_updates[1]))

    cleared_and_allocated_accumulators = array_ops.concat_v2(
        [accumulators_cleared, accumulators_allocated], 0)

    # Calculate values to put into scatter update for candidate counts.
    # Candidate split counts are always reset back to 0 for both cleared
    # and allocated accumulators. This means some accumulators might be doubly
    # reset to 0 if the were released and not allocated, then later allocated.
    split_values = array_ops.tile(
        array_ops.expand_dims(array_ops.expand_dims(
            array_ops.zeros_like(cleared_and_allocated_accumulators,
                                 dtype=dtypes.float32), 1), 2),
        [1, self.params.num_splits_to_consider, self.params.num_output_columns])
    updates.append(state_ops.scatter_update(
        self.variables.candidate_split_sums,
        cleared_and_allocated_accumulators, split_values))
    if self.params.regression:
      updates.append(state_ops.scatter_update(
          self.variables.candidate_split_squares,
          cleared_and_allocated_accumulators, split_values))

    # Calculate values to put into scatter update for total counts.
    total_cleared = array_ops.tile(
        array_ops.expand_dims(
            math_ops.neg(array_ops.ones_like(accumulators_cleared,
                                             dtype=dtypes.float32)), 1),
        [1, self.params.num_output_columns])
    total_reset = array_ops.tile(
        array_ops.expand_dims(
            array_ops.zeros_like(accumulators_allocated,
                                 dtype=dtypes.float32), 1),
        [1, self.params.num_output_columns])
    accumulator_updates = array_ops.concat_v2([total_cleared, total_reset], 0)
    updates.append(state_ops.scatter_update(
        self.variables.accumulator_sums,
        cleared_and_allocated_accumulators, accumulator_updates))
    if self.params.regression:
      updates.append(state_ops.scatter_update(
          self.variables.accumulator_squares,
          cleared_and_allocated_accumulators, accumulator_updates))

    # Calculate values to put into scatter update for candidate splits.
    split_features_updates = array_ops.tile(
        array_ops.expand_dims(
            math_ops.neg(array_ops.ones_like(
                cleared_and_allocated_accumulators)), 1),
        [1, self.params.num_splits_to_consider])
    updates.append(state_ops.scatter_update(
        self.variables.candidate_split_features,
        cleared_and_allocated_accumulators, split_features_updates))

    updates += self.finish_iteration()

    return control_flow_ops.group(*updates)

  def finish_iteration(self):
    """Perform any operations that should be done at the end of an iteration.

    This is mostly useful for subclasses that need to reset variables after
    an iteration, such as ones that are used to finish nodes.

    Returns:
      A list of operations.
    """
    return []

  def inference_graph(self, input_data, data_spec):
    """Constructs a TF graph for evaluating a random tree.

    Args:
      input_data: A tensor or SparseTensor or placeholder for input data.
      data_spec: A list of tf.dtype values specifying the original types of
        each column.

    Returns:
      The last op in the random tree inference graph.
    """
    sparse_indices = []
    sparse_values = []
    sparse_shape = []
    if isinstance(input_data, sparse_tensor.SparseTensor):
      sparse_indices = input_data.indices
      sparse_values = input_data.values
      sparse_shape = input_data.dense_shape
      input_data = []
    return tensor_forest_ops.tree_predictions(
        input_data,
        sparse_indices,
        sparse_values,
        sparse_shape,
        data_spec,
        self.variables.tree,
        self.variables.tree_thresholds,
        self.variables.node_sums,
        valid_leaf_threshold=self.params.valid_leaf_threshold)

  def average_impurity(self):
    """Constructs a TF graph for evaluating the average leaf impurity of a tree.

    If in regression mode, this is the leaf variance. If in classification mode,
    this is the gini impurity.

    Returns:
      The last op in the graph.
    """
    children = array_ops.squeeze(array_ops.slice(
        self.variables.tree, [0, 0], [-1, 1]), squeeze_dims=[1])
    is_leaf = math_ops.equal(constants.LEAF_NODE, children)
    leaves = math_ops.to_int32(array_ops.squeeze(array_ops.where(is_leaf),
                                                 squeeze_dims=[1]))
    counts = array_ops.gather(self.variables.node_sums, leaves)
    gini = self._weighted_gini(counts)
    # Guard against step 1, when there often are no leaves yet.
    def impurity():
      return gini
    # Since average impurity can be used for loss, when there's no data just
    # return a big number so that loss always decreases.
    def big():
      return array_ops.ones_like(gini, dtype=dtypes.float32) * 10000000.
    return control_flow_ops.cond(math_ops.greater(
        array_ops.shape(leaves)[0], 0), impurity, big)

  def size(self):
    """Constructs a TF graph for evaluating the current number of nodes.

    Returns:
      The current number of nodes in the tree.
    """
    return self.variables.end_of_tree - 1

  def get_stats(self, session):
    num_nodes = self.variables.end_of_tree.eval(session=session) - 1
    num_leaves = array_ops.where(
        math_ops.equal(array_ops.squeeze(array_ops.slice(
            self.variables.tree, [0, 0], [-1, 1])), constants.LEAF_NODE)
        ).eval(session=session).shape[0]
    return TreeStats(num_nodes, num_leaves)
