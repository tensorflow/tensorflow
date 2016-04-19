# Copyright 2016 Google Inc. All Rights Reserved.
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

import tensorflow as tf

from tensorflow.contrib.tensor_forest.python.ops import inference_ops
from tensorflow.contrib.tensor_forest.python.ops import training_ops


# If tree[i][0] equals this value, then i is a leaf node.
LEAF_NODE = -1


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

  def __init__(self, num_trees=100, max_nodes=10000, bagging_fraction=1.0,
               samples_to_decide=25, max_depth=0, num_splits_to_consider=0,
               max_fertile_nodes=0, split_after_samples=0,
               valid_leaf_threshold=0, **kwargs):
    self.num_trees = num_trees
    self.max_nodes = max_nodes
    self.bagging_fraction = bagging_fraction
    self.samples_to_decide = samples_to_decide
    self.max_depth = max_depth
    self.num_splits_to_consider = num_splits_to_consider
    self.max_fertile_nodes = max_fertile_nodes
    self.split_after_samples = split_after_samples
    self.valid_leaf_threshold = valid_leaf_threshold

    for name, value in kwargs.items():
      setattr(self, name, value)

  def values(self):
    return self.__dict__

  def fill(self):
    """Intelligently sets any non-specific parameters."""
    # Fail fast if num_classes isn't set.
    _ = getattr(self, 'num_classes')

    self.training_library_base_dir = getattr(
        self, 'training_library_base_dir', '')
    self.inference_library_base_dir = getattr(
        self, 'inference_library_base_dir', '')

    # Allow each tree to be unbalanced by up to a factor of 2.
    self.max_depth = (self.max_depth or
                      int(2 * math.ceil(math.log(self.max_nodes, 2))))

    # The Random Forest literature recommends sqrt(# features) for
    # classification problems, and p/3 for regression problems.
    # TODO(thomaswc): Consider capping this for large number of features.
    self.num_splits_to_consider = (
        self.num_splits_to_consider or
        max(10, int(math.ceil(math.sqrt(self.num_features)))))

    # max_fertile_nodes doesn't effect performance, only training speed.
    # We therefore set it primarily based upon space considerations.
    # Each fertile node takes up num_splits_to_consider times as much
    # as space as a non-fertile node.  We want the fertile nodes to in
    # total only take up as much space as the non-fertile nodes, so
    num_fertile = int(math.ceil(self.max_nodes / self.num_splits_to_consider))
    # But always use at least 1000 accumulate slots.
    num_fertile = max(num_fertile, 1000)
    self.max_fertile_nodes = self.max_fertile_nodes or num_fertile
    # But it also never needs to be larger than the number of leaves,
    # which is max_nodes / 2.
    self.max_fertile_nodes = min(self.max_fertile_nodes,
                                 int(math.ceil(self.max_nodes / 2.0)))

    # split_after_samples and valid_leaf_threshold should be about the same.
    # Therefore, if either is set, use it to set the other.  Otherwise, fall
    # back on samples_to_decide.
    samples_to_decide = self.split_after_samples or self.samples_to_decide

    self.split_after_samples = self.split_after_samples or samples_to_decide
    self.valid_leaf_threshold = self.valid_leaf_threshold or samples_to_decide

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


# A simple container to hold the training variables for a single tree.
class TreeTrainingVariables(object):

  def __init__(self, params):
    self.tree = tf.Variable(
        [[-1, -1]] + [[-2, -1]] * (params.max_nodes - 1), name='tree')
    self.tree_thresholds = tf.Variable(
        [-1.0] * (params.max_nodes), name='tree_thresholds')
    self.tree_depths = tf.Variable(
        [1] * (params.max_nodes), name='tree_depths')
    self.end_of_tree = tf.Variable([1], name='end_of_tree')

    self.non_fertile_leaves = tf.Variable([0], name='non_fertile_leaves')
    self.non_fertile_leaf_scores = tf.Variable(
        [1.0], name='non_fertile_leaf_scores')

    self.node_to_accumulator_map = tf.Variable(
        [-1] * params.max_nodes, name='node_to_accumulator_map')

    self.candidate_split_features = tf.Variable(
        [[-1] * params.num_splits_to_consider] * params.max_fertile_nodes,
        name='candidate_split_features')
    self.candidate_split_thresholds = tf.Variable(
        [[0.0] * params.num_splits_to_consider] * params.max_fertile_nodes,
        name='candidate_split_thresholds')

    self.node_per_class_weights = tf.Variable(
        [[0.0] * params.num_classes] * params.max_nodes,
        name='node_per_class_weights')
    self.candidate_split_per_class_weights = tf.Variable(
        [[[0.0] * params.num_classes] * params.num_splits_to_consider] *
        params.max_fertile_nodes,
        name='candidate_split_per_class_weights')
    self.total_split_per_class_weights = tf.Variable(
        [[-1.0] * params.num_classes] * params.max_fertile_nodes,
        name='total_split_per_class_weights')


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

  def __init__(self, params, device_assigner):
    self.variables = []
    for i in range(params.num_trees):
      with tf.device(device_assigner.get_device(i)):
        self.variables.append(TreeTrainingVariables(params))

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
      dummy = tf.constant(0)
      self.cached = dummy.device

    return self.cached


class RandomForestGraphs(object):
  """Builds TF graphs for random forest training and inference."""

  def __init__(self, params, device_assigner=None, variables=None):
    self.params = params
    self.device_assigner = device_assigner or RandomForestDeviceAssigner()
    tf.logging.info('Constructing forest with params = ')
    tf.logging.info(self.params.__dict__)
    self.variables = variables or ForestTrainingVariables(
        self.params, device_assigner=self.device_assigner)
    self.trees = [
        RandomTreeGraphs(
            self.variables[i], self.params,
            training_ops.Load(self.params.training_library_base_dir),
            inference_ops.Load(self.params.inference_library_base_dir))
        for i in range(self.params.num_trees)]

  def training_graph(self, input_data, input_labels):
    """Constructs a TF graph for training a random forest.

    Args:
      input_data: A tensor or placeholder for input data.
      input_labels: A tensor or placeholder for labels associated with
        input_data.

    Returns:
      The last op in the random forest training graph.
    """
    tree_graphs = []
    for i in range(self.params.num_trees):
      with tf.device(self.device_assigner.get_device(i)):
        seed = self.params.base_random_seed
        if seed != 0:
          seed += i
        # If using bagging, randomly select some of the input.
        tree_data = input_data
        tree_labels = input_labels
        if self.params.bagging_fraction < 1.0:
          # TODO(thomaswc): This does sampling without replacment.  Consider
          # also allowing sampling with replacement as an option.
          batch_size = tf.slice(tf.shape(input_data), [0], [1])
          r = tf.random_uniform(batch_size, seed=seed)
          mask = tf.less(r, tf.ones_like(r) * self.params.bagging_fraction)
          gather_indices = tf.squeeze(tf.where(mask), squeeze_dims=[1])
          # TODO(thomaswc): Calculate out-of-bag data and labels, and store
          # them for use in calculating statistics later.
          tree_data = tf.gather(input_data, gather_indices)
          tree_labels = tf.gather(input_labels, gather_indices)
        tree_graphs.append(
            self.trees[i].training_graph(tree_data, tree_labels, seed))
    return tf.group(*tree_graphs)

  def inference_graph(self, input_data):
    """Constructs a TF graph for evaluating a random forest.

    Args:
      input_data: A tensor or placeholder for input data.

    Returns:
      The last op in the random forest inference graph.
    """
    probabilities = []
    for i in range(self.params.num_trees):
      with tf.device(self.device_assigner.get_device(i)):
        probabilities.append(self.trees[i].inference_graph(input_data))
    with tf.device(self.device_assigner.get_device(0)):
      all_predict = tf.pack(probabilities)
      return tf.reduce_sum(all_predict, 0) / self.params.num_trees

  def average_size(self):
    """Constructs a TF graph for evaluating the average size of a forest.

    Returns:
      The average number of nodes over the trees.
    """
    sizes = []
    for i in range(self.params.num_trees):
      with tf.device(self.device_assigner.get_device(i)):
        sizes.append(self.trees[i].size())
    return tf.reduce_mean(tf.pack(sizes))

  def average_impurity(self):
    """Constructs a TF graph for evaluating the leaf impurity of a forest.

    Returns:
      The last op in the graph.
    """
    impurities = []
    for i in range(self.params.num_trees):
      with tf.device(self.device_assigner.get_device(i)):
        impurities.append(self.trees[i].average_impurity())
    return tf.reduce_mean(tf.pack(impurities))

  def get_stats(self, session):
    tree_stats = []
    for i in range(self.params.num_trees):
      with tf.device(self.device_assigner.get_device(i)):
        tree_stats.append(self.trees[i].get_stats(session))
    return ForestStats(tree_stats, self.params)


class RandomTreeGraphs(object):
  """Builds TF graphs for random tree training and inference."""

  def __init__(self, variables, params, t_ops, i_ops):
    self.training_ops = t_ops
    self.inference_ops = i_ops
    self.variables = variables
    self.params = params

  def _gini(self, class_counts):
    """Calculate the Gini impurity.

    If c(i) denotes the i-th class count and c = sum_i c(i) then
      score = 1 - sum_i ( c(i) / c )^2

    Args:
      class_counts: A 2-D tensor of per-class counts, from either
        candidate_split_per_class_weights or total_split_per_class_weights.

    Returns:
      A 1-D tensor of the Gini impurities for each row in the input.
    """
    smoothed = 1.0 + class_counts
    sums = tf.reduce_sum(smoothed, 1)
    sum_squares = tf.reduce_sum(tf.square(smoothed), 1)

    return 1.0 - sum_squares / (sums * sums)

  def _weighted_gini(self, class_counts):
    """Our split score is the Gini impurity times the number of examples.

    If c(i) denotes the i-th class count and c = sum_i c(i) then
      score = c * (1 - sum_i ( c(i) / c )^2 )
            = c - sum_i c(i)^2 / c
    Args:
      class_counts: A 2-D tensor of per-class counts, from either
        candidate_split_per_class_weights or total_split_per_class_weights.

    Returns:
      A 1-D tensor of the Gini impurities for each row in the input.
    """
    smoothed = 1.0 + class_counts
    sums = tf.reduce_sum(smoothed, 1)
    sum_squares = tf.reduce_sum(tf.square(smoothed), 1)

    return sums - sum_squares / sums

  def training_graph(self, input_data, input_labels, random_seed):
    """Constructs a TF graph for training a random tree.

    Args:
      input_data: A tensor or placeholder for input data.
      input_labels: A tensor or placeholder for labels associated with
        input_data.
      random_seed: The random number generator seed to use for this tree.  0
        means use the current time as the seed.

    Returns:
      The last op in the random tree training graph.
    """
    # Count extremely random stats.
    (pcw_node_delta, pcw_splits_indices, pcw_splits_delta, pcw_totals_indices,
     pcw_totals_delta, input_leaves) = (
         self.training_ops.count_extremely_random_stats(
             input_data, input_labels, self.variables.tree,
             self.variables.tree_thresholds,
             self.variables.node_to_accumulator_map,
             self.variables.candidate_split_features,
             self.variables.candidate_split_thresholds,
             num_classes=self.params.num_classes))
    node_update_op = tf.assign_add(self.variables.node_per_class_weights,
                                   pcw_node_delta)
    candidate_update_op = self.training_ops.scatter_add_ndim(
        self.variables.candidate_split_per_class_weights,
        pcw_splits_indices, pcw_splits_delta)

    totals_update_op = self.training_ops.scatter_add_ndim(
        self.variables.total_split_per_class_weights, pcw_totals_indices,
        pcw_totals_delta)

    # Sample inputs.
    update_indices, feature_updates, threshold_updates = (
        self.training_ops.sample_inputs(
            input_data, self.variables.node_to_accumulator_map,
            input_leaves, self.variables.candidate_split_features,
            self.variables.candidate_split_thresholds,
            split_initializations_per_input=(
                self.params.split_initializations_per_input),
            split_sampling_random_seed=random_seed))
    update_features_op = tf.scatter_update(
        self.variables.candidate_split_features, update_indices,
        feature_updates)
    update_thresholds_op = tf.scatter_update(
        self.variables.candidate_split_thresholds, update_indices,
        threshold_updates)

    # Calculate finished nodes.
    with tf.control_dependencies([totals_update_op]):
      children = tf.squeeze(tf.slice(self.variables.tree, [0, 0], [-1, 1]),
                            squeeze_dims=[1])
      is_leaf = tf.equal(LEAF_NODE, children)
      leaves = tf.to_int32(tf.squeeze(tf.where(is_leaf), squeeze_dims=[1]))
      finished = self.training_ops.finished_nodes(
          leaves, self.variables.node_to_accumulator_map,
          self.variables.total_split_per_class_weights,
          num_split_after_samples=self.params.split_after_samples)

    # Update leaf scores.
    # TODO(gilberth): Optimize this. It currently calculates counts for
    # every non-fertile leaf.
    with tf.control_dependencies([node_update_op]):
      def f1():
        return self.variables.non_fertile_leaf_scores

      def f2():
        counts = tf.gather(self.variables.node_per_class_weights,
                           self.variables.non_fertile_leaves)
        new_scores = self._weighted_gini(counts)
        return tf.assign(self.variables.non_fertile_leaf_scores, new_scores)

      # Because we can't have tf.self.variables of size 0, we have to put in a
      # garbage value of -1 in there.  Here we check for that so we don't
      # try to index into node_per_class_weights in a tf.gather with a negative
      # number.
      update_nonfertile_leaves_scores_op = tf.cond(tf.less(
          self.variables.non_fertile_leaves[0], 0), f1, f2)

    # Calculate best splits.
    with tf.control_dependencies([candidate_update_op, totals_update_op]):
      split_indices = self.training_ops.best_splits(
          finished, self.variables.node_to_accumulator_map,
          self.variables.candidate_split_per_class_weights,
          self.variables.total_split_per_class_weights)

    # Grow tree.
    with tf.control_dependencies([update_features_op, update_thresholds_op]):
      (tree_update_indices, tree_children_updates,
       tree_threshold_updates, tree_depth_updates, new_eot) = (
           self.training_ops.grow_tree(
               self.variables.end_of_tree, self.variables.tree_depths,
               self.variables.node_to_accumulator_map, finished, split_indices,
               self.variables.candidate_split_features,
               self.variables.candidate_split_thresholds))
      tree_update_op = tf.scatter_update(
          self.variables.tree, tree_update_indices, tree_children_updates)
      threhsolds_update_op = tf.scatter_update(
          self.variables.tree_thresholds, tree_update_indices,
          tree_threshold_updates)
      depth_update_op = tf.scatter_update(
          self.variables.tree_depths, tree_update_indices, tree_depth_updates)

    # Update fertile slots.
    with tf.control_dependencies([update_nonfertile_leaves_scores_op,
                                  depth_update_op]):
      (node_map_updates, accumulators_cleared, accumulators_allocated,
       new_nonfertile_leaves, new_nonfertile_leaves_scores) = (
           self.training_ops.update_fertile_slots(
               finished, self.variables.non_fertile_leaves,
               self.variables.non_fertile_leaf_scores,
               self.variables.end_of_tree, self.variables.tree_depths,
               self.variables.candidate_split_per_class_weights,
               self.variables.total_split_per_class_weights,
               self.variables.node_to_accumulator_map,
               max_depth=self.params.max_depth))

    # Ensure end_of_tree doesn't get updated until UpdateFertileSlots has
    # used it to calculate new leaves.
    gated_new_eot, = tf.tuple([new_eot], control_inputs=[new_nonfertile_leaves])
    eot_update_op = tf.assign(self.variables.end_of_tree, gated_new_eot)

    updates = []
    updates.append(eot_update_op)
    updates.append(tree_update_op)
    updates.append(threhsolds_update_op)
    updates.append(tf.assign(
        self.variables.non_fertile_leaves, new_nonfertile_leaves,
        validate_shape=False))
    updates.append(tf.assign(
        self.variables.non_fertile_leaf_scores,
        new_nonfertile_leaves_scores, validate_shape=False))

    updates.append(tf.scatter_update(
        self.variables.node_to_accumulator_map,
        tf.squeeze(tf.slice(node_map_updates, [0, 0], [1, -1]),
                   squeeze_dims=[0]),
        tf.squeeze(tf.slice(node_map_updates, [1, 0], [1, -1]),
                   squeeze_dims=[0])))

    cleared_and_allocated_accumulators = tf.concat(
        0, [accumulators_cleared, accumulators_allocated])
    # Calculate values to put into scatter update for candidate counts.
    # Candidate split counts are always reset back to 0 for both cleared
    # and allocated accumulators. This means some accumulators might be doubly
    # reset to 0 if the were released and not allocated, then later allocated.
    candidate_pcw_values = tf.tile(
        tf.expand_dims(tf.expand_dims(
            tf.zeros_like(cleared_and_allocated_accumulators, dtype=tf.float32),
            1), 2),
        [1, self.params.num_splits_to_consider, self.params.num_classes])
    updates.append(tf.scatter_update(
        self.variables.candidate_split_per_class_weights,
        cleared_and_allocated_accumulators, candidate_pcw_values))

    # Calculate values to put into scatter update for total counts.
    total_cleared = tf.tile(
        tf.expand_dims(
            tf.neg(tf.ones_like(accumulators_cleared, dtype=tf.float32)), 1),
        [1, self.params.num_classes])
    total_reset = tf.tile(
        tf.expand_dims(
            tf.zeros_like(accumulators_allocated, dtype=tf.float32), 1),
        [1, self.params.num_classes])
    total_pcw_updates = tf.concat(0, [total_cleared, total_reset])
    updates.append(tf.scatter_update(
        self.variables.total_split_per_class_weights,
        cleared_and_allocated_accumulators, total_pcw_updates))

    # Calculate values to put into scatter update for candidate splits.
    split_features_updates = tf.tile(
        tf.expand_dims(
            tf.neg(tf.ones_like(cleared_and_allocated_accumulators)), 1),
        [1, self.params.num_splits_to_consider])
    updates.append(tf.scatter_update(
        self.variables.candidate_split_features,
        cleared_and_allocated_accumulators, split_features_updates))

    return tf.group(*updates)

  def inference_graph(self, input_data):
    """Constructs a TF graph for evaluating a random tree.

    Args:
      input_data: A tensor or placeholder for input data.

    Returns:
      The last op in the random tree inference graph.
    """
    return self.inference_ops.tree_predictions(
        input_data, self.variables.tree, self.variables.tree_thresholds,
        self.variables.node_per_class_weights,
        valid_leaf_threshold=self.params.valid_leaf_threshold)

  def average_impurity(self):
    """Constructs a TF graph for evaluating the average leaf impurity of a tree.

    Returns:
      The last op in the graph.
    """
    children = tf.squeeze(tf.slice(self.variables.tree, [0, 0], [-1, 1]),
                          squeeze_dims=[1])
    is_leaf = tf.equal(LEAF_NODE, children)
    leaves = tf.to_int32(tf.squeeze(tf.where(is_leaf), squeeze_dims=[1]))
    counts = tf.gather(self.variables.node_per_class_weights, leaves)
    impurity = self._weighted_gini(counts)
    return tf.reduce_sum(impurity) / tf.reduce_sum(counts + 1.0)

  def size(self):
    """Constructs a TF graph for evaluating the current number of nodes.

    Returns:
      The current number of nodes in the tree.
    """
    return self.variables.end_of_tree - 1

  def get_stats(self, session):
    num_nodes = self.variables.end_of_tree.eval(session=session) - 1
    num_leaves = tf.where(
        tf.equal(tf.squeeze(tf.slice(self.variables.tree, [0, 0], [-1, 1])),
                 LEAF_NODE)).eval(session=session).shape[0]
    return TreeStats(num_nodes, num_leaves)
