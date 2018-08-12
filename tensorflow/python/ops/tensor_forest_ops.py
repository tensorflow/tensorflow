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
"""Ops for tensor_forest."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import gen_tensor_forest_ops


class TreeVariableSavable(saver.BaseSaverBuilder.SaveableObject):
  """SaveableObject implementation for TreeVariable."""

  def __init__(self, params, tree_handle, stats_handle, create_op, name):
    """Creates a TreeVariableSavable object.

    Args:
      params: A TensorForestParams object.
      tree_handle: handle to the tree variable.
      stats_handle: handle to the stats variable.
      create_op: the op to initialize the variable.
      name: the name to save the tree variable under.
    """
    self.params = params
    tensor = gen_tensor_forest_ops.tree_serialize(tree_handle)
    # slice_spec is useful for saving a slice from a variable.
    # It's not meaningful the tree variable. So we just pass an empty value.
    slice_spec = ""
    specs = [saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name), ]
    super(TreeVariableSavable,
          self).__init__(tree_handle, specs, name)
    self._tree_handle = tree_handle
    self._create_op = create_op

  def restore(self, restored_tensors, unused_restored_shapes):
    """Restores the associated tree from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint.
      unused_restored_shapes: the shapes this object should conform to after
        restore. Not meaningful for trees.

    Returns:
      The operation that restores the state of the tree variable.
    """
    with ops.control_dependencies([self._create_op]):
      return gen_tensor_forest_ops.tree_deserialize(
          self._tree_handle,
          restored_tensors[0],
          params=self.params.serialized_params_proto)


def tree_variable(params, tree_config, stats_handle, name, container=None):
  r"""Creates a tree model and returns a handle to it.

  Args:
    params: A TensorForestParams object.
    tree_config: A `Tensor` of type `string`. Serialized proto of the tree.
    stats_handle: Resource handle to the stats object.
    name: A name for the variable.
    container: An optional `string`. Defaults to `""`.

  Returns:
    A `Tensor` of type mutable `string`. The handle to the tree.
  """
  with ops.name_scope(name, "TreeVariable") as name:
    resource_handle = gen_tensor_forest_ops.decision_tree_resource_handle_op(
        container, shared_name=name, name=name)

    create_op = gen_tensor_forest_ops.create_tree_variable(
        resource_handle,
        tree_config,
        params=params.serialized_params_proto)
    is_initialized_op = gen_tensor_forest_ops.tree_is_initialized_op(
        resource_handle)
    # Adds the variable to the savable list.
    saveable = TreeVariableSavable(params, resource_handle, stats_handle,
                                   create_op,
                                   resource_handle.name)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
    resources.register_resource(resource_handle, create_op, is_initialized_op)
    return resource_handle


class FertileStatsVariableSavable(saver.BaseSaverBuilder.SaveableObject):
  """SaveableObject implementation for FertileStatsVariable."""

  def __init__(self, params, stats_handle, create_op, name):
    """Creates a FertileStatsVariableSavable object.

    Args:
      params: A TensorForestParams object.
      stats_handle: handle to the tree variable.
      create_op: the op to initialize the variable.
      name: the name to save the tree variable under.
    """
    self.params = params
    tensor = gen_tensor_forest_ops.fertile_stats_serialize(
        stats_handle, params=params.serialized_params_proto)
    # slice_spec is useful for saving a slice from a variable.
    # It's not meaningful the tree variable. So we just pass an empty value.
    slice_spec = ""
    specs = [saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name), ]
    super(FertileStatsVariableSavable,
          self).__init__(stats_handle, specs, name)
    self._stats_handle = stats_handle
    self._create_op = create_op

  def restore(self, restored_tensors, unused_restored_shapes):
    """Restores the associated tree from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint.
      unused_restored_shapes: the shapes this object should conform to after
        restore. Not meaningful for trees.

    Returns:
      The operation that restores the state of the tree variable.
    """
    with ops.control_dependencies([self._create_op]):
      return gen_tensor_forest_ops.fertile_stats_deserialize(
          self._stats_handle, restored_tensors[0],
          params=self.params.serialized_params_proto)


def fertile_stats_variable(params, stats_config, name,
                           container=None):
  r"""Creates a stats object and returns a handle to it.

  Args:
    params: A TensorForestParams object.
    stats_config: A `Tensor` of type `string`. Serialized proto of the stats.
    name: A name for the variable.
    container: An optional `string`. Defaults to `""`.

  Returns:
    A `Tensor` of type mutable `string`. The handle to the stats.
  """
  with ops.name_scope(name, "FertileStatsVariable") as name:
    resource_handle = gen_tensor_forest_ops.fertile_stats_resource_handle_op(
        container, shared_name=name, name=name)

    create_op = gen_tensor_forest_ops.create_fertile_stats_variable(
        resource_handle, stats_config,
        params=params.serialized_params_proto)
    is_initialized_op = gen_tensor_forest_ops.fertile_stats_is_initialized_op(
        resource_handle)
    # Adds the variable to the savable list.
    saveable = FertileStatsVariableSavable(params, resource_handle, create_op,
                                           resource_handle.name)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
    resources.register_resource(resource_handle, create_op, is_initialized_op)
    return resource_handle


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
      self.stats = gen_tensor_forest_ops.fertile_stats_variable(
          params, tree_stat, self.get_tree_name('stats', tree_num))
    self.tree = gen_tensor_forest_ops.tree_variable(
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
               configs,
               tree_configs=None,
               tree_stats=None,
               device_assigner=None,
               variables=None,
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
          tree_data = array_ops.gather(
              processed_dense_features, gather_indices)
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
    return math_ops.reduce_mean(math_ops.to_float(array_ops.stack(sizes)))

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
    unused_epoch = math_ops.to_int32(get_epoch_variable())

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
