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

from tensorflow.python.ops import resources

from tensorflow.python import ops
from tensorflow.python.ops import gen_tensor_forest_ops

from tensorflow.python.training import saver

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


def build_params_proto(params):
  proto = gen_ops.TensorForestParams()
  proto.num_trees = params.num_trees
  proto.max_nodes = params.max_nodes
  proto.is_regression = params.is_regression
  proto.num_outputs = params.num_classes
  proto.num_features = params.num_features

  proto.leaf_type = params.leaf_model_type
  proto.stats_type = params.stats_model_type
  proto.collection_type = _params_proto.COLLECTION_BASIC
  proto.finish_type.type = _params_proto.SPLIT_FINISH_BASIC

  proto.split_after_samples = 250
  proto.num_splits_to_consider = 0

  return proto.SerializeToString()


class _VariableSavable(saver.BaseSaverBuilder.SaveableObject):
  """SaveableObject implementation for Variable."""

  def __init__(self, params, config, resource_handle, create_op_func,
               is_initialized_op_func, serialize_op_func, deserialize_op_func, name):

    self.params = params
    self._resource_handle = resource_handle
    tensor = serialize_op_func(resource_handle)
    is_initialized_op = is_initialized_op_func(resource_handle)
    create_op = create_op_func(
        resource_handle,
        config,
        params=params.serialized_params_proto)
    # slice_spec is useful for saving a slice from a variable.
    # It's not meaningful the tree variable. So we just pass an empty value.
    slice_spec = ""
    specs = [saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name), ]
    super(VariableSavable,
          self).__init__(resource_handle, specs, name)
    self._create_op = create_op_func(
        resource_handle,
        config,
        params=params.serialized_params_proto)
    self._deserialize_op_func = deserialize_op_func
    self._resource_handle = resource_handle

    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self)
    resources.register_resource(resource_handle, create_op, is_initialized_op)

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
      return self._deserialize_op_func(
          self._resource_handle,
          restored_tensors[0],
          params=self.params.serialized_params_proto)

  @property
  def resource(self):
      return self._resource_hanlde


def _variable(params,
              config,
              name,
              container,
              type_name,
              resource_handle_op_func,
              create_op_func,
              serialize_op_func,
              deserialize_op_func,
              is_initialized_op_func):

  with ops.name_scope(name, type_name) as name:
    resource_handle = resource_handle_op_func(
        container, shared_name=name, name=name)

    create_op = create_op_func(
        resource_handle,
        config,
        params=params.serialized_params_proto)
    is_initialized_op = is_initialized_op_func(resource_handle)
    # Adds the variable to the savable list.
    saveable = VariableSavable(params, resource_handle,
                               create_op,
                               serialize_op_func,
                               deserialize_op_func,
                               resource_handle.name)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
    resources.register_resource(resource_handle, create_op, is_initialized_op)
    return resource_handle


def tree_variable(params, tree_config, name, container=None):
    return _variable(params,
                     tree_config,
                     name,
                     container,
                     "TreeVariable",
                     gen_tensor_forest_ops.decision_tree_resource_handle_op,
                     gen_tensor_forest_ops.create_tree_variable,
                     gen_tensor_forest_ops.tree_serialize,
                     gen_tensor_forest_ops.tree_deserialize,
                     gen_tensor_forest_ops.tree_is_initialized_op)


def fertile_stats_variable(params, stats_config, name, container=None):
    return _variable(params,
                     stats_config,
                     name,
                     container,
                     "FertileStatsVariable",
                     gen_tensor_forest_ops.fertile_stats_resource_handle_op,
                     gen_tensor_forest_ops.create_fertile_stats_variable,
                     gen_tensor_forest_ops.fertile_stats_serialize,
                     gen_tensor_forest_ops.fertile_stats_deserialize,
                     gen_tensor_forest_ops.fertile_stats_is_initialized_op)


class DecisionTreeVariables(object):
  """Stores tf.Variables for training a single desision tree.

  Uses tf.get_variable to get tree-specific names so that this can be used
  with a implementation (trains a model, saves it,
  then relies on restoring that model to evaluate).
  """

  def __init__(self, params, tree_num, training, tree_config='', tree_stat=''):
    if (not hasattr(params, 'params_proto') or
        not isinstance(params.params_proto,
                       tensor_forest_pb2.TensorForestParams)):
        raise Exception("TensorForestParams required")

    params.serialized_params_proto = params.params_proto.SerializeToString()
    self.stats = None
    if training:
      self.stats = fertile_stats_variable(
          params, tree_stat, self.get_tree_name('stats', tree_num))
    self.tree = tree_variable(
        params, tree_config, self.get_tree_name('tree', tree_num))

  def get_tree_name(self, name, num):
    return '{0}-{1}'.format(name, num)


class ForestVariables(object):
  """A container for a forests training data, consisting of multiple trees.

  Instantiates a DesicionTreeVariables object for each tree. We override the
  __getitem__ and __setitem__ function so that usage looks like this:

    forest_variables = ForestVariables(params)

    ... forest_variables.tree ...
  """

  def __init__(self, params, device_assigner, training=True,
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
        self.variables.append(DecisionTreeVariables(
            params, i, training, **kwargs))

  def __setitem__(self, t, val):
    self.variables[t] = val

  def __getitem__(self, t):
    return self.variables[t]
