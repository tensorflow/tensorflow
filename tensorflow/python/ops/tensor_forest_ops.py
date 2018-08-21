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


class VariableSavable(saver.BaseSaverBuilder.SaveableObject):
  """SaveableObject implementation for Variable."""

  def __init__(self, leaf_model_type, num_output, config, resource_handle_func, create_op_func,
               is_initialized_op_func, serialize_op_func, deserialize_op_func, name, type_name):

    with ops.name_scope(name, type_name) as name:
      self._resource_handle = resource_handle_func(
          container, shared_name=name, name=name)

    self._is_initialized_op = is_initialized_op_func(self._resource_handle)
    tensor = serialize_op_func(self._resource_handle)
    self._create_op = create_op_func(
        self._resource_handle,
        config,
        leaf_model_type=leaf_model_type,
        num_output=num_output)
    # slice_spec is useful for saving a slice from a variable.
    # It's not meaningful the tree variable. So we just pass an empty value.
    slice_spec = ""
    specs = [saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name), ]
    super(VariableSavable,
          self).__init__(self._resource_handle, specs, name)

    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self)

    resources.register_resource(
        self._resource_handle, self._create_op, self._is_initialized_op)
    self._deserialize_op_func = deserialize_op_func

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
      )

  @property
  def resource(self):
      return self._resource_hanlde


def tree_variable(leaf_model_type, num_output, tree_config, name, container=None):
  return VariableSavable(leaf_model_type,
                          num_output,
                          tree_config,
                          name,
                          container,
                          "TreeVariable",
                          gen_tensor_forest_ops.decision_tree_resource_handle_op,
                          gen_tensor_forest_ops.create_tree_variable,
                          gen_tensor_forest_ops.tree_serialize,
                          gen_tensor_forest_ops.tree_deserialize,
                          gen_tensor_forest_ops.tree_is_initialized_op).resource


def fertile_stats_variable(leaf_model_type, num_output, stats_config, name, container=None):
  return VariableSavable(leaf_model_type,
                          num_output,
                          stats_config,
                          name,
                          container,
                          "FertileStatsVariable",
                          gen_tensor_forest_ops.fertile_stats_resource_handle_op,
                          gen_tensor_forest_ops.create_fertile_stats_variable,
                          gen_tensor_forest_ops.fertile_stats_serialize,
                          gen_tensor_forest_ops.fertile_stats_deserialize,
                          gen_tensor_forest_ops.fertile_stats_is_initialized_op).resource


class DecisionTreeVariables(object):
        """Stores tf.Variables for training a single desision tree.

  Uses tf.get_variable to get tree-specific names so that this can be used
  with a implementation (trains a model, saves it,
  then relies on restoring that model to evaluate).
  """

  def __init__(self, leaf_model_type, num_output, tree_num, tree_config='', tree_stat=''):
    self.stats = fertile_stats_variable(
        leaf_model_type, num_output, tree_stat, self.get_tree_name('stats', tree_num))
    self.tree = tree_variable(
        leaf_model_type, num_output, tree_config, self.get_tree_name('tree', tree_num))

  def get_tree_name(self, name, num):
    return '{0}-{1}'.format(name, num)


class ForestVariables(object):
"""A container for a forests training data, consisting of multiple trees.

  Instantiates a DesicionTreeVariables object for each tree. We override the
  __getitem__ and __setitem__ function so that usage looks like this:

    forest_variables = ForestVariables(params)

    ... forest_variables.tree ...
"""

  def __init__(self, leaf_model_type, num_output, params
                     tree_configs=None, tree_stats=None):
    self.variables = []
    # Set up some scalar variables to run through the device assigner, then
    # we can use those to colocate everything related to a tree.

    for i in range(params.num_trees):
      kwargs = {}
      if tree_configs is not None:
        kwargs.update(dict(tree_config=tree_configs[i]))
      if tree_stats is not None:
        kwargs.update(dict(tree_stat=tree_stats[i]))

      self.variables.append(DecisionTreeVariables(
        leaf_model_type, num_output, i, **kwargs))

  def __setitem__(self, t, val):
    self.variables[t] = val

  def __getitem__(self, t):
    return self.variables[t]
