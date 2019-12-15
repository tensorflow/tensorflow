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

from tensorflow.python import ops
from tensorflow.python.ops import gen_tensor_forest_ops
from tensorflow.python.ops import resources
from tensorflow.python.training import saver


class TreeVariableSaveable(saver.BaseSaverBuilder.SaveableObject):
  """Resource that holds a tree."""

  def __init__(self, type_name, name, container, config, resource_handle_func,
               create_op_func, is_initialized_op_func, serialize_op_func,
               deserialize_op_func):

    with ops.name_scope(name, type_name) as name:
      self._resource_handle = resource_handle_func(
          container, shared_name=name, name=name)

    self._is_initialized_op = is_initialized_op_func(self._resource_handle)
    tensor = serialize_op_func(self._resource_handle)
    self._create_op = create_op_func(self._resource_handle, config)
    # slice_spec is useful for saving a slice from a variable.
    # It's not meaningful the tree variable. So we just pass an empty
    # value.
    slice_spec = ''
    specs = [saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name)]
    super(TreeVariableSaveable, self).__init__(self._resource_handle, specs,
                                               name)

    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self)

    resources.register_resource(self._resource_handle, self._create_op,
                                self._is_initialized_op)
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
    return self._resource_handle


def tree_variable(tree_config, name, container=None):
  return TreeVariableSaveable(
      'TreeVariable', name, container, tree_config,
      gen_tensor_forest_ops.tensor_forest_tree_resource_handle_op,
      gen_tensor_forest_ops.tensor_forest_create_tree_variable,
      gen_tensor_forest_ops.tensor_forest_tree_is_initialized_op,
      gen_tensor_forest_ops.tensor_forest_tree_serialize,
      gen_tensor_forest_ops.tensor_forest_tree_deserialize).resource


class ForestVariables(object):
  """Resource that holds all trees from a forest."""

  def __init__(self, params, tree_configs=None):

    self._variables = []

    for i in range(params.n_trees):
      tree_config = ''
      if tree_configs is not None:
        tree_config = tree_configs[i]
      self._variables.append(tree_variable(
          tree_config,
          'tree-%s' % i,
      ))

  def __getitem__(self, t):
    return self._variables[t]
