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
"""Model ops python wrappers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# pylint: disable=unused-import
from tensorflow.contrib.boosted_trees.python.ops import boosted_trees_ops_loader
# pylint: enable=unused-import
from tensorflow.contrib.boosted_trees.python.ops import gen_model_ops
from tensorflow.contrib.boosted_trees.python.ops.gen_model_ops import tree_ensemble_deserialize
from tensorflow.contrib.boosted_trees.python.ops.gen_model_ops import tree_ensemble_serialize
# pylint: disable=unused-import
from tensorflow.contrib.boosted_trees.python.ops.gen_model_ops import tree_ensemble_stamp_token
from tensorflow.contrib.boosted_trees.python.ops.gen_model_ops import tree_ensemble_used_handlers
# pylint: enable=unused-import

from tensorflow.python.framework import ops
from tensorflow.python.ops import resources
from tensorflow.python.training import saver
from tensorflow.python.training.checkpointable import tracking

ops.NotDifferentiable("TreeEnsembleVariable")
ops.NotDifferentiable("TreeEnsembleSerialize")
ops.NotDifferentiable("TreeEnsembleDeserialize")


class TreeEnsembleVariableSavable(saver.BaseSaverBuilder.SaveableObject):
  """SaveableObject implementation for TreeEnsembleVariable."""

  def __init__(self, tree_ensemble_handle, create_op, name):
    """Creates a TreeEnsembleVariableSavable object.

    Args:
      tree_ensemble_handle: handle to the tree ensemble variable.
      create_op: the op to initialize the variable.
      name: the name to save the tree ensemble variable under.
    """
    stamp_token, ensemble_config = tree_ensemble_serialize(tree_ensemble_handle)
    # slice_spec is useful for saving a slice from a variable.
    # It's not meaningful the tree ensemble variable. So we just pass an empty
    # value.
    slice_spec = ""
    specs = [
        saver.BaseSaverBuilder.SaveSpec(stamp_token, slice_spec,
                                        name + "_stamp"),
        saver.BaseSaverBuilder.SaveSpec(ensemble_config, slice_spec,
                                        name + "_config"),
    ]
    super(TreeEnsembleVariableSavable,
          self).__init__(tree_ensemble_handle, specs, name)
    self._tree_ensemble_handle = tree_ensemble_handle
    self._create_op = create_op

  def restore(self, restored_tensors, unused_restored_shapes):
    """Restores the associated tree ensemble from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint.
      unused_restored_shapes: the shapes this object should conform to after
        restore. Not meaningful for trees.

    Returns:
      The operation that restores the state of the tree ensemble variable.
    """
    with ops.control_dependencies([self._create_op]):
      return tree_ensemble_deserialize(
          self._tree_ensemble_handle,
          stamp_token=restored_tensors[0],
          tree_ensemble_config=restored_tensors[1])


class TreeEnsembleVariable(tracking.TrackableResource):
  """A Tree ensemble model."""

  def __init__(self, stamp_token, tree_ensemble_config, name, container=None):
    self._stamp_token = stamp_token
    self._tree_ensemble_config = tree_ensemble_config
    self._name = name
    self._container = container
    self._init_op = None
    super(TreeEnsembleVariable, self).__init__()

  def create_resource(self):
    return gen_model_ops.decision_tree_ensemble_resource_handle_op(
        self._container, shared_name=self._name, name=self._name)

  def initialize(self):
    return gen_model_ops.create_tree_ensemble_variable(
        self.resource_handle, self._stamp_token, self._tree_ensemble_config)

  @property
  def initializer(self):
    if self._init_op is None:
      self._init_op = self.initialize()
    return self._init_op

  def is_initialized(self):
    return gen_model_ops.tree_ensemble_is_initialized_op(self.resource_handle)

  def _gather_saveables_for_checkpoint(self):
    return {
        "tree_ensemble_variable":
            functools.partial(
                TreeEnsembleVariableSavable,
                tree_ensemble_handle=self.resource_handle,
                create_op=self.initializer)
    }


def tree_ensemble_variable(stamp_token,
                           tree_ensemble_config,
                           name,
                           container=None):
  r"""Creates a tree ensemble model and returns a handle to it.

  Args:
    stamp_token: The initial stamp token value for the ensemble resource.
    tree_ensemble_config: A `Tensor` of type `string`.
      Serialized proto of the tree ensemble.
    name: A name for the ensemble variable.
    container: An optional `string`. Defaults to `""`.

  Returns:
    A `Tensor` of type mutable `string`. The handle to the tree ensemble.
  """
  with ops.name_scope(name, "TreeEnsembleVariable") as name:
    tree_ensemble_var = TreeEnsembleVariable(stamp_token, tree_ensemble_config,
                                             name, container)
    resource_handle = tree_ensemble_var.resource_handle
    create_op = tree_ensemble_var.initializer
    is_initialized_op = tree_ensemble_var.is_initialized()
    # Adds the variable to the savable list.
    saveable = TreeEnsembleVariableSavable(resource_handle, create_op,
                                           resource_handle.name)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
    resources.register_resource(resource_handle, create_op, is_initialized_op)
    return resource_handle
