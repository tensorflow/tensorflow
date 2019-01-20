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

from tensorflow.contrib.tensor_forest.python.ops import gen_model_ops

# pylint: disable=unused-import
from tensorflow.contrib.tensor_forest.python.ops.gen_model_ops import feature_usage_counts
from tensorflow.contrib.tensor_forest.python.ops.gen_model_ops import traverse_tree_v4
from tensorflow.contrib.tensor_forest.python.ops.gen_model_ops import tree_predictions_v4
from tensorflow.contrib.tensor_forest.python.ops.gen_model_ops import tree_size
from tensorflow.contrib.tensor_forest.python.ops.gen_model_ops import update_model_v4
# pylint: enable=unused-import

from tensorflow.contrib.util import loader
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import resources
from tensorflow.python.platform import resource_loader
from tensorflow.python.training import saver
from tensorflow.python.training.checkpointable import tracking


_model_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("_model_ops.so"))


ops.NotDifferentiable("TreeVariable")
ops.NotDifferentiable("TreeSerialize")
ops.NotDifferentiable("TreeDeserialize")
ops.NotDifferentiable("TreeSize")
ops.NotDifferentiable("TreePredictionsV4")
ops.NotDifferentiable("FeatureUsageCounts")


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
    tensor = gen_model_ops.tree_serialize(tree_handle)
    # slice_spec is useful for saving a slice from a variable.
    # It's not meaningful the tree variable. So we just pass an empty value.
    slice_spec = ""
    specs = [saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name),]
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
      return gen_model_ops.tree_deserialize(
          self._tree_handle,
          restored_tensors[0],
          params=self.params.serialized_params_proto)


class TreeVariable(tracking.TrackableResource):
  """A tree model."""

  def __init__(self, params, tree_config, stats_handle, name, container=None):
    self._params = params
    self._tree_config = tree_config
    self._stats_handle = stats_handle
    self._name = name
    self._container = container
    self._init_op = None
    super(TreeVariable, self).__init__()
    self._resource_handle = self.create_resource()

  def create_resource(self):
    if context.executing_eagerly():
      # TODO(allenl): This will leak memory due to kernel caching by the
      # shared_name attribute value (but is better than the alternative of
      # sharing everything by default when executing eagerly; hopefully creating
      # tables in a loop is uncommon).
      shared_name = "tree_variable_%d" % (ops.uid(),)
    else:
      shared_name = self._name
    return gen_model_ops.decision_tree_resource_handle_op(
        self._container, shared_name=shared_name, name=self._name)

  def initialize(self):
    return gen_model_ops.create_tree_variable(
        self.resource_handle,
        self._tree_config,
        params=self._params.serialized_params_proto)

  @property
  def initializer(self):
    if self._init_op is None:
      self._init_op = self.initialize()
    return self._init_op

  def is_initialized(self):
    return gen_model_ops.tree_is_initialized_op(self.resource_handle)

  def _gather_saveables_for_checkpoint(self):
    """For object-based checkpointing."""
    return {
        "tree_variable":
            functools.partial(
                TreeVariableSavable,
                params=self._params,
                tree_handle=self.resource_handle,
                stats_handle=self._stats_handle,
                create_op=self._init_op)
    }


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
    tree_var = TreeVariable(params, tree_config, stats_handle, name, container)
    resource_handle = tree_var.resource_handle
    create_op = tree_var.initializer
    is_initialized_op = tree_var.is_initialized()
    # Adds the variable to the savable list.
    saveable = tree_var._gather_saveables_for_checkpoint()["tree_variable"](  # pylint: disable=protected-access
        name=resource_handle.name)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
    resources.register_resource(resource_handle, create_op, is_initialized_op)
    return resource_handle
