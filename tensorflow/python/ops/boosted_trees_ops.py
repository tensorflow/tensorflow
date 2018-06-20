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
"""Ops for boosted_trees."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import resources

# Re-exporting ops used by other modules.
# pylint: disable=unused-import
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_gains_per_feature as calculate_best_gains_per_feature
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_stats_summary as make_stats_summary
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_predict as predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_training_predict as training_predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_update_ensemble as update_ensemble
# pylint: enable=unused-import

from tensorflow.python.training import saver


class PruningMode(object):
  NO_PRUNING, PRE_PRUNING, POST_PRUNING = range(0, 3)


class _TreeEnsembleSavable(saver.BaseSaverBuilder.SaveableObject):
  """SaveableObject implementation for TreeEnsemble."""

  def __init__(self, resource_handle, create_op, name):
    """Creates a _TreeEnsembleSavable object.

    Args:
      resource_handle: handle to the decision tree ensemble variable.
      create_op: the op to initialize the variable.
      name: the name to save the tree ensemble variable under.
    """
    stamp_token, serialized = (
        gen_boosted_trees_ops.boosted_trees_serialize_ensemble(resource_handle))
    # slice_spec is useful for saving a slice from a variable.
    # It's not meaningful the tree ensemble variable. So we just pass an empty
    # value.
    slice_spec = ''
    specs = [
        saver.BaseSaverBuilder.SaveSpec(stamp_token, slice_spec,
                                        name + '_stamp'),
        saver.BaseSaverBuilder.SaveSpec(serialized, slice_spec,
                                        name + '_serialized'),
    ]
    super(_TreeEnsembleSavable, self).__init__(resource_handle, specs, name)
    self._resource_handle = resource_handle
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
      return gen_boosted_trees_ops.boosted_trees_deserialize_ensemble(
          self._resource_handle,
          stamp_token=restored_tensors[0],
          tree_ensemble_serialized=restored_tensors[1])


class TreeEnsemble(object):
  """Creates TreeEnsemble resource."""

  def __init__(self, name, stamp_token=0, is_local=False, serialized_proto=''):
    with ops.name_scope(name, 'TreeEnsemble') as name:
      self._resource_handle = (
          gen_boosted_trees_ops.boosted_trees_ensemble_resource_handle_op(
              container='', shared_name=name, name=name))
      create_op = gen_boosted_trees_ops.boosted_trees_create_ensemble(
          self.resource_handle,
          stamp_token,
          tree_ensemble_serialized=serialized_proto)
      is_initialized_op = (
          gen_boosted_trees_ops.is_boosted_trees_ensemble_initialized(
              self._resource_handle))
      # Adds the variable to the savable list.
      if not is_local:
        saveable = _TreeEnsembleSavable(self.resource_handle, create_op,
                                        self.resource_handle.name)
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
      resources.register_resource(
          self.resource_handle,
          create_op,
          is_initialized_op,
          is_shared=not is_local)

  @property
  def resource_handle(self):
    return self._resource_handle

  def get_stamp_token(self):
    """Returns the current stamp token of the resource."""
    stamp_token, _, _, _, _ = (
        gen_boosted_trees_ops.boosted_trees_get_ensemble_states(
            self.resource_handle))
    return stamp_token

  def get_states(self):
    """Returns states of the tree ensemble.

    Returns:
      stamp_token, num_trees, num_finalized_trees, num_attempted_layers and
      range of the nodes in the latest layer.
    """
    (stamp_token, num_trees, num_finalized_trees, num_attempted_layers,
     nodes_range) = (
         gen_boosted_trees_ops.boosted_trees_get_ensemble_states(
             self.resource_handle))
    # Use identity to give names.
    return (array_ops.identity(stamp_token, name='stamp_token'),
            array_ops.identity(num_trees, name='num_trees'),
            array_ops.identity(num_finalized_trees, name='num_finalized_trees'),
            array_ops.identity(
                num_attempted_layers, name='num_attempted_layers'),
            array_ops.identity(nodes_range, name='last_layer_nodes_range'))

  def serialize(self):
    """Serializes the ensemble into proto and returns the serialized proto.

    Returns:
      stamp_token: int64 scalar Tensor to denote the stamp of the resource.
      serialized_proto: string scalar Tensor of the serialized proto.
    """
    return gen_boosted_trees_ops.boosted_trees_serialize_ensemble(
        self.resource_handle)

  def deserialize(self, stamp_token, serialized_proto):
    """Deserialize the input proto and resets the ensemble from it.

    Args:
      stamp_token: int64 scalar Tensor to denote the stamp of the resource.
      serialized_proto: string scalar Tensor of the serialized proto.

    Returns:
      Operation (for dependencies).
    """
    return gen_boosted_trees_ops.boosted_trees_deserialize_ensemble(
        self.resource_handle, stamp_token, serialized_proto)
