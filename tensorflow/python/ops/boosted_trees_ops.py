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
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_bucketize
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_calculate_best_gains_per_feature as calculate_best_gains_per_feature
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_center_bias as center_bias
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_create_quantile_stream_resource as create_quantile_stream_resource
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_example_debug_outputs as example_debug_outputs
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_quantile_summaries as make_quantile_summaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_make_stats_summary as make_stats_summary
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_predict as predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_add_summaries as quantile_add_summaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_deserialize as quantile_resource_deserialize
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_flush as quantile_flush
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_get_bucket_boundaries as get_bucket_boundaries
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_quantile_stream_resource_handle_op as quantile_resource_handle_op
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_training_predict as training_predict
from tensorflow.python.ops.gen_boosted_trees_ops import boosted_trees_update_ensemble as update_ensemble
from tensorflow.python.ops.gen_boosted_trees_ops import is_boosted_trees_quantile_stream_resource_initialized as is_quantile_resource_initialized
# pylint: enable=unused-import

from tensorflow.python.training import saver
from tensorflow.python.training.checkpointable import tracking


class PruningMode(object):
  """Class for working with Pruning modes."""
  NO_PRUNING, PRE_PRUNING, POST_PRUNING = range(0, 3)

  _map = {'none': NO_PRUNING, 'pre': PRE_PRUNING, 'post': POST_PRUNING}

  @classmethod
  def from_str(cls, mode):
    if mode in cls._map:
      return cls._map[mode]
    else:
      raise ValueError('pruning_mode mode must be one of: {}'.format(', '.join(
          sorted(cls._map))))


class QuantileAccumulatorSaveable(saver.BaseSaverBuilder.SaveableObject):
  """SaveableObject implementation for QuantileAccumulator."""

  def __init__(self, resource_handle, create_op, num_streams, name):
    self._resource_handle = resource_handle
    self._num_streams = num_streams
    self._create_op = create_op
    bucket_boundaries = get_bucket_boundaries(self._resource_handle,
                                              self._num_streams)
    slice_spec = ''
    specs = []

    def make_save_spec(tensor, suffix):
      return saver.BaseSaverBuilder.SaveSpec(tensor, slice_spec, name + suffix)

    for i in range(self._num_streams):
      specs += [
          make_save_spec(bucket_boundaries[i], '_bucket_boundaries_' + str(i))
      ]
    super(QuantileAccumulatorSaveable, self).__init__(self._resource_handle,
                                                      specs, name)

  def restore(self, restored_tensors, unused_tensor_shapes):
    bucket_boundaries = restored_tensors
    with ops.control_dependencies([self._create_op]):
      return quantile_resource_deserialize(
          self._resource_handle, bucket_boundaries=bucket_boundaries)


class QuantileAccumulator(tracking.TrackableResource):
  """SaveableObject implementation for QuantileAccumulator.

     The bucket boundaries are serialized and deserialized from checkpointing.
  """

  def __init__(self,
               epsilon,
               num_streams,
               num_quantiles,
               name=None,
               max_elements=None):
    self._eps = epsilon
    self._num_streams = num_streams
    self._num_quantiles = num_quantiles
    super(QuantileAccumulator, self).__init__()

    with ops.name_scope(name, 'QuantileAccumulator') as name:
      self._name = name
      self._resource_handle = self.create_resource()
      self._init_op = self.initialize()
      is_initialized_op = self.is_initialized()
    resources.register_resource(self.resource_handle, self._init_op,
                                is_initialized_op)
    self._saveable = QuantileAccumulatorSaveable(
        self.resource_handle, self._init_op, self._num_streams,
        self.resource_handle.name)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self._saveable)

  def create_resource(self):
    return quantile_resource_handle_op(
        container='', shared_name=self._name, name=self._name)

  def initialize(self):
    return create_quantile_stream_resource(self.resource_handle, self._eps,
                                           self._num_streams)

  @property
  def initializer(self):
    if self._init_op is None:
      self._init_op = self.initialize()
    return self._init_op

  def is_initialized(self):
    return is_quantile_resource_initialized(self.resource_handle)

  @property
  def saveable(self):
    return self._saveable

  def _gather_saveables_for_checkpoint(self):
    return {'quantile_accumulator', self._saveable}

  def add_summaries(self, float_columns, example_weights):
    summaries = make_quantile_summaries(float_columns, example_weights,
                                        self._eps)
    summary_op = quantile_add_summaries(self.resource_handle, summaries)
    return summary_op

  def flush(self):
    return quantile_flush(self.resource_handle, self._num_quantiles)

  def get_bucket_boundaries(self):
    return get_bucket_boundaries(self.resource_handle, self._num_streams)


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


class TreeEnsemble(tracking.TrackableResource):
  """Creates TreeEnsemble resource."""

  def __init__(self, name, stamp_token=0, is_local=False, serialized_proto=''):
    self._stamp_token = stamp_token
    self._serialized_proto = serialized_proto
    self._is_local = is_local
    with ops.name_scope(name, 'TreeEnsemble') as name:
      self._name = name
      self._resource_handle = self.create_resource()
      self._init_op = self.initialize()
      is_initialized_op = self.is_initialized()
      # Adds the variable to the savable list.
      if not is_local:
        self._saveable = _TreeEnsembleSavable(
            self.resource_handle, self.initializer, self.resource_handle.name)
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self._saveable)
      resources.register_resource(
          self.resource_handle,
          self.initializer,
          is_initialized_op,
          is_shared=not is_local)

  def create_resource(self):
    return gen_boosted_trees_ops.boosted_trees_ensemble_resource_handle_op(
        container='', shared_name=self._name, name=self._name)

  def initialize(self):
    return gen_boosted_trees_ops.boosted_trees_create_ensemble(
        self.resource_handle,
        self._stamp_token,
        tree_ensemble_serialized=self._serialized_proto)

  @property
  def initializer(self):
    if self._init_op is None:
      self._init_op = self.initialize()
    return self._init_op

  def is_initialized(self):
    return gen_boosted_trees_ops.is_boosted_trees_ensemble_initialized(
        self.resource_handle)

  def _gather_saveables_for_checkpoint(self):
    if not self._is_local:
      return {'tree_ensemble': self._saveable}

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
