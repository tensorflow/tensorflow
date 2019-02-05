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
"""Stats Accumulator ops python wrappers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from tensorflow.contrib.boosted_trees.python.ops import batch_ops_utils
# pylint: disable=unused-import
from tensorflow.contrib.boosted_trees.python.ops import boosted_trees_ops_loader
# pylint: enable=unused-import
from tensorflow.contrib.boosted_trees.python.ops import gen_stats_accumulator_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import resources
from tensorflow.python.training import saver
from tensorflow.python.training.checkpointable import tracking

# Pattern to remove all non alpha numeric from a string.
_PATTERN = re.compile(r"[\W_]+")


class StatsAccumulatorSaveable(saver.BaseSaverBuilder.SaveableObject):
  """SaveableObject implementation for StatsAccumulator."""

  def __init__(self, resource_handle, create_op, is_scalar, name):
    self._create_op = create_op
    self._resource_handle = resource_handle
    self._is_scalar = is_scalar
    slice_spec = ""
    saver_name = self._resource_handle.name
    (stamp_token, num_updates, partition_ids, feature_ids, gradients,
     hessians) = self.serialize()
    specs = [
        saver.BaseSaverBuilder.SaveSpec(stamp_token, slice_spec,
                                        saver_name + "_stamp"),
        saver.BaseSaverBuilder.SaveSpec(num_updates, slice_spec,
                                        saver_name + "_num_updates"),
        saver.BaseSaverBuilder.SaveSpec(partition_ids, slice_spec,
                                        saver_name + "_partition_ids"),
        saver.BaseSaverBuilder.SaveSpec(feature_ids, slice_spec,
                                        saver_name + "_feature_ids"),
        saver.BaseSaverBuilder.SaveSpec(gradients, slice_spec,
                                        saver_name + "_gradients"),
        saver.BaseSaverBuilder.SaveSpec(hessians, slice_spec,
                                        saver_name + "hessians"),
    ]
    super(StatsAccumulatorSaveable, self).__init__(self._resource_handle, specs,
                                                   name)

  def serialize(self):
    """Serializes the stats accumulator state."""
    if self._is_scalar:
      return gen_stats_accumulator_ops.stats_accumulator_scalar_serialize(
          self._resource_handle)
    else:
      return gen_stats_accumulator_ops.stats_accumulator_tensor_serialize(
          self._resource_handle)

  def deserialize(self, stamp_token, num_updates, partition_ids, feature_ids,
                  gradients, hessians):
    """Resets the stats accumulator with the serialized state."""
    if self._is_scalar:
      return gen_stats_accumulator_ops.stats_accumulator_scalar_deserialize(
          self._resource_handle, stamp_token, num_updates, partition_ids,
          feature_ids, gradients, hessians)
    else:
      return gen_stats_accumulator_ops.stats_accumulator_tensor_deserialize(
          self._resource_handle, stamp_token, num_updates, partition_ids,
          feature_ids, gradients, hessians)

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
      return self.deserialize(
          stamp_token=restored_tensors[0],
          num_updates=restored_tensors[1],
          partition_ids=restored_tensors[2],
          feature_ids=restored_tensors[3],
          gradients=restored_tensors[4],
          hessians=restored_tensors[5])


class StatsAccumulator(tracking.TrackableResource):
  """A resource that allows to accumulate gradients and hessians.

  For consistency guarantees, we use read and write stamp tokens.
  The stamp token on the resource is updated with StatsAccumulator.flush.
  Calls to StatsAccumulator.add that don't provide the current stamp token are
  ignored.
  """

  def __init__(self,
               stamp_token,
               gradient_shape,
               hessian_shape,
               name=None,
               container=None):
    """Creates a stats accumulator and returns a handle to it.

    Args:
      stamp_token: An int64, initial value to use for the stamp token.
      gradient_shape: A TensorShape, containing shape of gradients.
      hessian_shape: A TensorShape, containing shape of hessians.
      name: A name for the stats accumulator variable.
      container: An optional `string`. Defaults to `""`.

    Returns:
      A `Tensor` of type mutable `string`. The handle to the stats accumulator.
    """
    self._stamp_token = stamp_token
    self._gradient_shape = gradient_shape
    self._hessian_shape = hessian_shape
    self._container = container

    if (gradient_shape == tensor_shape.scalar() and
        hessian_shape == tensor_shape.scalar()):
      self._is_scalar = True
    else:
      self._is_scalar = False

    if name is not None:
      name = _PATTERN.sub("", name)
    with ops.name_scope(name, "StatsAccumulator") as name:
      self._name = name
      self._resource_handle = self.create_resource()
      self._init_op = self.initialize()
      is_initialized_op = self.is_initialized()
    resources.register_resource(self.resource_handle, self.initializer,
                                is_initialized_op)
    self._saveable = StatsAccumulatorSaveable(
        self.resource_handle, self.initializer, self._is_scalar, name)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self._saveable)

  def create_resource(self):
    if self._is_scalar:
      return (
          gen_stats_accumulator_ops.stats_accumulator_scalar_resource_handle_op(
              self._container, self._name, name=self._name))
    else:
      return (
          gen_stats_accumulator_ops.stats_accumulator_tensor_resource_handle_op(
              self._container, self._name, name=self._name))

  def initialize(self):
    if self._is_scalar:
      return gen_stats_accumulator_ops.create_stats_accumulator_scalar(
          self.resource_handle, self._stamp_token)
    else:
      return gen_stats_accumulator_ops.create_stats_accumulator_tensor(
          self.resource_handle, self._stamp_token,
          self._gradient_shape.as_list(), self._hessian_shape.as_list())

  @property
  def initializer(self):
    if self._init_op is None:
      self._init_op = self.initialize()
    return self._init_op

  def is_initialized(self):
    if self._is_scalar:
      return gen_stats_accumulator_ops.stats_accumulator_scalar_is_initialized(
          self.resource_handle)
    else:
      return gen_stats_accumulator_ops.stats_accumulator_tensor_is_initialized(
          self.resource_handle)

  @property
  def saveable(self):
    return self._saveable

  def _gather_saveables_for_checkpoint(self):
    return {"stats_accumulator", self.saveable}

  def add(self, stamp_token, partition_ids, feature_ids, gradients, hessians):
    """Updates the stats accumulator."""
    partition_ids, feature_ids, gradients, hessians = (self._make_summary(
        partition_ids, feature_ids, gradients, hessians))
    if self._is_scalar:
      return gen_stats_accumulator_ops.stats_accumulator_scalar_add(
          [self.resource_handle], stamp_token, [partition_ids], [feature_ids],
          [gradients], [hessians])
    else:
      return gen_stats_accumulator_ops.stats_accumulator_tensor_add(
          [self.resource_handle], stamp_token, [partition_ids], [feature_ids],
          [gradients], [hessians])

  def schedule_add(self, partition_ids, feature_ids, gradients, hessians):
    """Schedules an update to the stats accumulator."""
    partition_ids, feature_ids, gradients, hessians = (self._make_summary(
        partition_ids, feature_ids, gradients, hessians))
    if self._is_scalar:
      return batch_ops_utils.ScheduledStampedResourceOp(
          op=gen_stats_accumulator_ops.stats_accumulator_scalar_add,
          resource_handle=self.resource_handle,
          partition_ids=partition_ids,
          feature_ids=feature_ids,
          gradients=gradients,
          hessians=hessians)
    else:
      return batch_ops_utils.ScheduledStampedResourceOp(
          op=gen_stats_accumulator_ops.stats_accumulator_tensor_add,
          resource_handle=self.resource_handle,
          partition_ids=partition_ids,
          feature_ids=feature_ids,
          gradients=gradients,
          hessians=hessians)

  def _make_summary(self, partition_ids, feature_ids, gradients, hessians):
    if self._is_scalar:
      return gen_stats_accumulator_ops.stats_accumulator_scalar_make_summary(
          partition_ids, feature_ids, gradients, hessians)
    else:
      return gen_stats_accumulator_ops.stats_accumulator_tensor_make_summary(
          partition_ids, feature_ids, gradients, hessians)

  def flush(self, stamp_token, next_stamp_token):
    """Flushes the stats accumulator."""
    if self._is_scalar:
      return gen_stats_accumulator_ops.stats_accumulator_scalar_flush(
          self.resource_handle, stamp_token, next_stamp_token)
    else:
      return gen_stats_accumulator_ops.stats_accumulator_tensor_flush(
          self.resource_handle, stamp_token, next_stamp_token)
