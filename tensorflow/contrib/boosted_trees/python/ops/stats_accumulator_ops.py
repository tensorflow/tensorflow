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
from tensorflow.contrib.boosted_trees.python.ops import gen_stats_accumulator_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import resources
from tensorflow.python.platform import resource_loader
from tensorflow.python.training import saver

# Pattern to remove all non alpha numeric from a string.
_PATTERN = re.compile(r"[\W_]+")


class StatsAccumulator(saver.BaseSaverBuilder.SaveableObject):
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
    if name is not None:
      name = _PATTERN.sub("", name)
    with ops.name_scope(name, "StatsAccumulator") as name:
      # Both values are scalars.
      if (gradient_shape == tensor_shape.scalar() and
          hessian_shape == tensor_shape.scalar()):
        self._is_scalar = True
        self._resource_handle = (gen_stats_accumulator_ops.
                                 stats_accumulator_scalar_resource_handle_op(
                                     container, name, name=name))

        create_op = gen_stats_accumulator_ops.create_stats_accumulator_scalar(
            self._resource_handle, stamp_token)
        is_initialized_op = (
            gen_stats_accumulator_ops.stats_accumulator_scalar_is_initialized(
                self._resource_handle))
      else:
        self._is_scalar = False
        self._resource_handle = (gen_stats_accumulator_ops.
                                 stats_accumulator_tensor_resource_handle_op(
                                     container, name, name=name))
        create_op = gen_stats_accumulator_ops.create_stats_accumulator_tensor(
            self._resource_handle, stamp_token, gradient_shape.as_list(),
            hessian_shape.as_list())
        is_initialized_op = (
            gen_stats_accumulator_ops.stats_accumulator_tensor_is_initialized(
                self._resource_handle))

    self._create_op = create_op
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

    super(StatsAccumulator, self).__init__(self._resource_handle, specs, name)
    resources.register_resource(self._resource_handle, create_op,
                                is_initialized_op)
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self)

  def add(self, stamp_token, partition_ids, feature_ids, gradients, hessians):
    """Updates the stats accumulator."""
    partition_ids, feature_ids, gradients, hessians = (self._make_summary(
        partition_ids, feature_ids, gradients, hessians))
    if self._is_scalar:
      return gen_stats_accumulator_ops.stats_accumulator_scalar_add(
          [self._resource_handle], stamp_token, [partition_ids], [feature_ids],
          [gradients], [hessians])
    else:
      return gen_stats_accumulator_ops.stats_accumulator_tensor_add(
          [self._resource_handle], stamp_token, [partition_ids], [feature_ids],
          [gradients], [hessians])

  def schedule_add(self, partition_ids, feature_ids, gradients, hessians):
    """Schedules an update to the stats accumulator."""
    partition_ids, feature_ids, gradients, hessians = (self._make_summary(
        partition_ids, feature_ids, gradients, hessians))
    if self._is_scalar:
      return batch_ops_utils.ScheduledStampedResourceOp(
          op=gen_stats_accumulator_ops.stats_accumulator_scalar_add,
          resource_handle=self._resource_handle,
          partition_ids=partition_ids,
          feature_ids=feature_ids,
          gradients=gradients,
          hessians=hessians)
    else:
      return batch_ops_utils.ScheduledStampedResourceOp(
          op=gen_stats_accumulator_ops.stats_accumulator_tensor_add,
          resource_handle=self._resource_handle,
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

  def flush(self, stamp_token, next_stamp_token):
    """Flushes the stats accumulator."""
    if self._is_scalar:
      return gen_stats_accumulator_ops.stats_accumulator_scalar_flush(
          self._resource_handle, stamp_token, next_stamp_token)
    else:
      return gen_stats_accumulator_ops.stats_accumulator_tensor_flush(
          self._resource_handle, stamp_token, next_stamp_token)

  def serialize(self):
    """Serializes the stats accumulator state."""
    if self._is_scalar:
      return gen_stats_accumulator_ops.stats_accumulator_scalar_serialize(
          self._resource_handle)
    else:
      return gen_stats_accumulator_ops.stats_accumulator_tensor_serialize(
          self._resource_handle)

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

  def resource(self):
    return self._resource_handle


# Conditionally load ops, they might already be statically linked in.
try:
  _stats_accumulator_ops = loader.load_op_library(
      resource_loader.get_path_to_datafile("_stats_accumulator_ops.so"))
except (errors.NotFoundError, IOError):
  print("Error loading _stats_accumulator_ops.so")
