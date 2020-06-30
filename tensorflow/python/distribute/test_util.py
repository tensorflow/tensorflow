# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.eager import def_function
from tensorflow.python.ops import array_ops
from tensorflow.python.types import core
from tensorflow.python.util import nest


def gather(strategy, value):
  """Gathers value from all workers.

  This is intended for tests before we implement an official all-gather API.

  Args:
    strategy: a `tf.distribute.Strategy`.
    value: a nested structure of n-dim `tf.distribute.DistributedValue` of
      `tf.Tensor`, or of a `tf.Tensor` if the strategy only has one replica.

  Returns:
    a (n+1)-dim `tf.Tensor`.
  """
  return nest.map_structure(functools.partial(_gather, strategy), value)


def _gather(strategy, value):
  """Gathers a single value."""
  # pylint: disable=protected-access
  if not isinstance(value, values.DistributedValues):
    assert isinstance(value, core.Tensor)
    value = values.PerReplica([value])
  if not isinstance(strategy.extended,
                    collective_all_reduce_strategy.CollectiveAllReduceExtended):
    return array_ops.stack(value._values)
  assert len(strategy.extended.worker_devices) == len(value._values)
  inputs = [array_ops.expand_dims_v2(v, axis=0) for v in value._values]
  collective_keys = strategy.extended._collective_keys
  devices = strategy.extended.worker_devices
  group_size = strategy.num_replicas_in_sync

  @def_function.function
  def gather_fn():
    gathered = cross_device_utils.build_collective_gather(
        inputs, devices, group_size, collective_keys)
    return distribute_utils.update_regroup(
        strategy.extended, gathered, group=True)

  return gather_fn()
  # pylint: enable=protected-access
