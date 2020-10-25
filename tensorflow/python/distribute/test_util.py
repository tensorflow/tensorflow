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

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest


def gather(strategy, value):
  """Gathers value from all workers.

  This is intended for tests before we implement an official all-gather API.

  Args:
    strategy: a `tf.distribute.Strategy`.
    value: a nested structure of n-dim `tf.distribute.DistributedValue` of
      `tf.Tensor`, or of a `tf.Tensor` if the strategy only has one replica.
      Cannot contain tf.sparse.SparseTensor.

  Returns:
    a (n+1)-dim `tf.Tensor`.
  """
  return nest.map_structure(functools.partial(_gather, strategy), value)


def _gather(strategy, value):
  """Gathers a single value."""
  # pylint: disable=protected-access
  if not isinstance(value, values.DistributedValues):
    value = values.PerReplica([ops.convert_to_tensor(value)])
  if not isinstance(strategy.extended,
                    collective_all_reduce_strategy.CollectiveAllReduceExtended):
    return array_ops.stack(value._values)
  assert len(strategy.extended.worker_devices) == len(value._values)
  inputs = [array_ops.expand_dims_v2(v, axis=0) for v in value._values]
  return strategy.gather(values.PerReplica(inputs), axis=0)
  # pylint: enable=protected-access


def set_logical_devices_to_at_least(device, num):
  """Create logical devices of at least a given number."""
  if num < 1:
    raise ValueError("`num` must be at least 1 not %r" % (num,))
  physical_devices = config.list_physical_devices(device)
  if not physical_devices:
    raise RuntimeError("No {} found".format(device))
  if len(physical_devices) >= num:
    return
  # By default each physical device corresponds to one logical device. We create
  # multiple logical devices for the last physical device so that we have `num`
  # logical devices.
  num = num - len(physical_devices) + 1
  logical_devices = []
  for _ in range(num):
    if device.upper() == "GPU":
      logical_devices.append(
          context.LogicalDeviceConfiguration(memory_limit=2048))
    else:
      logical_devices.append(context.LogicalDeviceConfiguration())
  # Create logical devices from the the last device since sometimes the first
  # GPU is the primary graphic card and may has less memory available.
  config.set_logical_device_configuration(physical_devices[-1], logical_devices)


def main(enable_v2_behavior=True):
  """All-in-one main function for tf.distribute tests."""
  if enable_v2_behavior:
    v2_compat.enable_v2_behavior()
  else:
    v2_compat.disable_v2_behavior()
  # TODO(b/131360402): configure default logical devices.
  multi_process_runner.test_main()
