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
"""TPU Distribution Strategy.

This is experimental.  It's not ready for general use.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import tpu
from tensorflow.contrib.distribute.python import one_device_strategy
from tensorflow.contrib.distribute.python import values
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops


# TODO(isaprykin):  Consider whether inheriting is really appropriate.
class TPUStrategy(one_device_strategy.OneDeviceStrategy):
  """Experimental TPU distribution strategy implementation."""

  def __init__(self,
               global_batch_size=2,
               num_cores_per_host=2,
               iterations_per_step=2):
    # TODO(isaprykin): Generalize the defaults.
    super(TPUStrategy, self).__init__('/cpu:0')
    # TODO(isaprykin): Auto-detect number of cores and hosts.
    self._num_cores_per_host = num_cores_per_host
    self._global_batch_size = global_batch_size
    # TODO(isaprykin): This might have to be per-call.
    self._iterations_per_step = iterations_per_step

  def distribute_dataset(self, dataset_fn):
    return values.PerIterationDataset(
        self._call_dataset_fn(dataset_fn), self._iterations_per_step)

  def _call_for_each_tower(self, fn, *args, **kwargs):
    kwargs.pop('run_concurrently', None)

    # TODO(isaprykin): Support variable arguments similar to PerDevice+regroup.
    inputs = args[0]

    sharded_shape = [None]  # Python 2 nonlocal.

    def infeed_input(i):
      """Get input, split it and then enqueue."""
      batches = array_ops.gather(inputs, i)

      # TODO(isaprykin):  Handle partial batch.
      global_shape = [self._global_batch_size] + list(batches.get_shape())[1:]
      sharded_shape[0] = ([self._global_batch_size / self._num_cores_per_host] +
                          list(global_shape)[1:])

      batches.set_shape(global_shape)
      batches = array_ops.split(batches, self._num_cores_per_host)

      infeeds = [
          tpu_ops.infeed_enqueue_tuple(
              inputs=[batches[j]], shapes=[sharded_shape[0]], device_ordinal=j)
          for j in range(self._num_cores_per_host)
      ]

      with ops.control_dependencies(infeeds):
        return i + 1

    with ops.device('/task:0/device:CPU:0'):
      enqueue_ops = control_flow_ops.while_loop(
          lambda i: i < self._iterations_per_step,
          infeed_input, [constant_op.constant(0)],
          parallel_iterations=1)

    assert sharded_shape[0]

    def dequeueing_fn(*args, **kwargs):
      del args, kwargs
      x, = tpu.infeed_dequeue_tuple(
          dtypes=[dtypes.float32], shapes=[sharded_shape[0]])
      return fn(x)

    def iterate_on_tpu():
      return tpu.repeat(self._iterations_per_step, dequeueing_fn, [])

    with one_device_strategy._OneDeviceTowerContext(self):  # pylint: disable=protected-access
      tpu_result = tpu.batch_parallel(
          iterate_on_tpu, [], num_shards=self._num_cores_per_host)

    return control_flow_ops.group(tpu_result, enqueue_ops)
