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

import itertools

from tensorflow.contrib import tpu
from tensorflow.contrib.distribute.python import one_device_strategy
from tensorflow.contrib.distribute.python import values
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest


class TPUStrategy(one_device_strategy.OneDeviceStrategy):
  """Experimental TPU distribution strategy implementation."""

  def __init__(self,
               num_cores_per_host=2,
               iterations_per_step=2):
    # TODO(isaprykin): Generalize the defaults.  They are currently tailored for
    # the unit test.
    super(TPUStrategy, self).__init__('/cpu:0')
    # TODO(isaprykin): Auto-detect number of cores and hosts.
    self._num_cores_per_host = num_cores_per_host
    # TODO(isaprykin): This might have to be per-call.
    self._iterations_per_step = iterations_per_step

  def distribute_dataset(self, dataset_fn):
    return values.PerIterationDataset(
        self._call_dataset_fn(dataset_fn), self._iterations_per_step,
        self._num_cores_per_host)

  def _call_for_each_tower(self, fn, *args, **kwargs):
    kwargs.pop('run_concurrently', None)

    inputs = {'args': args, 'kwargs': kwargs}
    flat_inputs = nest.flatten(inputs)

    feed_mask = [isinstance(f, values.PerIteration) for f in flat_inputs]

    feeds = lambda: itertools.compress(flat_inputs, feed_mask)
    shapes = [f.get_shape() for f in feeds()]
    if any([not s.is_fully_defined() for s in shapes]):
      raise ValueError(
          'TPU currently requires fully defined shapes. Either use '
          'set_shape() on the input tensors or use '
          'dataset.apply(map_and_batch(..., drop_remainder=True)).')
    types = [f.get_dtype() for f in feeds()]

    def infeed_input(i):
      """Get input, split it and then enqueue."""
      iteration_inputs = [f.get(i) for f in feeds()]
      infeed_inputs = [[inputs_per_core[core_id]
                        for inputs_per_core in iteration_inputs]
                       for core_id in range(self._num_cores_per_host)]

      infeed_ops = []
      for core_id, infeed_input in enumerate(infeed_inputs):
        infeed_ops.append(
            tpu_ops.infeed_enqueue_tuple(
                inputs=infeed_input, shapes=shapes, device_ordinal=core_id))

      with ops.control_dependencies(infeed_ops):
        return i + 1

    with ops.device('/task:0/device:CPU:0'):
      enqueue_ops = control_flow_ops.while_loop(
          lambda i: i < self._iterations_per_step,
          infeed_input, [constant_op.constant(0)],
          parallel_iterations=1)

    def dequeueing_fn(*args, **kwargs):
      """Dequeue input arguments and supply them to `fn`."""
      del args, kwargs
      dequeued = tpu.infeed_dequeue_tuple(dtypes=types, shapes=shapes)
      dequeued = iter(dequeued)

      fn_inputs = []
      for inp, is_feed in zip(flat_inputs, feed_mask):
        if is_feed:
          fn_inputs.append(next(dequeued))
        else:
          fn_inputs.append(inp)

      fn_inputs = nest.pack_sequence_as(inputs, fn_inputs)
      return fn(*fn_inputs['args'], **fn_inputs['kwargs'])

    def iterate_on_tpu():
      return tpu.repeat(self._iterations_per_step, dequeueing_fn, [])

    with one_device_strategy._OneDeviceTowerContext(self):  # pylint: disable=protected-access
      tpu_result = tpu.batch_parallel(
          iterate_on_tpu, [], num_shards=self._num_cores_per_host)

    return control_flow_ops.group(tpu_result, enqueue_ops)

  def _reduce(self, method_string, value, destinations):
    del destinations  # TPU is graph mode only.  Rely on implicit Send/Recv.
    if method_string == 'mean':
      # TODO(jhseu):  Revisit once we support model-parallelism.
      value *= (1. / self._num_cores_per_host)
    return tpu_ops.cross_replica_sum(value)

  @property
  def num_towers(self):
    return self._num_cores_per_host
