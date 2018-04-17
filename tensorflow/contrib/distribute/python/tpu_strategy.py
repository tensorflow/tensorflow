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
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops


# TODO(isaprykin):  Consider whether inheriting is really appropriate.
class TpuStrategy(one_device_strategy.OneDeviceStrategy):

  def __init__(self, master=None, iterations=None, model_dir=None):
    super(TpuStrategy, self).__init__('/cpu:0')

  def _call_for_each_tower(self, fn, *args, **kwargs):
    kwargs.pop('run_concurrently', None)

    # TODO(isaprykin): Give an API for many iterations per step.
    iterations = 1

    # TODO(isaprykin): Do not hard code shapes and input format :)
    # TODO(isaprykin): Detect the number of TPU cores automatically.

    def dequeueing_fn(*args, **kwargs):
      del args, kwargs
      x, = tpu.infeed_dequeue_tuple(dtypes=[dtypes.float32], shapes=[[1, 1, 1]])
      return fn(x)

    iterator = args[0]

    def infeed_input(i):
      """Get input, split it and then enqueue."""
      batches = iterator.get_next()
      batches = array_ops.split(batches, 2)

      infeeds = [
          tpu_ops.infeed_enqueue_tuple(
              inputs=[batches[j]], shapes=[[1, 1, 1]], device_ordinal=j)
          for j in range(2)
      ]

      with ops.control_dependencies(infeeds):
        return i + 1

    with ops.device('/task:0/device:CPU:0'):
      enqueue_ops = control_flow_ops.while_loop(
          lambda i: i < iterations,
          infeed_input, [constant_op.constant(0)],
          parallel_iterations=1)

    def iterate_on_tpu():
      return tpu.repeat(iterations, dequeueing_fn, [])

    with one_device_strategy._OneDeviceTowerContext(self):  # pylint: disable=protected-access
      tpu_result = tpu.batch_parallel(iterate_on_tpu, [], num_shards=2)

    return control_flow_ops.group(tpu_result, enqueue_ops)
