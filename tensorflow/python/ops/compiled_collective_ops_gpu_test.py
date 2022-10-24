# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Collective Operations that require GPU."""

import os

from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import test


class CompiledCollectiveOpGPUTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    """Set group_size = num_gpus = 2 for all tests in this class."""
    super(CompiledCollectiveOpGPUTest, cls).setUpClass()
    # Group size is the number of devices in a group communicating collectively.
    # This will be passed into the collective ops in the tests below.
    cls._group_size = 2
    cls._devices = ['/device:GPU:{}'.format(i) for i in range(2)]
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'

  def _setup_context(self, num_gpus=2):
    context._reset_context()
    gpus = config.list_physical_devices('GPU')
    if len(gpus) < num_gpus:
      self.skipTest('Expected at least {} GPUs but found {} GPUs'.format(
          num_gpus, len(gpus)))
    context.ensure_initialized()

  def testCompiledAllReduce(self):
    self._setup_context()

    def all_reduce_sum(v):
      return collective_ops.all_reduce_v2(
          t=v,
          group_size=2,
          group_key=1,
          instance_key=1,
          merge_op='Add',
          final_op='Id')

    strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])

    @def_function.function(jit_compile=True)
    def f():
      return control_flow_ops.while_loop(
          lambda i, _: i < 5, lambda i, t: (i + 1, all_reduce_sum(t)),
          (array_ops.zeros([]), constant_op.constant(1.0)))

    @def_function.function
    def run():
      return strategy.run(f)

    _, reduce = strategy.experimental_local_results(run())[0]
    self.assertEqual(reduce.numpy(), 32.0)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
