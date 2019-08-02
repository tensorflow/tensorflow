# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import test


class CollectiveOpGPUTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testBasicNcclReduce(self):
    inputs = [[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
              [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3]]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_size = len(inputs)
    group_key = 1
    instance_key = 1
    # Configure virtual GPU devices
    device_type = 'GPU'
    virtual_devices = [config_pb2.GPUOptions.Experimental.VirtualDevices(
        memory_limit_mb=([1 << 10] * group_size))]  # 1 GB per virtual GPU
    gpu_options = config_pb2.GPUOptions(
        visible_device_list='0',
        experimental=config_pb2.GPUOptions.Experimental(
            virtual_devices=virtual_devices))
    # Configure NCCL
    experimental = config_pb2.ConfigProto.Experimental(collective_nccl=True)
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'
    config = config_pb2.ConfigProto(gpu_options=gpu_options,
                                    experimental=experimental)
    devices = ['/{}:{}'.format(device_type, i) for i in range(group_size)]

    with self.session(config=config) as sess:
      if not test_util.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      colred = []
      for i in range(group_size):
        with ops.device(devices[i]):
          tensor = constant_op.constant(inputs[i])
          colred.append(collective_ops.all_reduce(tensor, group_size, group_key,
                                                  instance_key, 'Add', 'Div'))
      run_options = config_pb2.RunOptions()
      results = sess.run(colred, options=run_options)
    for i in range(group_size):
      self.assertAllClose(results[i], expected, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  test.main()
