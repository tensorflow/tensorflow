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
"""Basic tests for TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensorrt.python import trt_convert
# pylint: disable=unused-import
from tensorflow.contrib.tensorrt.python.ops import trt_engine_op
# pylint: enable=unused-import
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import googletest


class NoDeviceAssignmentTest(googletest.TestCase):

  def testNoDeviceAssignment(self):
    """Test that conversion should succeed when device is not specified."""
    if not trt_convert.is_tensorrt_enabled():
      return
    sess = session.Session()  # By default this will consume all the gpu memory.
    used_bytes = 0
    for device in sess.list_devices():
      if 'GPU:0' in device.name:
        used_bytes = device.memory_limit_bytes
    self.assertGreater(used_bytes, 0)

    input_dims = [100, 24, 24, 2]
    g = ops.Graph()
    with g.as_default():
      inp = array_ops.placeholder(
          dtype=dtypes.float32, shape=input_dims, name='input')
      for i in range(2):
        mul = inp * inp
        inp = mul + inp
      array_ops.squeeze(inp, name='output')

    trt_gdef = trt_convert.create_inference_graph(
        input_graph_def=g.as_graph_def(),
        outputs=['output'],
        max_batch_size=input_dims[0],
        # Use half of the allocated memory. It will fail if the converter
        # fallback to use native cudaMalloc(), so here it tests that converter
        # doesn't fallback.
        max_workspace_size_bytes=used_bytes // 4,
        minimum_segment_size=2,
        is_dynamic_op=False)
    self.assertEqual(1,
                     sum([node.op == 'TRTEngineOp' for node in trt_gdef.node]))


if __name__ == '__main__':
  test.main()
