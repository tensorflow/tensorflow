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
"""Tests for Collective Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import test

# TODO(tucker): Make these ops work in eager mode. b/79776476


class CollectiveOpTest(test.TestCase):

  def _testCollectiveReduce(self, t0, t1, expected):
    group_key = 1
    instance_key = 1
    with self.test_session(
        config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
      with ops.device('/CPU:0'):
        in0 = constant_op.constant(t0)
        colred0 = collective_ops.all_reduce(in0, 2, group_key, instance_key,
                                            'Add', 'Div', [0])
      with ops.device('/CPU:1'):
        in1 = constant_op.constant(t1)
        colred1 = collective_ops.all_reduce(in1, 2, group_key, instance_key,
                                            'Add', 'Div', [0])
      run_options = config_pb2.RunOptions()
      run_options.experimental.collective_graph_key = 1
      results = sess.run([colred0, colred1], options=run_options)
    self.assertAllClose(results[0], expected, rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[1], expected, rtol=1e-5, atol=1e-5)

  def testCollectiveReduce(self):
    self._testCollectiveReduce([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
                               [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3],
                               [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2])

  def _testCollectiveBroadcast(self, t0):
    group_key = 1
    instance_key = 1
    with self.test_session(
        config=config_pb2.ConfigProto(device_count={'CPU': 2})) as sess:
      with ops.device('/CPU:0'):
        in0 = constant_op.constant(t0)
        out0 = collective_ops.broadcast_send(in0, in0.shape, in0.dtype,
                                             2, group_key, instance_key)
      with ops.device('/CPU:1'):
        c1 = constant_op.constant(t0)
        out1 = collective_ops.broadcast_recv(c1.shape, c1.dtype,
                                             2, group_key, instance_key)
      run_options = config_pb2.RunOptions()
      run_options.experimental.collective_graph_key = 1
      results = sess.run([out0, out1], options=run_options)
    self.assertAllClose(results[0], t0, rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[1], t0, rtol=1e-5, atol=1e-5)

  def testCollectiveBroadcast(self):
    self._testCollectiveBroadcast([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1])


if __name__ == '__main__':
  test.main()
