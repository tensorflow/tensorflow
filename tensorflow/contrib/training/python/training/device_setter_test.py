# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.contrib.training.device_setter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.training.python.training import device_setter as device_setter_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.training import server_lib


class GreedyLoadBalancingStrategyTest(test.TestCase):
  _cluster_spec = server_lib.ClusterSpec({
      "ps": ["ps0:2222", "ps1:2222"],
      "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
  })

  def testUniformLoadEqualsRoundRobin(self):

    def _load_fn(unused_op):
      return 1

    with ops.device(
        device_setter.replica_device_setter(
            cluster=self._cluster_spec,
            ps_strategy=device_setter_lib.GreedyLoadBalancingStrategy(
                2, _load_fn))):
      u = variables.Variable(array_ops.zeros([2, 2]))
      v = variables.Variable(array_ops.zeros([2, 1]))
      w = variables.Variable(array_ops.zeros([2, 2]))
      x = variables.Variable(array_ops.zeros([1, 3]))
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", u.device)
      self.assertDeviceEqual("/job:ps/task:0", u.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", v.device)
      self.assertDeviceEqual("/job:ps/task:1", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:0", w.device)
      self.assertDeviceEqual("/job:ps/task:0", w.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", x.device)
      self.assertDeviceEqual("/job:ps/task:1", x.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  def testByteSizeLoadFn(self):
    with ops.device(
        device_setter.replica_device_setter(
            cluster=self._cluster_spec,
            ps_strategy=device_setter_lib.GreedyLoadBalancingStrategy(
                2, device_setter_lib.byte_size_load_fn))):
      u = variables.Variable(array_ops.zeros([2, 2]))
      v = variables.Variable(array_ops.zeros([2, 1]))
      w = variables.Variable(array_ops.zeros([2, 2]))
      x = variables.Variable(array_ops.zeros([1, 3]))
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", u.device)
      self.assertDeviceEqual("/job:ps/task:0", u.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", v.device)
      self.assertDeviceEqual("/job:ps/task:1", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:ps/task:0", x.device)
      self.assertDeviceEqual("/job:ps/task:0", x.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  def testByteSizeLoadFnWithScalar(self):
    with ops.device(
        device_setter.replica_device_setter(
            cluster=self._cluster_spec,
            ps_strategy=device_setter_lib.GreedyLoadBalancingStrategy(
                2, device_setter_lib.byte_size_load_fn))):
      # Note: we must test the load function as part of the device function
      # instead of passing u.op to the function directly, because the only
      # time that the output Tensor has unknown shape for scalars is during
      # Variable construction.
      u = variables.Variable(0)
      self.assertDeviceEqual("/job:ps/task:0", u.device)
      self.assertDeviceEqual("/job:ps/task:0", u.initializer.device)


if __name__ == "__main__":
  test.main()
