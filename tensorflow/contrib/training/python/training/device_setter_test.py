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

import tensorflow as tf


class GreedyLoadBalancingStrategyTest(tf.test.TestCase):
  _cluster_spec = tf.train.ClusterSpec({
      "ps": ["ps0:2222", "ps1:2222"],
      "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]})

  def testUniformLoadEqualsRoundRobin(self):
    def _load_fn(unused_op):
      return 1

    with tf.device(tf.train.replica_device_setter(
        cluster=self._cluster_spec,
        ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
            2, _load_fn))):
      u = tf.Variable(tf.zeros([2, 2]))
      v = tf.Variable(tf.zeros([2, 1]))
      w = tf.Variable(tf.zeros([2, 2]))
      x = tf.Variable(tf.zeros([1, 3]))
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
    with tf.device(tf.train.replica_device_setter(
        cluster=self._cluster_spec,
        ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
            2, tf.contrib.training.byte_size_load_fn))):
      u = tf.Variable(tf.zeros([2, 2]))
      v = tf.Variable(tf.zeros([2, 1]))
      w = tf.Variable(tf.zeros([2, 2]))
      x = tf.Variable(tf.zeros([1, 3]))
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
    with tf.device(tf.train.replica_device_setter(
        cluster=self._cluster_spec,
        ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
            2, tf.contrib.training.byte_size_load_fn))):
      # Note: we must test the load function as part of the device function
      # instead of passing u.op to the function directly, because the only
      # time that the output Tensor has unknown shape for scalars is during
      # Variable construction.
      u = tf.Variable(0)
      self.assertDeviceEqual("/job:ps/task:0", u.device)
      self.assertDeviceEqual("/job:ps/task:0", u.initializer.device)

if __name__ == "__main__":
  tf.test.main()
