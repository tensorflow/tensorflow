# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for control_flow_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import standard_ops as tf
from tensorflow.python.platform import googletest
from tensorflow.python.training import momentum

from tensorflow.contrib.immediate.python.immediate import test_util
import tensorflow.contrib.immediate as immediate

env = immediate.Env({"tf": tf, "control_flow_ops": control_flow_ops,
                     "embedding_ops": embedding_ops})
tf = env.tf
control_flow_ops = env.control_flow_ops
embedding_ops = env.embedding_ops


class ShapeTestCase(test_util.TensorFlowTestCase):

  def testShape(self):
    with ops.Graph().as_default():
      tensor = tf.constant([1.0, 2.0])
      self.assertEquals([2], tensor.get_shape())


class SwitchTestCase(test_util.TensorFlowTestCase):

  def testIndexedSlicesWithDenseShape(self):
    with self.test_session():
      data = ops.IndexedSlices(tf.constant([1, 2, 3]),
                               tf.constant([0, 1]),
                               dense_shape=tf.constant([3]))
      zero = tf.constant(0)
      one = tf.constant(1)
      less_op = tf.less(zero, one)
      switch_false, switch_true = control_flow_ops.switch(data, less_op)
      self.assertAllEqual([1, 2, 3], switch_true.values.eval())
      self.assertAllEqual([0, 1], switch_true.indices.eval())


if __name__ == "__main__":
  googletest.main()
