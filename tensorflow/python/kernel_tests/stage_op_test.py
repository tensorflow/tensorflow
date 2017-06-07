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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class StageTest(test.TestCase):

  def testSimple(self):
    with self.test_session(use_gpu=True) as sess:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device('/gpu:0'):
        stager = data_flow_ops.StagingArea([dtypes.float32])
        stage = stager.put([v])
        y = stager.get()
        y = math_ops.reduce_max(math_ops.matmul(y, y))
      sess.run(stage, feed_dict={x: -1})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i})
        self.assertAllClose(4 * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  def testMultiple(self):
    with self.test_session(use_gpu=True) as sess:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea([dtypes.float32, dtypes.float32])
        stage = stager.put([x, v])
        z, y = stager.get()
        y = math_ops.reduce_max(z * math_ops.matmul(y, y))
      sess.run(stage, feed_dict={x: -1})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i})
        self.assertAllClose(
            4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  def testDictionary(self):
    with self.test_session(use_gpu=True) as sess:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea(
            [dtypes.float32, dtypes.float32],
            shapes=[[], [128, 128]],
            names=['x', 'v'])
        stage = stager.put({'x': x, 'v': v})
        ret = stager.get()
        z = ret['x']
        y = ret['v']
        y = math_ops.reduce_max(z * math_ops.matmul(y, y))
      sess.run(stage, feed_dict={x: -1})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i})
        self.assertAllClose(
            4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  def testColocation1(self):
    with ops.device('/cpu:0'):
      x = array_ops.placeholder(dtypes.float32)
      v = 2. * (array_ops.zeros([128, 128]) + x)
    with ops.device('/gpu:0'):
      stager = data_flow_ops.StagingArea([dtypes.float32])
      y = stager.put([v])
      self.assertEqual(y.device, '/device:GPU:0')
    with ops.device('/cpu:0'):
      x = stager.get()
      self.assertEqual(x.device, '/device:CPU:0')


if __name__ == '__main__':
  test.main()
