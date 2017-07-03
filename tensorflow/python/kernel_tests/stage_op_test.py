# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

TIMEOUT = 1

class StageTest(test.TestCase):


  def testSimple(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea([dtypes.float32])
        stage = stager.put([v])
        y = stager.get()
        y = math_ops.reduce_max(math_ops.matmul(y, y))

    G.finalize()

    with self.test_session(use_gpu=True, graph=G) as sess:
      sess.run(stage, feed_dict={x: -1})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i})
        self.assertAllClose(4 * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  def testMultiple(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea([dtypes.float32, dtypes.float32])
        stage = stager.put([x, v])
        z, y = stager.get()
        y = math_ops.reduce_max(z * math_ops.matmul(y, y))

    G.finalize()

    with self.test_session(use_gpu=True, graph=G) as sess:
      sess.run(stage, feed_dict={x: -1})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i})
        self.assertAllClose(
            4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  def testDictionary(self):
    with ops.Graph().as_default() as G:
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

    G.finalize()

    with self.test_session(use_gpu=True, graph=G) as sess:
      sess.run(stage, feed_dict={x: -1})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i})
        self.assertAllClose(
            4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  def testColocation(self):
    gpu_dev = test.gpu_device_name()

    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(gpu_dev):
        stager = data_flow_ops.StagingArea([dtypes.float32])
        y = stager.put([v])
        expected_name = gpu_dev if 'gpu' not in gpu_dev else '/device:GPU:0'
        self.assertEqual(y.device, expected_name)
      with ops.device('/cpu:0'):
        x = stager.get()
        self.assertEqual(x.device, '/device:CPU:0')

    G.finalize()

  def testPeek(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.int32, name='x')
        p = array_ops.placeholder(dtypes.int32, name='p')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea([dtypes.int32, ], shapes=[[]])
        stage = stager.put([x])
        peek = stager.peek(p)
        ret = stager.get()

    G.finalize()

    with self.test_session(use_gpu=True, graph=G) as sess:
      for i in range(10):
        sess.run(stage, feed_dict={x:i})

      for i in range(10):
        self.assertTrue(sess.run(peek, feed_dict={p:i}) == i)

  def testSizeAndClear(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32, name='x')
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea(
            [dtypes.float32, dtypes.float32],
            shapes=[[], [128, 128]],
            names=['x', 'v'])
        stage = stager.put({'x': x, 'v': v})
        ret = stager.get()
        size = stager.size()
        clear = stager.clear()

    G.finalize()

    with self.test_session(use_gpu=True, graph=G) as sess:
      sess.run(stage, feed_dict={x: -1})
      self.assertEqual(sess.run(size), 1)
      sess.run(stage, feed_dict={x: -1})
      self.assertEqual(sess.run(size), 2)
      sess.run(clear)
      self.assertEqual(sess.run(size), 0)

  def testCapacity(self):
    capacity = 3

    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.int32, name='x')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea([dtypes.int32, ],
          capacity=capacity, shapes=[[]])
        stage = stager.put([x])
        ret = stager.get()
        size = stager.size()

    G.finalize()

    from six.moves import queue as Queue
    import threading

    queue = Queue.Queue()
    n = 8

    with self.test_session(use_gpu=True, graph=G) as sess:
      # Stage data in a separate thread which will block
      # when it hits the staging area's capacity and thus
      # not fill the queue with n tokens
      def thread_run():
        for i in range(n):
          sess.run(stage, feed_dict={x: i})
          queue.put(0)

      t = threading.Thread(target=thread_run)
      t.daemon = True
      t.start()

      # Get tokens from the queue until a timeout occurs
      try:
        for i in range(n):
          queue.get(timeout=TIMEOUT)
      except Queue.Empty:
        pass

      # Should've timed out on the iteration 'capacity'
      if not i == capacity:
        self.fail("Expected to timeout on iteration '{}' "
                  "but instead timed out on iteration '{}' "
                  "Staging Area size is '{}' and configured "
                  "capacity is '{}'.".format(capacity, i,
                                            sess.run(size),
                                            capacity))

      # Should have capacity elements in the staging area
      self.assertTrue(sess.run(size) == capacity)

      # Clear the staging area completely
      for i in range(n):
        self.assertTrue(sess.run(ret) == i)

      # It should now be empty
      self.assertTrue(sess.run(size) == 0)

  def testMemoryLimit(self):
    memory_limit = 512*1024  # 512K
    chunk = 200*1024 # 256K
    capacity = memory_limit // chunk

    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.uint8, name='x')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea([dtypes.uint8, ],
          memory_limit=memory_limit, shapes=[[]])
        stage = stager.put([x])
        ret = stager.get()
        size = stager.size()

    G.finalize()

    from six.moves import queue as Queue
    import threading
    import numpy as np

    queue = Queue.Queue()
    n = 8

    with self.test_session(use_gpu=True, graph=G) as sess:
      # Stage data in a separate thread which will block
      # when it hits the staging area's capacity and thus
      # not fill the queue with n tokens
      def thread_run():
        for i in range(n):
          sess.run(stage, feed_dict={x: np.full(chunk, i, dtype=np.uint8)})
          queue.put(0)

      t = threading.Thread(target=thread_run)
      t.daemon = True
      t.start()

      # Get tokens from the queue until a timeout occurs
      try:
        for i in range(n):
          queue.get(timeout=TIMEOUT)
      except Queue.Empty:
        pass

      # Should've timed out on the iteration 'capacity'
      if not i == capacity:
        self.fail("Expected to timeout on iteration '{}' "
                  "but instead timed out on iteration '{}' "
                  "Staging Area size is '{}' and configured "
                  "capacity is '{}'.".format(capacity, i,
                                            sess.run(size),
                                            capacity))

      # Should have capacity elements in the staging area
      self.assertTrue(sess.run(size) == capacity)

      # Clear the staging area completely
      for i in range(n):
        self.assertTrue(np.all(sess.run(ret) == i))

      self.assertTrue(sess.run(size) == 0)

if __name__ == '__main__':
  test.main()
