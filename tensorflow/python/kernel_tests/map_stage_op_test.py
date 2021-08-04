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

from tensorflow.python.framework import errors
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

TIMEOUT = 1


class MapStageTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testSimple(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea([dtypes.float32])
        stage = stager.put(pi, [v], [0])
        k, y = stager.get(gi)
        y = math_ops.reduce_max(math_ops.matmul(y, y))

    G.finalize()

    with self.session(graph=G) as sess:
      sess.run(stage, feed_dict={x: -1, pi: 0})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i, pi: i + 1, gi: i})
        self.assertAllClose(4 * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  @test_util.run_deprecated_v1
  def testMultiple(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32])
        stage = stager.put(pi, [x, v], [0, 1])
        k, (z, y) = stager.get(gi)
        y = math_ops.reduce_max(z * math_ops.matmul(y, y))

    G.finalize()

    with self.session(graph=G) as sess:
      sess.run(stage, feed_dict={x: -1, pi: 0})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i, pi: i + 1, gi: i})
        self.assertAllClose(
            4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  @test_util.run_deprecated_v1
  def testDictionary(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea(
            [dtypes.float32, dtypes.float32],
            shapes=[[], [128, 128]],
            names=['x', 'v'])
        stage = stager.put(pi, {'x': x, 'v': v})
        key, ret = stager.get(gi)
        z = ret['x']
        y = ret['v']
        y = math_ops.reduce_max(z * math_ops.matmul(y, y))

    G.finalize()

    with self.session(graph=G) as sess:
      sess.run(stage, feed_dict={x: -1, pi: 0})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i, pi: i + 1, gi: i})
        self.assertAllClose(
            4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  def testColocation(self):
    gpu_dev = test.gpu_device_name()

    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(gpu_dev):
        stager = data_flow_ops.MapStagingArea([dtypes.float32])
        y = stager.put(1, [v], [0])
        expected_name = gpu_dev if 'gpu' not in gpu_dev else '/device:GPU:0'
        self.assertEqual(y.device, expected_name)
      with ops.device('/cpu:0'):
        _, x = stager.get(1)
        y = stager.peek(1)[0]
        _, z = stager.get()
        self.assertEqual(x[0].device, '/device:CPU:0')
        self.assertEqual(y.device, '/device:CPU:0')
        self.assertEqual(z[0].device, '/device:CPU:0')

    G.finalize()

  @test_util.run_deprecated_v1
  def testPeek(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.int32, name='x')
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
        p = array_ops.placeholder(dtypes.int32, name='p')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea(
            [
                dtypes.int32,
            ], shapes=[[]])
        stage = stager.put(pi, [x], [0])
        peek = stager.peek(gi)
        size = stager.size()

    G.finalize()

    n = 10

    with self.session(graph=G) as sess:
      for i in range(n):
        sess.run(stage, feed_dict={x: i, pi: i})

      for i in range(n):
        self.assertTrue(sess.run(peek, feed_dict={gi: i})[0] == i)

      self.assertTrue(sess.run(size) == 10)

  @test_util.run_deprecated_v1
  def testSizeAndClear(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32, name='x')
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea(
            [dtypes.float32, dtypes.float32],
            shapes=[[], [128, 128]],
            names=['x', 'v'])
        stage = stager.put(pi, {'x': x, 'v': v})
        size = stager.size()
        clear = stager.clear()

    G.finalize()

    with self.session(graph=G) as sess:
      sess.run(stage, feed_dict={x: -1, pi: 3})
      self.assertEqual(sess.run(size), 1)
      sess.run(stage, feed_dict={x: -1, pi: 1})
      self.assertEqual(sess.run(size), 2)
      sess.run(clear)
      self.assertEqual(sess.run(size), 0)

  @test_util.run_deprecated_v1
  def testCapacity(self):
    capacity = 3

    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.int32, name='x')
        pi = array_ops.placeholder(dtypes.int64, name='pi')
        gi = array_ops.placeholder(dtypes.int64, name='gi')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea(
            [
                dtypes.int32,
            ], capacity=capacity, shapes=[[]])

      stage = stager.put(pi, [x], [0])
      get = stager.get()
      size = stager.size()

    G.finalize()

    from six.moves import queue as Queue
    import threading

    queue = Queue.Queue()
    n = 8

    with self.session(graph=G) as sess:
      # Stage data in a separate thread which will block
      # when it hits the staging area's capacity and thus
      # not fill the queue with n tokens
      def thread_run():
        for i in range(n):
          sess.run(stage, feed_dict={x: i, pi: i})
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
                  "capacity is '{}'.".format(capacity, i, sess.run(size),
                                             capacity))

      # Should have capacity elements in the staging area
      self.assertTrue(sess.run(size) == capacity)

      # Clear the staging area completely
      for i in range(n):
        sess.run(get)

      self.assertTrue(sess.run(size) == 0)

  @test_util.run_deprecated_v1
  def testMemoryLimit(self):
    memory_limit = 512 * 1024  # 512K
    chunk = 200 * 1024  # 256K
    capacity = memory_limit // chunk

    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.uint8, name='x')
        pi = array_ops.placeholder(dtypes.int64, name='pi')
        gi = array_ops.placeholder(dtypes.int64, name='gi')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea(
            [dtypes.uint8], memory_limit=memory_limit, shapes=[[]])
        stage = stager.put(pi, [x], [0])
        get = stager.get()
        size = stager.size()

    G.finalize()

    from six.moves import queue as Queue
    import threading
    import numpy as np

    queue = Queue.Queue()
    n = 8

    with self.session(graph=G) as sess:
      # Stage data in a separate thread which will block
      # when it hits the staging area's capacity and thus
      # not fill the queue with n tokens
      def thread_run():
        for i in range(n):
          data = np.full(chunk, i, dtype=np.uint8)
          sess.run(stage, feed_dict={x: data, pi: i})
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
                  "capacity is '{}'.".format(capacity, i, sess.run(size),
                                             capacity))

      # Should have capacity elements in the staging area
      self.assertTrue(sess.run(size) == capacity)

      # Clear the staging area completely
      for i in range(n):
        sess.run(get)

      self.assertTrue(sess.run(size) == 0)

  @test_util.run_deprecated_v1
  def testOrdering(self):
    import six
    import random

    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.int32, name='x')
        pi = array_ops.placeholder(dtypes.int64, name='pi')
        gi = array_ops.placeholder(dtypes.int64, name='gi')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea(
            [
                dtypes.int32,
            ], shapes=[[]], ordered=True)
        stage = stager.put(pi, [x], [0])
        get = stager.get()
        size = stager.size()

    G.finalize()

    n = 10

    with self.session(graph=G) as sess:
      # Keys n-1..0
      keys = list(reversed(six.moves.range(n)))

      for i in keys:
        sess.run(stage, feed_dict={pi: i, x: i})

      self.assertTrue(sess.run(size) == n)

      # Check that key, values come out in ascending order
      for i, k in enumerate(reversed(keys)):
        get_key, values = sess.run(get)
        self.assertTrue(i == k == get_key == values)

      self.assertTrue(sess.run(size) == 0)

  @test_util.run_deprecated_v1
  def testPartialDictInsert(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        f = array_ops.placeholder(dtypes.float32)
        v = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
      with ops.device(test.gpu_device_name()):
        # Test barrier with dictionary
        stager = data_flow_ops.MapStagingArea(
            [dtypes.float32, dtypes.float32, dtypes.float32],
            names=['x', 'v', 'f'])
        stage_xf = stager.put(pi, {'x': x, 'f': f})
        stage_v = stager.put(pi, {'v': v})
        key, ret = stager.get(gi)
        size = stager.size()
        isize = stager.incomplete_size()

    G.finalize()

    with self.session(graph=G) as sess:
      # 0 complete and incomplete entries
      self.assertTrue(sess.run([size, isize]) == [0, 0])
      # Stage key 0, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 0, x: 1, f: 2})
      self.assertTrue(sess.run([size, isize]) == [0, 1])
      # Stage key 1, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 1, x: 1, f: 2})
      self.assertTrue(sess.run([size, isize]) == [0, 2])

      # Now complete key 0 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 0, v: 1})
      # 1 complete and 1 incomplete entry
      self.assertTrue(sess.run([size, isize]) == [1, 1])
      # We can now obtain tuple associated with key 0
      self.assertTrue(
          sess.run([key, ret], feed_dict={
              gi: 0
          }) == [0, {
              'x': 1,
              'f': 2,
              'v': 1
          }])

      # 0 complete and 1 incomplete entry
      self.assertTrue(sess.run([size, isize]) == [0, 1])
      # Now complete key 1 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 1, v: 3})
      # We can now obtain tuple associated with key 1
      self.assertTrue(
          sess.run([key, ret], feed_dict={
              gi: 1
          }) == [1, {
              'x': 1,
              'f': 2,
              'v': 3
          }])

  @test_util.run_deprecated_v1
  def testPartialIndexInsert(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        f = array_ops.placeholder(dtypes.float32)
        v = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea(
            [dtypes.float32, dtypes.float32, dtypes.float32])
        stage_xf = stager.put(pi, [x, f], [0, 2])
        stage_v = stager.put(pi, [v], [1])
        key, ret = stager.get(gi)
        size = stager.size()
        isize = stager.incomplete_size()

    G.finalize()

    with self.session(graph=G) as sess:
      # 0 complete and incomplete entries
      self.assertTrue(sess.run([size, isize]) == [0, 0])
      # Stage key 0, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 0, x: 1, f: 2})
      self.assertTrue(sess.run([size, isize]) == [0, 1])
      # Stage key 1, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 1, x: 1, f: 2})
      self.assertTrue(sess.run([size, isize]) == [0, 2])

      # Now complete key 0 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 0, v: 1})
      # 1 complete and 1 incomplete entry
      self.assertTrue(sess.run([size, isize]) == [1, 1])
      # We can now obtain tuple associated with key 0
      self.assertTrue(sess.run([key, ret], feed_dict={gi: 0}) == [0, [1, 1, 2]])

      # 0 complete and 1 incomplete entry
      self.assertTrue(sess.run([size, isize]) == [0, 1])
      # Now complete key 1 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 1, v: 3})
      # We can now obtain tuple associated with key 1
      self.assertTrue(sess.run([key, ret], feed_dict={gi: 1}) == [1, [1, 3, 2]])

  @test_util.run_deprecated_v1
  def testPartialDictGetsAndPeeks(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        f = array_ops.placeholder(dtypes.float32)
        v = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        pei = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
      with ops.device(test.gpu_device_name()):
        # Test barrier with dictionary
        stager = data_flow_ops.MapStagingArea(
            [dtypes.float32, dtypes.float32, dtypes.float32],
            names=['x', 'v', 'f'])
        stage_xf = stager.put(pi, {'x': x, 'f': f})
        stage_v = stager.put(pi, {'v': v})
        peek_xf = stager.peek(pei, ['x', 'f'])
        peek_v = stager.peek(pei, ['v'])
        key_xf, get_xf = stager.get(gi, ['x', 'f'])
        key_v, get_v = stager.get(gi, ['v'])
        pop_key_xf, pop_xf = stager.get(indices=['x', 'f'])
        pop_key_v, pop_v = stager.get(pi, ['v'])
        size = stager.size()
        isize = stager.incomplete_size()

    G.finalize()

    with self.session(graph=G) as sess:
      # 0 complete and incomplete entries
      self.assertTrue(sess.run([size, isize]) == [0, 0])
      # Stage key 0, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 0, x: 1, f: 2})
      self.assertTrue(sess.run([size, isize]) == [0, 1])
      # Stage key 1, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 1, x: 1, f: 2})
      self.assertTrue(sess.run([size, isize]) == [0, 2])

      # Now complete key 0 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 0, v: 1})
      # 1 complete and 1 incomplete entry
      self.assertTrue(sess.run([size, isize]) == [1, 1])

      # We can now peek at 'x' and 'f' values associated with key 0
      self.assertTrue(sess.run(peek_xf, feed_dict={pei: 0}) == {'x': 1, 'f': 2})
      # Peek at 'v' value associated with key 0
      self.assertTrue(sess.run(peek_v, feed_dict={pei: 0}) == {'v': 1})
      # 1 complete and 1 incomplete entry
      self.assertTrue(sess.run([size, isize]) == [1, 1])

      # We can now obtain 'x' and 'f' values associated with key 0
      self.assertTrue(
          sess.run([key_xf, get_xf], feed_dict={
              gi: 0
          }) == [0, {
              'x': 1,
              'f': 2
          }])
      # Still have 1 complete and 1 incomplete entry
      self.assertTrue(sess.run([size, isize]) == [1, 1])

      # We can no longer get 'x' and 'f' from key 0
      with self.assertRaises(errors.InvalidArgumentError) as cm:
        sess.run([key_xf, get_xf], feed_dict={gi: 0})

      exc_str = ("Tensor at index '0' for key '0' " 'has already been removed.')

      self.assertTrue(exc_str in cm.exception.message)

      # Obtain 'v' value associated with key 0
      self.assertTrue(
          sess.run([key_v, get_v], feed_dict={
              gi: 0
          }) == [0, {
              'v': 1
          }])
      # 0 complete and 1 incomplete entry
      self.assertTrue(sess.run([size, isize]) == [0, 1])

      # Now complete key 1 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 1, v: 1})
      # 1 complete and 1 incomplete entry
      self.assertTrue(sess.run([size, isize]) == [1, 0])

      # Pop without key to obtain 'x' and 'f' values associated with key 1
      self.assertTrue(sess.run([pop_key_xf, pop_xf]) == [1, {'x': 1, 'f': 2}])
      # still 1 complete and 1 incomplete entry
      self.assertTrue(sess.run([size, isize]) == [1, 0])
      # We can now obtain 'x' and 'f' values associated with key 1
      self.assertTrue(
          sess.run([pop_key_v, pop_v], feed_dict={
              pi: 1
          }) == [1, {
              'v': 1
          }])
      # Nothing is left
      self.assertTrue(sess.run([size, isize]) == [0, 0])

  @test_util.run_deprecated_v1
  def testPartialIndexGets(self):
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        f = array_ops.placeholder(dtypes.float32)
        v = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        pei = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
      with ops.device(test.gpu_device_name()):
        # Test again with partial index gets
        stager = data_flow_ops.MapStagingArea(
            [dtypes.float32, dtypes.float32, dtypes.float32])
        stage_xvf = stager.put(pi, [x, v, f], [0, 1, 2])
        key_xf, get_xf = stager.get(gi, [0, 2])
        key_v, get_v = stager.get(gi, [1])
        size = stager.size()
        isize = stager.incomplete_size()

    G.finalize()

    with self.session(graph=G) as sess:
      # Stage complete tuple
      sess.run(stage_xvf, feed_dict={pi: 0, x: 1, f: 2, v: 3})

      self.assertTrue(sess.run([size, isize]) == [1, 0])

      # Partial get using indices
      self.assertTrue(
          sess.run([key_xf, get_xf], feed_dict={
              gi: 0
          }) == [0, [1, 2]])

      # Still some of key 0 left
      self.assertTrue(sess.run([size, isize]) == [1, 0])

      # Partial get of remaining index
      self.assertTrue(sess.run([key_v, get_v], feed_dict={gi: 0}) == [0, [3]])

      # All gone
      self.assertTrue(sess.run([size, isize]) == [0, 0])


if __name__ == '__main__':
  test.main()
