# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.data_flow_ops.PaddingFIFOQueue."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


@test_util.run_v1_only("PaddingFIFOQueue removed from v2")
class PaddingFIFOQueueTest(test.TestCase):

  def testConstructor(self):
    with ops.Graph().as_default():
      q = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((None,),), name="Q")
    self.assertTrue(isinstance(q.queue_ref, ops.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'PaddingFIFOQueueV2'
      attr { key: 'component_types' value { list { type: DT_FLOAT } } }
      attr { key: 'shapes' value { list { shape { dim { size: -1 } } } } }
      attr { key: 'capacity' value { i: 10 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      """, q.queue_ref.op.node_def)

  def testMultiQueueConstructor(self):
    with ops.Graph().as_default():
      q = data_flow_ops.PaddingFIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32), ((), ()),
          shared_name="foo",
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, ops.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'PaddingFIFOQueueV2'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list { shape { } shape { } } } }
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: 'foo' } }
      """, q.queue_ref.op.node_def)

  def testConstructorWithShapes(self):
    with ops.Graph().as_default():
      q = data_flow_ops.PaddingFIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32),
          shapes=(tensor_shape.TensorShape([1, 1, 2, 3]),
                  tensor_shape.TensorShape([5, 8])),
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, ops.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'PaddingFIFOQueueV2'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list {
        shape { dim { size: 1 }
                dim { size: 1 }
                dim { size: 2 }
                dim { size: 3 } }
        shape { dim { size: 5 }
                dim { size: 8 } }
      } } }
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      """, q.queue_ref.op.node_def)

  def testEnqueue(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      enqueue_op = q.enqueue((10.0,))
      enqueue_op.run()

  def testEnqueueWithShape(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=((3, 2),))
      enqueue_correct_op = q.enqueue(([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],))
      enqueue_correct_op.run()
      with self.assertRaises(ValueError):
        q.enqueue(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],))
      self.assertEqual(1, q.size().eval())

  def testEnqueueManyWithShape(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(
          10, [dtypes_lib.int32, dtypes_lib.int32], shapes=[(), (2,)])
      q.enqueue_many([[1, 2, 3, 4], [[1, 1], [2, 2], [3, 3], [4, 4]]]).run()
      self.assertEqual(4, q.size().eval())

  def testParallelEnqueue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        self.evaluate(enqueue_op)

      threads = [
          self.checkedThread(
              target=enqueue, args=(e,)) for e in enqueue_ops
      ]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      # Dequeue every element using a single thread.
      results = []
      for _ in xrange(len(elems)):
        results.append(self.evaluate(dequeued_t))
      self.assertItemsEqual(elems, results)

  def testParallelDequeue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Enqueue every element using a single thread.
      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      # Run one consumer thread for each element in elems.
      results = []

      def dequeue():
        results.append(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in enqueue_ops]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, results)

  def testDequeue(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      for i in xrange(len(elems)):
        vals = self.evaluate(dequeued_t)
        self.assertEqual([elems[i]], vals)

  def testEnqueueAndBlockingDequeue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(3, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      def enqueue():
        # The enqueue_ops should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        for enqueue_op in enqueue_ops:
          self.evaluate(enqueue_op)

      results = []

      def dequeue():
        for _ in xrange(len(elems)):
          results.append(self.evaluate(dequeued_t))

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      for elem, result in zip(elems, results):
        self.assertEqual([elem], result)

  def testMultiEnqueueAndDequeue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10,
                                         (dtypes_lib.int32, dtypes_lib.float32),
                                         ((), ()))
      elems = [(5, 10.0), (10, 20.0), (15, 30.0)]
      enqueue_ops = [q.enqueue((x, y)) for x, y in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      for i in xrange(len(elems)):
        x_val, y_val = self.evaluate(dequeued_t)
        x, y = elems[i]
        self.assertEqual([x], x_val)
        self.assertEqual([y], y_val)

  def testQueueSizeEmpty(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      self.assertEqual([0], q.size().eval())

  def testQueueSizeAfterEnqueueAndDequeue(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue()
      size = q.size()
      self.assertEqual([], size.get_shape())

      enqueue_op.run()
      self.assertEqual(1, self.evaluate(size))
      dequeued_t.op.run()
      self.assertEqual(0, self.evaluate(size))

  def testEnqueueMany(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue()
      enqueue_op.run()
      enqueue_op.run()

      for i in range(8):
        vals = self.evaluate(dequeued_t)
        self.assertEqual([elems[i % 4]], vals)

  def testEmptyEnqueueMany(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, (
          (None, None),))
      empty_t = constant_op.constant(
          [], dtype=dtypes_lib.float32, shape=[0, 2, 3])
      enqueue_op = q.enqueue_many((empty_t,))
      size_t = q.size()

      self.assertEqual([0], self.evaluate(size_t))
      enqueue_op.run()
      self.assertEqual([0], self.evaluate(size_t))

  def testEmptyDequeueMany(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, shapes=((),))
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue_many(0)

      self.assertEqual([], self.evaluate(dequeued_t).tolist())
      enqueue_op.run()
      self.assertEqual([], self.evaluate(dequeued_t).tolist())

  def testEmptyDequeueManyWithDynamicShape(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=((None,),))
      enqueue_op = q.enqueue(([10.0],))
      dequeued_t = q.dequeue_many(0)

      self.assertEqual([], self.evaluate(dequeued_t).tolist())
      enqueue_op.run()
      self.assertEqual([], self.evaluate(dequeued_t).tolist())

  def testEmptyDequeueUpToWithDynamicShape(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=((None,),))
      enqueue_op = q.enqueue(([10.0],))
      dequeued_t = q.dequeue_up_to(0)

      self.assertEqual([], self.evaluate(dequeued_t).tolist())
      enqueue_op.run()
      self.assertEqual([], self.evaluate(dequeued_t).tolist())

  def testConstructPaddingFIFOQueueWithNoShape(self):
    with self.cached_session():
      with self.assertRaisesRegex(
          ValueError,
          r"When providing partial shapes, a list of shapes must be provided."):
        self.evaluate(
            data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32,
                                           None).queue_ref)

  def testMultiEnqueueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10,
                                         (dtypes_lib.float32, dtypes_lib.int32),
                                         ((), (2,)))
      float_elems = [10.0, 20.0, 30.0, 40.0]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue()

      enqueue_op.run()
      enqueue_op.run()

      for i in range(8):
        float_val, int_val = self.evaluate(dequeued_t)
        self.assertEqual(float_elems[i % 4], float_val)
        self.assertAllEqual(int_elems[i % 4], int_val)

  def testMultiEnqueueManyWithPartiallyKnownShapes(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (None,)))
      float_elems = [10.0, 20.0, 30.0, 40.0]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue()

      enqueue_op.run()
      enqueue_op.run()

      for i in range(8):
        float_val, int_val = self.evaluate(dequeued_t)
        self.assertEqual(float_elems[i % 4], float_val)
        self.assertAllEqual(int_elems[i % 4], int_val)

  def testDequeueMany(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(4)

      enqueue_op.run()

      self.assertAllEqual(elems[0:4], self.evaluate(dequeued_t))
      self.assertAllEqual(elems[4:8], self.evaluate(dequeued_t))

  def testDequeueUpToNoBlocking(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_up_to(4)

      enqueue_op.run()

      self.assertAllEqual(elems[0:4], self.evaluate(dequeued_t))
      self.assertAllEqual(elems[4:8], self.evaluate(dequeued_t))

  def testMultiDequeueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (2,)))
      float_elems = [
          10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
      ]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                   [15, 16], [17, 18], [19, 20]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue_many(4)
      dequeued_single_t = q.dequeue()

      enqueue_op.run()

      float_val, int_val = self.evaluate(dequeued_t)
      self.assertAllEqual(float_elems[0:4], float_val)
      self.assertAllEqual(int_elems[0:4], int_val)
      self.assertEqual(float_val.shape, dequeued_t[0].get_shape())
      self.assertEqual(int_val.shape, dequeued_t[1].get_shape())

      float_val, int_val = self.evaluate(dequeued_t)
      self.assertAllEqual(float_elems[4:8], float_val)
      self.assertAllEqual(int_elems[4:8], int_val)

      float_val, int_val = self.evaluate(dequeued_single_t)
      self.assertAllEqual(float_elems[8], float_val)
      self.assertAllEqual(int_elems[8], int_val)
      self.assertEqual(float_val.shape, dequeued_single_t[0].get_shape())
      self.assertEqual(int_val.shape, dequeued_single_t[1].get_shape())

  def testMultiDequeueManyWithPartiallyKnownShapes(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (None,)))
      float_elems = [
          10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
      ]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                   [15, 16], [17, 18], [19, 20]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue_many(4)
      dequeued_single_t = q.dequeue()

      enqueue_op.run()

      float_val, int_val = self.evaluate(dequeued_t)
      self.assertAllEqual(float_elems[0:4], float_val)
      self.assertAllEqual(int_elems[0:4], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(float_val.shape).is_compatible_with(
              dequeued_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(dequeued_t[
              1].get_shape()))

      float_val, int_val = self.evaluate(dequeued_t)
      self.assertAllEqual(float_elems[4:8], float_val)
      self.assertAllEqual(int_elems[4:8], int_val)

      float_val, int_val = self.evaluate(dequeued_single_t)
      self.assertAllEqual(float_elems[8], float_val)
      self.assertAllEqual(int_elems[8], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(float_val.shape).is_compatible_with(
              dequeued_single_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(
              dequeued_single_t[1].get_shape()))

  def testMultiDequeueManyWithPartiallyKnownShapesAndVariableSizeInput(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.string, dtypes_lib.int32),
          shapes=((None,), (1, None)))
      str_elems = [["a"], ["ab"], ["abc"], ["abc", "d"], ["abc", "d", "e"],
                   ["abc", "d", "e", "f"]]

      int_elems = [[[1]], [[2]], [[3]], [[1, 2]], [[1, 2, 3]], [[1, 2, 3, 4]]]

      enqueue_ops = [q.enqueue((str_elems[i], int_elems[i])) for i in range(6)]

      dequeued_t = q.dequeue_many(5)
      dequeued_single_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()
      string_val, int_val = self.evaluate(dequeued_t)

      self.assertAllEqual([[b"a", b"", b""], [b"ab", b"", b""],
                           [b"abc", b"", b""], [b"abc", b"d", b""],
                           [b"abc", b"d", b"e"]], string_val)
      self.assertAllEqual([[[1, 0, 0]], [[2, 0, 0]], [[3, 0, 0]], [[1, 2, 0]],
                           [[1, 2, 3]]], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(string_val.shape).is_compatible_with(
              dequeued_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(dequeued_t[
              1].get_shape()))

      string_val, int_val = self.evaluate(dequeued_single_t)
      self.assertAllEqual([b"abc", b"d", b"e", b"f"], string_val)
      self.assertAllEqual([[1, 2, 3, 4]], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(string_val.shape).is_compatible_with(
              dequeued_single_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(
              dequeued_single_t[1].get_shape()))

  def testMultiDequeueUpToPartiallyKnownShapesAndVariableInputNoBlocking(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.string, dtypes_lib.int32),
          shapes=((None,), (1, None)))
      str_elems = [["a"], ["ab"], ["abc"], ["abc", "d"], ["abc", "d", "e"],
                   ["abc", "d", "e", "f"]]

      int_elems = [[[1]], [[2]], [[3]], [[1, 2]], [[1, 2, 3]], [[1, 2, 3, 4]]]

      enqueue_ops = [q.enqueue((str_elems[i], int_elems[i])) for i in range(6)]

      dequeued_t = q.dequeue_up_to(5)
      dequeued_single_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()
      string_val, int_val = self.evaluate(dequeued_t)

      self.assertAllEqual([[b"a", b"", b""], [b"ab", b"", b""],
                           [b"abc", b"", b""], [b"abc", b"d", b""],
                           [b"abc", b"d", b"e"]], string_val)
      self.assertAllEqual([[[1, 0, 0]], [[2, 0, 0]], [[3, 0, 0]], [[1, 2, 0]],
                           [[1, 2, 3]]], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(string_val.shape).is_compatible_with(
              dequeued_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(dequeued_t[
              1].get_shape()))

      string_val, int_val = self.evaluate(dequeued_single_t)
      self.assertAllEqual([b"abc", b"d", b"e", b"f"], string_val)
      self.assertAllEqual([[1, 2, 3, 4]], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(string_val.shape).is_compatible_with(
              dequeued_single_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(
              dequeued_single_t[1].get_shape()))

  def testHighDimension(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.int32, ((4, 4, 4, 4),))
      elems = np.array([[[[[x] * 4] * 4] * 4] * 4 for x in range(10)], np.int32)
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(10)

      enqueue_op.run()
      self.assertAllEqual(dequeued_t, elems)

  def testPartiallyKnownHighDimension(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.int32, (
          (4, None, 4, None),))
      elems = np.array([[[[[x] * 4] * 4] * 4] * 4 for x in range(10)], np.int32)
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(10)

      enqueue_op.run()
      self.assertAllEqual(dequeued_t, elems)

  def testEnqueueWrongShape(self):
    q = data_flow_ops.PaddingFIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32),
                                       ((), (2,)))

    with self.assertRaises(ValueError):
      q.enqueue(([1, 2], [2, 2]))

    with self.assertRaises(ValueError):
      q.enqueue_many((7, [[1, 2], [3, 4], [5, 6]]))

  def testBatchSizeMismatch(self):
    q = data_flow_ops.PaddingFIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32,
                                            dtypes_lib.int32), ((), (), ()))

    with self.assertRaises(ValueError):
      q.enqueue_many(([1, 2, 3], [1, 2], [1, 2, 3]))

    with self.assertRaises(ValueError):
      q.enqueue_many(
          ([1, 2, 3], [1, 2], array_ops.placeholder(dtypes_lib.int32)))

    with self.assertRaises(ValueError):
      q.enqueue_many(
          (array_ops.placeholder(dtypes_lib.int32), [1, 2], [1, 2, 3]))

  def testEnqueueManyEmptyTypeConversion(self):
    q = data_flow_ops.PaddingFIFOQueue(10,
                                       (dtypes_lib.int32, dtypes_lib.float32), (
                                           (), ()))
    enq = q.enqueue_many(([], []))
    self.assertEqual(dtypes_lib.int32, enq.inputs[1].dtype)
    self.assertEqual(dtypes_lib.float32, enq.inputs[2].dtype)

  def testEnqueueWrongType(self):
    q = data_flow_ops.PaddingFIFOQueue(10,
                                       (dtypes_lib.int32, dtypes_lib.float32), (
                                           (), ()))

    with self.assertRaises(ValueError):
      q.enqueue((array_ops.placeholder(dtypes_lib.int32),
                 array_ops.placeholder(dtypes_lib.int32)))

    with self.assertRaises(ValueError):
      q.enqueue_many((array_ops.placeholder(dtypes_lib.int32),
                      array_ops.placeholder(dtypes_lib.int32)))

  def testEnqueueWrongPartiallyKnownShapeAtRuntime(self):
    with self.cached_session() as sess:
      # First dimension of second component is unknown, second
      # dimension must be 3.
      q = data_flow_ops.PaddingFIFOQueue(10,
                                         (dtypes_lib.int32, dtypes_lib.int32), (
                                             (2, 2), (None, 3)))
      elems_ok = np.array([1] * 4).reshape((2, 2)).astype(np.int32)
      elems_bad = array_ops.placeholder(dtypes_lib.int32)
      enqueue_op = q.enqueue((elems_ok, elems_bad))
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  r"Expected \[\?,3\], got \[3,4\]"):
        sess.run([enqueue_op],
                 feed_dict={elems_bad: np.array([1] * 12).reshape((3, 4))})

  def testEnqueueDequeueManyWrongPartiallyKnownShape(self):
    with self.cached_session() as sess:
      # First dimension of second component is unknown, second
      # dimension must be 3.
      q = data_flow_ops.PaddingFIFOQueue(10,
                                         (dtypes_lib.int32, dtypes_lib.int32), (
                                             (2, 2), (None, 3)))
      elems_ok = np.array([1] * 8).reshape((2, 2, 2)).astype(np.int32)
      elems_bad = array_ops.placeholder(dtypes_lib.int32)
      enqueue_op = q.enqueue_many((elems_ok, elems_bad))
      dequeued_t = q.dequeue_many(2)
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch in tuple component 1. "
          r"Expected \[2,\?,3\], got \[2,3,4\]"):
        sess.run([enqueue_op],
                 feed_dict={elems_bad: np.array([1] * 24).reshape((2, 3, 4))})
        self.evaluate(dequeued_t)

  def testParallelEnqueueMany(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(1000, dtypes_lib.float32, shapes=((),))
      elems = [10.0 * x for x in range(100)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(1000)

      # Enqueue 100 items in parallel on 10 threads.
      def enqueue():
        self.evaluate(enqueue_op)

      threads = [self.checkedThread(target=enqueue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      self.assertCountEqual(self.evaluate(dequeued_t), elems * 10)

  def testParallelDequeueMany(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(1000, dtypes_lib.float32, shapes=((),))
      elems = [10.0 * x for x in range(1000)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(100)

      enqueue_op.run()

      # Dequeue 100 items in parallel on 10 threads.
      dequeued_elems = []

      def dequeue():
        dequeued_elems.extend(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testParallelDequeueUpTo(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(1000, dtypes_lib.float32, shapes=((),))
      elems = [10.0 * x for x in range(1000)]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(101)

      enqueue_op.run()
      close_op.run()

      # Dequeue up to 101 items in parallel on 10 threads, from closed queue.
      dequeued_elems = []

      def dequeue():
        dequeued_elems.extend(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testParallelEnqueueAndDequeue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(50, dtypes_lib.float32, shapes=((),))
      initial_elements = [10.0] * 49
      q.enqueue_many((initial_elements,)).run()

      enqueue_op = q.enqueue((20.0,))
      dequeued_t = q.dequeue()

      def enqueue():
        for _ in xrange(100):
          self.evaluate(enqueue_op)

      def dequeue():
        for _ in xrange(100):
          self.assertTrue(self.evaluate(dequeued_t) in (10.0, 20.0))

      enqueue_threads = [self.checkedThread(target=enqueue) for _ in range(10)]
      dequeue_threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for enqueue_thread in enqueue_threads:
        enqueue_thread.start()
      for dequeue_thread in dequeue_threads:
        dequeue_thread.start()
      for enqueue_thread in enqueue_threads:
        enqueue_thread.join()
      for dequeue_thread in dequeue_threads:
        dequeue_thread.join()

      # Dequeue the initial count of elements to clean up.
      cleanup_elems = q.dequeue_many(49).eval()
      for elem in cleanup_elems:
        self.assertTrue(elem in (10.0, 20.0))

  def testMixtureOfEnqueueAndEnqueueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.int32, shapes=((),))
      enqueue_placeholder = array_ops.placeholder(dtypes_lib.int32, shape=())
      enqueue_op = q.enqueue((enqueue_placeholder,))
      enqueuemany_placeholder = array_ops.placeholder(
          dtypes_lib.int32, shape=(None,))
      enqueuemany_op = q.enqueue_many((enqueuemany_placeholder,))

      dequeued_t = q.dequeue()
      close_op = q.close()

      def dequeue():
        for i in xrange(250):
          self.assertEqual(i, self.evaluate(dequeued_t))

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()

      elements_enqueued = 0
      while elements_enqueued < 250:
        # With equal probability, run Enqueue or enqueue_many.
        if random.random() > 0.5:
          enqueue_op.run({enqueue_placeholder: elements_enqueued})
          elements_enqueued += 1
        else:
          count = random.randint(0, min(20, 250 - elements_enqueued))
          range_to_enqueue = np.arange(
              elements_enqueued, elements_enqueued + count, dtype=np.int32)
          enqueuemany_op.run({enqueuemany_placeholder: range_to_enqueue})
          elements_enqueued += count

      close_op.run()
      dequeue_thread.join()
      self.assertEqual(0, q.size().eval())

  def testMixtureOfDequeueAndDequeueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.int32, shapes=((),))
      enqueue_op = q.enqueue_many((np.arange(250, dtype=np.int32),))
      dequeued_t = q.dequeue()
      count_placeholder = array_ops.placeholder(dtypes_lib.int32, shape=())
      dequeuemany_t = q.dequeue_many(count_placeholder)

      def enqueue():
        self.evaluate(enqueue_op)

      enqueue_thread = self.checkedThread(target=enqueue)
      enqueue_thread.start()

      elements_dequeued = 0
      while elements_dequeued < 250:
        # With equal probability, run Dequeue or dequeue_many.
        if random.random() > 0.5:
          self.assertEqual(elements_dequeued, self.evaluate(dequeued_t))
          elements_dequeued += 1
        else:
          count = random.randint(0, min(20, 250 - elements_dequeued))
          expected_range = np.arange(
              elements_dequeued, elements_dequeued + count, dtype=np.int32)
          self.assertAllEqual(expected_range,
                              dequeuemany_t.eval({
                                  count_placeholder: count
                              }))
          elements_dequeued += count

      q.close().run()
      enqueue_thread.join()
      self.assertEqual(0, q.size().eval())

  def testBlockingDequeueMany(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(4)

      dequeued_elems = []

      def enqueue():
        # The enqueue_op should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.evaluate(enqueue_op)

      def dequeue():
        dequeued_elems.extend(self.evaluate(dequeued_t).tolist())

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      self.assertAllEqual(elems, dequeued_elems)

  def testBlockingDequeueUpTo(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_up_to(4)

      dequeued_elems = []

      def enqueue():
        # The enqueue_op should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.evaluate(enqueue_op)

      def dequeue():
        dequeued_elems.extend(self.evaluate(dequeued_t).tolist())

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      self.assertAllEqual(elems, dequeued_elems)

  def testDequeueManyWithTensorParameter(self):
    with self.cached_session():
      # Define a first queue that contains integer counts.
      dequeue_counts = [random.randint(1, 10) for _ in range(100)]
      count_q = data_flow_ops.PaddingFIFOQueue(100, dtypes_lib.int32, ((),))
      enqueue_counts_op = count_q.enqueue_many((dequeue_counts,))
      total_count = sum(dequeue_counts)

      # Define a second queue that contains total_count elements.
      elems = [random.randint(0, 100) for _ in range(total_count)]
      q = data_flow_ops.PaddingFIFOQueue(total_count, dtypes_lib.int32, ((),))
      enqueue_elems_op = q.enqueue_many((elems,))

      # Define a subgraph that first dequeues a count, then DequeuesMany
      # that number of elements.
      dequeued_t = q.dequeue_many(count_q.dequeue())

      enqueue_counts_op.run()
      enqueue_elems_op.run()

      dequeued_elems = []
      for _ in dequeue_counts:
        dequeued_elems.extend(self.evaluate(dequeued_t))
      self.assertEqual(elems, dequeued_elems)

  def testDequeueFromClosedQueue(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()
      close_op.run()
      for elem in elems:
        self.assertEqual([elem], self.evaluate(dequeued_t))

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                  "is closed and has insufficient"):
        self.evaluate(dequeued_t)

  def testBlockingDequeueFromClosedQueue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def dequeue():
        for elem in elems:
          self.assertEqual([elem], self.evaluate(dequeued_t))
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testDequeueUpToFromClosedQueueReturnsRemainder(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(3)

      enqueue_op.run()

      def dequeue():
        self.assertAllEqual(elems[:3], self.evaluate(dequeued_t))
        self.assertAllEqual(elems[3:], self.evaluate(dequeued_t))

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueFromClosedEmptyQueue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      close_op = q.close()
      dequeued_t = q.dequeue()

      def dequeue():
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueManyFromClosedQueue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(4)

      enqueue_op.run()

      def dequeue():
        self.assertAllEqual(elems, self.evaluate(dequeued_t))
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueManyButNotAllFromClosedQueue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(3)

      enqueue_op.run()

      def dequeue():
        self.assertAllEqual(elems[:3], self.evaluate(dequeued_t))
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testEnqueueManyLargerThanCapacityWithConcurrentDequeueMany(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(3)
      cleanup_dequeue_t = q.dequeue()

      def enqueue():
        self.evaluate(enqueue_op)

      def dequeue():
        self.assertAllEqual(elems[0:3], self.evaluate(dequeued_t))
        with self.assertRaises(errors_impl.OutOfRangeError):
          self.evaluate(dequeued_t)
        self.assertEqual(elems[3], self.evaluate(cleanup_dequeue_t))

      def close():
        self.evaluate(close_op)

      enqueue_thread = self.checkedThread(target=enqueue)
      enqueue_thread.start()

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      close_thread = self.checkedThread(target=close)
      close_thread.start()

      enqueue_thread.join()
      dequeue_thread.join()
      close_thread.join()

  def testClosedBlockingDequeueManyRestoresPartialBatch(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, (dtypes_lib.float32,
                                             dtypes_lib.float32), ((), ()))
      elems_a = [1.0, 2.0, 3.0]
      elems_b = [10.0, 20.0, 30.0]
      enqueue_op = q.enqueue_many((elems_a, elems_b))
      dequeued_a_t, dequeued_b_t = q.dequeue_many(4)
      cleanup_dequeue_a_t, cleanup_dequeue_b_t = q.dequeue()
      close_op = q.close()

      enqueue_op.run()

      def dequeue():
        with self.assertRaises(errors_impl.OutOfRangeError):
          self.evaluate([dequeued_a_t, dequeued_b_t])

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      close_op.run()
      dequeue_thread.join()
      # Test that the elements in the partially-dequeued batch are
      # restored in the correct order.
      for elem_a, elem_b in zip(elems_a, elems_b):
        val_a, val_b = self.evaluate([cleanup_dequeue_a_t, cleanup_dequeue_b_t])
        self.assertEqual(elem_a, val_a)
        self.assertEqual(elem_b, val_b)
      self.assertEqual(0, q.size().eval())

  def testBlockingDequeueManyFromClosedEmptyQueue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      close_op = q.close()
      dequeued_t = q.dequeue_many(4)

      def dequeue():
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueUpToFromClosedEmptyQueue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(4)

      def dequeue():
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testEnqueueToClosedQueue(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      enqueue_op = q.enqueue((10.0,))
      close_op = q.close()

      enqueue_op.run()
      close_op.run()

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegex(errors_impl.CancelledError, "is closed"):
        enqueue_op.run()

  def testEnqueueManyToClosedQueue(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()

      enqueue_op.run()
      close_op.run()

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegex(errors_impl.CancelledError, "is closed"):
        enqueue_op.run()

  def testBlockingEnqueueToFullQueue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue((50.0,))
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        self.evaluate(blocking_enqueue_op)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()
      # The dequeue ops should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      for elem in elems:
        self.assertEqual([elem], self.evaluate(dequeued_t))
      self.assertEqual([50.0], self.evaluate(dequeued_t))
      thread.join()

  def testBlockingEnqueueManyToFullQueue(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue_many(([50.0, 60.0],))
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        self.evaluate(blocking_enqueue_op)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()
      # The dequeue ops should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      for elem in elems:
        self.assertEqual([elem], self.evaluate(dequeued_t))
        time.sleep(0.01)
      self.assertEqual([50.0], self.evaluate(dequeued_t))
      self.assertEqual([60.0], self.evaluate(dequeued_t))

      # Make sure the thread finishes before exiting.
      thread.join()

  def testBlockingEnqueueBeforeClose(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue((50.0,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        # Expect the operation to succeed once the dequeue op runs.
        self.evaluate(blocking_enqueue_op)

      enqueue_thread = self.checkedThread(target=blocking_enqueue)
      enqueue_thread.start()

      # The close_op should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      def close():
        self.evaluate(close_op)

      close_thread = self.checkedThread(target=close)
      close_thread.start()

      # The dequeue will unblock both threads.
      self.assertEqual(10.0, self.evaluate(dequeued_t))
      enqueue_thread.join()
      close_thread.join()

      for elem in [20.0, 30.0, 40.0, 50.0]:
        self.assertEqual(elem, self.evaluate(dequeued_t))
      self.assertEqual(0, q.size().eval())

  def testBlockingEnqueueManyBeforeClose(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue_many(([50.0, 60.0],))
      close_op = q.close()
      dequeued_t = q.dequeue()
      enqueue_op.run()

      def blocking_enqueue():
        self.evaluate(blocking_enqueue_op)

      enqueue_thread = self.checkedThread(target=blocking_enqueue)
      enqueue_thread.start()

      # The close_op should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      def close():
        self.evaluate(close_op)

      close_thread = self.checkedThread(target=close)
      close_thread.start()

      # The dequeue will unblock both threads.
      self.assertEqual(10.0, self.evaluate(dequeued_t))
      enqueue_thread.join()
      close_thread.join()
      for elem in [20.0, 30.0, 50.0, 60.0]:
        self.assertEqual(elem, self.evaluate(dequeued_t))

  def testDoesNotLoseValue(self):
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(1, dtypes_lib.float32, ((),))
      enqueue_op = q.enqueue((10.0,))
      size_t = q.size()

      enqueue_op.run()
      for _ in range(500):
        self.assertEqual(self.evaluate(size_t), [1])

  def testSharedQueueSameSession(self):
    with self.cached_session():
      q1 = data_flow_ops.PaddingFIFOQueue(
          1, dtypes_lib.float32, ((),), shared_name="shared_queue")
      q1.enqueue((10.0,)).run()

      q2 = data_flow_ops.PaddingFIFOQueue(
          1, dtypes_lib.float32, ((),), shared_name="shared_queue")

      q1_size_t = q1.size()
      q2_size_t = q2.size()

      self.assertEqual(self.evaluate(q1_size_t), [1])
      self.assertEqual(self.evaluate(q2_size_t), [1])

      self.assertEqual(q2.dequeue().eval(), [10.0])

      self.assertEqual(self.evaluate(q1_size_t), [0])
      self.assertEqual(self.evaluate(q2_size_t), [0])

      q2.enqueue((20.0,)).run()

      self.assertEqual(self.evaluate(q1_size_t), [1])
      self.assertEqual(self.evaluate(q2_size_t), [1])

      self.assertEqual(q1.dequeue().eval(), [20.0])

      self.assertEqual(self.evaluate(q1_size_t), [0])
      self.assertEqual(self.evaluate(q2_size_t), [0])

  def testIncompatibleSharedQueueErrors(self):
    with self.cached_session():
      q_a_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((),), shared_name="q_a")
      q_a_2 = data_flow_ops.PaddingFIFOQueue(
          15, dtypes_lib.float32, ((),), shared_name="q_a")
      q_a_1.queue_ref.op.run()
      with self.assertRaisesOpError("capacity"):
        q_a_2.queue_ref.op.run()

      q_b_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((),), shared_name="q_b")
      q_b_2 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.int32, ((),), shared_name="q_b")
      q_b_1.queue_ref.op.run()
      with self.assertRaisesOpError("component types"):
        q_b_2.queue_ref.op.run()

      q_c_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((),), shared_name="q_c")
      q_c_2 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_c")
      q_c_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_c_2.queue_ref.op.run()

      q_d_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_d")
      q_d_2 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((),), shared_name="q_d")
      q_d_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_d_2.queue_ref.op.run()

      q_e_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_e")
      q_e_2 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 4)], shared_name="q_e")
      q_e_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_e_2.queue_ref.op.run()

      q_f_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((),), shared_name="q_f")
      q_f_2 = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), ((), ()),
          shared_name="q_f")
      q_f_1.queue_ref.op.run()
      with self.assertRaisesOpError("component types"):
        q_f_2.queue_ref.op.run()

  def testSelectQueue(self):
    with self.cached_session():
      num_queues = 10
      qlist = []
      for _ in xrange(num_queues):
        qlist.append(
            data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),)))
      # Enqueue/Dequeue into a dynamically selected queue
      for _ in xrange(20):
        index = np.random.randint(num_queues)
        q = data_flow_ops.PaddingFIFOQueue.from_list(index, qlist)
        q.enqueue((10.,)).run()
        self.assertEqual(q.dequeue().eval(), 10.0)

  def testSelectQueueOutOfRange(self):
    with self.cached_session():
      q1 = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      q2 = data_flow_ops.PaddingFIFOQueue(15, dtypes_lib.float32, ((),))
      enq_q = data_flow_ops.PaddingFIFOQueue.from_list(3, [q1, q2])
      with self.assertRaisesOpError("is not in"):
        enq_q.dequeue().eval()

  def _blockingDequeue(self, sess, dequeue_op):
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(dequeue_op)

  def _blockingDequeueMany(self, sess, dequeue_many_op):
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(dequeue_many_op)

  def _blockingEnqueue(self, sess, enqueue_op):
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(enqueue_op)

  def _blockingEnqueueMany(self, sess, enqueue_many_op):
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(enqueue_many_op)

  @test_util.run_deprecated_v1
  def testResetOfBlockingOperation(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q_empty = data_flow_ops.PaddingFIFOQueue(5, dtypes_lib.float32, ((),))
      dequeue_op = q_empty.dequeue()
      dequeue_many_op = q_empty.dequeue_many(1)

      q_full = data_flow_ops.PaddingFIFOQueue(5, dtypes_lib.float32, ((),))
      sess.run(q_full.enqueue_many(([1.0, 2.0, 3.0, 4.0, 5.0],)))
      enqueue_op = q_full.enqueue((6.0,))
      enqueue_many_op = q_full.enqueue_many(([6.0],))

      threads = [
          self.checkedThread(
              self._blockingDequeue, args=(sess, dequeue_op)),
          self.checkedThread(
              self._blockingDequeueMany, args=(sess, dequeue_many_op)),
          self.checkedThread(
              self._blockingEnqueue, args=(sess, enqueue_op)),
          self.checkedThread(
              self._blockingEnqueueMany, args=(sess, enqueue_many_op))
      ]
      for t in threads:
        t.start()
      time.sleep(0.1)
      sess.close()  # Will cancel the blocked operations.
      for t in threads:
        t.join()

  def testBigEnqueueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(5, dtypes_lib.int32, ((),))
      elem = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      enq = q.enqueue_many((elem,))
      deq = q.dequeue()
      size_op = q.size()

      enq_done = []

      def blocking_enqueue():
        enq_done.append(False)
        # This will fill the queue and then block until enough dequeues happen.
        self.evaluate(enq)
        enq_done.append(True)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()

      # The enqueue should start and then block.
      results = []
      results.append(
          self.evaluate(deq))  # Will only complete after the enqueue starts.
      self.assertEqual(len(enq_done), 1)
      self.assertEqual(self.evaluate(size_op), 5)

      for _ in range(3):
        results.append(self.evaluate(deq))

      time.sleep(0.1)
      self.assertEqual(len(enq_done), 1)
      self.assertEqual(self.evaluate(size_op), 5)

      # This dequeue will unblock the thread.
      results.append(self.evaluate(deq))
      time.sleep(0.1)
      self.assertEqual(len(enq_done), 2)
      thread.join()

      for i in range(5):
        self.assertEqual(self.evaluate(size_op), 5 - i)
        results.append(self.evaluate(deq))
        self.assertEqual(self.evaluate(size_op), 5 - i - 1)

      self.assertAllEqual(elem, results)

  def testBigDequeueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(2, dtypes_lib.int32, ((),))
      elem = np.arange(4, dtype=np.int32)
      enq_list = [q.enqueue((e,)) for e in elem]
      deq = q.dequeue_many(4)

      results = []

      def blocking_dequeue():
        # Will only complete after 4 enqueues complete.
        results.extend(self.evaluate(deq))

      thread = self.checkedThread(target=blocking_dequeue)
      thread.start()
      # The dequeue should start and then block.
      for enq in enq_list:
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.assertEqual(len(results), 0)
        self.evaluate(enq)

      # Enough enqueued to unblock the dequeue
      thread.join()
      self.assertAllEqual(elem, results)

  def testDtypes(self):
    with self.cached_session() as sess:
      dtypes = [
          dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
          dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8, dtypes_lib.int64,
          dtypes_lib.bool, dtypes_lib.complex64, dtypes_lib.complex128
      ]
      shape = (32, 4, 128)
      q = data_flow_ops.PaddingFIFOQueue(32, dtypes, [shape[1:]] * len(dtypes))

      input_tuple = []
      for dtype in dtypes:
        np_dtype = dtype.as_numpy_dtype
        np_array = np.random.randint(-10, 10, shape)
        if dtype == dtypes_lib.bool:
          np_array = np_array > 0
        elif dtype in (dtypes_lib.complex64, dtypes_lib.complex128):
          np_array = np.sqrt(np_array.astype(np_dtype))
        else:
          np_array = np_array.astype(np_dtype)
        input_tuple.append(np_array)

      q.enqueue_many(input_tuple).run()

      output_tuple_t = q.dequeue_many(32)
      output_tuple = self.evaluate(output_tuple_t)

      for (input_elem, output_elem) in zip(input_tuple, output_tuple):
        self.assertAllEqual(input_elem, output_elem)

  def testUnknownRank(self):
    with self.assertRaisesRegex(ValueError, "must have a defined rank"):
      data_flow_ops.PaddingFIFOQueue(32, [dtypes_lib.float32],
                                     [tensor_shape.TensorShape(None)])


class QueueFromListTest(test.TestCase):

  def testQueueFromListShapes(self):
    which = constant_op.constant(1)

    def _cmp(expected, *shapes):
      qs = [
          data_flow_ops.PaddingFIFOQueue(10, [dtypes_lib.float32],
                                         [tensor_shape.TensorShape(s)])
          for s in shapes
      ]
      s_expected = tensor_shape.TensorShape(expected)
      s = data_flow_ops.QueueBase.from_list(which, qs).shapes[0]
      if s_expected.ndims is None:
        self.assertEqual(s_expected.ndims, s.ndims)
      else:
        self.assertEqual(s_expected.as_list(), s.as_list())

    _cmp(None, [1, None], [None])
    _cmp([None], [1], [2])
    _cmp([1, None], [1, 1], [1, 2])
    _cmp([1, None], [1, 1], [1, None])
    _cmp([None, None], [None, 1], [1, None])
    _cmp([1], [1], [1], [1])
    _cmp([None], [1], [None], [1])
    _cmp(None, [1, None], [1], [1])

  def testQueueFromListShapesMultipleComponents(self):
    q_u_u = data_flow_ops.PaddingFIFOQueue(
        10, [dtypes_lib.float32, dtypes_lib.int32],
        [tensor_shape.TensorShape([None]), tensor_shape.TensorShape([None])])
    q_u_f = data_flow_ops.PaddingFIFOQueue(
        10, [dtypes_lib.float32, dtypes_lib.int32],
        [tensor_shape.TensorShape([None]), tensor_shape.TensorShape([1, 2])])
    q_f_f = data_flow_ops.PaddingFIFOQueue(
        10, [dtypes_lib.float32, dtypes_lib.int32],
        [tensor_shape.TensorShape([3, 4]), tensor_shape.TensorShape([1, 2])])
    which = constant_op.constant(1)

    s_cmp_1 = data_flow_ops.QueueBase.from_list(which,
                                                [q_u_u, q_u_u, q_u_u]).shapes
    self.assertEqual([1, 1], [x.ndims for x in s_cmp_1])
    self.assertEqual([None, None], [x.as_list()[0] for x in s_cmp_1])

    s_cmp_2 = data_flow_ops.QueueBase.from_list(which,
                                                [q_u_u, q_u_u, q_u_f]).shapes
    self.assertEqual([1, None], [x.ndims for x in s_cmp_2])
    self.assertEqual([None], s_cmp_2[0].as_list())

    s_cmp_3 = data_flow_ops.QueueBase.from_list(which, [q_f_f, q_f_f]).shapes
    self.assertEqual([2, 2], [x.ndims for x in s_cmp_3])
    self.assertEqual([[3, 4], [1, 2]], [x.as_list() for x in s_cmp_3])


if __name__ == "__main__":
  test.main()
