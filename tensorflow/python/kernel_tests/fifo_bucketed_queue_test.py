"""Tests for tensorflow.ops.data_flow_ops.FIFOBucketedQueue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class FIFOBucketedQueueTest(tf.test.TestCase):

  def testConstructor(self):
    with tf.Graph().as_default():
      q = tf.FIFOBucketedQueue(2, 10, 20, (tf.int32, tf.float32),
                               shapes=(tf.TensorShape([1]),
                                       tf.TensorShape([5, 8])), name="Q")
    self.assertTrue(isinstance(q.queue_ref, tf.Tensor))
    self.assertEquals(tf.string_ref, q.queue_ref.dtype)
    self.assertProtoEquals("""
      name:'Q' op:'FIFOBucketedQueue'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list {
        shape { dim { size: 1 } }
        shape { dim { size: 5 }
                dim { size: 8 } }
      } } }
      attr { key: 'buckets' value { i: 2 } }
      attr { key: 'batch_size' value { i: 10 } }
      attr { key: 'capacity' value { i: 20 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      """, q.queue_ref.op.node_def)

  def testMultiEnqueueAndDequeueMany(self):
    with self.test_session() as sess:
      batch_size = 2
      q = tf.FIFOBucketedQueue(2, batch_size, 10, (tf.int32, tf.float32), shapes=((), ()))
      elems = [(0, 10.0), (1, 20.0), (0, 30.0)]
      enqueue_ops = [q.enqueue((x, y)) for x, y in elems]
      dequeued_t = q.dequeue_many(2)

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      b_val, y_val = sess.run(dequeued_t)

      self.assertAllEqual(b_val, [0, 0])
      self.assertEqual(y_val[0], elems[0][1])
      self.assertEqual(y_val[1], elems[2][1])

  def testMultiEnqueueManyAndDequeueMany(self):
    with self.test_session() as sess:
      batch_size = 2
      q = tf.FIFOBucketedQueue(2, batch_size, 10, (tf.int32, tf.float32), shapes=((), ()))
      bucket_ids = [0, 1, 0]
      float_elems = [10.0, 20.0, 30.0]
      enqueue_op = q.enqueue_many((bucket_ids, float_elems))
      dequeued_t = q.dequeue_many(2)

      enqueue_op.run()

      b_val, y_val = sess.run(dequeued_t)

      self.assertAllEqual(b_val, [0, 0])
      self.assertEqual(y_val[0], float_elems[0])
      self.assertEqual(y_val[1], float_elems[2])

if __name__ == "__main__":
  tf.test.main()
