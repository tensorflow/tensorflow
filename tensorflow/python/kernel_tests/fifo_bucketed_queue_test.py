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
      q = tf.FIFOBucketedQueue(2, 5, (tf.int32, tf.float32),
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
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      """, q.queue_ref.op.node_def)

if __name__ == "__main__":
  tf.test.main()
