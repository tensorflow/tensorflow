# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

class IpuIpuModelTest(test_util.TensorFlowTestCase):

  def testIpuWhilePerfTest(self):
    def cond(i, v):
      return tf.less(i, 10)

    def body(i, v):
      v = v + i
      i = i + 1
      return (i, v)

    with tf.device("/device:IPU:0"):
      i = tf.constant(0)
      v = tf.placeholder(tf.int32, [500])
      r = tf.while_loop(cond, body, [i, v])

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:

        result = sess.run(r, {v:np.zeros([500], np.int32)})
        self.assertAllClose(result[1], np.broadcast_to(45, [500]))

        rep = sess.run(report)

        print(tu.extract_all_strings_from_event_trace(rep))

if __name__ == "__main__":
    googletest.main()
