from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util


class ContribIpuOpsTest(test_util.TensorFlowTestCase):

  def testSummary(self):
    with tf.device("/device:IPU:0"):
      a = tf.placeholder(tf.float32, [1], name="a")
      b = tf.placeholder(tf.float32, [1], name="b")
      out = a + b

    summary = tf.contrib.ipu.ops.ipu_compile_summary('comp', out)

    cfg = tf.contrib.ipu.utils.create_ipu_config(True)
    with tf.Session(config=tf.ConfigProto(ipu_options=cfg)) as sess:
      fd = {
        a: [1.0],
        b: [2.0],
      }
      result, s = sess.run([out, summary], fd)
      self.assertAllClose(result, [3.0])
      self.assertTrue(len(s) > 100)

if __name__ == "__main__":
    googletest.main()
