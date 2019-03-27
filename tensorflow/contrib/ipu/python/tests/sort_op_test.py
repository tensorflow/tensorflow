import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class ContribIpuOpsTest(test_util.TensorFlowTestCase):
  def testSortOp(self):
    with ops.device("/device:IPU:0"):
      with tf.Session() as session:
        t1 = tf.random_uniform([1000], dtype=tf.float32)
        t2 = tf.contrib.framework.sort(t1, name="t2")
        h1, h2 = session.run([t1, t2])
        self.assertEqual(sorted(h1), list(h2))


if __name__ == "__main__":
  googletest.main()
