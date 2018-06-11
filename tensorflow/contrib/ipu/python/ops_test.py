from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

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

  def testEventFetchAndStringDecode(self):
    with tf.device("/device:IPU:0"):
      a = tf.placeholder(tf.float32, [1], name="a")
      b = tf.placeholder(tf.float32, [1], name="b")
      out = a + b

    events = gen_ipu_ops.ipu_event_trace()

    cfg = tf.contrib.ipu.utils.create_ipu_config(True)
    with tf.Session(config=tf.ConfigProto(ipu_options=cfg)) as sess:
      # Discard any existing events
      sess.run(events)

      fd = {
        a: [1.0],
        b: [2.0],
      }
      result = sess.run(out, fd)
      self.assertAllClose(result, [3.0])

      # Event trace (1x compile, 1x load engine, 1x execute)
      e = sess.run(events)
      self.assertEqual(len(e), 3)

      dump = tf.contrib.ipu.utils.extract_all_strings_from_event_trace(e);
      self.assertTrue(len(dump) > 100)

  def testIpuScope(self):
    # 1: design is targetted at the device
    # 2: variables are resource variables
    # 3: training a while_loop is possible
    def RNN(x):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(1, forget_bias=1.0)
      outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
      return outputs[-1]

    with tf.contrib.ipu.ops.ipu_scope("/device:IPU:0"):

      a = tf.placeholder(tf.float32, [1,8,1], name="a")
      b = tf.placeholder(tf.float32, [1,8,1], name="b")

      c = tf.get_variable('c', shape=[1],
                          initializer=tf.constant_initializer(1.0))

      logits = RNN(a) * c

      res = tf.reshape(logits, [1,8,1])

      l = tf.losses.mean_squared_error(res, b)

      optimizer = tf.train.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(l)

    self.assertTrue("ResourceVariable" in str(type(c)))
    self.assertEqual(logits.device, "/device:IPU:0")

    cfg = tf.contrib.ipu.utils.create_ipu_config(True)
    with tf.Session(config=tf.ConfigProto(ipu_options=cfg)) as sess:
      # Initialize and then discard events relating to initialization
      sess.run(tf.global_variables_initializer())

      fd = {
        a: [[[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]],
        b: [[[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]],
      }

      l_initial, _ = sess.run([l, train], fd)

      for _ in range(100):
        _, _ = sess.run([l, train], fd)

      l_final, _ = sess.run([l, train], fd)

      self.assertTrue(l_initial > l_final)


if __name__ == "__main__":
    googletest.main()
