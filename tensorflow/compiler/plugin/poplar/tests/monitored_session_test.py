# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.training import monitored_session as ms

class IpuMonitoredSessionTest(test_util.TensorFlowTestCase):

    def testMonitoredSession(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [2,2], name="a")
            pb = tf.placeholder(tf.float32, [2,2], name="b")
            output = pa + pb

        with ms.MonitoredSession(
            session_creator=ms.ChiefSessionCreator()) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            result = sess.run(output, fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])

            fd = {pa: [[0.,0.],[1.,1.]], pb: [[2.,1.],[4.,5.]]}
            result = sess.run(output, fd)
            self.assertAllClose(result, [[2.,1.],[5.,6.]])

    def testTrainingLoop(self):

        # Model
        with tf.device("/device:IPU:0"):
          with tf.variable_scope("vs", use_resource=True):
            x = tf.placeholder(tf.float32, [4,1,4], name="a")
            l = tf.placeholder(tf.float32, [4,1,1], name="b")

            y = tf.layers.dense(x, 1, activation=tf.nn.sigmoid)


            loss = tf.losses.log_loss(l, y)
            train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

            init = tf.global_variables_initializer()

        # Test data
        image_data = [[[1, 1, 1, 1]],
                      [[2, 2, 2, 2]],
                      [[3, 3, 3, 3]],
                      [[4, 4, 4, 4]]]
        label_data = [[[1]],
                      [[2]],
                      [[3]],
                      [[4]]]

        # Run training.
        with ms.MonitoredTrainingSession(is_chief=True,
                                         chief_only_hooks=None,
                                         save_summaries_steps=None,
                                         save_summaries_secs=None) as sess:
            sess.run(init)
            measured_loss,_ = sess.run([loss,train_op],
                                       feed_dict={x: image_data, l: label_data})
            self.assertTrue(measured_loss < 5.0)

    def testMonitoredSessionStopAtStepHook(self):
      with tf.device("/device:IPU:0"):
        pa = tf.placeholder(tf.float32, [2,2], name="a")
        pb = tf.placeholder(tf.float32, [2,2], name="b")
        output = pa + pb

      with tf.variable_scope('gs', use_resource=True):
        tf.train.create_global_step()

      hook = tf.train.StopAtStepHook(num_steps=2)

      with ms.MonitoredSession(
              session_creator=ms.ChiefSessionCreator(),
              hooks=[hook]) as sess:

        fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[1.,2.],[6.,8.]])

        fd = {pa: [[0.,0.],[1.,1.]], pb: [[2.,1.],[4.,5.]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[2.,1.],[5.,6.]])

if __name__ == "__main__":
    googletest.main()
