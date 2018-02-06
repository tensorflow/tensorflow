# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util

class ConditionalTest(test_util.TensorFlowTestCase):

    def testSimpleCond(self):

        with tf.device("/device:XLA_IPU:0"):
            pcond = tf.placeholder(tf.bool, [], name="pred")
            pa = tf.placeholder(tf.float32, [], name="a")
            pb = tf.placeholder(tf.float32, [], name="b")
            pc = tf.placeholder(tf.float32, [], name="c")
            output = tf.cond(pcond,
                             true_fn=lambda: pa+pb+pc,
                             false_fn=lambda: pa-pb-pc)

        with tf.Session() as sess:

            fd = {pcond: True, pa: 1., pb: 2., pc: 3.}
            result = sess.run(output, fd)
            self.assertAllClose(result, 6)

            fd = {pcond: False, pa: 1., pb: 2., pc: 3.}
            result = sess.run(output, fd)
            self.assertAllClose(result, -4.)

    def testDifferentArgs(self):

      with tf.device("/device:XLA_IPU:0"):
        pcond = tf.placeholder(tf.bool, [], name="pred")
        pa = tf.placeholder(tf.float32, [], name="a")
        pb = tf.placeholder(tf.float32, [], name="b")
        pc = tf.placeholder(tf.float32, [], name="c")
        output = tf.cond(pcond,
                         true_fn=lambda: pa+pb,
                         false_fn=lambda: pb-pc)

      with tf.Session() as sess:

        fd = {pcond: True, pa: 1., pb: 2., pc: 3.}
        result = sess.run(output, fd)
        self.assertAllClose(result, 3.)

        fd = {pcond: False, pa: 1., pb: 2., pc: 3.}
        result = sess.run(output, fd)
        self.assertAllClose(result, -1.)

    def testReadResourceVar(self):

      with tf.device("/device:XLA_IPU:0"):
        with tf.variable_scope('vs', use_resource=True):
          pcond = tf.placeholder(tf.bool, [], name="pred")
          va = tf.get_variable("x", shape=[], dtype=tf.float32,
                               initializer=tf.constant_initializer(1))

          output = tf.cond(pcond,
                           true_fn=lambda: va.read_value(),
                           false_fn=lambda: tf.constant(0.))

      with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        fd = {pcond: True}
        result = sess.run(output, fd)
        self.assertAllClose(result, 1.)

        fd = {pcond: False}
        result = sess.run(output, fd)
        self.assertAllClose(result, 0.)


    def testWriteResourceVar(self):

      with tf.device("/device:XLA_IPU:0"):
        with tf.variable_scope('vs', use_resource=True):
          pcond = tf.placeholder(tf.bool, [], name="pred")
          va = tf.get_variable("x", shape=[], dtype=tf.float32,
                               initializer=tf.constant_initializer(1))

          output = tf.cond(pcond,
                           true_fn=lambda: tf.assign(va, 1.),
                           false_fn=lambda: tf.assign(va, 2.))

      with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        fd = {pcond: True}
        result = sess.run(output, fd)
        self.assertAllClose(result, 1.)

        self.assertAllClose(sess.run(va.read_value()), 1.)

        fd = {pcond: False}
        result = sess.run(output, fd)
        self.assertAllClose(result, 2.)

        self.assertAllClose(sess.run(va.read_value()), 2.)

if __name__ == "__main__":
    googletest.main()
