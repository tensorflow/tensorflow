# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.core.protobuf import config_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

class IpuIpuModelTest(test_util.TensorFlowTestCase):

    def testIpuModelDevice(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [2,2], name="a")
            pb = tf.placeholder(tf.float32, [2,2], name="b")
            output = pa + pb

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            result = sess.run(output, fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])

    def testIpuModelDeviceWithNoReport(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [2,2], name="a")
            pb = tf.placeholder(tf.float32, [2,2], name="b")
            output = pa + pb

        with tf.device('cpu'):
            with tf.control_dependencies([output]):
                report = gen_ipu_ops.ipu_summary()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            result, rep = sess.run([output, report], fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])
            self.assertTrue(len(rep) == 0)

    def testIpuModelDeviceWithReport(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [2,2], name="a")
            pb = tf.placeholder(tf.float32, [2,2], name="b")
            output = pa + pb

        with tf.device('cpu'):
            with tf.control_dependencies([output]):
                report = gen_ipu_ops.ipu_summary()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.enable_profile = True
        with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            result, rep = sess.run([output, report], fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])
            self.assertTrue(len(rep) == 1)

    def testIpuModelDeviceWithMultipleReport(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [2,2], name="a")
            pb = tf.placeholder(tf.float32, [2,2], name="b")
            out1 = pa + pb
            out2 = pa - pb

        with tf.device('cpu'):
            with tf.control_dependencies([out1, out2]):
                report = gen_ipu_ops.ipu_summary()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.enable_profile = True
        with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            result = sess.run(out1, fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])

            result, rep = sess.run([out2, report], fd)
            self.assertAllClose(result, [[1.,0.],[-2.,-2.]])

            self.assertTrue(len(rep) == 2)

if __name__ == "__main__":
    googletest.main()
