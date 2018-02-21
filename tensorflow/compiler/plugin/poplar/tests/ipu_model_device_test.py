# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.core.protobuf import config_pb2

class IpuIpuModelTest(test_util.TensorFlowTestCase):

    def testIpuModelDevice(self):
        with tf.device("/device:IPU:0"):
            opts = config_pb2.IPUOptions()
            dev = opts.device_config.add()
            dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
            with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:
                pa = tf.placeholder(tf.float32, [2,2], name="a")
                pb = tf.placeholder(tf.float32, [2,2], name="b")
                output = pa + pb

                fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
                result = sess.run(output, fd)
                self.assertAllClose(result, [[1.,2.],[6.,8.]])

if __name__ == "__main__":
    googletest.main()
