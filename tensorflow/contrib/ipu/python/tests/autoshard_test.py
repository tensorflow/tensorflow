# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.compiler import xla
from tensorflow.contrib.ipu.python import autoshard
from tensorflow.contrib.ipu.python import sharded_optimizer as so
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops as nn
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent as gd

class AutoshardTest(test_util.TensorFlowTestCase):

    def testSimpleXlaCompileInference(self):

        def my_model(inp):
          output = inp * inp
          return [output]

        with ops.device("cpu"):
          inp = array_ops.placeholder(np.float32, [], name="a")

        with ops.device("/device:IPU:0"):
            out = xla.compile(my_model, inputs=[inp])

        autoshard.automatic_sharding(2, inp, out[0])

        op_list = ops.get_default_graph().get_operations()
        for o in op_list:
          if o.device == '/device:IPU:0' and o.type != 'NoOp':
            self.assertTrue(o.get_attr('_XlaSharding') is not None)


    def testSimpleXlaCompileTraining(self):

      def my_model(x, y):
        x = convolutional.conv2d(x, 8, 3, padding='same', name="conv1",
                                 use_bias=False)
        x = convolutional.conv2d(x, 8, 3, padding='same', name="conv2",
                                 use_bias=False)
        x = convolutional.conv2d(x, 8, 3, padding='same', name="conv3",
                                 use_bias=False)
        x = math_ops.reduce_max(x,  axis=[1, 2])

        cross_entropy = nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
        loss = math_ops.reduce_mean(cross_entropy)
        optim = so.ShardedOptimizer(gd.GradientDescentOptimizer(0.01))
        train = optim.minimize(cross_entropy)
        return [loss, train]

      with ops.device("cpu"):
        inp = array_ops.placeholder(np.float32, [1, 12, 12, 4], name="data")
        lab = array_ops.placeholder(np.float32, [1, 8], name="labl")

      with ops.device("/device:IPU:0"):
        out = xla.compile(my_model, inputs=[inp, lab])

      autoshard.automatic_sharding(2, inp, out[0])

      op_set = autoshard.dependencies([out[0]])

      for o in op_set:
        if o.device == '/device:IPU:0' and o.type not in ['NoOp', 'Identity']:
          self.assertTrue(o.get_attr('_XlaSharding') is not None)

    def testSimpleTraining(self):

      def my_model(x, y):
        x = convolutional.conv2d(x, 8, 3, padding='same', name="conv1",
                                 use_bias=False)
        x = convolutional.conv2d(x, 8, 3, padding='same', name="conv2",
                                 use_bias=False)
        x = convolutional.conv2d(x, 8, 3, padding='same', name="conv3",
                                 use_bias=False)
        x = math_ops.reduce_max(x,  axis=[1, 2])

        cross_entropy = nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
        loss = math_ops.reduce_mean(cross_entropy)
        optim = so.ShardedOptimizer(gd.GradientDescentOptimizer(0.01))
        train = optim.minimize(cross_entropy)
        return [loss, train]

      with ops.device("cpu"):
        inp = array_ops.placeholder(np.float32, [1, 12, 12, 4], name="data")
        lab = array_ops.placeholder(np.float32, [1, 8], name="labl")

      with ops.device("/device:IPU:0"):
        l, t = my_model(inp, lab)

      autoshard.automatic_sharding(2, inp, l, [t])

      op_set = autoshard.dependencies([l, t])

      for o in op_set:
        if o.device == '/device:IPU:0' and o.type not in ['NoOp', 'Identity']:
          self.assertTrue(o.get_attr('_XlaSharding') is not None)



if __name__ == "__main__":
    googletest.main()
