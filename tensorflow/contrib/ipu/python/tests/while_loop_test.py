# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import gradient_descent
from tensorflow.python.platform import googletest
from tensorflow.contrib.ipu import ipu_compiler

class WhileLoopTest(test_util.TensorFlowTestCase):

  def testWhileLoopTupleOfTuples(self):
    # This test makes sure that we can handle tuple of tuples for while loops
    random_seed.set_random_seed(1)
    dataType = dtypes.float32
    num_input = 14
    timesteps = 2
    num_units = 128

    def RNN(x):
      # Define a GRU cell with tensorflow
      gru_cell = nn.rnn_cell.GRUCell(num_units, name="GRU")
      # Get gru cell output
      outputs, states = nn.dynamic_rnn(gru_cell, x, dtype=dataType)
      return outputs[-1]

    def my_net(X, Y):
      # Forward pass
      logits = RNN(X)
      # Loss
      cross_entropy = math_ops.reduce_mean(nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
      # Training
      train = gradient_descent.GradientDescentOptimizer(0.01).minimize(cross_entropy)
      return [cross_entropy, train]

    with ops.device('cpu'):
      X = array_ops.placeholder(dataType, [1, timesteps, num_input])
      Y = array_ops.placeholder(dataType, [1, timesteps, num_units])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[X, Y])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {X: np.ones(X.shape), Y: np.ones(Y.shape)})
      # Compare the value - check that the loss is within 1 of the expected
      # value obtained by running on XLA_CPU.
      self.assertAllClose(result[0], 621.9, rtol=1)

  def testGather(self):
    def my_net(p, i):
      # Forward pass
      a = array_ops.gather(p, i, axis=0)
      return [a]

    with ops.device('cpu'):
      X = array_ops.placeholder(dtypes.int32, [2, 4])
      Y = array_ops.placeholder(dtypes.int32, [2])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[X, Y])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {X: [[1,3,5,7],[0,2,4,6]], Y: [1, 0]})
      self.assertAllClose(result[0], [[0,2,4,6],[1,3,5,7]])

  def testGatherTransposed(self):
    def my_net(p, i):
      # Forward pass
      p = array_ops.transpose(p, [1, 0])
      a = array_ops.gather(p, i, axis=0)
      return [a]

    with ops.device('cpu'):
      X = array_ops.placeholder(dtypes.int32, [2, 4])
      Y = array_ops.placeholder(dtypes.int32, [2])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[X, Y])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {X: [[1,3,5,7],[0,2,4,6]], Y: [1, 0]})
      self.assertAllClose(result[0], [[3,2],[1,0]])

  def testInplaceOpsInRepeats(self):
    def my_net(x):
      def cond(i, x):
        return i < 3

      def body(i, x):
        i = i + 1
        x = nn.relu(x * x)
        return (i, x)

      i = 0
      return control_flow_ops.while_loop(cond, body, (i, x))

    with ops.device('cpu'):
      x = array_ops.placeholder(dtypes.float32, [4])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[x])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      (c, x) = sess.run(r, {x: np.full([4], 2)})
      self.assertEqual(c, 3)
      self.assertAllClose(x, np.full([4], 256))

  def testNestedWhileLoopsSimplified(self):
    def my_net(x):
      def cond(i, x):
        return i < 3

      def cond1(j, x):
        return j < 2

      def body1(j, x):
        j = j + 1
        x = x * 2
        return (j, x)

      def body(i, x):
        i = i + 1
        j = 0
        _, x = control_flow_ops.while_loop(cond1, body1, (j, x), maximum_iterations=10)
        return (i, x)

      i = 0
      a, b = control_flow_ops.while_loop(cond, body, (i, x), maximum_iterations=10)
      return (a, b)

    with ops.device('cpu'):
      x = array_ops.placeholder(dtypes.int32, [4])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[x])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      c, val = sess.run(r, {x: np.full([4], 2, dtype=np.int32)})
      self.assertEqual(c, 3)
      self.assertAllClose(val, np.full([4], 128))

  def testFusionsInWhileLoops(self):
    def my_net():
      def cond(i, x):
        return i < 3

      def body(i, loss):
        i = i + 1
        init = init_ops.random_normal_initializer(0.0, 1.0, seed=1, dtype=np.float32)
        x = variable_scope.get_variable("v2", dtype=np.float32, shape=[1, 4, 4, 2],
                                         initializer=init)
        with variable_scope.variable_scope("vs", use_resource=True):
          y = convolutional.conv2d(x, 2, 1, use_bias=True,
                                 kernel_initializer=init_ops.ones_initializer(),
                                 name='conv1')
          y = convolutional.conv2d(y, 2, 1, use_bias=True,
                                 kernel_initializer=init_ops.ones_initializer(),
                                 name='conv2')
          y = convolutional.conv2d(y, 2, 1, use_bias=True,
                                 kernel_initializer=init_ops.ones_initializer(),
                                 name='conv3')
        loss = math_ops.reduce_sum(y)
        optimizer = gradient_descent.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)
        with ops.control_dependencies([train]):
          i = array_ops.identity(i)
          loss = array_ops.identity(loss)
          return (i, loss)
      i = 0
      loss = 0.0
      return control_flow_ops.while_loop(cond, body, (i, loss), maximum_iterations=10)

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      c, val = sess.run(r, {})
      self.assertEqual(c, 3)

if __name__ == "__main__":
    googletest.main()
