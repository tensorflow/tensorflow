# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.compiler import xla
from tensorflow.contrib import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.training import gradient_descent
from tensorflow.python.platform import googletest
from tensorflow.python.ops import variables

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
      r = xla.compile(my_net, inputs=[X, Y])

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(r, {X: np.ones(X.shape), Y: np.ones(Y.shape)})
      # Compare the value - check that the loss is within 1 of the expected
      # value obtained by running on XLA_CPU.
      self.assertAllClose(result[0], 621.9, rtol=1)

if __name__ == "__main__":
    googletest.main()
