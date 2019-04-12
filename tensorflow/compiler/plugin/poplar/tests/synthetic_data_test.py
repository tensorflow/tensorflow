# Copyright 2017, 2018, 2019 Graphcore Ltd
#

import os
import numpy as np

from tensorflow.python.platform import googletest
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent

import test_utils as tu


class IpuXlaVariableTestSyntheticData(test_util.TensorFlowTestCase):
  # This test is in its separate class to prevent messing up the enviroment for
  # other tests.

  def testResourceCountsAreCorrect(self):
    # Same test as above, but no copying to and form device should occur.
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        w1 = variable_scope.get_variable(
            "w1",
            shape=[4, 2],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(
                np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)))
        b1 = variable_scope.get_variable(
            "b1",
            shape=[2],
            dtype=np.float32,
            trainable=False,
            initializer=init_ops.constant_initializer(
                np.array([2, 3], dtype=np.float32)))
        w2 = variable_scope.get_variable(
            "w2",
            shape=[2, 2],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(
                np.array([[1, 2], [3, 4]], dtype=np.float32)))
        b2 = variable_scope.get_variable(
            "b2",
            shape=[2],
            dtype=np.float32,
            trainable=False,
            initializer=init_ops.constant_initializer(
                np.array([2, 3], dtype=np.float32)))

      x = array_ops.placeholder(np.float32, shape=[1, 4])
      y = math_ops.matmul(x, w1) + b1
      y = math_ops.matmul(y, w2) + b2

      loss = math_ops.reduce_sum(y)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(loss)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run([train, loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)
      self.assertEqual(len(list(io_evts)), 0)

      # Explicitly fetch the first set of weights and biases
      sess.run([w1, b1])

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)
      self.assertEqual(len(list(io_evts)), 0)


if __name__ == "__main__":
  os.environ["TF_POPLAR_FLAGS"] = "--use_synthetic_data --use_ipu_model"
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
