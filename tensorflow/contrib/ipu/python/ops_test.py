from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.contrib import ipu
from tensorflow.python.training import gradient_descent


class ContribIpuOpsTest(test_util.TensorFlowTestCase):

  def testSummary(self):
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, [1], name="a")
      b = array_ops.placeholder(np.float32, [1], name="b")
      out = a + b

    summary = ipu.ops.ipu_compile_summary('comp', out)

    cfg = ipu.utils.create_ipu_config(profiling=True, type='IPU_MODEL')
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      fd = {
        a: [1.0],
        b: [2.0],
      }
      result, s = sess.run([out, summary], fd)
      self.assertAllClose(result, [3.0])
      self.assertTrue(len(s) > 100)

  def testCreateConfig(self):
    cfg = ipu.utils.create_ipu_config(type='IPU')
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))

    cfg = ipu.utils.create_ipu_config(type='IPU_MODEL')
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))

    cfg = ipu.utils.create_ipu_config(type='CPU')
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))

    with self.assertRaises(Exception):
      cfg = ipu.utils.create_ipu_config(type='Other')

  def testEventFetchAndStringDecode(self):
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, [1], name="a")
      b = array_ops.placeholder(np.float32, [1], name="b")
      out = a + b

    events = gen_ipu_ops.ipu_event_trace()

    cfg = ipu.utils.create_ipu_config(profiling=True, type='IPU_MODEL')
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      # Discard any existing events
      sess.run(events)

      fd = {
        a: [1.0],
        b: [2.0],
      }
      result = sess.run(out, fd)
      self.assertAllClose(result, [3.0])

      # 1x compile begin, 1x compile end, 1x load engine, 1x execute,
      # 1x device->host
      e = sess.run(events)
      self.assertEqual(len(e), 5)

      dump = ipu.utils.extract_all_strings_from_event_trace(e);
      self.assertTrue(len(dump) > 100)

  def testIpuScope(self):
    # 1: design is targetted at the device
    # 2: variables are resource variables
    # 3: training a while_loop is possible
    def RNN(x):
      lstm_cell = rnn_cell.BasicLSTMCell(1, forget_bias=1.0)
      outputs, states = rnn.dynamic_rnn(lstm_cell, x, dtype=np.float32)
      return outputs[-1]

    with ipu.ops.ipu_scope("/device:IPU:0"):

      a = array_ops.placeholder(np.float32, [1,8,1], name="a")
      b = array_ops.placeholder(np.float32, [1,8,1], name="b")

      c = variable_scope.get_variable('c', initializer=[1.0])

      logits = RNN(a) * c

      res = array_ops.reshape(logits, [1,8,1])

      l = losses.mean_squared_error(res, b)

      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(l)

    self.assertTrue("ResourceVariable" in str(type(c)))
    self.assertEqual(logits.device, "/device:IPU:0")

    cfg = ipu.utils.create_ipu_config(profiling=True, type='IPU_MODEL')
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      # Initialize and then discard events relating to initialization
      sess.run(variables.global_variables_initializer())

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
