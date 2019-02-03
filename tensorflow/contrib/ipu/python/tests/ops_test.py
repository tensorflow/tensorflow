from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.contrib.compiler import xla
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.contrib import ipu
from tensorflow.python.training import gradient_descent

def count_compile_end_events(events):
  fn = (lambda x: 1 if x.type==IpuTraceEvent.COMPILE_END and len(x.compile_end.compilation_report) > 10
                  else 0)
  return sum(map(fn, events))

class ContribIpuOpsTest(test_util.TensorFlowTestCase):

  def testSummary(self):
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, [1], name="a")
      b = array_ops.placeholder(np.float32, [1], name="b")
      out = a + b

    summary = ipu.ops.ipu_compile_summary('comp', out)

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      fd = {
        a: [1.0],
        b: [2.0],
      }
      result, s = sess.run([out, summary], fd)
      self.assertAllClose(result, [3.0])
      self.assertTrue(len(s) > 100)

  def testCreateConfig(self):
    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, [1,1])
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))
    self.assertTrue(len(cfg.device_config), 2)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.auto_select_ipus(cfg, [4, 4])
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))
    self.assertTrue(len(cfg.device_config), 2)
    self.assertTrue(cfg.device_config[0].auto_count, 4)
    self.assertTrue(cfg.device_config[1].auto_count, 4)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.select_ipus(cfg, [2, 3])
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))
    self.assertTrue(len(cfg.device_config), 2)
    self.assertTrue(cfg.device_config[0].cfg_index, 2)
    self.assertTrue(cfg.device_config[1].cfg_index, 3)

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_compilation_options(cfg, {'A':'B', 'C':'D'})
    self.assertTrue(len(cfg.compilation_options), 2)
    self.assertTrue(cfg.compilation_options[0].option, "A")
    self.assertTrue(cfg.compilation_options[0].value, "B")
    self.assertTrue(cfg.compilation_options[1].option, "C")
    self.assertTrue(cfg.compilation_options[1].value, "D")

    with self.assertRaises(Exception):
      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.auto_select_ipus(cfg, [4, 4])
      cfg = ipu.utils.select_ipus(cfg, [4, 4])

    with self.assertRaises(Exception):
      cfg = ipu.utils.create_ipu_config(profiling=True, enable_ipu_events=True)

  def testEventFetchAndStringDecode(self):
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, [1], name="a")
      b = array_ops.placeholder(np.float32, [1], name="b")
      out = a + b

    events = gen_ipu_ops.ipu_event_trace()

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      # Discard any existing events
      sess.run(events)

      fd = {
        a: [1.0],
        b: [2.0],
      }
      result = sess.run(out, fd)
      self.assertAllClose(result, [3.0])

      # 1x compile begin, 1x compile end, 1x load engine, 1x execute
      e = sess.run(events)
      self.assertEqual(len(e), 4)

      dump = ipu.utils.extract_all_strings_from_event_trace(e);
      self.assertTrue(len(dump) > 100)

  def testIpuSimpleScopeAndExecutionReport(self):
    def my_net(a, b):
      c = a + b
      return [c]

    with ops.device('cpu'):
      a = array_ops.placeholder(np.float32, [1], name="a")
      b = array_ops.placeholder(np.float32, [1], name="b")
      events = gen_ipu_ops.ipu_event_trace()

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[a, b])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:

      fd = {
        a: [1],
        b: [2],
      }

      sess.run(events)

      res = sess.run(r[0], fd)
      self.assertEqual(res, [3])

      e = sess.run(events)
      evts = ipu.utils.extract_all_events(e)
      self.assertEqual(count_compile_end_events(evts), 1)

      t_list = ipu.utils.extract_execution_state_timing_list_from_events(e)
      self.assertEqual(len(t_list), 1)
      self.assertEqual(type(t_list), list)
      self.assertEqual(type(t_list[0]), tuple)
      self.assertTrue(t_list[0][0].startswith("cluster"))

      lines = t_list[0][1].split()
      for l in lines:
        m = re.match(r'\d+,\d+,\d+,\d+,\d+$', l)
        self.assertTrue(m)

  def testIpuWhileScope(self):
    # 1: design is targetted at the device
    # 2: variables are resource variables
    # 3: training a while_loop is possible
    def my_net(a, b):
      c = variable_scope.get_variable('c', initializer=[1.0])
      self.assertTrue("ResourceVariable" in str(type(c)))

      lstm_cell = rnn_cell.LSTMCell(1, forget_bias=1.0)
      outputs, states = rnn.dynamic_rnn(lstm_cell, a, dtype=np.float32)

      logits = outputs[-1] * c
      self.assertEqual(logits.device, "/device:IPU:0")

      res = array_ops.reshape(logits, [1,8,1])

      l = losses.mean_squared_error(res, b)

      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(l)

      return [l, train]

    with ops.device('cpu'):
      a = array_ops.placeholder(np.float32, [1,8,1], name="a")
      b = array_ops.placeholder(np.float32, [1,8,1], name="b")

    with ipu.ops.ipu_scope("/device:IPU:0"):

      l = xla.compile(my_net, inputs=[a, b])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      # Initialize and then discard events relating to initialization
      sess.run(variables.global_variables_initializer())

      fd = {
        a: [[[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]],
        b: [[[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]],
      }

      l_initial = sess.run([l], fd)

      for _ in range(100):
        _ = sess.run([l], fd)

      l_final = sess.run([l], fd)

      self.assertTrue(l_initial > l_final)

  def testInitializerDeviceChange(self):

    inp = array_ops.placeholder(np.float32, [1,8,8,4])
    with ipu.ops.ipu_scope("/device:IPU:0"):
      out = convolutional.conv2d(inp, 8, 1)

    events = gen_ipu_ops.ipu_event_trace()

    ipu.utils.move_variable_initialization_to_cpu()

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      # Discard any pending events
      sess.run(events)

      # Run initializer (should be on CPU)
      sess.run(variables.global_variables_initializer())

      e = sess.run(events)
      self.assertEqual(len(e), 2) # compile begin/end, no load/execute


  def testMultiScopeTest(self):
    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, [2, 2])
      y = array_ops.placeholder(np.float32, [2, 2])
      report = gen_ipu_ops.ipu_event_trace()

    with ipu.ops.ipu_scope('/device:IPU:0'):
      z = math_ops.matmul(x, y)
    with ipu.ops.ipu_scope('/device:IPU:0'):
      z2 = math_ops.matmul(x, z)

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      sess.run(report)
      result = sess.run(z2, {x: np.ones([2, 2]),  y: np.ones([2, 2])})

      self.assertAllEqual(result, [[4, 4], [4, 4]])

      rep = sess.run(report)
      evts = ipu.utils.extract_all_types_from_event_trace(rep)

      num_compiles = 0
      num_executions = 0
      for e in evts:
        if e == IpuTraceEvent.COMPILE_END:
          num_compiles += 1
        if e == IpuTraceEvent.EXECUTE:
          num_executions += 1

      self.assertEqual(num_compiles, 1)
      self.assertEqual(num_executions, 1)


if __name__ == "__main__":
    googletest.main()
