from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import json
import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.contrib.ipu import ipu_compiler
from tensorflow.contrib import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.framework import errors


class MultiIpuTest(test_util.TensorFlowTestCase):
  def testMultiIpu(self):
    def my_graph(pa, pb, pc):
      with ops.device("/device:IPU:0"):
        with ipu.ops.ipu_shard(0):
          o1 = pa + pb

        with ipu.ops.ipu_shard(1):
          o2 = pa + pc
          out = o1 + o2

      return [out]

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [2], name="a")
      pb = array_ops.placeholder(np.float32, [2], name="b")
      pc = array_ops.placeholder(np.float32, [2], name="c")
      report = gen_ipu_ops.ipu_event_trace()

    out = ipu_compiler.compile(my_graph, [pa, pb, pc])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:

      sess.run(report)

      fd = {pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]}
      result = sess.run(out, fd)
      self.assertAllClose(result[0], [3., 8.])

      rep = sess.run(report)

      evts = ipu.utils.extract_all_events(rep)
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_END:
          js = json.loads(evt.compile_end.tensor_map.decode('utf-8'))

          mods = list(js['mappings'].keys())
          self.assertEqual(len(mods), 1)

          tiles = set()
          for tensor in js['mappings'][mods[0]]:
            for tile in tensor['tiles']:
              tiles.add(tile['tile_id'])

          self.assertEqual(len(tiles), 2)
          self.assertTrue(0 in tiles)
          self.assertTrue(1216 in tiles)

  def testMultipleConfigureIpuShouldFail(self):
    def my_graph(pa, pb, pc):
      with ops.device("/device:IPU:0"):
        o1 = pa + pb
        o2 = pa + pc
        out = o1 + o2

      return [out]

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [2], name="a")
      pb = array_ops.placeholder(np.float32, [2], name="b")
      pc = array_ops.placeholder(np.float32, [2], name="c")
      report = gen_ipu_ops.ipu_event_trace()

    out = ipu_compiler.compile(my_graph, [pa, pb, pc])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    with self.assertRaises(Exception):
      cfg = ipu.utils.create_ipu_config(profiling=True)
      cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=True)
      ipu.utils.configure_ipu_system(cfg)

  def testNotEnoughIpus(self):
    def my_graph(pa, pb, pc):
      with ipu.ops.ipu_shard(0):
        o1 = pa + pb
      with ipu.ops.ipu_shard(1):
        o2 = pa + pc
      with ipu.ops.ipu_shard(2):
        out = o1 + o2
        return out

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [2], name="a")
      pb = array_ops.placeholder(np.float32, [2], name="b")
      pc = array_ops.placeholder(np.float32, [2], name="c")
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = ipu_compiler.compile(my_graph, [pa, pb, pc])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:
      with self.assertRaisesRegexp(errors.ResourceExhaustedError,
                                   'Trying to compile a graph for'):
        sess.run(out, {pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]})

  def testMultiIpuVariables(self):
    def my_graph(pa, pb, pc):
      with variable_scope.variable_scope("", use_resource=True):
        with ipu.ops.ipu_scope("/device:IPU:0"):
          with ipu.ops.ipu_shard(0):
            init1 = init_ops.constant_initializer([1.0, 3.0])
            v1 = variable_scope.get_variable(
                "v1", dtype=np.float32, shape=[2], initializer=init1)
            o1 = pa + pb + v1

          with ipu.ops.ipu_shard(1):
            init2 = init_ops.constant_initializer([1.0, 2.0])
            v2 = variable_scope.get_variable(
                "v2", dtype=np.float32, shape=[2], initializer=init2)
            o2 = pa + pc + v2
            out = o1 + o2

      return [out]

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [2], name="a")
      pb = array_ops.placeholder(np.float32, [2], name="b")
      pc = array_ops.placeholder(np.float32, [2], name="c")
      report = gen_ipu_ops.ipu_event_trace()

    out = ipu_compiler.compile(my_graph, [pa, pb, pc])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:

      sess.run(report)
      sess.run(variables.global_variables_initializer())
      sess.run(report)

      fd = {pa: [1., 1.], pb: [0., 1.], pc: [1., 5.]}
      result = sess.run(out, fd)
      self.assertAllClose(result[0], [5., 13.])

      rep = sess.run(report)

      evts = ipu.utils.extract_all_events(rep)
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_END:
          js = json.loads(evt.compile_end.tensor_map.decode('utf-8'))

          mods = list(js['mappings'].keys())
          self.assertEqual(len(mods), 1)

          tiles = set()
          for tensor in js['mappings'][mods[0]]:
            for tile in tensor['tiles']:
              tiles.add(tile['tile_id'])

          self.assertEqual(len(tiles), 2)
          self.assertTrue(0 in tiles)
          self.assertTrue(1216 in tiles)

  def testMultiIpuTraining(self):
    def my_graph(inp, lab):
      with ops.device("/device:IPU:0"):
        with ipu.ops.ipu_shard(0):
          x = convolutional.conv2d(inp, 8, 3, padding='same', name="convA")

        with ipu.ops.ipu_shard(1):
          x = convolutional.conv2d(x, 8, 1, padding='same', name="convB")
          x = math_ops.reduce_mean(x, axis=[1, 2])

          loss = nn.softmax_cross_entropy_with_logits(logits=x, labels=lab)
          loss = math_ops.reduce_mean(loss)

        opt = ipu.sharded_optimizer.ShardedOptimizer(
            gradient_descent.GradientDescentOptimizer(0.000001))
        train = opt.minimize(loss)

      return [loss, train]

    with ops.device('cpu'):
      inp = array_ops.placeholder(np.float32, [1, 32, 32, 4], name="data")
      lab = array_ops.placeholder(np.float32, [1, 8], name="labels")
      report = gen_ipu_ops.ipu_event_trace()

    out = ipu_compiler.compile(my_graph, [inp, lab])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:

      sess.run(report)
      sess.run(variables.global_variables_initializer())
      sess.run(report)

      fd = {inp: np.ones([1, 32, 32, 4]), lab: np.ones([1, 8])}
      sess.run(out, fd)

      rep = sess.run(report)

      num_compiles = 0

      evts = ipu.utils.extract_all_events(rep)
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_END:
          num_compiles = num_compiles + 1

      self.assertEqual(num_compiles, 1)

      compile_report = ipu.utils.extract_compile_reports(rep)
      self.assertEqual(len(compile_report), 1)

      js = json.loads(compile_report[0][1])
      cs_list = js['computeSets']['names']

      # There are 2 inter-ipu copies
      n_inter_ipu_copies = 0
      for n in cs_list:
        if fnmatch.fnmatch(n, '*custom-call*/GlobalPre/*'):
          n_inter_ipu_copies = n_inter_ipu_copies + 1

      self.assertEqual(n_inter_ipu_copies, 2)

  def testConvAndBiasAddDifferentIPUs(self):
    def my_graph(inp, bias):
      with ops.device("/device:IPU:0"):
        with ipu.ops.ipu_shard(0):
          x = convolutional.conv2d(
              inp, 8, 3, padding='same', name="conv", use_bias=False)

        with ipu.ops.ipu_shard(1):
          x = nn_ops.bias_add(x, bias, name='biasAdd')

      return x

    with ops.device('cpu'):
      inp = array_ops.placeholder(np.float32, [1, 32, 32, 4], name="data")
      bias = array_ops.placeholder(np.float32, [8], name="bias")
      report = gen_ipu_ops.ipu_event_trace()

    out = ipu_compiler.compile(my_graph, [inp, bias])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:

      sess.run(report)
      sess.run(variables.global_variables_initializer())
      sess.run(report)

      fd = {inp: np.ones([1, 32, 32, 4]), bias: np.ones([8])}
      sess.run(out, fd)

      rep = sess.run(report)

      num_compiles = 0
      ge_list = []
      evts = ipu.utils.extract_all_events(rep)
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_END:
          num_compiles = num_compiles + 1
          ge_list = tu.get_all_global_exchange_from_json_report(evt)

      self.assertEqual(num_compiles, 1)

      # There is 1 piece of global exchange (aprt from progId)
      wl = ['progIdCopy/GlobalPreAll', '*_to_/custom-call/GlobalPreAll']
      self.assertTrue(tu.check_all_compute_sets_and_list(ge_list, wl))


if __name__ == "__main__":
  googletest.main()
