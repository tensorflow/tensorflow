from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import re

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.contrib.compiler import xla
from tensorflow.contrib import ipu
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent

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

    out = xla.compile(my_graph, [pa, pb, pc])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2, True)
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:

      sess.run(report)

      fd = {pa: [1., 1.], pb: [0., 1.], pc:[1., 5.]}
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


  def testMultiIpuVariables(self):
    def my_graph(pa, pb, pc):
      with variable_scope.variable_scope("", use_resource=True):
        with ipu.ops.ipu_scope("/device:IPU:0"):
          with ipu.ops.ipu_shard(0):
            init1 = init_ops.constant_initializer([1.0, 3.0]);
            v1 = variable_scope.get_variable("v1", dtype=np.float32, shape=[2],
                                             initializer=init1)
            o1 = pa + pb + v1

          with ipu.ops.ipu_shard(1):
            init2 = init_ops.constant_initializer([1.0, 2.0]);
            v2 = variable_scope.get_variable("v2", dtype=np.float32, shape=[2],
                                             initializer=init2)
            o2 = pa + pc + v2
            out = o1 + o2

      return [out]

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [2], name="a")
      pb = array_ops.placeholder(np.float32, [2], name="b")
      pc = array_ops.placeholder(np.float32, [2], name="c")
      report = gen_ipu_ops.ipu_event_trace()

    out = xla.compile(my_graph, [pa, pb, pc])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2, True)
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:

      sess.run(report)
      sess.run(variables.global_variables_initializer())
      sess.run(report)

      fd = {pa: [1., 1.], pb: [0., 1.], pc:[1., 5.]}
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

    out = xla.compile(my_graph, [inp, lab])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    cfg = ipu.utils.auto_select_ipus(cfg, 2, True)
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:

      sess.run(report)
      sess.run(variables.global_variables_initializer())
      sess.run(report)

      fd = {inp: np.ones([1, 32, 32, 4]), lab: np.ones([1, 8])}
      sess.run(out, fd)

      rep = sess.run(report)

      num_compiles = 0
      gdef = None
      evts = ipu.utils.extract_all_events(rep)
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_BEGIN:
          gdef = ipu.utils.extract_xla_graph_def_from_compilation_event(evt)
        if evt.type == IpuTraceEvent.COMPILE_END:
          num_compiles = num_compiles + 1

      self.assertEqual(num_compiles, 1)

      # Convolutions are on correct IPUs
      for n in gdef.node:
        if n.op == 'HloConvolution':
          if re.match(r'.*/convA/Conv2D/', n.name):
            self.assertTrue(n.device == '/device/XLA:0')
          if re.match(r'.*/convB/Conv2D/', n.name):
            self.assertTrue(n.device == '/device/XLA:1')
          if re.match(r'.*/convB/Conv2D_grad/Conv2DBackpropInput/', n.name):
            self.assertTrue(n.device == '/device/XLA:1')
          if re.match(r'.*/convB/Conv2D_grad/Conv2DBackpropFilter', n.name):
            self.assertTrue(n.device == '/device/XLA:1')
          if re.match(r'.*/convA/Conv2D_grad/Conv2DBackpropFilter', n.name):
            self.assertTrue(n.device == '/device/XLA:0')

      # There are 2 inter-ipu copies and they copy something 'data' shaped
      n_inter_ipu_copies = 0
      for n in gdef.node:
        if n.op == 'HloCustomCall':
          a = n.attr.get('custom_call_target')
          s = n.attr.get('_output_shapes').list.shape[0]
          if a.s == b'inter_ipu_copy':
            n_inter_ipu_copies = n_inter_ipu_copies + 1
            self.assertEqual([int(i.size) for i in s.dim], [1, 32, 32, 8])

      self.assertEqual(n_inter_ipu_copies, 2)


if __name__ == "__main__":
    googletest.main()
