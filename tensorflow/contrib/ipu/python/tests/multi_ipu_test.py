from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.contrib.compiler import xla
from tensorflow.contrib import ipu
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

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


if __name__ == "__main__":
    googletest.main()
