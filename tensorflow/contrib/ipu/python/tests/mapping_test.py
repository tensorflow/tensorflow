# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.contrib.compiler import xla
from tensorflow.contrib import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.contrib.ipu import ipu_compiler


class MappingTest(test_util.TensorFlowTestCase):
  def testGather(self):
    def my_net(w, i):
      out = array_ops.gather(w, i)
      return [out]

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [256])
      w = array_ops.placeholder(np.float32, [8192])
      report = gen_ipu_ops.ipu_event_trace()

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[w, i])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:

      result = sess.run(r, {i: np.arange(0, 3 * 256, 3), w: np.arange(8192)})
      self.assertAllClose(result[0], np.arange(0, 3 * 256, 3))

      rep = sess.run(report)

      events = ipu.utils.extract_all_events(rep)

      for e in events:
        if e.type == IpuTraceEvent.COMPILE_END:
          j = e.compile_end.tensor_map.decode('utf-8')
          if len(j) > 0:
            tm = json.loads(e.compile_end.tensor_map.decode('utf-8'))

            bad_maps = []
            for g in tm['mappings']:
              for tensor in tm['mappings'][g]:
                if tensor['total_elements'] > 16:
                  if tensor['tiles_used'] == 1 and tensor['constant'] == 0:
                    bad_maps += [tensor['inst_name']]

      self.assertEqual(len(bad_maps), 0)


if __name__ == "__main__":
  googletest.main()
