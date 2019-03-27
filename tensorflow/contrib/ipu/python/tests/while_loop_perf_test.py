# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
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


def count_event_type(events, type):
  return sum(map((lambda x: 1 if x.type == type else 0), events))


class WhileLoopPerfTest(test_util.TensorFlowTestCase):
  def testIpuWhilePerfTest(self):
    def cond(i, v):
      return math_ops.less(i, 15)

    def body(i, v):
      v = v + i
      i = i + 1
      return (i, v)

    def my_net(v):
      i = constant_op.constant(0)
      r = control_flow_ops.while_loop(
          cond, body, (i, v), maximum_iterations=10)
      return [r[1]]

    with ops.device('cpu'):
      v = array_ops.placeholder(np.int32, [500])
      report = gen_ipu_ops.ipu_event_trace()

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[v])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:

      result = sess.run(r, {v: np.zeros([500], np.int32)})
      self.assertAllClose(result[0], np.broadcast_to(45, [500]))

      rep = sess.run(report)

      events = ipu.utils.extract_all_events(rep)

      # Check that there is only one compile
      self.assertEqual(
          count_event_type(events, IpuTraceEvent.COMPILE_BEGIN), 1)

      # Check that there is only one execute
      self.assertEqual(count_event_type(events, IpuTraceEvent.EXECUTE), 1)


if __name__ == "__main__":
  googletest.main()
