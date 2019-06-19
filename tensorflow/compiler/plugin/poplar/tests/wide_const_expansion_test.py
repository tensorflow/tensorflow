from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import test_utils as tu

from tensorflow.python.compiler.xla import xla
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import constant_op
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class WideConstExpansionTest(test_util.TensorFlowTestCase):
  def testCheckMaxTileSize(self):
    dtype = np.float32
    shape = (1024, 2048)
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("", use_resource=True):
        a = variable_scope.get_variable(
            "a",
            shape=shape,
            initializer=init_ops.constant_initializer(2),
            dtype=dtype)
      pb = array_ops.placeholder(shape=shape, dtype=dtype, name="b")
      c = constant_op.constant(4, shape=shape, dtype=dtype, name="c")
      output = a + pb + c

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(execution_trace=False)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      max_tile_size = tu.get_maximum_tile_size_from_events(s)
      self.assertTrue(max_tile_size < 17000)

      out = sess.run(output, {pb: np.ones(shape=shape, dtype=dtype)})
      self.assertAllClose(np.full(shape, 7, dtype=dtype), out)
      result = sess.run(report)
      s = tu.extract_all_strings_from_event_trace(result)
      max_tile_size = tu.get_maximum_tile_size_from_events(s)

      self.assertTrue(max_tile_size < 41000)

  def testWideConstantWithAllocationTarget(self):
    # This test will fail if the dynamic slice is not mapped correctly.
    dtype = np.float32
    shape = (512, 2, 2048)

    def my_net(y):
      def cond(i, x, y):
        return i < 2

      def body(i, x, y):
        s = array_ops.slice(x, [i, i, i], [1, 1, 2048])
        y = y + math_ops.reduce_mean(s)
        i = i + 1
        return (i, x, y)

      i = 0
      c = constant_op.constant(4, shape=shape, dtype=dtype, name="c")
      return control_flow_ops.while_loop(cond, body, (i, c, y))[2]

    with ops.device('cpu'):
      y = array_ops.placeholder(dtype, [1])
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system()

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[y])

    with tu.ipu_session() as sess:
      sess.run(report)
      y = sess.run(r, {y: [10]})
      self.assertAllClose(y[0], [18])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = [
          '__seed*', 'Copy_*_to_*', 'while/Slice/dynamic-slice*/dynamicSlice',
          'while/Mean/reduce', 'while/Mean/multiply', 'while/add*/add*/AddTo'
      ]
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

      max_tile_size = tu.get_maximum_tile_size_from_events(s)
      self.assertTrue(max_tile_size < 60000)
      always_live_size = tu.get_always_live_size_from_events(s)
      self.assertTrue(max_tile_size < 4500000)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
