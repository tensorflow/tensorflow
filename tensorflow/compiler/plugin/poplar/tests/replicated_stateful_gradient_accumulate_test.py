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
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops


class ReplicatedStatefulGradientAccumulateTest(test_util.TensorFlowTestCase):
  def testStatefulGradientAccumulate(self):
    dtype = np.float32

    def my_net(y):
      def cond(i, x, y):
        return i < 10

      def body(i, x, y):
        x = x + gen_poputil_ops.ipu_stateful_gradient_accumulate(
            array_ops.ones_like(x), num_mini_batches=5)
        y = y + array_ops.ones_like(x)
        i = i + 1
        return (i, x, y)

      i = 0
      return control_flow_ops.while_loop(cond, body, (i, y, y))

    with ops.device('cpu'):
      y = array_ops.placeholder(dtype, [1])
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(replicated=True)

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[y])

    with tu.ipu_session() as sess:
      sess.run(report)
      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      self.assertAllEqual(y[1], [20])
      self.assertAllEqual(y[2], [20])

  def testWideConstantWithAllocationTarget(self):
    dtype = np.float32

    def my_net(y):
      def cond(i, x, y):
        return i < 10

      def body(i, x, y):
        x = x + gen_poputil_ops.ipu_stateful_gradient_accumulate(
            array_ops.ones_like(x), num_mini_batches=4)
        y = y + array_ops.ones_like(x)
        i = i + 1
        return (i, x, y)

      i = 0
      return control_flow_ops.while_loop(cond, body, (i, y, y))

    with ops.device('cpu'):
      y = array_ops.placeholder(dtype, [1])
      report = gen_ipu_ops.ipu_event_trace()

    tu.configure_ipu_system(replicated=True)

    with ops.device("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[y])

    with tu.ipu_session() as sess:
      sess.run(report)
      y = sess.run(r, {y: [10]})
      self.assertEqual(y[0], 10)
      # Note that num_mini_batches doesn't divide the number of iterations.
      self.assertAllEqual(y[1], [18])
      self.assertAllEqual(y[2], [20])


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
