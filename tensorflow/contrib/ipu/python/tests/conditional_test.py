# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.ipu.python import ipu_compiler
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class ConditionalTest(test_util.TensorFlowTestCase):
  def testSimpleCond(self):
    def my_model(pcond, pa, pb, pc):
      output = control_flow_ops.cond(
          pcond, true_fn=lambda: pa + pb + pc, false_fn=lambda: pa - pb - pc)
      return [output]

    with ops.device("cpu"):
      pcond = array_ops.placeholder(np.bool, [], name="pred")
      pa = array_ops.placeholder(np.float32, [], name="a")
      pb = array_ops.placeholder(np.float32, [], name="b")
      pc = array_ops.placeholder(np.float32, [], name="c")

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_model, inputs=[pcond, pa, pb, pc])

    with session_lib.Session() as sess:

      fd = {pcond: True, pa: 1., pb: 2., pc: 3.}
      result = sess.run(r[0], fd)
      self.assertAllClose(result, 6)

      fd = {pcond: False, pa: 1., pb: 2., pc: 3.}
      result = sess.run(r[0], fd)
      self.assertAllClose(result, -4.)

  def testDifferentArgs(self):
    def my_model(pcond, pa, pb, pc):
      output = control_flow_ops.cond(
          pcond, true_fn=lambda: pa + pb, false_fn=lambda: pb - pc)
      return [output]

    with ops.device("cpu"):
      pcond = array_ops.placeholder(np.bool, [], name="pred")
      pa = array_ops.placeholder(np.float32, [], name="a")
      pb = array_ops.placeholder(np.float32, [], name="b")
      pc = array_ops.placeholder(np.float32, [], name="c")

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_model, inputs=[pcond, pa, pb, pc])

    with session_lib.Session() as sess:

      fd = {pcond: True, pa: 1., pb: 2., pc: 3.}
      result = sess.run(r[0], fd)
      self.assertAllClose(result, 3.)

      fd = {pcond: False, pa: 1., pb: 2., pc: 3.}
      result = sess.run(r[0], fd)
      self.assertAllClose(result, -1.)

  def testReadResourceVar(self):
    def my_model(pcond):
      va = variable_scope.get_variable(
          "x",
          shape=[],
          dtype=np.float32,
          initializer=init_ops.constant_initializer(1))

      o = control_flow_ops.cond(
          pcond,
          true_fn=lambda: va.read_value(),
          false_fn=lambda: constant_op.constant(0.))
      return [o]

    with ops.device("cpu"):
      pcond = array_ops.placeholder(np.bool, [], name="pred")

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_model, inputs=[pcond])

    with session_lib.Session() as sess:

      sess.run(variables.global_variables_initializer())

      fd = {pcond: True}
      result = sess.run(r[0], fd)
      self.assertAllClose(result, 1.)

      fd = {pcond: False}
      result = sess.run(r[0], fd)
      self.assertAllClose(result, 0.)

  def testWriteResourceVar(self):
    def my_model(pcond):
      va = variable_scope.get_variable(
          "x",
          shape=[],
          dtype=np.float32,
          initializer=init_ops.constant_initializer(1))

      o = control_flow_ops.cond(
          pcond,
          true_fn=lambda: state_ops.assign(va, 1.),
          false_fn=lambda: state_ops.assign(va, 2.))

      return [o, va.read_value()]

    with ops.device("cpu"):
      pcond = array_ops.placeholder(np.bool, [], name="pred")

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_model, inputs=[pcond])

    with session_lib.Session() as sess:

      sess.run(variables.global_variables_initializer())

      fd = {pcond: True}
      result = sess.run(r[0], fd)
      self.assertAllClose(result, 1.)

      self.assertAllClose(sess.run(r[1], fd), 1.)

      fd = {pcond: False}
      result = sess.run(r[0], fd)
      self.assertAllClose(result, 2.)

      self.assertAllClose(sess.run(r[1], fd), 2.)


if __name__ == "__main__":
  googletest.main()
