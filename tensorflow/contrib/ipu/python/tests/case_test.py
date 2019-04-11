# Copyright 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.contrib.ipu import ipu_compiler
from tensorflow.contrib import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.eager import function as eager_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


class CaseTest(test_util.TensorFlowTestCase):
  def testCaseSimple(self):
    def my_graph(pa, pb, pc):
      with ipu.ops.ipu_scope("/device:IPU:0"):

        @eager_function.defun
        def b0(x, y):
          return x + y

        @eager_function.defun
        def b1(x, y):
          return x - y

        @eager_function.defun
        def b2(x, y):
          return x * y

        branches = [
            f.get_concrete_function(
                array_ops.zeros_like(pb), array_ops.zeros_like(pc))
            for f in [b0, b1, b2]
        ]

        c_out = gen_functional_ops.case(
            pa, input=[pb, pc], Tout=[dtypes.float32], branches=branches)

        return [c_out[0]]

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.int32, [], name="a")
      pb = array_ops.placeholder(np.float32, [2], name="b")
      pc = array_ops.placeholder(np.float32, [2], name="c")
      report = gen_ipu_ops.ipu_event_trace()

    out = ipu_compiler.compile(my_graph, [pa, pb, pc])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:

      sess.run(report)

      result = sess.run(out, {pa: 0, pb: [0., 1.], pc: [1., 5.]})
      self.assertAllClose(result[0], [1., 6.])

      result = sess.run(out, {pa: 1, pb: [0., 1.], pc: [1., 5.]})
      self.assertAllClose(result[0], [-1., -4.])

      result = sess.run(out, {pa: 2, pb: [0., 1.], pc: [1., 5.]})
      self.assertAllClose(result[0], [0., 5.])

      result = sess.run(out, {pa: 10, pb: [0., 1.], pc: [1., 5.]})
      self.assertAllClose(result[0], [0., 5.])

      rep = sess.run(report)
      evts = ipu.utils.extract_all_events(rep)

      num_compiles = 0
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_END:
          num_compiles = num_compiles + 1

      self.assertEqual(num_compiles, 1)

  def testCaseVariables(self):
    def my_graph(pa, pb):
      with ipu.ops.ipu_scope("/device:IPU:0"):

        @eager_function.defun
        def b0(x, y):
          return x + y

        @eager_function.defun
        def b1(x, y):
          return x - y

        @eager_function.defun
        def b2(x, y):
          return x * y

        v = variable_scope.get_variable(
            'b0', dtype=dtypes.float32, initializer=[1., 5.])

        branches = [
            f.get_concrete_function(
                array_ops.zeros_like(pb), array_ops.zeros_like(v))
            for f in [b0, b1, b2]
        ]

        c_out = gen_functional_ops.case(
            pa, input=[pb, v], Tout=[dtypes.float32], branches=branches)

        return [c_out[0]]

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.int32, [], name="a")
      pb = array_ops.placeholder(np.float32, [2], name="b")
      report = gen_ipu_ops.ipu_event_trace()

    out = ipu_compiler.compile(my_graph, [pa, pb])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with sl.Session() as sess:

      sess.run(variables_lib.global_variables_initializer())

      sess.run(report)

      result = sess.run(out, {pa: 0, pb: [0., 1.]})
      self.assertAllClose(result[0], [1., 6.])

      result = sess.run(out, {pa: 1, pb: [0., 1.]})
      self.assertAllClose(result[0], [-1., -4.])

      result = sess.run(out, {pa: 2, pb: [0., 1.]})
      self.assertAllClose(result[0], [0., 5.])

      result = sess.run(out, {pa: 10, pb: [0., 1.]})
      self.assertAllClose(result[0], [0., 5.])

      rep = sess.run(report)
      evts = ipu.utils.extract_all_events(rep)

      num_compiles = 0
      for evt in evts:
        if evt.type == IpuTraceEvent.COMPILE_END:
          num_compiles = num_compiles + 1

      self.assertEqual(num_compiles, 1)

  # Case operation does not have a gradient function at the moment
  # def testCaseTraining(self):
  #   def my_graph(pa, pb, ref):
  #     with ipu.ops.ipu_scope("/device:IPU:0"):
  #
  #       @eager_function.defun
  #       def b0(x, y):
  #         return x + y
  #
  #       @eager_function.defun
  #       def b1(x, y):
  #         return x - y
  #
  #       @eager_function.defun
  #       def b2(x, y):
  #         return x * y
  #
  #       v = variable_scope.get_variable('b0', dtype=dtypes.float32,
  #                                       initializer=[1.])
  #
  #       branches = [f.get_concrete_function(array_ops.zeros_like(pb),
  #                                           array_ops.zeros_like(v))
  #                   for f in [b0, b1, b2]]
  #
  #       c_out = gen_functional_ops.case(
  #         pa,
  #         input=[pb, v],
  #         Tout=[dtypes.float32],
  #         branches=branches)
  #
  #       loss = losses.mean_squared_error(c_out[0], ref)
  #
  #       opt = gradient_descent.GradientDescentOptimizer(0.01)
  #       train = opt.minimize(loss)
  #
  #       return [train]
  #
  #   with ops.device('cpu'):
  #     pa = array_ops.placeholder(np.int32, [], name="a")
  #     pb = array_ops.placeholder(np.float32, [1], name="b")
  #     ref = array_ops.placeholder(np.float32, [1], name="ref")
  #     report = gen_ipu_ops.ipu_event_trace()
  #
  #   out = ipu_compiler.compile(my_graph, [pa, pb, ref])
  #
  #   cfg = ipu.utils.create_ipu_config(profiling=True)
  #   cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
  #   ipu.utils.configure_ipu_system(cfg)
  #
  #   with sl.Session() as sess:
  #
  #     sess.run(variables_lib.global_variables_initializer())
  #
  #     sess.run(report)
  #
  #     for _ in range(1000):
  #       f = np.random.randint(2)
  #       pb_d = np.random.random([1])
  #
  #       ref = np.array([0.])
  #       if f == 0:
  #         ref = pb_d + np.array([2.])
  #       if f == 1:
  #         ref = pb_d - np.array([2.])
  #       if f == 2:
  #         ref = pb_d * np.array([2.])
  #
  #       sess.run(out, {pa: f, pb: pb_d, ref: ref})
  #
  #
  #     rep = sess.run(report)
  #     evts = ipu.utils.extract_all_events(rep)
  #
  #     num_compiles = 0
  #     for evt in evts:
  #       if evt.type == IpuTraceEvent.COMPILE_END:
  #         num_compiles = num_compiles + 1
  #
  #     self.assertEqual(num_compiles, 1)


if __name__ == "__main__":
  googletest.main()
