
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

class UpdateOpDependenciesTest(test_util.TensorFlowTestCase):

  def testDontOutlineInplaceExpression(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float16, [])
      pb = array_ops.placeholder(np.float16, [])
      pc = array_ops.placeholder(np.float16, [])
      pd = array_ops.placeholder(np.float16, [])
      e = pa + pb - pc + pd

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)
      fd = {
        pa: 1,
        pb: 2,
        pc: 3,
        pd: 4
      }
      result = sess.run(e, fd)
      self.assertAllClose(result, 4)

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['progIdCopy',
            'host-exchange-local-copy-',
            'add/add.*/AddTo',
            'sub/subtract.*/AddTo',
            'add_1/add.*/AddTo']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testAddCopyToViewChangingShape(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float16, [2, 2])
      b = array_ops.transpose(pa)
      c = pa + pa
      d = c / b

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)
      fd = {
        pa: [[10, -20], [5, 1]],
      }
      result = sess.run(d, fd)
      self.assertAllClose(result, [[2, -8], [-.5, 2]])

      result = sess.run(report)
      self.assertTrue(len(result) == 5)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['progIdCopy',
            'host-exchange-local-copy-',
            'Copy_XLA_Args/arg0.*_to_transpose.*.clone/OnTileCopy',
            'ArithmeticOptimizer/SimplifyAggregation_Mul_add/multiply.*/Op/Multiply',
            'truediv/divide.*/Op/Divide']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testAddCopyBeforeInplaceOpWithViewChangingParent(self):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [3])
      pb = array_ops.placeholder(np.float32, [3])
      pc = array_ops.placeholder(np.float32, [])
      c = array_ops.slice(pa, [0], [2])
      d = array_ops.slice(pb, [0], [2])
      e = c + d
      f = e / pc
      g = array_ops.slice(pa, [1], [2])
      h = f + g

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)
      fd = {
        pa: [1, 2, 3],
        pb: [5, 6, 7],
        pc: 2,
      }
      result = sess.run(h, fd)
      self.assertAllClose(result, [5, 7])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

      s = tu.extract_all_strings_from_event_trace(result)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['progIdCopy',
            'host-exchange-local-copy-',
            'Copy_XLA_Args/arg0.*_to_add/add.*/OnTileCopy',
            'add/add.*/AddTo',
            'truediv/divide.*/Op/Divide',
            'add_1/add.*/AddTo']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

if __name__ == "__main__":
    googletest.main()
