# Copyright 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest


class PopopsCustomOpsTest(test_util.TensorFlowTestCase):

  shape = [2, 2]
  datatype = np.float32

  def _testPopopsCustomUnaryOp(self, op, x):
    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(self.datatype, self.shape, name="a")
      output = op(pa)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      sess.run(report)

      result = sess.run(output, {pa: x})

      rep = sess.run(report)
      s = tu.extract_all_strings_from_event_trace(rep)
      cs_list = tu.get_compute_sets_from_report(s)

    return result, cs_list

  def testSqrt(self):
    x = np.full(self.shape, 4.0, dtype=self.datatype)
    expected_result = np.sqrt(x)

    result, cs_list = self._testPopopsCustomUnaryOp(math_ops.sqrt, x)

    self.assertAllClose(result, expected_result)

    ok = ['Sqrt/custom-call.*/Op/Sqrt']
    self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testRsqrt(self):
    x = np.full(self.shape, 4.0, dtype=self.datatype)
    expected_result = np.power(x, -0.5)

    result, cs_list = self._testPopopsCustomUnaryOp(math_ops.rsqrt, x)

    self.assertAllClose(result, expected_result)

    ok = ['Rsqrt/custom-call.*/Op/Rsqrt']
    self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))



if __name__ == "__main__":
  googletest.main()
