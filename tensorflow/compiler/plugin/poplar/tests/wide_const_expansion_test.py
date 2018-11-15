from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re

import test_utils as tu

from tensorflow.python.client import session as session_lib
from tensorflow.python.platform import googletest
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import constant_op
from tensorflow.core.protobuf import config_pb2
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

class WideConstExpansionTest(test_util.TensorFlowTestCase):

  def testCheckMaxTileSize(self):
    dtype = np.float32
    shape = (1024,2048)
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("", use_resource=True):
      	a = variable_scope.get_variable("a", shape=shape,
      		    initializer=init_ops.constant_initializer(2), dtype=dtype)
      pb = array_ops.placeholder(shape=shape, dtype=dtype, name="b")
      output = a + pb

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(execution_trace=False, compile_ipu_code=True) as sess:
      sess.run(variables.global_variables_initializer())
      result = sess.run(report)

      s = tu.extract_all_strings_from_event_trace(result)
      max_tile_size = tu.get_maximum_tile_size_from_events(s)
      print(max_tile_size)
      self.assertTrue(max_tile_size < 100000)

if __name__ == "__main__":
  googletest.main()