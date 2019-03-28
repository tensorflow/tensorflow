from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
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
      self.assertTrue(max_tile_size < 13000)

      out = sess.run(output, {pb: np.ones(shape=shape, dtype=dtype)})
      self.assertAllClose(np.full(shape, 7, dtype=dtype), out)
      result = sess.run(report)
      s = tu.extract_all_strings_from_event_trace(result)
      max_tile_size = tu.get_maximum_tile_size_from_events(s)
      self.assertTrue(max_tile_size < 40000)


if __name__ == "__main__":
  googletest.main()
