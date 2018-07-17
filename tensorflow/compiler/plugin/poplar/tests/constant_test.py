# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import test_utils as tu
import numpy as np

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest

class ConstantTest(test_util.TensorFlowTestCase):

    def testScalar(self):
        with ops.device("/device:IPU:0"):
            output = array_ops.constant(1.0)

        with ops.device('cpu'):
            report = gen_ipu_ops.ipu_event_trace()

        with tu.ipu_session() as sess:
            sess.run(report)

            result = sess.run(output, {})
            self.assertAllClose(result, 1.0)

            result = sess.run(report)
            s = tu.extract_all_types_from_event_trace(result)
            self.assertEqual(len(s), 2) # compile begin and end

    def testTensor(self):
        with ops.device("/device:IPU:0"):
            output = array_ops.constant([1.0, 2.0])

        with ops.device('cpu'):
            report = gen_ipu_ops.ipu_event_trace()

        with tu.ipu_session() as sess:
            sess.run(report)

            result = sess.run(output, {})
            self.assertAllClose(result, [1.0, 2.0])

            result = sess.run(report)
            s = tu.extract_all_types_from_event_trace(result)
            self.assertEqual(len(s), 2) # compile begin and end

    def testMultipleTensors(self):
        with ops.device("/device:IPU:0"):
            with variable_scope.variable_scope("", use_resource=True):
                var0 = resource_variable_ops.ResourceVariable(
                    np.array([1.0, 2.0], dtype=np.float32), name="v0")
                var1 = resource_variable_ops.ResourceVariable(
                    np.array([3.0, 4.0], dtype=np.float32), name="v1")

        with ops.device('cpu'):
            report = gen_ipu_ops.ipu_event_trace()

        with tu.ipu_session() as sess:
            sess.run(report)

            sess.run(variables.global_variables_initializer())

            result = sess.run(report)
            s = tu.extract_all_types_from_event_trace(result)
            self.assertEqual(len(s), 2) # compile begin and end

if __name__ == "__main__":
    googletest.main()
