# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import googletest
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables

import numpy as np

class IpuXlaBatchNormTest(test_util.TensorFlowTestCase):

    def testBatchNormalize1(self):

        with ops.device("/device:IPU:0"):
            with session_lib.Session() as sess:
                with variable_scope.variable_scope("ascope", use_resource=True):
                    x = array_ops.placeholder(np.float32, [1,64,64,4], name="a")

                    beta = variable_scope.get_variable("x", shape=[4],
                            dtype=np.float32,
                            initializer=init_ops.constant_initializer(0.0))
                    gamma = variable_scope.get_variable("y", shape=[4],
                            dtype=np.float32,
                            initializer=init_ops.constant_initializer(1.0))

                    b_mean, b_var = nn.moments(x, [0,1,2], name='moments')

                    normed = nn.batch_normalization(x,
                                                       b_mean, b_var,
                                                       beta, gamma,
                                                       1e-3)

                    fd = {
                        x: np.zeros([1,64,64,4])
                    }

                    sess.run(variables.global_variables_initializer())

                    result = sess.run(normed, fd)
                    self.assertAllClose(result,
                                        np.zeros([1,64,64,4]))

if __name__ == "__main__":
    googletest.main()
