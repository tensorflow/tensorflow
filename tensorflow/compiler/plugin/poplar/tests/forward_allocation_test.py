# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.layers import normalization as layers_norm
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class ForwardAllocationTest(test_util.TensorFlowTestCase):
  def testPrefixPathWithReshape(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      z = array_ops.placeholder(np.float32, shape=[32])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())
      res = gen_array_ops.reshape(y, [32]) + z

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      result = sess.run(res, {
          x: np.reshape(np.arange(32), [1, 4, 4, 2]),
          z: np.ones([32])
      })
      # Confirmed with values on the CPU.
      self.assertAllClose(result, [
          2., 2., 6., 6., 10., 10., 14., 14., 18., 18., 22., 22., 26., 26.,
          30., 30., 34., 34., 38., 38., 42., 42., 46., 46., 50., 50., 54., 54.,
          58., 58., 62., 62.
      ])

  def testPrefixPathWithTranspose(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      z = array_ops.placeholder(np.float32, shape=[4, 4, 2, 1])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())
      res = array_ops.transpose(y, [1, 2, 3, 0]) + z

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      result = sess.run(res, {
          x: np.reshape(np.arange(32), [1, 4, 4, 2]),
          z: np.ones([4, 4, 2, 1])
      })
      self.assertAllClose(
          result,
          [[[[2.], [2.]], [[6.], [6.]], [[10.], [10.]], [[14.], [14.]]],
           [[[18.], [18.]], [[22.], [22.]], [[26.], [26.]], [[30.], [30.]]],
           [[[34.], [34.]], [[38.], [38.]], [[42.], [42.]], [[46.], [46.]]],
           [[[50.], [50.]], [[54.], [54.]], [[58.], [58.]], [[62.], [62.]]]])

  def testPrefixPathWithElementwiseInPath(self):
    with ops.device("/device:IPU:0"):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      z = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      s = array_ops.placeholder(np.float32, shape=[])

      with variable_scope.variable_scope("vs", use_resource=True):
        y = convolutional.conv2d(
            x,
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.ones_initializer())
      res = y + z * s

    tu.configure_ipu_system(True, True, True)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())

      result = sess.run(
          res, {
              x: np.reshape(np.arange(32), [1, 4, 4, 2]),
              z: np.reshape(np.arange(32), [1, 4, 4, 2]),
              s: 2.0
          })
      # Confirmed with values on the CPU.
      self.assertAllClose(
          result, [[[[1., 3.], [9., 11.], [17., 19.], [25., 27.]],
                    [[33., 35.], [41., 43.], [49., 51.], [57., 59.]],
                    [[65., 67.], [73., 75.], [81., 83.], [89., 91.]],
                    [[97., 99.], [105., 107.], [113., 115.], [121., 123.]]]])


if __name__ == "__main__":
  googletest.main()
