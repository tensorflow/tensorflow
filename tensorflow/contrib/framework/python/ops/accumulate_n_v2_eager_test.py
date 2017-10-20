# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for new version of accumulate_n op that will eventually go into 
`ops.math_ops`.

These test cases spefically exercise the `eager` APIs. They need to be in a 
separate file from the remaining tests because eager mode is currently something
you can turn on but can't turn off for the lifetime of the current process."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.framework.python.ops import accumulate_n_v2 as av2

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import tape


from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test



class AccumulateNV2EagerTest(test_util.TensorFlowTestCase):
  """Tests of the new, differentiable version of accumulate_n"""

  def testMinimalEagerMode(self):
    forty = constant_op.constant(40)
    two = constant_op.constant(2)
    answer = av2.accumulate_n_v2([forty, two])
    self.assertEqual(42, answer.numpy())


  def testFloat(self):
    np.random.seed(12345)
    x = [np.random.random((1, 2, 3, 4, 5)) - 0.5 for _ in range(5)]
    tf_x = ops.convert_n_to_tensor(x)
    with self.test_session(use_gpu=True):
      self.assertAllClose(sum(x), av2.accumulate_n_v2(tf_x).numpy())
      self.assertAllClose(x[0] * 5, av2.accumulate_n_v2([tf_x[0]] * 5).numpy())

  def testGrad(self):
    np.random.seed(42)
    num_inputs = 3
    input_vars = [
        resource_variable_ops.ResourceVariable(10.0 * np.random.random())
        for i in range(0, num_inputs)
    ]

    def fn(first, second, third):
      return av2.accumulate_n_v2([first, second, third])

    grad_fn = backprop.gradients_function(fn)      
    grad = grad_fn(input_vars[0], input_vars[1], input_vars[2])
    self.assertAllEqual(np.repeat(1.0, num_inputs), # d/dx (x + y + ...) = 1
                        [elem.numpy() for elem in grad])



if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()

