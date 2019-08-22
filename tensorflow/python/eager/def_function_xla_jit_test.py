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

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class DefFunctionTest(test.TestCase):

  def testBasic(self):

    def fn(x, a):
      return x + a

    func = def_function.function(fn, experimental_compile=False)
    xla_func = def_function.function(fn, experimental_compile=True)

    inputs = constant_op.constant([1, 2, 2, 3, 3])
    self.assertAllClose([2, 3, 3, 4, 4], func(inputs, 1))
    self.assertAllClose([2, 3, 3, 4, 4], xla_func(inputs, 1))

  def testUnsupportedOps(self):

    def fn(x):
      return array_ops.unique(x).y  # Unique is not supported by XLA

    func = def_function.function(fn, experimental_compile=False)
    xla_func = def_function.function(fn, experimental_compile=True)

    inputs = constant_op.constant([1, 2, 2, 3, 3])
    self.assertAllClose([1, 2, 3], func(inputs))
    with self.assertRaisesRegexp(errors.InvalidArgumentError, 'not compilable'):
      xla_func(inputs)

  def testFunctionGradient(self):
    v = resource_variable_ops.ResourceVariable(2.0)

    def fn(x):
      return v * x

    func = def_function.function(fn, experimental_compile=False)
    xla_func = def_function.function(fn, experimental_compile=True)

    x = constant_op.constant(3.0)
    with backprop.GradientTape() as tape_1:
      y_1 = func(x)
    with backprop.GradientTape() as tape_2:
      y_2 = xla_func(x)
    dy_1 = tape_1.gradient(y_1, v)
    dy_2 = tape_2.gradient(y_2, v)

    self.assertAllClose(6.0, y_1)
    self.assertAllClose(6.0, y_2)
    self.assertAllClose(3.0, dy_1)
    self.assertAllClose(3.0, dy_2)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
