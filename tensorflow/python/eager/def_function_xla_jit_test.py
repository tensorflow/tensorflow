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
    if not test.is_built_with_rocm():
      # XLA support is not yet enabled for TF ROCm
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

    def run_and_check(test_func):
      x = constant_op.constant(3.0)
      with backprop.GradientTape() as tape:
        y = test_func(x)
      dy = tape.gradient(y, v)

      self.assertAllClose(6.0, y)
      self.assertAllClose(3.0, dy)

    run_and_check(func)
    if not test.is_built_with_rocm():
      # XLA support is not yet enabled for TF ROCm
      run_and_check(xla_func)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
