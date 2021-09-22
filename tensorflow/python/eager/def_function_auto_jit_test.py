# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class DefFunctionAutoclusteringTest(xla_test.XLATestCase):

  def testUpdateVariableMemoryUsage(self):
    if 'tpu' in self.device.lower():
      self.skipTest('Autoclustering does not run on TPU')

    v = variables.Variable(random_ops.random_normal([128, 128]))

    @def_function.function(jit_compile=False)
    def update_var(a, b, c):
      # create 4 ops here, which is the required minimum cluster size.
      x = a * b
      y = x + c
      v.assign(y + v)

    arg1 = random_ops.random_normal([1, 1])
    arg2 = random_ops.random_normal([1, 1])
    arg3 = random_ops.random_normal([1, 1])

    with context.collect_graphs(optimized=True) as graphs:
      initial_usage = context.context().get_memory_info(
          v.device)['current']
      update_var(arg1, arg2, arg3)
      final_usage = context.context().get_memory_info(
          v.device)['current']

    self.assertIn('_XlaRun', [n.op for n in graphs[0].node])
    self.assertEqual(initial_usage, final_usage)

  def testUpdateVariableWithCompileTimeConstMemoryUsage(self):
    if 'tpu' in self.device.lower():
      self.skipTest('Autoclustering does not run on TPU')

    v = variables.Variable(random_ops.random_normal([128, 128]))

    # test a signature of (compile-time const, args, res_var). The compile-time
    # const will be optimized away so that the kernel signature will become
    # (args, res_var).
    @def_function.function(jit_compile=False)
    def update_var(shape, a, b, c):
      # create 4 ops here, which is the required minimum cluster size.
      x = a * b
      y = x + c
      v.assign_add(array_ops.reshape(y, shape))

    arg1 = random_ops.random_normal([1])
    arg2 = random_ops.random_normal([1])
    arg3 = random_ops.random_normal([1])

    with context.collect_graphs(optimized=True) as graphs:
      initial_usage = context.context().get_memory_info(
          v.device)['current']
      update_var(constant_op.constant([1, 1]), arg1, arg2, arg3)
      final_usage = context.context().get_memory_info(
          v.device)['current']

    self.assertIn('_XlaRun', [n.op for n in graphs[0].node])
    self.assertEqual(initial_usage, final_usage)

if __name__ == '__main__':
  ops.enable_eager_execution()
  context.context().optimizer_jit = True
  test.main()
