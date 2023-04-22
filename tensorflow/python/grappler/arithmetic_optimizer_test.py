# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Grappler Arithmetic Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ArithmeticOptimizerTest(test.TestCase):

  # See b/146524878.
  def testFunctionArgShapeInference(self):

    @def_function.function
    def f(x, y):
      return math_ops.matmul(
          x, array_ops.reshape(array_ops.transpose(y), [384, 1536]))

    with context.eager_mode():
      x = array_ops.ones((1, 384))
      y = array_ops.ones((1536, 384))
      with context.collect_graphs(optimized=True) as graphs:
        f(x, y).numpy()
      self.assertLen(graphs, 1)
      self.assertLen(graphs[0].node, 4)
      self.assertEqual(graphs[0].node[2].name,
                       'ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul')


if __name__ == '__main__':
  test.main()
