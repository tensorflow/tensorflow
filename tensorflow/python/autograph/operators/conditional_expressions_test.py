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
"""Tests for conditional_expressions module."""

from tensorflow.python.autograph.operators import conditional_expressions
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


def _basic_expr(cond):
  return conditional_expressions.if_exp(
      cond,
      lambda: constant_op.constant(1),
      lambda: constant_op.constant(2),
      'cond')


@test_util.run_all_in_graph_and_eager_modes
class IfExpTest(test.TestCase):

  def test_tensor(self):
    self.assertEqual(self.evaluate(_basic_expr(constant_op.constant(True))), 1)
    self.assertEqual(self.evaluate(_basic_expr(constant_op.constant(False))), 2)

  def test_tensor_mismatched_type(self):
    # tf.function required because eager cond degenerates to Python if.
    @def_function.function
    def test_fn():
      conditional_expressions.if_exp(
          constant_op.constant(True), lambda: 1.0, lambda: 2, 'expr_repr')

    with self.assertRaisesRegex(
        TypeError,
        "'expr_repr' has dtype float32 in the main.*int32 in the else"):
      test_fn()

  def test_python(self):
    self.assertEqual(self.evaluate(_basic_expr(True)), 1)
    self.assertEqual(self.evaluate(_basic_expr(False)), 2)
    self.assertEqual(
        conditional_expressions.if_exp(True, lambda: 1, lambda: 2, ''), 1)
    self.assertEqual(
        conditional_expressions.if_exp(False, lambda: 1, lambda: 2, ''), 2)


if __name__ == '__main__':
  test.main()
