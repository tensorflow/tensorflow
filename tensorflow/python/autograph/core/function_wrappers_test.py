# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for function_wrappers module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class FunctionWrappersTest(test.TestCase):

  def test_name_scope(self):
    if context.executing_eagerly():
      self.skipTest('Tensor names are disabled in eager')

    with function_wrappers.FunctionScope(
        'test_name', None,
        converter.ConversionOptions(
            optional_features=converter.Feature.NAME_SCOPES)):
      t = constant_op.constant(1)
    self.assertIn('test_name', t.name)

  def test_auto_control_deps(self):
    v = variables.Variable(1)
    with function_wrappers.FunctionScope(
        '_', None,
        converter.ConversionOptions(
            optional_features=converter.Feature.AUTO_CONTROL_DEPS)) as scope:
      v.assign(2)
      op = scope.ret(constant_op.constant(1), True)
    self.evaluate(op)
    self.assertEqual(self.evaluate(v.read_value()), 2)

  def test_all_disabled(self):
    with function_wrappers.FunctionScope(None, None,
                                         converter.STANDARD_OPTIONS):
      t = constant_op.constant(1)
    self.assertEqual(self.evaluate(t), 1)


if __name__ == '__main__':
  test.main()
