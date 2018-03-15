# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for single_return module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.converters import converter_test_base
from tensorflow.contrib.py2tf.converters import single_return
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.platform import test


class SingleReturnTest(converter_test_base.TestCase):

  def compiled_fn(self, test_fn, *args):
    node = self.parse_and_analyze(test_fn, {})
    node = single_return.transform(node, self.ctx)
    module = self.compiled(node, *args)
    return module

  def test_noop(self):
    # Noop
    def test_fn(x):
      return x

    with self.compiled_fn(test_fn) as result:
      self.assertEqual(test_fn(2.0), result.test_fn(2.0))

  def test_return_expression(self):
    # ANF
    def test_fn(x):
      return x * x

    with self.compiled_fn(test_fn) as result:
      x = 2
      self.assertEqual(test_fn(x), result.test_fn(x))

  def test_merge(self):
    # Simple merge
    def test_fn(x):
      if x > 0:
        return x
      else:
        return x * x

    with self.compiled_fn(test_fn) as result:
      for x in [-2, 2]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_orphan_branch(self):

    def test_fn(x):
      if x > 0:
        return x

    with self.assertRaises(ValueError):
      self.compiled_fn(test_fn)

  def test_lift_body_into_false_branch(self):

    def test_fn(x):
      if x > 0:
        return x
      return x * x

    with self.compiled_fn(test_fn) as result:
      for x in [-2, 2]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_lift_body_into_true_branch(self):

    def test_fn(x):
      if x < 0:
        x *= x
      else:
        # TODO(alexbw): linter bug here that requires us suppress this warning.
        return x  # pylint: disable=undefined-loop-variable
      return x

    with self.compiled_fn(test_fn) as result:
      for x in [-2, 2]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_nested_if(self):

    def test_fn(x):
      if x > 0:
        if x < 5:
          return x
        else:
          return x * x
      else:
        return x * x * x

    with self.compiled_fn(test_fn) as result:
      for x in [-2, 2, 5]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_context_manager(self):

    def test_fn(x):

      with name_scope(''):
        return x * x

    with self.compiled_fn(test_fn) as result:
      result.name_scope = name_scope
      for x in [-2, 2]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_context_manager_in_conditional(self):

    def test_fn(x):
      if x > 0:
        with name_scope(''):
          return x * x
      else:
        return x

    with self.compiled_fn(test_fn, name_scope) as result:
      result.name_scope = name_scope
      for x in [-2, 2]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def text_conditional_in_context_manager(self):

    def test_fn(x):
      with name_scope(''):
        if x > 0:
          return x * x
        else:
          return x

    with self.compiled_fn(test_fn) as result:
      result.name_scope = name_scope
      for x in [-2, 2]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_no_return(self):

    def test_fn(x):
      x *= x

    with self.compiled_fn(test_fn) as result:
      self.assertEqual(test_fn(2), result.test_fn(2))

  def test_nested_functiondefs(self):

    def test_fn(x):

      def inner_fn(y):
        if y > 0:
          return y * y
        else:
          return y

      return inner_fn(x)

    with self.compiled_fn(test_fn) as result:
      for x in [-2, 2]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_loop(self):

    def test_fn(x):
      for _ in range(10):
        return x
      return x

    with self.assertRaises(ValueError):
      self.compiled_fn(test_fn)


if __name__ == '__main__':
  test.main()
