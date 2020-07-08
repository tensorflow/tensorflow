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
"""Tests for return_statements module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class SingleReturnTest(converter_testing.TestCase):

  def assertTransformedEquivalent(self, f, *inputs):
    tr = self.transform(f, (functions, return_statements))
    self.assertEqual(f(*inputs), tr(*inputs))

  def test_straightline(self):

    def f(x):
      return x * x

    self.assertTransformedEquivalent(f, 2)

  def test_superfluous_returns(self):

    def f():
      retval = 1
      return retval
      retval = 2  # pylint:disable=unreachable
      return retval

    self.assertTransformedEquivalent(f)

  def test_superfluous_returns_adjacent(self):

    def f():
      return 1
      return 2  # pylint:disable=unreachable

    self.assertTransformedEquivalent(f)

  def test_conditional(self):

    def f(x):
      if x > 0:
        return x
      else:
        return x * x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_conditional_missing_else(self):

    def f(x):
      if x > 0:
        return x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_conditional_missing_else_then_default(self):

    def f(x):
      if x > 0:
        return x
      return x * x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_conditional_else_only_then_default(self):

    def f(x):
      if x < 0:
        x *= x
      else:
        return x
      return x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_conditional_nested(self):

    def f(x):
      if x > 0:
        if x < 5:
          return x
        else:
          return x * x
      else:
        return x * x * x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)
    self.assertTransformedEquivalent(f, 5)

  def test_context_manager(self):

    def f(x):
      with ops.name_scope(''):
        return x * x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_context_manager_in_conditional(self):

    def f(x):
      if x > 0:
        with ops.name_scope(''):
          return x * x
      else:
        return x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def text_conditional_in_context_manager(self):

    def f(x):
      with ops.name_scope(''):
        if x > 0:
          return x * x
        else:
          return x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_no_return(self):

    def f(x):
      x *= x

    self.assertTransformedEquivalent(f, 2)

  def test_nested_function(self):

    def f(x):

      def inner_fn(y):
        if y > 0:
          return y * y
        else:
          return y

      return inner_fn(x)

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_nested_function_in_control_flow(self):

    def f(x):

      if x:
        def inner_fn(y):
          return y
        inner_fn(x)

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_for_loop(self):

    def f(n):
      for _ in range(n):
        return 1

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, 0)

  def test_while_loop(self):

    def f(n):
      i = 0
      s = 0
      while i < n:
        i += 1
        s += i
        if s > 4:
          return s
      return -1

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, 4)

  def test_null_return(self):

    def f(n):
      if n > 4:
        return
      return

    self.assertTransformedEquivalent(f, 4)
    self.assertTransformedEquivalent(f, 5)

  def test_nested_multiple_withs(self):

    def f(x):
      v = []
      while x > 0:
        x -= 1
        with ops.name_scope(''):
          if x % 2 == 0:
            return v
        with ops.name_scope(''):
          v.append(x)
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 4)

  def test_multiple_returns_in_nested_scope(self):

    def f(a):
      v = []
      for x in a:
        x -= 1
        if x > 100:
          return v
        try:
          raise ValueError('intentional')
        except ValueError:  # pylint:disable=bare-except
          return v
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, [])
    self.assertTransformedEquivalent(f, [1])
    self.assertTransformedEquivalent(f, [2])
    self.assertTransformedEquivalent(f, [1, 2, 3])

if __name__ == '__main__':
  test.main()
