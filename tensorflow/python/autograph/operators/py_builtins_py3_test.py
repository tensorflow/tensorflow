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
"""Tests for py_builtins_py3 module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.platform import test


class PyBuiltinsTest(test.TestCase):

  def test_super_with_no_arg_in_original_context(self):
    test_case_self = self

    class TestBase(object):

      def plus_twenty(self, x):
        return x + 20

    class TestSubclass(TestBase):

      def plus_twenty(self, x):
        test_case_self.fail('This should never be called.')

      def no_arg(self):
        test_base = py_builtins.super_in_original_context(super, (), 0)
        return test_base.plus_twenty(1)

    tc = TestSubclass()
    self.assertEqual(tc.no_arg(), 21)

  def test_super_in_original_context_with_locals(self):
    test_case_self = self

    class TestBase(object):

      def plus_twenty(self, x):
        return x + 20

    class TestSubclass(TestBase):

      def plus_twenty(self, x):
        test_case_self.fail('This should never be called.')

      def with_locals(self):
        x = 1
        y = 7
        z = 7

        test_base = py_builtins.super_in_original_context(super, (), 0)
        return test_base.plus_twenty(x + y - z)

    tc = TestSubclass()
    self.assertEqual(tc.with_locals(), 21)

  def test_super_in_original_context_with_args(self):
    test_case_self = self

    class TestBase(object):

      def plus_twenty(self, x):
        return x + 20

    class TestSubclass(TestBase):

      def plus_twenty(self, x):
        test_case_self.fail('This should never be called.')

      def with_args(self, x, y, z):
        test_base = py_builtins.super_in_original_context(super, (), 0)
        return test_base.plus_twenty(x + y - z)

    tc = TestSubclass()
    self.assertEqual(tc.with_args(1, 7, 7), 21)
    self.assertEqual(tc.with_args.__func__(*[tc, 1, 7, 7]), 21)

  def test_super_in_original_context_with_varargs(self):
    test_case_self = self

    class TestBase(object):

      def plus_twenty(self, x):
        return x + 20

    class TestSubclass(TestBase):

      def plus_twenty(self, x):
        test_case_self.fail('This should never be called.')

      def with_varargs(self, *args):
        test_base = py_builtins.super_in_original_context(super, (), 0)
        return test_base.plus_twenty(args[0] + args[1] - args[2])

    tc = TestSubclass()
    self.assertEqual(tc.with_varargs.__func__(*[tc, 1, 7, 7]), 21)

  def test_super_in_original_context_with_kwargs(self):
    test_case_self = self

    class TestBase(object):

      def plus_twenty(self, x):
        return x + 20

    class TestSubclass(TestBase):

      def plus_twenty(self, x):
        test_case_self.fail('This should never be called.')

      def with_kwargs(self, **kwargs):
        test_base = py_builtins.super_in_original_context(super, (), 0)
        return test_base.plus_twenty(kwargs['x'] + kwargs['y'] - kwargs['z'])

    tc = TestSubclass()
    self.assertEqual(tc.with_kwargs.__func__(tc, x=1, y=7, z=7), 21)


if __name__ == '__main__':
  test.main()
