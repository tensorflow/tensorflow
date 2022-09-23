# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for function_type."""

from tensorflow.core.function.function_type import function_type
from tensorflow.python.platform import test


class FunctionTypeTest(test.TestCase):

  def test_required_only(self):

    def foo(x, y, z):  # pylint: disable=unused-argument
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    self.assertEqual(
        constraint,
        function_type.FunctionType(
            (function_type.Parameter(
                "x", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                None),
             function_type.Parameter(
                 "y", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                 None),
             function_type.Parameter(
                 "z", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                 None))))
    self.assertEqual(function_type.FunctionType.get_default_values(foo), {})

  def test_optional_only(self):

    def foo(x=1, y=2, z=3):  # pylint: disable=unused-argument
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    self.assertEqual(
        constraint,
        function_type.FunctionType(
            (function_type.Parameter(
                "x", function_type.Parameter.POSITIONAL_OR_KEYWORD, True, None),
             function_type.Parameter(
                 "y", function_type.Parameter.POSITIONAL_OR_KEYWORD, True,
                 None),
             function_type.Parameter(
                 "z", function_type.Parameter.POSITIONAL_OR_KEYWORD, True,
                 None))))
    self.assertEqual(
        function_type.FunctionType.get_default_values(foo), {
            "x": 1,
            "y": 2,
            "z": 3
        })

  def test_required_and_optional(self):

    def foo(x, y, z=3):  # pylint: disable=unused-argument
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    self.assertEqual(
        constraint,
        function_type.FunctionType(
            (function_type.Parameter(
                "x", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                None),
             function_type.Parameter(
                 "y", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                 None),
             function_type.Parameter(
                 "z", function_type.Parameter.POSITIONAL_OR_KEYWORD, True,
                 None))))
    self.assertEqual(
        function_type.FunctionType.get_default_values(foo), {"z": 3})

  def test_method_bound(self):

    class MyClass:

      def foo(self, x, y=1):
        pass

    constraint = function_type.FunctionType.from_callable(MyClass().foo)
    self.assertEqual(
        constraint,
        function_type.FunctionType(
            (function_type.Parameter(
                "self", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                None),
             function_type.Parameter(
                 "x", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                 None),
             function_type.Parameter(
                 "y", function_type.Parameter.POSITIONAL_OR_KEYWORD, True,
                 None))))
    self.assertEqual(
        function_type.FunctionType.get_default_values(MyClass().foo), {"y": 1})

  def test_method_unbound(self):

    class MyClass:

      def foo(self, x, y=1):
        pass

    constraint = function_type.FunctionType.from_callable(MyClass.foo)
    self.assertEqual(
        constraint,
        function_type.FunctionType(
            (function_type.Parameter(
                "self", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                None),
             function_type.Parameter(
                 "x", function_type.Parameter.POSITIONAL_OR_KEYWORD, False,
                 None),
             function_type.Parameter(
                 "y", function_type.Parameter.POSITIONAL_OR_KEYWORD, True,
                 None))))
    self.assertEqual(
        function_type.FunctionType.get_default_values(MyClass.foo), {"y": 1})

  def test_required_only_validation(self):

    def foo(x, y):  # pylint: disable=unused-argument
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    constraint.bind(*(1, 2))
    constraint.bind(*(), **{"x": 1, "y": 2})
    constraint.bind(*(), **{"y": 1, "x": 2})
    constraint.bind(*(1,), **{"y": 2})

    with self.assertRaisesRegex(TypeError, "too many positional arguments"):
      constraint.bind(*(1, 2, 3))

    with self.assertRaisesRegex(TypeError, "multiple values for argument 'x'"):
      constraint.bind(*(1,), **{"x": 2})

    with self.assertRaisesRegex(TypeError,
                                "got an unexpected keyword argument 'z'"):
      constraint.bind(*(1, 2), **{"z": 2})

    with self.assertRaisesRegex(TypeError, "missing a required argument: 'x'"):
      constraint.bind(*(), **{"z": 3})

    with self.assertRaisesRegex(TypeError, "missing a required argument: 'x'"):
      constraint.bind(*(), **{"y": 3})

  def test_optional_only_validation(self):

    def foo(x=1, y=2):  # pylint: disable=unused-argument
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    constraint.bind(*(1, 2))
    constraint.bind(*(), **{"x": 1, "y": 2})
    constraint.bind(*(1,), **{"y": 2})
    constraint.bind(*(1,))
    constraint.bind(*(), **{"x": 1})

    with self.assertRaisesRegex(TypeError, "too many positional arguments"):
      constraint.bind(*(1, 2, 3))

    with self.assertRaisesRegex(TypeError, "multiple values for argument 'x'"):
      constraint.bind(*(1,), **{"x": 2})

    with self.assertRaisesRegex(TypeError,
                                "got an unexpected keyword argument 'z'"):
      constraint.bind(*(1, 2), **{"z": 2})

    with self.assertRaisesRegex(TypeError,
                                "got an unexpected keyword argument 'z'"):
      constraint.bind(*(), **{"z": 3})

  def test_required_and_optional_validation(self):

    def foo(x, y=2):  # pylint: disable=unused-argument
      pass

    constraint = function_type.FunctionType.from_callable(foo)
    constraint.bind(*(1, 2))
    constraint.bind(*(), **{"x": 1, "y": 2})
    constraint.bind(*(1,), **{"y": 2})
    constraint.bind(*(1,))
    constraint.bind(*(), **{"x": 1})

    with self.assertRaisesRegex(TypeError, "too many positional arguments"):
      constraint.bind(*(1, 2, 3))

    with self.assertRaisesRegex(TypeError, "multiple values for argument 'x'"):
      constraint.bind(*(1,), **{"x": 2})

    with self.assertRaisesRegex(TypeError,
                                "got an unexpected keyword argument 'z'"):
      constraint.bind(*(1, 2), **{"z": 2})

    with self.assertRaisesRegex(TypeError, "missing a required argument: 'x'"):
      constraint.bind(*(), **{"z": 3})


if __name__ == "__main__":
  test.main()
