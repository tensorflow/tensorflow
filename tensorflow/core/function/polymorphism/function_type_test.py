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

import collections
import pickle
import platform

from absl.testing import parameterized

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.framework import func_graph
from tensorflow.python.platform import test
from tensorflow.python.types import trace


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

  def test_pickle(self):
    original = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, None),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, None),
        function_type.Parameter("z", function_type.Parameter.KEYWORD_ONLY,
                                False, None)
    ])
    cloned = pickle.loads(pickle.dumps(original))
    self.assertEqual(original, cloned)


class CanonicalizationTest(test.TestCase, parameterized.TestCase):
  args_1_2_3 = {"args": (1, 2, 3), "kwargs": {}}
  args_1_2_kwargs_z_3 = {"args": (1, 2), "kwargs": {"z": 3}}
  kwargs_x_1_y_2_z_3 = {"args": (), "kwargs": {"x": 1, "y": 2, "z": 3}}
  args_1_2 = {"args": (1, 2), "kwargs": {}}
  args_1_kwargs_y_2 = {"args": (1,), "kwargs": {"y": 2}}
  kwargs_x_1_y_2 = {"args": (), "kwargs": {"x": 1, "y": 2}}

  @parameterized.parameters(
      args_1_2_3,
      args_1_2_kwargs_z_3,
      kwargs_x_1_y_2_z_3,
  )
  def test_required_only(self, args, kwargs):

    def foo(x, y, z):
      del x, y, z

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    bound_args, mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)

    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  @parameterized.parameters(
      args_1_2_3,
      args_1_2_kwargs_z_3,
      kwargs_x_1_y_2_z_3,
  )
  def test_optional_all(self, args, kwargs):

    def foo(x=1, y=2, z=3):
      del x, y, z

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    bound_args, mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)

    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  @parameterized.parameters(args_1_2, args_1_kwargs_y_2, kwargs_x_1_y_2)
  def test_optional_some(self, args, kwargs):

    def foo(x=1, y=2, z=3):
      del x, y, z

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    bound_args, mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, function_type.FunctionType.get_default_values(foo), {},
        polymorphic_type)

    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  @parameterized.parameters(
      args_1_2_3,
      args_1_2_kwargs_z_3,
      kwargs_x_1_y_2_z_3,
  )
  def test_mixed(self, args, kwargs):

    def foo(x, y, z=3):
      del x, y, z

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    bound_args, mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)

    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  def test_varargs(self):

    def foo(*my_var_args):
      del my_var_args

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    bound_args, mono_type, _ = function_type.canonicalize_to_monomorphic(
        (1, 2, 3), {}, {}, {}, polymorphic_type)

    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("my_var_args_0",
                                function_type.Parameter.POSITIONAL_ONLY, False,
                                trace_type.from_value(1, type_context)),
        function_type.Parameter("my_var_args_1",
                                function_type.Parameter.POSITIONAL_ONLY, False,
                                trace_type.from_value(2, type_context)),
        function_type.Parameter("my_var_args_2",
                                function_type.Parameter.POSITIONAL_ONLY, False,
                                trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  def test_varkwargs(self):

    def foo(**kwargs):
      del kwargs

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    bound_args, mono_type, _ = function_type.canonicalize_to_monomorphic((), {
        "x": 1,
        "y": 2,
        "z": 3
    }, {}, {}, polymorphic_type)

    self.assertEqual(bound_args.args, ())
    self.assertEqual(bound_args.kwargs, {"x": 1, "y": 2, "z": 3})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  def test_varargs_and_varkwargs(self):

    def foo(*args, **kwargs):
      del args, kwargs

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    bound_args, mono_type, _ = function_type.canonicalize_to_monomorphic((1,), {
        "y": 2,
        "z": 3
    }, {}, {}, polymorphic_type)

    self.assertEqual(bound_args.args, (1,))
    self.assertEqual(bound_args.kwargs, {"y": 2, "z": 3})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("args_0",
                                function_type.Parameter.POSITIONAL_ONLY, False,
                                trace_type.from_value(1, type_context)),
        function_type.Parameter("y", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  @parameterized.parameters(
      args_1_2_kwargs_z_3,
      kwargs_x_1_y_2_z_3,
  )
  def test_kwonly(self, args, kwargs):

    def foo(x, y, *, z):
      del x, y, z

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    bound_args, mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)

    self.assertEqual(bound_args.args, (1, 2))
    self.assertEqual(bound_args.kwargs, {"z": 3})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)

  @parameterized.parameters(
      args_1_2_3,
      args_1_2_kwargs_z_3,
  )
  def test_posonly(self, args, kwargs):
    major, minor, _ = platform.python_version_tuple()
    if not (major == "3" and int(minor) >= 8):
      self.skipTest("Positional only args are supported in Python 3.8+")

    # Raises syntax error in 3.7 but is important coverage for 3.8+.
    foo = eval("lambda x, y, /, z: x + y + z")  # pylint: disable=eval-used

    polymorphic_type = function_type.FunctionType.from_callable(foo)
    bound_args, mono_type, _ = function_type.canonicalize_to_monomorphic(
        args, kwargs, {}, {}, polymorphic_type)

    self.assertEqual(bound_args.args, (1, 2, 3))
    self.assertEqual(bound_args.kwargs, {})

    type_context = trace_type.InternalTracingContext()
    expected_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(3, type_context)),
    ])

    self.assertEqual(mono_type, expected_type)


class TypeHierarchyTest(test.TestCase):

  def test_same_type(self):
    foo_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(1))
    ])
    self.assertEqual(foo_type, foo_type)
    self.assertTrue(foo_type.is_supertype_of(foo_type))
    self.assertEqual(
        foo_type,
        foo_type.most_specific_common_subtype([foo_type, foo_type, foo_type]))

  def test_unrelated_types(self):
    foo_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(1))
    ])
    bar_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(2))
    ])
    self.assertNotEqual(foo_type, bar_type)
    self.assertFalse(foo_type.is_supertype_of(bar_type))
    self.assertIsNone(
        foo_type.most_specific_common_subtype([bar_type, bar_type]))
    self.assertIsNone(
        foo_type.most_specific_common_subtype([bar_type, foo_type]))

  def test_partial_raises_error(self):
    foo_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(1)),
    ])
    bar_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, None)
    ])
    self.assertNotEqual(foo_type, bar_type)

    with self.assertRaises(TypeError):
      foo_type.is_supertype_of(bar_type)

    with self.assertRaises(TypeError):
      bar_type.is_supertype_of(foo_type)

    with self.assertRaises(TypeError):
      foo_type.most_specific_common_subtype([bar_type, bar_type])

    with self.assertRaises(TypeError):
      bar_type.most_specific_common_subtype([foo_type, bar_type])

  def test_related_types(self):

    class MockAlwaysSuperType(trace.TraceType):

      def is_subtype_of(self, other: trace.TraceType) -> bool:
        return False

      def most_specific_common_supertype(self, others):
        return self

      def placeholder_value(self, placeholder_context):
        raise NotImplementedError

      def __eq__(self, other):
        return self is other

      def __hash__(self):
        return 0

    supertype = MockAlwaysSuperType()

    class MockAlwaysSubtype(trace.TraceType):

      def is_subtype_of(self, other) -> bool:
        return True

      def most_specific_common_supertype(self, others):
        return supertype

      def placeholder_value(self, placeholder_context):
        raise NotImplementedError

      def __eq__(self, other):
        return self is other

      def __hash__(self):
        return 1

    subtype = MockAlwaysSubtype()

    foo_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, supertype),
    ])
    bar_type = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, subtype)
    ])

    self.assertNotEqual(foo_type, bar_type)
    self.assertTrue(bar_type.is_supertype_of(foo_type))
    self.assertFalse(foo_type.is_supertype_of(bar_type))

    self.assertEqual(
        foo_type.most_specific_common_subtype([bar_type, foo_type]), foo_type)
    self.assertEqual(
        bar_type.most_specific_common_subtype([bar_type, foo_type]), foo_type)

  def test_placeholder_arg(self):
    type_context = trace_type.InternalTracingContext()
    foo = function_type.FunctionType([
        function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                                False, trace_type.from_value(1, type_context)),
        function_type.Parameter("y",
                                function_type.Parameter.POSITIONAL_OR_KEYWORD,
                                False, trace_type.from_value(2, type_context)),
        function_type.Parameter("z", function_type.Parameter.KEYWORD_ONLY,
                                False, trace_type.from_value(3, type_context)),
    ])
    context_graph = func_graph.FuncGraph("test")
    placeholder_context = trace_type.InternalPlaceholderContext(context_graph)
    self.assertEqual(
        foo.placeholder_arguments(placeholder_context).args, (1, 2))
    self.assertEqual(
        foo.placeholder_arguments(placeholder_context).kwargs, {"z": 3})


class CapturesTest(test.TestCase):

  def setUp(self):
    super(CapturesTest, self).setUp()

    def gen_type_fn(mapping):
      return function_type.FunctionType([], collections.OrderedDict(mapping))

    self.type_a1_b1 = gen_type_fn({
        "a": trace_type.from_value(1),
        "b": trace_type.from_value(1)
    })
    self.type_a1_b1_c1 = gen_type_fn({
        "a": trace_type.from_value(1),
        "b": trace_type.from_value(1),
        "c": trace_type.from_value(1)
    })
    self.type_a2_b2_c2 = gen_type_fn({
        "a": trace_type.from_value(2),
        "b": trace_type.from_value(2),
        "c": trace_type.from_value(2)
    })
    self.type_a1_b1_c2 = gen_type_fn({
        "a": trace_type.from_value(1),
        "b": trace_type.from_value(1),
        "c": trace_type.from_value(2)
    })
    self.type_d1 = gen_type_fn({"d": trace_type.from_value(1)})

  def testCapturesSubtype(self):
    self.assertFalse(self.type_a1_b1.is_supertype_of(self.type_a1_b1_c1))
    self.assertTrue(self.type_a1_b1_c1.is_supertype_of(self.type_a1_b1))
    self.assertFalse(self.type_a1_b1_c1.is_supertype_of(self.type_a2_b2_c2))
    self.assertFalse(self.type_a1_b1_c1.is_supertype_of(self.type_a2_b2_c2))
    self.assertFalse(self.type_d1.is_supertype_of(self.type_a1_b1))

  def testCapturesSupertype(self):
    supertype_1 = self.type_a1_b1_c1.most_specific_common_subtype(
        [self.type_a1_b1_c1])
    self.assertLen(supertype_1.captures, 3)

    supertype_2 = self.type_a1_b1.most_specific_common_subtype(
        [self.type_a1_b1_c1, self.type_a2_b2_c2])
    self.assertIsNone(supertype_2)

    supertype_3 = self.type_a1_b1.most_specific_common_subtype(
        [self.type_a1_b1_c2])
    self.assertLen(supertype_3.captures, 2)

    supertype_4 = self.type_a1_b1_c1.most_specific_common_subtype(
        [self.type_a1_b1_c2])
    self.assertIsNone(supertype_4)

    supertype_5 = self.type_a1_b1_c1.most_specific_common_subtype(
        [self.type_d1])
    self.assertEmpty(supertype_5.captures)


class SanitizationTest(test.TestCase):

  def testRename(self):
    self.assertEqual("arg_42", function_type.sanitize_arg_name("42"))
    self.assertEqual("a42", function_type.sanitize_arg_name("a42"))
    self.assertEqual("arg__42", function_type.sanitize_arg_name("_42"))
    self.assertEqual("a___", function_type.sanitize_arg_name("a%$#"))
    self.assertEqual("arg____", function_type.sanitize_arg_name("%$#"))
    self.assertEqual("foo", function_type.sanitize_arg_name("foo"))
    self.assertEqual("Foo", function_type.sanitize_arg_name("Foo"))
    self.assertEqual("arg_96ab_cd___53",
                     function_type.sanitize_arg_name("96ab.cd//?53"))

  def testLogWarning(self):

    with self.assertLogs(level="WARNING") as logs:
      result = function_type.sanitize_arg_name("96ab.cd//?53")

    self.assertEqual(result, "arg_96ab_cd___53")

    expected_message = (
        "WARNING:absl:`96ab.cd//?53` is not a valid tf.function parameter name."
        " Sanitizing to `arg_96ab_cd___53`.")
    self.assertIn(expected_message, logs.output)


class SerializationTest(test.TestCase, parameterized.TestCase):

  @parameterized.product(
      name=["arg_0", "param"],
      kind=[
          function_type.Parameter.POSITIONAL_ONLY,
          function_type.Parameter.POSITIONAL_OR_KEYWORD
      ],
      optional=[True, False],
      type_contraint=[None, trace_type.from_value(1)])
  def testParameter(self, name, kind, optional, type_contraint):
    original = function_type.Parameter(name, kind, optional, type_contraint)
    expected_type_constraint = serialization.serialize(
        type_contraint) if type_contraint else None
    expected = function_type_pb2.Parameter(
        name=name,
        kind=function_type.PY_TO_PROTO_ENUM[kind],
        is_optional=optional,
        type_constraint=expected_type_constraint)
    self.assertEqual(original.to_proto(), expected)
    self.assertEqual(function_type.Parameter.from_proto(expected), original)

  def testFunctionType(self):
    original = function_type.FunctionType([
        function_type.Parameter("a", function_type.Parameter.POSITIONAL_ONLY,
                                False, None),
    ], collections.OrderedDict([("b", trace_type.from_value(1))]))
    expected = function_type_pb2.FunctionType(
        parameters=[
            function_type_pb2.Parameter(
                name="a",
                kind=function_type_pb2.Parameter.Kind.POSITIONAL_ONLY,
                is_optional=False)
        ],
        captures=[
            function_type_pb2.Capture(
                name="b",
                type_constraint=serialization.serialize(
                    trace_type.from_value(1)))
        ])
    self.assertEqual(original.to_proto(), expected)
    self.assertEqual(function_type.FunctionType.from_proto(expected), original)


class FromStructuredSignatureTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {
          "signature": ((1, 2, 3), {}),
          "expected_types": (
              trace_type.from_value(1),
              trace_type.from_value(2),
              trace_type.from_value(3),
          ),
      },
      {
          "signature": (([1, 2, 3],), {}),
          "expected_types": (
              trace_type.from_value([1, 2, 3]),
          ),
      },
      {
          "signature": ((), {}),
          "expected_types": (),
      },
  )
  def testArgs(self, signature, expected_types):
    generated_type = function_type.from_structured_signature(signature)
    self.assertIsNone(generated_type.output)
    for i, p in enumerate(generated_type.parameters.values()):
      self.assertEqual(p.kind, function_type.Parameter.POSITIONAL_ONLY)
      self.assertEqual(p.type_constraint, expected_types[i])

  @parameterized.parameters(
      {
          "signature": ((), {"a": 1, "b": 2, "c": 3}),
          "expected_types": {
              "a": trace_type.from_value(1),
              "b": trace_type.from_value(2),
              "c": trace_type.from_value(3),
          },
      },
      {
          "signature": ((), {"a": [1, 2, 3]}),
          "expected_types": {
              "a": trace_type.from_value([1, 2, 3]),
          },
      },
      {
          "signature": ((), {}),
          "expected_types": {},
      },
  )
  def testKwargs(self, signature, expected_types):
    generated_type = function_type.from_structured_signature(signature)
    self.assertIsNone(generated_type.output)
    for p in generated_type.parameters.values():
      self.assertEqual(p.kind, function_type.Parameter.KEYWORD_ONLY)
      self.assertEqual(p.type_constraint, expected_types[p.name])

  @parameterized.parameters(
      {"output_signature": 1},
      {"output_signature": [1, 2, 3]},
      {"output_signature": ()},
  )
  def testOutput(self, output_signature):
    generated_type = function_type.from_structured_signature(
        ((), {}), output_signature
    )
    self.assertEqual(
        generated_type.output,
        trace_type.from_value(
            output_signature,
            trace_type.InternalTracingContext(is_legacy_signature=True),
        )
    )

if __name__ == "__main__":
  test.main()
