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
"""Tests for function_type_utils."""

from absl.testing import parameterized

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import tensor_spec
from tensorflow.python.platform import test
from tensorflow.python.util import tf_decorator


def dummy_tf_decorator(func):

  def wrapper(*args, **kwargs):
    return func(*args, **kwargs)

  return tf_decorator.make_decorator(func, wrapper)


def transparent_decorator(func):
  return func


class FunctionSpecTest(test.TestCase, parameterized.TestCase):

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_required_only(self, input_signature, type_constraint, decorator):

    @decorator
    def foo(x, y, z):  # pylint: disable=unused-argument
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['x', 'y', 'z'], None, None, None, [], None, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[0]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[1]),
            function_type_lib.Parameter(
                'z', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[2])
        ]))

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint':
              (tensor_spec.TensorSpec(shape=None),
               tensor_spec.TensorSpec(shape=None), trace_type.from_value(3))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),),
          'type_constraint':
              (tensor_spec.TensorSpec(shape=None), trace_type.from_value(2),
               trace_type.from_value(3))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_optional_only(self, input_signature, type_constraint, decorator):

    @decorator
    def foo(x=1, y=2, z=3):  # pylint: disable=unused-argument
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['x', 'y', 'z'], None, None, (1, 2, 3), [], None, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {'x': 1, 'y': 2, 'z': 3})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[0]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[1]),
            function_type_lib.Parameter(
                'z', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[2])
        ]))

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint':
              (tensor_spec.TensorSpec(shape=None),
               tensor_spec.TensorSpec(shape=None), trace_type.from_value(3))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_required_and_optional(self, input_signature, type_constraint,
                                 decorator):

    @decorator
    def foo(x, y, z=3):  # pylint: disable=unused-argument
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['x', 'y', 'z'], None, None, (3,), [], None, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {'z': 3})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[0]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[1]),
            function_type_lib.Parameter(
                'z', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[2])
        ]))

  @parameterized.product(
      ({
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_varargs(self, input_signature, type_constraint, decorator):

    @decorator
    def foo(*my_var_args):  # pylint: disable=unused-argument
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['my_var_args_0', 'my_var_args_1', 'my_var_args_2'
         ], None, None, None, [], None, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'my_var_args_0', function_type_lib.Parameter.POSITIONAL_ONLY,
                False, type_constraint[0]),
            function_type_lib.Parameter(
                'my_var_args_1', function_type_lib.Parameter.POSITIONAL_ONLY,
                False, type_constraint[1]),
            function_type_lib.Parameter(
                'my_var_args_2', function_type_lib.Parameter.POSITIONAL_ONLY,
                False, type_constraint[2])
        ]))

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint':
              (tensor_spec.TensorSpec(shape=None),
               tensor_spec.TensorSpec(shape=None), trace_type.from_value(3))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_kwonly(self, input_signature, type_constraint, decorator):

    @decorator
    def foo(x, y, *, z=3):  # pylint: disable=unused-argument
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec), (['x', 'y'], None, None, None, ['z'], {
            'z': 3
        }, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {'z': 3})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[0]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[1]),
            function_type_lib.Parameter(
                'z', function_type_lib.Parameter.KEYWORD_ONLY, True,
                type_constraint[2])
        ]))

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (None, tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),),
          'type_constraint': (None, tensor_spec.TensorSpec(shape=None),
                              trace_type.from_value(1))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None)),
          'type_constraint':
              (None,
               trace_type.from_value([
                   tensor_spec.TensorSpec(shape=None),
                   tensor_spec.TensorSpec(shape=None)
               ], trace_type.InternalTracingContext(is_legacy_signature=True)),
               tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_method_bound_internal(
      self, input_signature, type_constraint, decorator
  ):

    def testing_decorator(func):
      spec = function_type_utils.FunctionSpec.from_function_and_signature(
          func, input_signature
      )
      self.assertEqual(
          tuple(spec.fullargspec),
          (['self', 'x', 'y'], None, None, (1,), [], None, {}),
      )

      self.assertEqual(spec.default_values, {'y': 1})

      self.assertEqual(
          spec.function_type,
          function_type_lib.FunctionType([
              function_type_lib.Parameter(
                  'self',
                  function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
                  False,
                  type_constraint[0],
              ),
              function_type_lib.Parameter(
                  'x',
                  function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
                  False,
                  type_constraint[1],
              ),
              function_type_lib.Parameter(
                  'y',
                  function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
                  True,
                  type_constraint[2],
              ),
          ]),
      )

      return func

    class MyClass:

      @testing_decorator
      def foo(self, x, y=1):
        pass

    MyClass().foo(1)

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),),
          'type_constraint': (tensor_spec.TensorSpec(shape=None),
                              trace_type.from_value(1))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (trace_type.from_value([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], trace_type.InternalTracingContext(is_legacy_signature=True)),
                              tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_method_bound_external(
      self, input_signature, type_constraint, decorator
  ):
    class MyClass:

      @decorator
      def foo(self, x, y=1):
        pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        MyClass().foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['x', 'y'], None, None, (1,), [], None, {}),
    )
    self.assertEqual(spec.default_values, {'y': 1})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[0]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[1])
        ]))

  @parameterized.product(
      ({
          'input_signature': None,
          'type_constraint': (None, None, None)
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None)),
          'type_constraint': (None, tensor_spec.TensorSpec(shape=None),
                              tensor_spec.TensorSpec(shape=None))
      }, {
          'input_signature': (tensor_spec.TensorSpec(shape=None),),
          'type_constraint': (None, tensor_spec.TensorSpec(shape=None),
                              trace_type.from_value(1))
      }, {
          'input_signature': ([
              tensor_spec.TensorSpec(shape=None),
              tensor_spec.TensorSpec(shape=None)
          ], tensor_spec.TensorSpec(shape=None)),
          'type_constraint':
              (None,
               trace_type.from_value([
                   tensor_spec.TensorSpec(shape=None),
                   tensor_spec.TensorSpec(shape=None)
               ], trace_type.InternalTracingContext(is_legacy_signature=True)),
               tensor_spec.TensorSpec(shape=None))
      }),
      decorator=(dummy_tf_decorator, transparent_decorator),
  )
  def test_method_unbound(self, input_signature, type_constraint, decorator):

    class MyClass:

      @decorator
      def foo(self, x, y=1):
        pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        MyClass.foo, input_signature)
    self.assertEqual(
        tuple(spec.fullargspec),
        (['self', 'x', 'y'], None, None, (1,), [], None, {}))
    self.assertEqual(spec.input_signature, input_signature)
    self.assertEqual(spec.default_values, {'y': 1})
    self.assertEqual(
        spec.function_type,
        function_type_lib.FunctionType([
            function_type_lib.Parameter(
                'self', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
                False, type_constraint[0]),
            function_type_lib.Parameter(
                'x', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, False,
                type_constraint[1]),
            function_type_lib.Parameter(
                'y', function_type_lib.Parameter.POSITIONAL_OR_KEYWORD, True,
                type_constraint[2])
        ]))

  def test_spec_summary(self):
    input_signature = (
        tensor_spec.TensorSpec(shape=None),
        tensor_spec.TensorSpec(shape=None),
    )

    @dummy_tf_decorator
    def foo(x=2, y=3):  # pylint: disable=unused-argument
      pass

    spec = function_type_utils.FunctionSpec.from_function_and_signature(
        foo, input_signature
    )
    self.assertEqual(
        spec.signature_summary(True),
        'Input Parameters:\n'
        + '  x (POSITIONAL_OR_KEYWORD):'
        ' TensorSpec(shape=<unknown>, dtype=tf.float32, name=None)\n'
        + '  y'
        ' (POSITIONAL_OR_KEYWORD): TensorSpec(shape=<unknown>,'
        ' dtype=tf.float32, name=None)\n'
        + 'Output Type:\n'
        + '  None\n'
        + 'Captures:\n'
        + '  None\n'
        + 'Defaults:\n'
        + '  x: 2\n'
        + '  y: 3',
    )


# TODO(fmuham): Remove when is_same_structure is removed.
class SameStructureTest(test.TestCase):

  def test_same_structure(self):
    self.assertTrue(
        function_type_utils.is_same_structure([1, 2, 3], [1, 2, 3], True)
    )
    self.assertTrue(
        function_type_utils.is_same_structure([1, 2, 3], [1, 2, 4], False)
    )

    self.assertFalse(
        function_type_utils.is_same_structure([1, 2, 3], [1, 2, 4], True)
    )
    self.assertFalse(
        function_type_utils.is_same_structure([1, 2, 3], [1, 2, 3, 4], False)
    )


if __name__ == '__main__':
  test.main()
