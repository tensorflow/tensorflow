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
"""Tests for script operations."""
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops.script_ops import numpy_function
from tensorflow.python.platform import test


class ArgsAndDecoratorsTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_decorator(self):
    count = 0

    @script_ops.numpy_function(Tout=dtypes.int32)
    def plus(a, b):
      nonlocal count
      count += 1
      return a + b

    actual_result = plus(1, 2)
    expect_result = constant_op.constant(3, dtypes.int32)
    self.assertAllEqual(actual_result, expect_result)
    self.assertEqual(count, 1)

  @test_util.run_in_graph_and_eager_modes
  def test_inline_decorator(self):
    count = 0

    def plus(a, b):
      nonlocal count
      count += 1
      return a + b

    py_plus = script_ops.eager_py_func(Tout=dtypes.int32)(plus)

    actual_result = py_plus(1, 2)
    expect_result = constant_op.constant(3, dtypes.int32)
    self.assertAllEqual(actual_result, expect_result)
    self.assertEqual(count, 1)

  def test_bad_args(self):
    def minus(a, b):
      return a - b

    # No Tout
    with self.assertRaisesRegex(TypeError, r"Missing.*Tout"):

      @script_ops.eager_py_func
      def plus1(a, b):
        return a + b

    # Forgot to set arg-name
    with self.assertRaisesRegex(TypeError, r"Missing.*Tout"):

      @script_ops.eager_py_func(dtypes.int32)
      def plus2(a, b):
        return a + b

    # inputs in the decoration doesn't make sense
    with self.assertRaisesRegex(TypeError, r"Don't.*inp.*decorator"):

      @script_ops.eager_py_func(inp=[], Tout=dtypes.int32)
      def plus3(a, b):
        return a + b

    # Forgot to set the function is still a type-error.
    with self.assertRaisesRegex(TypeError, r"Don't.*inp.*decorator"):
      script_ops.eager_py_func(inp=[], Tout=dtypes.int32)

    # Still an error to call without inp
    with self.assertRaisesRegex(TypeError, r"Missing.*inp"):
      script_ops.eager_py_func(minus, Tout=dtypes.int32)

    # Why would you do this?
    with self.assertRaisesRegex(TypeError, r"Missing.*inp"):

      @script_ops.eager_py_func(func=minus, Tout=dtypes.int32)
      def plus4(a, b):
        return a + b

  def test_decorator_pass_through_extra_args(self):
    got_extra = None

    def dummy_script_op(func, inp, extra=None, **kwargs):
      del kwargs
      nonlocal got_extra
      got_extra = extra
      return func(*inp)

    decorator = script_ops._check_args_and_maybe_make_decorator(
        dummy_script_op, "dummy", Tout=dtypes.int32, extra="extra"
    )

    @decorator
    def plus(a, b):
      return a + b

    self.assertIsNone(got_extra)
    self.assertEqual(plus(1, 2), 3)
    self.assertEqual(got_extra, "extra")


class NumpyFunctionTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_numpy_arguments(self):

    def plus(a, b):
      return a + b

    actual_result = script_ops.numpy_function(plus, [1, 2], dtypes.int32)
    expect_result = constant_op.constant(3, dtypes.int32)
    self.assertAllEqual(actual_result, expect_result)

  def test_stateless(self):
    call_count = 0

    def plus(a, b):
      nonlocal call_count
      call_count += 1
      return a + b

    @def_function.function
    def numpy_func_stateless(a, b):
      return numpy_function(plus, [a, b], dtypes.int32, stateful=False)

    @def_function.function
    def func_stateless(a, b):
      sum1 = numpy_func_stateless(a, b)
      sum2 = numpy_func_stateless(a, b)
      return sum1 + sum2

    self.evaluate(func_stateless(
        constant_op.constant(1),
        constant_op.constant(2),
    ))

    self.assertIn(call_count, (1, 2))  # as stateless, func may be deduplicated

  def test_stateful(self):
    call_count = 0

    def plus(a, b):
      nonlocal call_count
      call_count += 1
      return a + b

    @def_function.function
    def numpy_func_stateful(a, b):
      return numpy_function(plus, [a, b], dtypes.int32, stateful=True)

    @def_function.function
    def func_stateful(a, b):
      sum1 = numpy_func_stateful(a, b)
      sum2 = numpy_func_stateful(a, b)
      return sum1 + sum2

    self.evaluate(func_stateful(
        constant_op.constant(1),
        constant_op.constant(2),
    ))

    self.assertEqual(call_count,
                     2)  # as stateful, func is guaranteed to execute twice


class PyFunctionTest(test.TestCase):
  @test_util.run_in_graph_and_eager_modes
  def test_variable_arguments(self):

    def plus(a, b):
      return a + b

    v1 = resource_variable_ops.ResourceVariable(1)
    self.evaluate(v1.initializer)

    actual_result = script_ops.eager_py_func(plus, [v1, 2], dtypes.int32)
    expect_result = constant_op.constant(3, dtypes.int32)
    self.assertAllEqual(actual_result, expect_result)

  @test_util.run_in_graph_and_eager_modes
  def test_fail_on_non_utf8_token(self):
    value = constant_op.constant(value=[1, 2])
    token = b"\xb0"
    data_type = [dtypes.int32]
    with self.assertRaises((errors.InternalError, UnicodeDecodeError)):
      self.evaluate(
          gen_script_ops.py_func(input=[value], token=token, Tout=data_type))


if __name__ == "__main__":
  test.main()
