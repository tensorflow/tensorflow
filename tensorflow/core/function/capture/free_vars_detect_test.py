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
"""Tests for detecting free vars in tf.function."""

import functools
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.core.function.capture import free_vars_detect
from tensorflow.python.util import tf_decorator


def get_var_name(d):
  return [var.name for var in d]


class GetSelfObjFromClosureTest(parameterized.TestCase):

  def test_single_enclosing_class(self):

    class Foo():

      def __init__(self):
        self.val = 1

      def bar(self):
        x = 2

        def fn():
          return self.val + x

        return fn

    foo = Foo()
    fn = foo.bar()
    self_obj = free_vars_detect._get_self_obj_from_closure(fn)
    self.assertIs(self_obj, foo)


class FreeVarDetectionTest(parameterized.TestCase):

  def test_func_arg(self):
    x = 1  # pylint: disable=unused-variable

    def f(x):
      return x + 1

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertEmpty(func_map)

  def test_func_local_var(self):

    def f():
      x = 1
      return x + 1

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertEmpty(func_map)

  def test_global_var_int(self):
    x = 1

    def f():
      return x + 1

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_builtin_func(self):

    def f(x):
      return len(x)

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertEmpty(func_map)

  def test_global_var_dict(self):
    glob = {"a": 1}

    def f():
      return glob["a"] + 1

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_global_var_dict_w_var_index(self):
    glob = {"a": 1}
    key = "a"

    def f():
      return glob[key] + 1

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["glob", "key"])

  def test_duplicate_global_var(self):
    x = 1

    def f():
      return x + x

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["x"])

  @parameterized.named_parameters(
      ("lambda_1", lambda _x: 3,), ("lambda_2", lambda _x: 3,))
  def test_multiple_lambda_w_same_line_num_and_args(self, fn):
    func_map = free_vars_detect._detect_function_free_vars(fn)
    self.assertEmpty(func_map)

  def test_lambda_wo_free_var(self):
    f = lambda x: x + x
    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertEmpty(func_map)

  def test_lambda_w_free_var(self):
    glob = 1
    f = lambda x: x + glob
    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_multi_lambda_w_free_var(self):
    glob = 1
    g = lambda x: x + glob
    h = lambda: glob + 1

    def f(x):
      return g(x) + h()

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertLen(func_map, 3)
    self.assertIn("f", func_map.keys())
    self.assertIn("g", func_map.keys())
    self.assertIn("h", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["g", "h"])
    free_vars = get_var_name(func_map["g"])
    self.assertSequenceEqual(free_vars, ["glob"])
    free_vars = get_var_name(func_map["h"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_lambda_inline(self):
    glob = 1

    def f(x):
      return lambda: x + glob

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_glob_numpy_var(self):
    a = 0
    b = np.asarray(1)

    def f():
      c = np.asarray(2)
      res = a + b + c
      return res

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["a", "b"])

  def test_global_var_in_nested_func(self):
    x = 1

    def f():

      def g():
        return x + 1

      return g()

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    self.assertLen(func_map.keys(), 1)
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_global_var_from_outer_func(self):
    x = 1

    def g():
      return x + 1

    def f():
      return g()

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    self.assertIn("g", func_map.keys())
    self.assertLen(func_map.keys(), 2)
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["g"])
    free_vars = get_var_name(func_map["g"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_method(self):
    x = 1

    class Foo():

      def f(self):
        return x

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 1)
    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_method_w_method_call(self):
    x = 0

    class Foo():

      def f(self):
        return self.g

      def g(self):
        return [x]

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["self.g"])

    self.assertIn("Foo.g", func_map.keys())
    free_vars = get_var_name(func_map["Foo.g"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_method_w_self_as_arg(self):
    x = 1

    class Foo():

      def f(self):
        return self.g(self)

      def g(self, obj):
        if obj != self:
          return x
        else:
          return -x

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["self.g"])

    self.assertIn("Foo.g", func_map.keys())
    free_vars = get_var_name(func_map["Foo.g"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_self_inside_method(self):
    x = 1

    class Foo():

      def __init__(self):
        self.val = 2

      def bar(self):

        def tf_func():
          return self.val + x

        return tf_func

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.bar())
    self.assertLen(func_map.keys(), 1)

    self.assertIn("tf_func", func_map.keys())
    free_vars = get_var_name(func_map["tf_func"])
    self.assertSequenceEqual(free_vars, ["self", "self.val", "x"])

  def test_self_inside_function_w_multiple_closures(self):
    # Test when a function contins multiple closures

    class Foo():

      def method(self):

        class Baz():

          def baz_str(self):
            return "Baz"

        baz = Baz()
        x = "x"

        class Bar():

          def bar_str(self):
            return x + "Bar"

          def method(self):

            def fn():
              return self.bar_str() + baz.baz_str()

            return fn

        bar = Bar()
        return bar.method()

    foo = Foo()
    fn = foo.method()
    # cells for `self.bar_str()`, `baz.baz_str()`
    self.assertLen(fn.__closure__, 2)

    func_map = free_vars_detect._detect_function_free_vars(fn)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("fn", func_map.keys())
    free_vars = get_var_name(func_map["fn"])
    self.assertSequenceEqual(free_vars, ["baz", "self", "self.bar_str"])

    self.assertIn("Bar.bar_str", func_map.keys())
    free_vars = get_var_name(func_map["Bar.bar_str"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_method_w_self_attribute(self):
    x = 0

    class Foo():

      def __init__(self):
        self.x = 1
        self.y = 2

      def f(self):
        return self.g + self.x + self.y

      def g(self):
        return x

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["self.g", "self.x", "self.y"])

    self.assertIn("Foo.g", func_map.keys())
    free_vars = get_var_name(func_map["Foo.g"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_method_w_multiple_attributes(self):
    glob = "dummy_value"

    class Foo():

      def f(self):
        return self.g.h.x.y.z

      def g(self):
        return glob

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["self.g"])

    self.assertIn("Foo.g", func_map.keys())
    free_vars = get_var_name(func_map["Foo.g"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_classmethod_decorator(self):
    glob = 1

    class Foo():

      @classmethod
      def f(cls):
        return glob

    func_map = free_vars_detect._detect_function_free_vars(Foo.f)
    self.assertLen(func_map.keys(), 1)
    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_method_call_classmethod(self):
    glob = 1

    class Foo():

      def f(self):
        return self.g()

      @classmethod
      def g(cls):
        return glob

    foo = Foo()
    func_map = free_vars_detect._detect_function_free_vars(foo.f)
    self.assertLen(func_map.keys(), 2)

    self.assertIn("Foo.f", func_map.keys())
    free_vars = get_var_name(func_map["Foo.f"])
    self.assertSequenceEqual(free_vars, ["self.g"])

    self.assertIn("Foo.g", func_map.keys())
    free_vars = get_var_name(func_map["Foo.g"])
    self.assertSequenceEqual(free_vars, ["glob"])

  def test_global_var_from_renamed_outer_func(self):
    x = 1

    def g():
      return x + 1

    def f():
      h = g
      return h()

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    self.assertIn("g", func_map.keys())
    self.assertLen(func_map.keys(), 2)
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["g"])
    free_vars = get_var_name(func_map["g"])
    self.assertSequenceEqual(free_vars, ["x"])

  def test_decorated_method_w_self_no_exception(self):
    """Test this pattern does not raise any exceptions."""

    def dummy_tf_function(func):

      func_map = free_vars_detect._detect_function_free_vars(func)
      self.assertLen(func_map, 1)
      self.assertIn("foo", func_map.keys())
      free_vars = get_var_name(func_map["foo"])
      self.assertSequenceEqual(free_vars, ["dummy_tf_function"])

      def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

      return wrapper

    glob = 1

    # This pattern is not fully supported yet in the sense that `self.bar()` is
    # not inspected so `glob` cannot be detected.
    # The reason is the neither `self` nor `self.bar` is accessible from the
    # perspective of dummy_tf_function decorator.
    # One possible solution is parsing the source code of the whole module,
    # instead of single function. And probably get the source of `self.bar`
    # from the AST of the module where `Foo` is defined. One potentail challenge
    # of this approach is how to locate the decorated function in the AST.
    class Foo():

      @dummy_tf_function
      def foo(self):
        return self.bar()

      def bar(self):
        return glob

    _ = Foo()

  # Use `wrapper_first` to control different arguments order
  @parameterized.parameters(
      (functools.update_wrapper, True),
      (tf_decorator.make_decorator, False))
  def test_func_w_decorator(self, make_decorator, wrapper_first):
    x = 1

    def decorator_foo(func):

      def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

      if wrapper_first:
        return make_decorator(wrapper, func)
      else:
        return make_decorator(func, wrapper)

    @decorator_foo
    @decorator_foo
    def f():

      @decorator_foo
      @decorator_foo
      def g():
        return x + 1

      return g()

    func_map = free_vars_detect._detect_function_free_vars(f)
    self.assertIn("f", func_map.keys())
    self.assertLen(func_map.keys(), 2)
    free_vars = get_var_name(func_map["f"])
    self.assertSequenceEqual(free_vars, ["decorator_foo", "x"])

  # TODO(panzf): test the pattern when callable function args are supported
  @unittest.skip("Feature not implemented")
  def test_global_var_from_arg_func(self):
    x = 1

    def g():
      return x + 1

    def f(h):
      return h()

    _ = f(g)


class GenerateLoggingTest(parameterized.TestCase):

  def _remove_explanation(self, logging_txt):
    free_vars = logging_txt.split("\n")
    self.assertGreater(len(free_vars), 2)
    return "\n".join(free_vars[2:])

  def test_none_input(self):
    txt = free_vars_detect.generate_free_var_logging(None)
    self.assertIsNone(txt)

  def test_non_function_input(self):
    x = 1

    class Foo():

      def bar(self):
        return x

    foo = Foo()
    txt = free_vars_detect.generate_free_var_logging(foo)
    self.assertIsNone(txt)

  def test_func_wo_source_code(self):
    code = "def f_exec():\n  return 1"
    # Use `exec` to generate a function without source code
    exec(code, globals())  # pylint: disable=exec-used
    txt = free_vars_detect.generate_free_var_logging(f_exec)  # pylint: disable=undefined-variable
    self.assertIsNone(txt)

  def test_no_free_var(self):

    def f(x):
      return x + 1

    txt = free_vars_detect.generate_free_var_logging(f)
    self.assertIsNone(txt)

  def test_single_func(self):
    x = 1
    y = 2

    def f(a):
      return a + x + y

    txt = free_vars_detect.generate_free_var_logging(f)
    txt = self._remove_explanation(txt)
    self.assertEqual(txt, "Inside function f(): x, y")

  def test_nested_func(self):
    x = 1
    y = 2

    def g():
      return y

    def f():
      return g() + x

    txt = free_vars_detect.generate_free_var_logging(f)
    txt = self._remove_explanation(txt)
    lines = txt.split("\n")
    self.assertLen(lines, 2)
    self.assertEqual(lines[0], "Inside function f(): g, x")
    self.assertEqual(lines[1], "Inside function g(): y")

  def test_method_w_method_call(self):
    x = 0

    class Foo():

      def f(self):
        return self.g

      def g(self):
        return [x]

    foo = Foo()
    txt = free_vars_detect.generate_free_var_logging(foo.f)
    txt = self._remove_explanation(txt)
    lines = txt.split("\n")
    self.assertLen(lines, 2)
    self.assertEqual(lines[0], "Inside function Foo.f(): self.g")
    self.assertEqual(lines[1], "Inside function Foo.g(): x")

  def test_partial_func(self):
    x = 1
    y = 2

    def f(a):
      return a + x + y

    partial_f = functools.partial(f, a=0)

    txt = free_vars_detect.generate_free_var_logging(partial_f)
    txt = self._remove_explanation(txt)
    self.assertEqual(txt, "Inside function f(): x, y")

  def test_partial_method(self):
    x = 0

    class Foo():

      def f(self):
        return self.g

      def g(self):
        return [x]

      partial_f = functools.partialmethod(f)

    foo = Foo()
    txt = free_vars_detect.generate_free_var_logging(foo.partial_f)
    txt = self._remove_explanation(txt)
    lines = txt.split("\n")
    self.assertLen(lines, 2)
    self.assertEqual(lines[0], "Inside function Foo.f(): self.g")
    self.assertEqual(lines[1], "Inside function Foo.g(): x")

  def test_partial_wrapped_partial_func(self):

    def decorator_foo(func):

      def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

      return functools.update_wrapper(wrapper, func)

    x = 1
    y = 2

    def f(a, b):
      return a + b + x + y

    f = functools.partial(f, a=0)
    f = decorator_foo(f)
    f = functools.partial(f, b=0)

    txt = free_vars_detect.generate_free_var_logging(f)
    txt = self._remove_explanation(txt)
    self.assertEqual(txt, "Inside function f(): x, y")

  def test_freevar_threshold(self):
    a = b = c = d = e = 1

    def f():
      return a + b + c + d + e

    txt = free_vars_detect.generate_free_var_logging(f, var_threshold=3)
    txt = self._remove_explanation(txt)
    self.assertEqual(txt, "Inside function f(): a, b, c...")

  def test_func_threshold(self):
    x = 1

    def g():
      return x

    def h():
      return x

    def f():
      return g() + h()

    txt = free_vars_detect.generate_free_var_logging(f, fn_threshold=2)
    txt = self._remove_explanation(txt)
    lines = txt.split("\n")
    self.assertLen(lines, 3)
    self.assertEqual(lines[0], "Inside function f(): g, h")
    self.assertEqual(lines[1], "Inside function g(): x")
    self.assertEqual(lines[2], "...")

  def test_func_second_call_return_none(self):
    x = 1

    def f():
      return x

    logging_txt = free_vars_detect.generate_free_var_logging(f)
    self.assertIsNotNone(logging_txt)
    logging_txt = free_vars_detect.generate_free_var_logging(f)
    self.assertIsNone(logging_txt)


if __name__ == "__main__":
  unittest.main()
