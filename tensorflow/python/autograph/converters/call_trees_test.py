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
"""Tests for call_trees module."""

import random
import types

from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class MockConvertedCall(object):

  def __init__(self):
    self.calls = []

  def __call__(self, f, args, kwargs, caller_fn_scope=None, options=None):
    del caller_fn_scope, options
    self.calls.append((args, kwargs))
    kwargs = kwargs or {}
    return f(*args, **kwargs)


class CallTreesTest(converter_testing.TestCase):
  """Tests for call_trees converter module.
  
  Validates that function calls are properly transformed and tracked,
  including handling of arguments, keyword arguments, star args, and
  special cases like method calls and debugger integration.
  """

  def setUp(self):
    """Reset warning flags before each test."""
    super(CallTreesTest, self).setUp()
    call_trees.python_random_warned = False

  def _transform_with_mock(self, f):
    mock = MockConvertedCall()
    tr = self.transform(
        f, (functions, call_trees),
        ag_overrides={'converted_call': mock})
    return tr, mock

  def test_function_no_args(self):
    """Test transformation of function with no arguments.
    
    Verifies that a simple function call without arguments is properly
    wrapped and executed through converted_call.
    """
    def f(f):
      return f() + 20

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda: 1), 21)
    self.assertListEqual(mock.calls, [((), None)])

  def test_function_with_expression_in_argument(self):
    """Test function call with expression evaluation in arguments.
    
    Verifies that nested function calls within arguments are properly
    evaluated and converted, maintaining correct execution order.
    """
    def f(f, g):
      return f(g() + 20) + 4000

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda x: x + 300, lambda: 1), 4321)
    self.assertListEqual(mock.calls, [
        ((), None),
        ((21,), None),
    ])

  def test_function_with_call_in_argument(self):
    """Test function call with another function call as argument.
    
    Verifies that nested function calls are tracked separately in
    converted_call invocations, ensuring proper call ordering.
    """
    def f(f, g):
      return f(g()) + 300

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda x: x + 20, lambda: 1), 321)
    self.assertListEqual(mock.calls, [
        ((), None),
        ((1,), None),
    ])

  def test_function_chaining(self):
    """Test chained method calls on function return values.
    
    Verifies that methods called on the return value of a function
    are properly handled and tracked as separate conversions.
    """
    def get_one():
      return 1

    def f():
      return get_one().__add__(20)

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(), 21)
    self.assertListEqual(mock.calls, [
        ((), None),
        ((20,), None),
    ])

  def test_function_with_single_arg(self):
    """Test function call with a single argument.
    
    Verifies that positional arguments are correctly passed through
    converted_call as a tuple.
    """
    def f(f, a):
      return f(a) + 20

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda a: a, 1), 21)
    self.assertListEqual(mock.calls, [((1,), None)])

  def test_function_with_args_only(self):
    """Test function call with multiple positional arguments.
    
    Verifies that multiple positional arguments are correctly converted
    and passed as a tuple to converted_call.
    """
    def f(f, a, b):
      return f(a, b) + 300

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda a, b: a + b, 1, 20), 321)
    self.assertListEqual(mock.calls, [((1, 20), None)])

  def test_function_with_kwarg(self):
    """Test function call with mixed positional and keyword arguments.
    
    Verifies that keyword arguments are properly separated and passed
    as a dictionary to converted_call.
    """
    def f(f, a, b):
      return f(a, c=b) + 300

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(lambda a, c: a + c, 1, 20), 321)
    self.assertListEqual(mock.calls, [((1,), {'c': 20})])

  def test_function_with_kwargs_starargs(self):
    """Test function call with all argument types (*args and **kwargs).
    
    Verifies that star arguments and star keyword arguments are properly
    unpacked and converted to flat tuples and dictionaries.
    """
    def f(f, a, *args, **kwargs):
      return f(a, *args, **kwargs) + 5

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(
        tr(lambda *args, **kwargs: 7, 1, *[2, 3], **{
            'b': 4,
            'c': 5
        }), 12)
    self.assertListEqual(mock.calls, [((1, 2, 3), {'b': 4, 'c': 5})])

  def test_function_with_starargs_only(self):
    """Test function call with unpacked list as star arguments.
    
    Verifies that list unpacking with *args is correctly flattened
    into the arguments tuple.
    """
    def g(*args):
      return sum(args)

    def f():
      args = [1, 20, 300]
      return g(*args) + 4000

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(), 4321)
    self.assertListEqual(mock.calls, [((1, 20, 300), None)])

  def test_function_with_starargs_mixed(self):
    """Test function call with mixed positional and unpacked arguments.
    
    Verifies that mixing regular positional args with unpacked star args
    is handled correctly, maintaining proper argument order.
    """
    def g(a, b, c, d):
      return a * 1000 + b * 100 + c * 10 + d

    def f():
      args1 = (1,)
      args2 = [3]
      return g(*args1, 2, *args2, 4)

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(tr(), 1234)
    self.assertListEqual(mock.calls, [((1, 2, 3, 4), None)])

  def test_function_with_kwargs_keywords(self):
    """Test function call with unpacked dictionary as kwargs.
    
    Verifies that dictionary unpacking with **kwargs is properly merged
    with explicit keyword arguments in the kwargs dictionary.
    """
    def f(f, a, b, **kwargs):
      return f(a, b=b, **kwargs) + 5

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(
        tr(lambda *args, **kwargs: 7, 1, 2, **{'c': 3}), 12)
    self.assertListEqual(mock.calls, [((1,), {'b': 2, 'c': 3})])

  def test_function_with_multiple_kwargs(self):
    """Test function call with multiple unpacked keyword dictionaries.
    
    Verifies that multiple **kwargs expansions are merged correctly,
    with proper precedence for duplicate keys.
    """
    def f(f, a, b, c, kwargs1, kwargs2):
      return f(a, b=b, **kwargs1, c=c, **kwargs2) + 5

    tr, mock = self._transform_with_mock(f)

    self.assertEqual(
        tr(lambda *args, **kwargs: 7, 1, 2, 3, {'d': 4}, {'e': 5}), 12)
    self.assertListEqual(mock.calls, [((1,), {
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5
    })])

  def test_function_with_call_in_lambda_argument(self):
    """Test lambda function containing calls as argument.
    
    Verifies that lambda functions containing nested function calls
    are properly handled by the converter.
    """
    def h(l, a):
      return l(a) + 4000

    def g(a, *args):
      return a + sum(args)

    def f(h, g, a, *args):
      return h(lambda x: g(x, *args), a)

    tr, _ = self._transform_with_mock(f)

    self.assertEqual(tr(h, g, 1, *(20, 300)), 4321)

  def test_debugger_set_trace(self):
    """Test that debugger functions are not converted.
    
    Verifies that pdb.set_trace() and similar debugger functions
    bypass conversion and execute directly.
    """
    tracking_list = []

    pdb = types.ModuleType('fake_pdb')
    pdb.set_trace = lambda: tracking_list.append(1)

    def f():
      return pdb.set_trace()

    tr, _ = self._transform_with_mock(f)

    tr()
    self.assertListEqual(tracking_list, [1])

  def test_class_method(self):
    """Test transformation of unbound class methods.
    
    Verifies that class methods are properly converted when accessed
    through the class (unbound method form).
    """
    class TestClass(object):

      def other_method(self, x):
        return x + 20

      def test_method(self, a):
        return self.other_method(a) + 300

    tc = TestClass()
    tr, mock = self._transform_with_mock(TestClass.test_method)

    self.assertEqual(321, tr(tc, 1))
    self.assertListEqual(mock.calls, [((1,), None)])

  def test_object_method(self):
    """Test transformation of bound instance methods.
    
    Verifies that instance methods are properly converted when accessed
    through an object instance (bound method form).
    """
    class TestClass(object):

      def other_method(self, x):
        return x + 20

      def test_method(self, a):
        return self.other_method(a) + 300

    tc = TestClass()
    tr, mock = self._transform_with_mock(tc.test_method)

    self.assertEqual(321, tr(tc, 1))
    self.assertListEqual(mock.calls, [((1,), None)])

  def test_python_random_warning(self):
    """Test that using Python random.randint triggers a warning.
    
    Verifies that using Python's random.randint() inside a tf.function
    triggers a warning to guide users toward tf.random alternatives.
    """
    def f():
      return random.randint(1, 10)

    with self.assertLogs(level='WARNING') as logs:
      tr, mock = self._transform_with_mock(f)

    self.assertLen(logs.output, 1)
    self.assertIn(
        "Detected use of Python's `random.randint()` inside a tf.function.",
        logs.output[0],
    )

    # The function should still be callable
    result = tr()
    self.assertIsInstance(result, int)
    self.assertGreaterEqual(result, 1)
    self.assertLessEqual(result, 10)

  def test_python_random_randrange_warning(self):
    """Test that using Python random.randrange triggers a warning.
    
    Verifies that using Python's random.randrange() inside a tf.function
    triggers a warning to guide users toward tf.random alternatives.
    """
    def f():
      return random.randrange(0, 100)

    with self.assertLogs(level='WARNING') as logs:
      tr, mock = self._transform_with_mock(f)

    self.assertLen(logs.output, 1)
    self.assertIn(
        "Detected use of Python's `random.randrange()` inside a tf.function.",
        logs.output[0],
    )

    # The function should still be callable
    result = tr()
    self.assertIsInstance(result, int)
    self.assertGreaterEqual(result, 0)
    self.assertLess(result, 100)

  def test_python_random_choice_warning(self):
    """Test that using Python random.choice triggers a warning.
    
    Verifies that using Python's random.choice() inside a tf.function
    triggers a warning to guide users toward tf.random alternatives.
    """
    def f():
      return random.choice([1, 2, 3])

    with self.assertLogs(level='WARNING') as logs:
      tr, mock = self._transform_with_mock(f)

    self.assertLen(logs.output, 1)
    self.assertIn(
        "Detected use of Python's `random.choice()` inside a tf.function.",
        logs.output[0],
    )

    # The function should still be callable
    result = tr()
    self.assertIn(result, [1, 2, 3])


if __name__ == '__main__':
  test.main()
