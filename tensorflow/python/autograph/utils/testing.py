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
"""Testing utilities."""

import re
import sys
import types
import unittest

from tensorflow.python.eager import def_function
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class AutoGraphTestCase(test.TestCase):
  """Tests specialized for AutoGraph, which run as tf.functions.

  These tests use a staged programming-like approach: most of the test code runs
  as-is inside a tf.function, but the assertions are lifted outside the
  function, and run with the corresponding function values instead.

  For example, the test:

      def test_foo(self):
        baz = bar();
        self.assertEqual(baz, value)

  is equivalent to writing:

      def test_foo(self):
        @tf.function
        def test_fn():
          baz = bar();
          return baz, value

        baz_actual, value_actual = test_fn()
        self.assertEqual(baz_actual, value_actual)

  Only assertions that require evaluation outside the function are lifted
  outside the function scope. The rest execute inline, at function creation
  time.
  """

  def __new__(cls, *args):
    obj = super().__new__(cls)

    for name in cls.__dict__:
      if not name.startswith(unittest.TestLoader.testMethodPrefix):
        continue
      m = getattr(obj, name)
      if callable(m):
        wrapper = obj._run_as_tf_function(m)
        setattr(obj, name, types.MethodType(wrapper, obj))

    return obj

  def _op_callback(
      self, op_type, inputs, attrs, outputs, op_name=None, graph=None):
    self.trace_log.append(op_type)

  def _run_as_tf_function(self, fn):

    def wrapper(self):
      @def_function.function(autograph=False)  # Testing autograph itself.
      def fn_wrapper():
        self.assertions = []
        self.raises_cm = None
        self.graph_assertions = []
        self.trace_log = []
        fn()
        targets = [args for _, args in self.assertions]
        return targets

      try:
        tensors = fn_wrapper()

        for assertion in self.graph_assertions:
          assertion(fn_wrapper.get_concrete_function().graph)

        actuals = self.evaluate(tensors)

      except:  # pylint:disable=bare-except
        if self.raises_cm is not None:
          # Note: Yes, the Raises and function contexts cross.
          self.raises_cm.__exit__(*sys.exc_info())
          return
        else:
          raise

      for (assertion, _), values in zip(self.assertions, actuals):
        assertion(*values)

    return wrapper

  def variable(self, name, value, dtype):
    with ops.init_scope():
      if name not in self.variables:
        self.variables[name] = variables.Variable(value, dtype=dtype)
        self.evaluate(self.variables[name].initializer)
    return self.variables[name]

  def setUp(self):
    super().setUp()
    self.variables = {}
    self.trace_log = []
    self.raises_cm = None
    op_callbacks.add_op_callback(self._op_callback)

  def tearDown(self):
    op_callbacks.remove_op_callback(self._op_callback)
    self.trace_log = None
    self.variables = None
    super().tearDown()

  def assertGraphContains(self, op_regex, n):
    def assertion(graph):
      matches = []
      for node in graph.as_graph_def().node:
        if re.match(op_regex, node.name):
          matches.append(node)
      for fn in graph.as_graph_def().library.function:
        for node_def in fn.node_def:
          if re.match(op_regex, node_def.name):
            matches.append(node_def)
      self.assertLen(matches, n)

    self.graph_assertions.append(assertion)

  def assertOpCreated(self, op_type):
    self.assertIn(op_type, self.trace_log)

  def assertOpsNotCreated(self, op_types):
    self.assertEmpty(set(op_types) & set(self.trace_log))

  def assertNoOpsCreated(self):
    self.assertEmpty(self.trace_log)

  def assertEqual(self, *args):
    self.assertions.append((super().assertEqual, list(args)))

  def assertLess(self, *args):
    self.assertions.append((super().assertLess, list(args)))

  def assertGreaterEqual(self, *args):
    self.assertions.append((super().assertGreaterEqual, list(args)))

  def assertDictEqual(self, *args):
    self.assertions.append((super().assertDictEqual, list(args)))

  def assertRaisesRuntime(self, *args):
    if self.raises_cm is not None:
      raise ValueError('cannot use more than one assertRaisesRuntime in a test')
    self.raises_cm = self.assertRaisesRegex(*args)
    self.raises_cm.__enter__()
