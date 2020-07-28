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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types
import unittest

from tensorflow.python.eager import def_function
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

  def _run_as_tf_function(self, fn):

    def wrapper(self):
      @def_function.function(autograph=False)  # Testing autograph itself.
      def fn_wrapper():
        self.assertions = []
        fn()
        targets = [args for _, args in self.assertions]
        return targets
      actuals = self.evaluate(fn_wrapper())
      for (_, args), value in zip(self.assertions, actuals):
        args[:] = value
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

  def tearDown(self):
    for fn, args in self.assertions:
      fn(*args)
    super().tearDown()

  def assertEqual(self, *args):
    self.assertions.append((super().assertEqual, list(args)))
