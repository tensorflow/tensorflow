# Lint as: python2, python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests to improve the consistency of tf.function I/O."""

from absl.testing import parameterized

import tensorflow as tf

from tensorflow.python.platform import test
from tensorflow.tools.consistency_integration_test.consistency_test_base import ConsistencyTestBase


class TfFunctionIOConsistencyTests(ConsistencyTestBase, parameterized.TestCase):
  """Test cases for known issues or bugs related to tf.function I/O."""

  def testDynamicIndirectVariableCreation(self):
    """Tests tf.function that tries to re-create `tf.Variable`s.

    Bugs:   b/147231209
    Status: Known issue
            (In the short term, we should allow `tf.Variable`s to be lifted out
            of each trace, rather than only one per `tf.function`.
            In the long term, we could allow `tf.Variable`s to be created
            arbitrarily (go/tf-mutable-refs).)
    Issue:  Re-creating `tf.Variables` inside tf.function is not allowed and
            the error message thrown is ambiguous (i.e. missing information
            about which variable it causing the failure and where it happened).

    Error message:
      "Creating variables on a non-first call to a function decorated with
      tf.function."

    Improve error message? Needed. (b/187847612)

    Notes:
    * If `tf.Variable` creation is detected in the initial trace, tf.function
      will retrace the function. For example:
      ```
      class Foo:
        def __init__(self):
          self.var = None

        @tf.function
        def __call__(self, x):
          print("#tracing")
          if self.var is None:
            self.var = tf.Variable(x)
          return self.var

      foo = Foo()
      foo(True)  # traced twice instead of once; tracing + variable lifting
                 # '#tracing' prints 2 times.
      foo(True)  # not traced; `#tracing` doesn't get printed.
      foo(False)  # retraced once; '#tracing' prints once since `self.var` is
                  # not None
      ```
      If `tf.Variable` creation is detected in a different trace for the same
      tf.function, it will fail during the retrace's variable lifting stage.
      (This is a simpler example of the test case.)
      ```
      class Baz:
        def __init__(self):
          self.cnt = 0

        @tf.function
        def __call__(self, x):
          print("#tracing")
          if self.cnt == 0:
            self._var = tf.Variable(x)
          elif self.cnt > 1:
            self._var = tf.Variable(x)
          self.cnt += 1

      baz = Baz()
      baz(True)  # traced twice instead of once; tracing + variable lifting
                 # '#tracing' prints 2 times.
      baz(True)  # not traced; no `tf.Variable` creation when `self.cnt == 1`.
                 # `#tracing` doesn't get printed.
      baz(False)  # retraced twice; retracing + variable lifting
                  # '#tracing' prints once since it fails at variable lifting
                  # stage.
      ```
    * The issue is prevalent when working with `tf.metrics.Mean` inside a
      tf.function (b/187445546):
      ```
      class Foo:

        def __init__(self):
          self._metrics = collections.defaultdict(tf.metrics.Mean)

        def __call__(self, is_training):
          self.compute(is_training)

        @tf.function
        def compute(self, is_training):
          if is_training:
            self._metrics['test'].update_state([1., 2.])

      foo = Foo()

      # Calling `foo` here with `False` will trigger tracing; retriggering
      # the tracing with `True` will cause the error.
      foo(False)  # tracing
      foo(True)  # error
      ```
    * Improve error message. It should mention the variable name and which
      function tried to re-create `tf.Variable`s
    * go/tf-mutable-refs is a work-in-progress, longer term project designed to
      address this issue.
    """
    self.skipTest('b/147231209')

    class Foo:
      """Foo class for demonstrating the issue."""

      def __init__(self):
        self._flag_keyed_vars = {}

      def __call__(self, var_creation_flag):
        self.compute(var_creation_flag)

      @tf.function
      def compute(self, var_creation_flag):
        if var_creation_flag not in self._flag_keyed_vars:
          self._flag_keyed_vars[var_creation_flag] = tf.Variable(1.0)

    foo = Foo()
    foo(True)  # traced twice, with variable lifting
    foo(True)  # not traced, reuses variables from first trace
    foo(False)  # re-traced twice, variable lifting raises error; but we don't
                # need to raise, we can just lift like in the first trace.

  @parameterized.named_parameters([('_RunFunctionEagerly', True),
                                   ('_RunFunctionNonEagerly', False)])
  def testVariableCreationCustomModule(self, run_eagerly):
    """Tests tf.function variable creation with custom objects (`tf.Module`).

    Bugs:   b/184210116
    Status: Working as intended
            (However, moving forward, we should support re-creating
            `tf.Variables` inside tf.function for each trace. This test case
            should pass eventually.)
    Issue:  `tf.Variable` creation in a custom module causes 'non-first call
            variable creation' error in a tf.function.

    Error message:
      "tf.function-decorated function tried to create variables on non-first
      call."

    Notes:
    * This is a simplified version of `testVariableCreationKerasLayers` test in
      //tensorflow/tools/consistency_integration_test/keras_integration_tests.py
      without involving Keras.
    * Inconsistent behavior between eager and non-eager mode execution of the
      tf.function.
    * In non-eager mode (graph mode), double tracing (i.e. first one during
      function tracing and second one in execution) causes variable creation in
      non-first call error.
    * go/tf-mutable-refs is a work-in-progress, longer term project designed to
      address this issue.

    Args:
      run_eagerly: Boolean deciding whether to run tf.function decorated
        functions eagerly or not.
    """
    self.skipTest('b/184210116')

    try:
      original_setting = tf.config.functions_run_eagerly()
      tf.config.run_functions_eagerly(run_eagerly)

      class Dense(tf.Module):
        """Custom Dense class for demonstration."""

        def __init__(self, in_features, out_features):
          super().__init__()
          self.w = tf.Variable(tf.random.normal([in_features, out_features]))
          self.b = tf.Variable(tf.zeros([out_features]))

        def __call__(self, x):
          y = tf.matmul(x, self.w) + self.b
          return tf.nn.relu(y)

      @tf.function
      def f(x):
        layer = Dense(3, 3)(x)
        return layer

      in_val = tf.constant([[1., 2., 3]])

      if run_eagerly:
        self.assertAllEqual(
            tf.constant([[0., 2.037801, 0.]], dtype=tf.float32), f(in_val))
      else:
        f(in_val)

    finally:
      tf.config.run_functions_eagerly(original_setting)

  def testRetraceOnObjectPropertyChange(self):
    """Tests retracing behavior of tf.function when object property has changed.

    Bugs:   b/162221622
    Status: Broken
            (When the property of an object has changed, tf.function should
            detect the update and retrace.)
    Issue:  Changing the property of an object does not trigger retracing and
            outputs wrong results.

    Error message:
      There isn't an error message thrown out; things work but wrongly because
      the correct conditional branch didn't get traced initially and because
      retracing doesn't take place.
    """
    self.skipTest('b/162221622')
    trace = []

    class Foo:
      """Foo class for demonstration."""

      def __init__(self):
        self.condition = True
        self.n = 1.0

      @tf.function
      def f(self, x):
        """Function `f` for demonstration."""
        nonlocal trace
        trace.append('#tracing')

        if not self.condition:
          trace.append('#retracing')
          self.n = x

        return self.n

    foo = Foo()
    a = 2.0

    out0 = foo.f(a)
    self.assertEqual(out0, tf.constant(1.))
    self.assertEqual(trace, ['#tracing'])

    trace = []
    foo.condition = False

    out1 = foo.f(a)
    # `out1` is 1.0 and `trace` is `[]` because tf.function did not retrace
    # despite that `foo`'s property has changed.
    self.assertEqual(out1, tf.constant(2.))
    self.assertEqual(trace, ['#tracing', '#retracing'])

  def testRetraceOnObjectPropertyChangeOneWorkaround(self):
    """Tests a possible workaround for handling changes in object property.

    Bugs:   b/162221622
    Status: Broken
            (The workaround demonstrated in this test case, however, works.
            The eventual goal though should be to improve the behavior by
            allowing retracing upon object property changes.)
    Issue:  n/a

    Error message: n/a

    Notes:
    * This is a workaround for issue demonstrated in
      `testRetraceOnObjectPropertyChange` test case. We are explicitly
      passing in the conditional variable in order to trigger retracing.
    """
    trace = []

    class Foo:
      """Foo class for demonstration."""

      def __init__(self):
        self.condition = True
        self.n = 1.0
        self.var = None

      @tf.function
      def f(self, x, condition):
        """Function `f` for demonstration."""
        nonlocal trace
        trace.append('#tracing')

        self.condition = condition

        if self.var is None:
          self.var = tf.Variable(x)

        if not self.condition:
          trace.append('#retracing')
          self.n = 5.0

        return self.var.assign_add(self.n)

    foo = Foo()
    a = 2.0

    out0 = foo.f(a, True)
    self.assertEqual(out0, tf.constant(3.))
    self.assertEqual(trace, ['#tracing', '#tracing'])

    trace = []

    out1 = foo.f(a, False)
    self.assertEqual(out1, tf.constant(8.))
    self.assertEqual(trace, ['#tracing', '#retracing'])

  def testDataResourcesIO(self):
    """Tests returning iterators from tf.function.

    Bugs:   b/170436338, b/170497789 (feature request)
    Status: Broken
    Issue:  Unable to return iterators from tf.function.

    Error message:
      "InvalidArgumentError: 6 nodes in a cycle [Op:__inference_f_11]"

    Improve error message? Needed. (b/187850865)

    Notes:
    * Current error message is not helpful; we need to improve it to explain
      what is causing the error where and suggest the known workaround.
    * One workaround is to keep the iterator as a global variable:
        ```
        its = []

        class Model(tf.Module):

          @tf.function
          def train(self):
            global its
            it = iter(tf.data.Dataset.from_tensors([0.0]).repeat())
            its.append(it)
            return it

        model = Model()
        model.train()
        ```
    * Another workaround is to create it upon `Model` initialization.
        ```
        class Model(tf.Module):

          def __init__(self):
            self.traced = False
            self.dataset = tf.data.Dataset.from_tensor_slices([1., 2.])
            self.iterator = iter(self.dataset)

          def create_variables(self):
            self.w = tf.Variable(0.0)

          @tf.function
          def train(self):
            if not self.traced:
              self.traced = True
              self.create_variables()
            return next(self.iterator)

        model = Model()
        model.train()
        ```
    """
    self.skipTest('b/170436338')

    class Model(tf.Module):
      """Model class for demonstrating the issue."""

      @tf.function
      def f(self):
        dataset = iter(tf.data.Dataset.from_tensors([0.0]).repeat())
        iterator = iter(dataset)
        return iterator

    m = Model()
    it0 = m.f()
    it1 = iter(tf.data.Dataset.from_tensors([0.0]).repeat())
    self.assertEqual(type(it0), type(it1))

  def testCachedTensor(self):
    """Tests tf.function behavior with cached tensors (side I/O).

    Bugs:   b/149094965
    Status: Working as intended
    Issue:  When there exists a trace that has cached tensors, retracing the
            function (upon receiving new input signature) will result in an
            error as the cached tensor is from the previous trace.

    Error message:
      "tf.Graph captured an external symbolic tensor."

    Improve error message? Needed. (b/187850615)

    Notes:
    * `self._cached_value` is already a cached tensor when the program tries to
      retrace upon receiving `tf.constant([1, 2])` as input.
    * The error message is returned during graph execution. Try to detect
      illegal capture and raise an exception during tracing.
    """
    self.skipTest('b/149094965')

    class Context(object):
      """Context class for demonstrating the issue."""

      def __init__(self):
        self._cached_value = None

      def f(self, x):
        result = x + 1
        if self._cached_value is not None:
          result += self._cached_value

        self._cached_value = x
        return result

    @tf.function
    def some_func(ctx, x):
      return ctx.f(x + 1)

    ctx = Context()
    some_func(ctx, tf.constant(1))
    some_func(ctx, tf.constant(2))
    self.assertAllEqual(
        some_func(ctx, tf.constant([1, 2])), tf.constant([6, 7]))


if __name__ == '__main__':
  test.main()
