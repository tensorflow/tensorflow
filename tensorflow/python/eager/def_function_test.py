# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import weakref

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam


class _ModelWithOptimizer(training.Model):

  def __init__(self):
    super(_ModelWithOptimizer, self).__init__()
    self.dense = core.Dense(1)
    self.optimizer = adam.AdamOptimizer(0.01)

  @def_function.function(
      input_signature=(tensor_spec.TensorSpec([None, 2], dtypes.float32),
                       tensor_spec.TensorSpec([None], dtypes.float32)))
  def call(self, x, y):
    with backprop.GradientTape() as tape:
      loss = math_ops.reduce_mean((self.dense(x) - y) ** 2.)
    trainable_variables = self.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return {'loss': loss}


class _HasDecoratedMethod(object):

  @def_function.function
  def f(self, x):
    return x * 3.


class DefFunctionTest(test.TestCase):

  def testNoVariables(self):

    @def_function.function
    def fn(x):
      return 2 * x

    self.assertAllEqual(fn(constant_op.constant(4.0)), 8.0)

  def testFailIfVariablesAreCreatedMoreThanOnce(self):

    @def_function.function
    def fn(x):
      return variables.Variable(1.0) + x

    with self.assertRaises(ValueError):
      fn(1.0)

  def testFailIfVariablesAreCreatedMoreThanOnceNoWeakRef(self):
    state = []

    @def_function.function
    def fn(x):
      state.append(variables.Variable(1.0))
      return state[-1] + x

    with self.assertRaises(ValueError):
      fn(1.0)

  def testCorrectVariableCreation(self):

    state = []

    @def_function.function
    def fn(x):
      if not state:
        state.append(variables.Variable(2.0))
      return state[0] * x

    self.assertAllEqual(fn(constant_op.constant(1.0)), 2.0)
    self.assertAllEqual(fn(constant_op.constant(3.0)), 6.0)

  def testFunctionInitializer(self):

    state = []

    @def_function.function
    def fn(x):
      if not state:
        state.append(variables.Variable(lambda: 2.0))
      return state[0] * x

    self.assertAllEqual(fn(constant_op.constant(1.0)), 2.0)

  def testFunctionInitializationFunction(self):

    state = []

    @def_function.function
    def fn(x):
      if not state:
        state.append(variables.Variable(2.0))
      return state[0] * x

    init_fn = fn.get_initialization_function(constant_op.constant(1.0))
    self.assertEqual(len(state), 1)
    self.assertFalse(
        resource_variable_ops.var_is_initialized_op(state[0].handle))
    init_fn()
    self.assertEqual(state[0].numpy(), 2.0)

  def testVariableInitializerNotConstant(self):

    state = []

    @def_function.function
    def fn(x):
      if not state:
        state.append(variables.Variable(2.0 * x))
      return state[0] * x

    self.assertAllEqual(fn(constant_op.constant(1.0)), 2.0)
    self.assertAllEqual(fn(constant_op.constant(3.0)), 6.0)

  def testLegacyGraphModeVariables(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      state = []

      @def_function.function
      def fn(x):
        if not state:
          state.append(variables.Variable(2.0))
        return state[0] * x

      result = fn(3.0)

      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(sess.run(state[0]), 2.0)
      self.assertAllEqual(self.evaluate(result), 6.0)

  def testLegacyGraphModeVariablesNonTrivialInitializer(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      state = []

      @def_function.function
      def fn(x):
        if not state:
          two = constant_op.constant(2.0)
          four = two * two
          two_again = math_ops.sqrt(four)
          state.append(variables.Variable(two_again + four))
        return state[0] * x

      result = fn(3.0)

      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(sess.run(state[0]), 6.0)
      self.assertAllEqual(self.evaluate(result), 18.0)

  def testLegacyGraphModeInputDependentInitializerFails(self):
    with ops.Graph().as_default():
      state = []

      @def_function.function
      def fn(x):
        if not state:
          state.append(variables.Variable(2.0 * x))
        return state[0] * x

      with self.assertRaises(ValueError):
        fn(constant_op.constant(3.0))

  def testMethod(self):

    class MyModel(object):

      def __init__(self):
        self.var = None

      @def_function.function
      def apply(self, x):
        if self.var is None:
          self.var = variables.Variable(2.0)
        return self.var * x

    m0 = MyModel()
    self.assertAllEqual(m0.apply(3.0), 6.0)
    # Calling twice to exercise that we do not recreate variables.
    m0.var.assign(3.0)
    self.assertAllEqual(m0.apply(3.0), 9.0)

    m1 = MyModel()
    self.assertAllEqual(m1.apply(3.0), 6.0)

  def test_functools_partial(self):
    self.assertAllClose(
        3.,
        def_function.function(functools.partial(lambda x, y: x + y, 1.))(
            constant_op.constant(2.)))

  def test_unspecified_default_argument(self):
    wrapped = def_function.function(
        lambda x, y=2: x + y,
        input_signature=[tensor_spec.TensorSpec((), dtypes.int32)])
    self.assertEqual(3, wrapped(constant_op.constant(1)).numpy())

  def test_optimizer(self):
    x = constant_op.constant([[3., 4.]])
    y = constant_op.constant([2.])
    model = _ModelWithOptimizer()
    model(x, y)

  def test_concrete_function_from_signature(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.float32)])
    def compute(x):
      return 2. * x

    concrete = compute.get_concrete_function()
    self.assertAllClose(1., concrete(constant_op.constant(0.5)))
    concrete = compute.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.float32))
    self.assertAllClose(4., concrete(constant_op.constant(2.)))
    signature_args, _ = concrete.structured_input_signature
    self.assertEqual(signature_args,
                     (tensor_spec.TensorSpec(None, dtypes.float32),))

  def test_serialization_signature_cache(self):

    @def_function.function
    def f(x, y):
      return x, y

    f(constant_op.constant([[3., 4.]]), constant_op.constant([2.]))
    f(constant_op.constant([[3, 4, 5]]), constant_op.constant([2]))

    signatures_args = set()
    concrete_functions = f._list_all_concrete_functions_for_serialization()
    for concrete_function in concrete_functions:
      args, kwargs = concrete_function.structured_input_signature
      signatures_args.add(args)
      self.assertEqual(dict(), kwargs)

    self.assertEqual(
        signatures_args,
        set(((tensor_spec.TensorSpec([1, 2], dtypes.float32),
              tensor_spec.TensorSpec([1], dtypes.float32)),
             (tensor_spec.TensorSpec([1, 3], dtypes.int32),
              tensor_spec.TensorSpec([1], dtypes.int32)))))

  @test_util.assert_no_garbage_created
  def testFunctionReferenceCycles(self):
    fn = def_function.function(lambda x: 2. * x)
    fn(constant_op.constant(4.0))
    weak_fn = weakref.ref(fn)
    del fn
    # Tests that the weak reference we made to the function is now dead, which
    # means the object has been deleted. This should be true as long as the
    # function itself is not involved in a reference cycle.
    self.assertIs(None, weak_fn())

  @test_util.assert_no_garbage_created
  def testMethodReferenceCycles(self):
    has_decorated_method = _HasDecoratedMethod()
    has_decorated_method.f(constant_op.constant(5.))
    weak_fn = weakref.ref(has_decorated_method.f)
    del has_decorated_method
    # Tests that the weak reference we made to the function is now dead, which
    # means the object has been deleted. This should be true as long as the
    # function itself is not involved in a reference cycle.
    self.assertIs(None, weak_fn())


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
