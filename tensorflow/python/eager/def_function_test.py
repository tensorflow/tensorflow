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
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
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

# pylint: disable=bad-continuation,anomalous-backslash-in-string
MIXING_GRAPH_EAGER_TENSORS_ERROR = (
"""An op outside of the function building code is being passed
a "Graph" tensor. It is possible to have Graph tensors
leak out of the function building context by including a
tf.init_scope in your function building code.
For example, the following function will fail:
  @tf.function
  def has_init_scope\(\):
    my_constant = tf.constant\(1.\)
    with tf.init_scope\(\):
      added = my_constant \* 2
The graph tensor has name: Const:0""")
# pylint: enable=bad-continuation,anomalous-backslash-in-string


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

      with self.assertRaises(lift_to_graph.UnliftableError):
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
                     (tensor_spec.TensorSpec(
                         None, dtypes.float32, name='x'),))

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
        set(((tensor_spec.TensorSpec([1, 2], dtypes.float32, name='x'),
              tensor_spec.TensorSpec([1], dtypes.float32, name='y')),
             (tensor_spec.TensorSpec([1, 3], dtypes.int32, name='x'),
              tensor_spec.TensorSpec([1], dtypes.int32, name='y')))))

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

  def testErrorMessageWhenGraphTensorIsPassedToEager(self):

    @def_function.function
    def failing_function():
      a = constant_op.constant(1.)

      with ops.init_scope():
        _ = a + a

    with self.assertRaisesRegexp(TypeError, MIXING_GRAPH_EAGER_TENSORS_ERROR):
      failing_function()

  def testVariableCreatorScope(self):
    created_variables = []
    captured_variables = []

    @def_function.function
    def f():
      if not created_variables:
        created_variables.append(variables.Variable(1.))
      return created_variables[0] + 1.

    def capture_creator(next_creator, **kwargs):
      created = next_creator(**kwargs)
      captured_variables.append(created)
      return created

    with variable_scope.variable_creator_scope(capture_creator):
      f()
    self.assertEqual(created_variables, captured_variables)

  def testVarAlreadyInitializedNoClobbering(self):
    v_holder = []

    @def_function.function
    def add_var(x):
      if not v_holder:
        v = variables.Variable([1., 2.])
        v_holder.append(v)
        already_initialized = variables.Variable(3.)
        with ops.init_scope():
          already_initialized.assign(10.)
        v_holder.append(already_initialized)
      return v_holder[0] + v_holder[1] + x

    add_var.get_concrete_function(constant_op.constant(2.))
    self.assertAllClose([13., 14.], add_var(constant_op.constant(2.)))

  def testSameVariableTwice(self):

    v = variables.Variable(1.0)

    @def_function.function
    def add(a, b):
      return a + b

    self.assertAllEqual(add(v, v), 2.0)

  def testShapeCache(self):
    @def_function.function
    def func(x):
      return 2 * x

    func_a = func.get_concrete_function(
        tensor_spec.TensorSpec([None], dtypes.int32))
    func_b = func.get_concrete_function(
        tensor_spec.TensorSpec([None], dtypes.int32))

    self.assertIs(func_a, func_b)

  def testInitializationInNestedCall(self):
    v_holder = []

    @def_function.function
    def add_var(x):
      if not v_holder:
        v = variables.Variable([1., 2.])
        v_holder.append(v)
        already_initialized = variables.Variable(3.)
        with ops.init_scope():
          already_initialized.assign(10.)
        v_holder.append(already_initialized)
      return v_holder[0] + v_holder[1] + x

    @def_function.function
    def wrapper(x):
      return add_var(x)

    self.assertAllClose([13., 14.], wrapper(constant_op.constant(2.)))
    v_holder[1].assign(11.)
    self.assertAllClose([14., 15.], wrapper(constant_op.constant(2.)))

  def testDeviceAnnotationRespected(self):
    if not context.num_gpus():
      self.skipTest("Needs multiple devices")

    a = []

    @def_function.function()
    def create_variable():
      with ops.init_scope():
        initial_value = random_ops.random_uniform(
            (2, 2), maxval=1000000, dtype=dtypes.int64)

      if not a:
        with ops.device("CPU:0"):
          a.append(resource_variable_ops.ResourceVariable(initial_value))

      return a[0].read_value()

    created_variable_read = create_variable()
    self.assertRegexpMatches(created_variable_read.device, "CPU")


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
