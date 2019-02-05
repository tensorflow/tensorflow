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

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import auto_control_deps as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import momentum


class AutomaticControlDependenciesTest(test.TestCase):

  def testBasic(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies() as c:
        v.assign(v + 1)
        v.assign(2 * v)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(), 4.0)

  @test_util.run_v1_only("b/120545219")
  def testCondMustRun(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          v.assign(v + 1)
          return 0.0

        def false_fn():
          v.assign(v + 4)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 5.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 6.0)

  @test_util.run_v1_only("b/120545219")
  def testCondMustRunSeparateRead(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          v.assign(v + 1)
          return 0.0

        def false_fn():
          v.assign(v + 4)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        one = constant_op.constant(1.0)
        one = c.mark_as_return(one)
      one.eval(feed_dict={p: False})
      self.assertAllEqual(v.read_value().eval(), 5.0)
      one.eval(feed_dict={p: True})
      self.assertAllEqual(v.read_value().eval(), 6.0)

  @test_util.run_v1_only("b/120545219")
  def testCondNested(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      q = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          v.assign(v + 1, name='true')
          return 1.0

        def false_fn():

          def inner_true_fn():
            v.assign(v * 2, name='false_true')
            return 2.0

          def inner_false_fn():
            v.assign(v * 3, name='false_false')
            return 3.0

          control_flow_ops.cond(q, inner_true_fn, inner_false_fn)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        with ops.name_scope('final'):
          val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False, q: False}), 3.0)
      self.assertAllEqual(val.eval(feed_dict={p: False, q: True}), 6.0)
      self.assertAllEqual(val.eval(feed_dict={p: True, q: True}), 7.0)
      self.assertAllEqual(val.eval(feed_dict={p: True, q: False}), 8.0)

  @test_util.run_v1_only("b/120545219")
  def testCondOneBranch(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          return 0.0

        def false_fn():
          v.assign(v + 4)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 5.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 5.0)

  @test_util.run_v1_only("b/120545219")
  def testCondOneBranchUpdateBefore(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:
        v.assign(v * 2)

        def true_fn():
          return 0.0

        def false_fn():
          v.assign(v + 4)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 6.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 12.0)

  @test_util.run_v1_only("b/120545219")
  def testCondOneBranchUpdateAfter(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          return 0.0

        def false_fn():
          v.assign(v + 4)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        v.assign(v * 2)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val.eval(feed_dict={p: False}), 10.0)
      self.assertAllEqual(val.eval(feed_dict={p: True}), 20.0)

  def testDefunWhileLoopWithCapturedLoopVars(self):
    n = 3
    x = constant_op.constant(list(range(n)))

    @function.defun
    def loop():
      c = lambda i, x: i < n
      b = lambda i, x: (i + 1, x + 1)
      i, out = control_flow_ops.while_loop(c, b, (0, x))
      return i, out

    i, out = loop()
    self.assertEqual(int(i), 3)
    self.assertAllEqual(out, [3, 4, 5])

  def testDecorator(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())

      @acd.automatic_control_dependencies
      def f():
        v.assign(v + 1)
        v.assign(2 * v)
        return v.read_value()

      self.assertAllEqual(f().eval(), 4.0)

  def testOptimizerInDefun(self):
    def loss(v):
      return v**2

    optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)

    @function.defun
    def train():
      self.v = resource_variable_ops.ResourceVariable(1.0)
      grad = backprop.implicit_grad(loss)(self.v)
      optimizer.apply_gradients(grad)
      return self.v.read_value()

    value = train()
    self.assertEqual(value.numpy(), -1.0)

  def testReturningNonTensorRaisesError(self):
    optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)
    optimizer.apply_gradients = function.defun(optimizer.apply_gradients)
    v = resource_variable_ops.ResourceVariable(1.0)
    grad = backprop.implicit_grad(lambda v: v**2)(v)

    with self.assertRaisesRegexp(TypeError,
                                 '.*must return zero or more Tensors.*'):
      # TODO(akshayka): We might want to allow defun-ing Python functions
      # that return operations (and just execute the op instead of running it).
      optimizer.apply_gradients(grad)

  # TODO(b/111663004): This should work when the outer context is graph
  # building.
  def testOptimizerNonSlotVarsInDefunNoError(self):
    def loss(v):
      return v**2

    optimizer = adam.AdamOptimizer(learning_rate=1.0)

    @function.defun
    def train():
      self.v = resource_variable_ops.ResourceVariable(1.0)
      grad = backprop.implicit_grad(loss)(self.v)
      optimizer.apply_gradients(grad)
      return self.v.read_value()

    train()

  def testOptimizerInDefunWithCapturedVariable(self):
    v = resource_variable_ops.ResourceVariable(1.0)
    def loss():
      return v**2

    optimizer = momentum.MomentumOptimizer(learning_rate=1.0, momentum=1.0)

    @function.defun
    def train():
      grad = backprop.implicit_grad(loss)()
      optimizer.apply_gradients(grad)

    train()
    self.assertEqual(v.numpy(), -1.0)

  def testRepeatedResourceInput(self):
    var = resource_variable_ops.ResourceVariable(1.0)

    @def_function.function
    def inner(var1, var2):
      return (resource_variable_ops.read_variable_op(var1, dtypes.float32) +
              resource_variable_ops.read_variable_op(var2, dtypes.float32))

    @def_function.function
    def outer():
      return inner(var.handle, var.handle)

    self.assertEqual(self.evaluate(outer()), 2.0)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
