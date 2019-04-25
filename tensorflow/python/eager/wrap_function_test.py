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
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class WrapFunctionTest(test.TestCase):

  def testDocString(self):

    def f(x, do_add):
      v = variables.Variable(5.0)
      if do_add:
        op = v.assign_add(x)
      else:
        op = v.assign_sub(x)
      with ops.control_dependencies([op]):
        return v.read_value()

    f_add = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32), True])

    self.assertAllEqual(f_add(1.0), 6.0)
    self.assertAllEqual(f_add(1.0), 7.0)

    # Can call tf.compat.v1.wrap_function again to get a new trace, a new set
    # of variables, and possibly different non-template arguments.
    f_sub = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32), False])

    self.assertAllEqual(f_sub(1.0), 4.0)
    self.assertAllEqual(f_sub(1.0), 3.0)

  def testPrune(self):

    x_in = []
    x_out = []

    def f(x, y):
      x_in.append(x)
      xx = x * x
      x_out.append(xx)
      return xx, 2 * y*y

    f_wrapped = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32)] * 2)

    f_pruned = f_wrapped.prune(x_in[0], [x_out[0]])
    self.assertAllEqual(f_pruned(ops.convert_to_tensor(2.0)), [4.0])

  def testNoArguments(self):

    def f():
      return constant_op.constant(1.)

    f_wrapped = wrap_function.wrap_function(f, [])
    self.assertAllEqual(1.0, f_wrapped())

  def testPruneCaptures(self):

    v1 = variables.Variable(2.)

    def f():
      v2 = variables.Variable(3.)
      return array_ops.identity(v1 * v2 * constant_op.constant(1.), 'fetch')

    f_wrapped = wrap_function.wrap_function(f, [])
    self.assertAllEqual(6.0, f_wrapped())

    # Test pruning directly on the inputs
    pruned = f_wrapped.prune(
        feeds=f_wrapped.inputs,
        fetches=f_wrapped.graph.get_tensor_by_name('fetch:0'))
    self.assertAllEqual(6.0, pruned())

    # Test pruning with no inputs
    pruned = f_wrapped.prune(
        feeds=(),
        fetches=f_wrapped.graph.get_tensor_by_name('fetch:0'))
    self.assertAllEqual(6.0, pruned())

  def testCollectionsIsolation(self):

    v1 = variables.Variable(2.)
    v2_holder = []
    def f():
      v2 = variables.Variable(3.)
      v2_holder.append(v2)
      ops.add_to_collection(ops.GraphKeys.LOSSES, v2 * constant_op.constant(3.))
      return array_ops.identity(v1 * v2 * constant_op.constant(1.), 'fetch')

    f_wrapped = wrap_function.wrap_function(f, [])
    self.assertAllEqual(6.0, f_wrapped())
    self.assertEqual(
        len(f_wrapped.graph.get_collection(ops.GraphKeys.LOSSES)), 1)
    f_var_collection = f_wrapped.graph.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertEqual(len(f_var_collection), 1)
    self.assertIs(f_var_collection[0], v2_holder[0])

    v3_holder = []
    def g():
      v3 = variables.Variable(4.)
      v3_holder.append(v3)
      ops.add_to_collection(ops.GraphKeys.LOSSES, v3 * constant_op.constant(3.))
      return array_ops.identity(v1 * v3 * constant_op.constant(1.), 'fetch')

    g_wrapped = wrap_function.wrap_function(g, [])
    self.assertAllEqual(8.0, g_wrapped())
    self.assertEqual(
        len(g_wrapped.graph.get_collection(ops.GraphKeys.LOSSES)), 1)
    g_var_collection = g_wrapped.graph.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertEqual(len(g_var_collection), 1)
    self.assertIs(g_var_collection[0], v3_holder[0])

    # Both have only one value, and their values aren't equal. So no sharing.
    self.assertNotEqual(g_wrapped.graph.get_collection(ops.GraphKeys.LOSSES),
                        f_wrapped.graph.get_collection(ops.GraphKeys.LOSSES))

  def testGradientsOfPrune(self):

    v1 = variables.Variable(2.)
    v2_holder = []

    def f(z):
      v2 = variables.Variable(3.)
      v2_holder.append(v2)
      return array_ops.identity(v1 * v2 * z, 'fetch')

    f_wrapped = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtype=dtypes.float32)])

    x = constant_op.constant(1.)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      out = f_wrapped(x)
    grads = tape.gradient(out, [x, v1, v2_holder[0]])

    self.assertAllEqual(6.0, out)
    self.assertAllEqual([6.0, 3.0, 2.0], grads)

    pruned = f_wrapped.prune(
        feeds=f_wrapped.inputs,
        fetches=f_wrapped.graph.get_tensor_by_name('fetch:0'))

    x = constant_op.constant(1.)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      out = pruned(x)
    grads = tape.gradient(out, [x, v1, v2_holder[0]])

    self.assertAllEqual(6.0, out)
    self.assertAllEqual([6.0, 3.0, 2.0], grads)

  def testPruneOperations(self):

    v = variables.Variable(0)

    def f():
      v.assign_add(1, name='increment', read_value=False)

    f_wrapped = wrap_function.wrap_function(f, [])
    pruned = f_wrapped.prune(
        feeds=(),
        fetches=(f_wrapped.graph.get_operation_by_name('increment'),))
    self.assertEqual((None,), pruned())
    self.assertEqual(1, self.evaluate(v))

    del f, f_wrapped

    def f1():
      v.assign_add(
          array_ops.placeholder(shape=[], dtype=dtypes.int32, name='step'),
          name='increment', read_value=False)
      return constant_op.constant(1, name='other')

    f_wrapped = wrap_function.wrap_function(f1, [])
    increments = f_wrapped.prune(
        feeds=(f_wrapped.graph.get_tensor_by_name('step:0')),
        fetches=(f_wrapped.graph.get_operation_by_name('increment'),
                 f_wrapped.graph.get_tensor_by_name('other:0')))
    first_output, second_output = increments(constant_op.constant(2))
    self.assertEqual(['step:0', 'increment/resource:0'],
                     [t.name for t in increments.inputs])
    self.assertIs(None, first_output)
    self.assertEqual(1, second_output.numpy())
    self.assertEqual(3, v.numpy())
    does_not_increment = f_wrapped.prune(
        feeds=(f_wrapped.graph.get_tensor_by_name('step:0')),
        fetches=f_wrapped.graph.get_tensor_by_name('other:0'))
    self.assertEqual(1, does_not_increment(constant_op.constant(3)).numpy())
    self.assertEqual(3, v.numpy())

  def testPruneStatefulOpsFromWrappedFunc(self):

    v0 = variables.Variable(0)
    v1 = variables.Variable(0)

    # When we wrap a function, we expect it to be executed with 'tf.Graph`
    # rules: it's allowed to prune all ops that are not in transitive fanin of
    # the fetches.
    def f(x):
      v0.assign_add(1, name='increment_v0')
      v1.assign_add(1, name='increment_v1')
      return x

    f_wrapped = wrap_function.wrap_function(f, [1])

    self.assertEqual(1, f_wrapped().numpy())
    self.assertEqual(0, v0.numpy())
    self.assertEqual(0, v1.numpy())

    f_wrapped_with_name = wrap_function.wrap_function(f, [2], name='func')

    self.assertEqual(2, f_wrapped_with_name().numpy())
    self.assertEqual(0, v0.numpy())
    self.assertEqual(0, v1.numpy())

  def test_operation_returned(self):

    v = variables.Variable(0)

    def f():
      v.assign(1, read_value=False, name='assign_to_v')

    f_wrapped = wrap_function.wrap_function(f, [])
    operation_to_fetch = f_wrapped.graph.get_operation_by_name('assign_to_v')
    f_pruned = f_wrapped.prune(
        [], operation_to_fetch)
    self.assertEqual(
        ['assign_to_v'],
        [operation.name for operation in f_pruned.graph.control_outputs])
    self.assertEqual(0, v.numpy())
    f_pruned()
    self.assertEqual(1, v.numpy())

  def test_function_from_graph_def(self):
    @def_function.function
    def make_graph_def(x):
      return x + 1.

    original_func_graph = make_graph_def.get_concrete_function(
        tensor_spec.TensorSpec([None, 2], dtypes.float32)).graph
    graph_def = original_func_graph.as_graph_def()
    revived_function = wrap_function.function_from_graph_def(
        graph_def, inputs=original_func_graph.inputs[0].name,
        outputs=original_func_graph.outputs[0].name)
    self.assertEqual(2., revived_function(constant_op.constant(1.)).numpy())

  def test_create_variables_with_same_name(self):
    def f():
      v1 = variables.Variable(0, name='v')
      v2 = variables.Variable(1, name='v')
      return v1, v2

    f_wrapped = wrap_function.wrap_function(f, [])
    self.assertDictEqual(
        {'v:0': 0, 'v_1:0': 1},  # assert that variable names are uniquified
        {v.name: v.numpy()
         for v in f_wrapped._variable_holder.variables.values()})

    # Uniquification should reset in separate calls to wrap_function.
    def f2():
      v1 = variables.Variable(3, name='v')
      v2 = variables.Variable(4, name='v')
      return v1, v2

    f_wrapped_2 = wrap_function.wrap_function(f2, [])
    self.assertDictEqual(
        {'v:0': 3, 'v_1:0': 4},
        {v.name: v.numpy()
         for v in f_wrapped_2._variable_holder.variables.values()})


class WrappedGraphTest(test.TestCase):

  def testAddFunction(self):

    def fn(x):
      v = variables.Variable(3, name='v')
      v2 = variable_scope.get_variable(
          'v', initializer=init_ops.Constant(4), shape=[], dtype=dtypes.int32)
      return v + v2 + x

    with self.cached_session() as sess:
      result = fn(constant_op.constant(5))
      sess.run(variables.global_variables_initializer())
      expected = sess.run(result)

    g = wrap_function.WrappedGraph()
    signature = [tensor_spec.TensorSpec([], dtypes.int32)]
    wrapped_fn = g.wrap_function(fn, signature)
    self.assertEqual(expected, wrapped_fn(constant_op.constant(5)).numpy())

  def testCollections(self):

    def fn(x):
      v = variables.VariableV1(3, name='v', trainable=False, collections=['a'])
      v2 = variable_scope.get_variable(
          'v', initializer=init_ops.Constant(4), shape=[], dtype=dtypes.int32,
          collections=['a', 'b'])
      return v + v2 + x

    def assert_collections(graph):
      self.assertLen(graph.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES), 1)
      self.assertLen(graph.get_collection('a'), 2)
      self.assertLen(graph.get_collection('b'), 1)

    g = wrap_function.WrappedGraph()
    g.wrap_function(fn, [tensor_spec.TensorSpec([], dtypes.int32)])
    assert_collections(g.graph)

    def assert_fn():
      assert_collections(ops.get_default_graph())
      return 1  # Return is required

    # Assert that collections are accessible within a wrapped function.
    g.wrap_function(assert_fn, [])

  def testShareVariablesSameGraph(self):

    def add_v1(x):
      with variable_scope.variable_scope(
          'reuse', reuse=variable_scope.AUTO_REUSE):
        v = variable_scope.get_variable(
            'v', initializer=init_ops.Constant(3), shape=[], dtype=dtypes.int32)
      return v + x

    def subtract_v1(x):
      with variable_scope.variable_scope(
          'reuse', reuse=variable_scope.AUTO_REUSE):
        v = variable_scope.get_variable(
            'v', initializer=init_ops.Constant(4), shape=[], dtype=dtypes.int32)
      return v - x

    def different_variable_fn_v1(x):
      with variable_scope.variable_scope(
          'no_reuse', reuse=variable_scope.AUTO_REUSE):
        v = variable_scope.get_variable(
            'v', initializer=init_ops.Constant(5), shape=[], dtype=dtypes.int32)
      return v * x

    def increment_variable_v1(x):
      with variable_scope.variable_scope(
          'reuse', reuse=variable_scope.AUTO_REUSE):
        v = variable_scope.get_variable(
            'v', initializer=init_ops.Constant(6), shape=[], dtype=dtypes.int32)
      return v.assign_add(x)

    g = wrap_function.WrappedGraph()
    signature = [tensor_spec.TensorSpec([], dtypes.int32)]
    add = g.wrap_function(add_v1, signature)
    subtract = g.wrap_function(subtract_v1, signature)
    different_variable_fn = g.wrap_function(different_variable_fn_v1, signature)
    increment_variable = g.wrap_function(increment_variable_v1, signature)

    self.assertEqual(10, add(constant_op.constant(7)).numpy())
    self.assertEqual(35, different_variable_fn(constant_op.constant(7)).numpy())

    # The shared variable has a starting value of 3 because add_v1 was wrapped
    # first.
    self.assertEqual(-4, subtract(constant_op.constant(7)).numpy())
    self.assertEqual(10, increment_variable(constant_op.constant(7)).numpy())

    # Check that variable updates
    self.assertEqual(17, add(constant_op.constant(7)).numpy())
    self.assertEqual(3, subtract(constant_op.constant(7)).numpy())

    # Sanity check - result from this function shouldn't change.
    self.assertEqual(35, different_variable_fn(constant_op.constant(7)).numpy())

    self.assertAllEqual({'reuse/v', 'no_reuse/v'}, set(g.variables.keys()))

  def testShareVariablesDifferentGraphs(self):

    def add_v1(x):
      v = variables.Variable(3, name='v')
      return v + x

    def subtract_v1(x):
      v = variables.Variable(4, name='v')
      return v - x

    def different_variable_fn_v1(x):
      with ops.name_scope('different_scope'):
        v = variables.Variable(5, name='v')
      return v * x

    def increment_variable_v1(x):
      v = variables.Variable(6, name='v')
      return v.assign_add(x)

    signature = [tensor_spec.TensorSpec([], dtypes.int32)]
    vh = wrap_function.VariableHolder(share_variables=True)
    new_graph = lambda: wrap_function.WrappedGraph(variable_holder=vh)

    add = new_graph().wrap_function(add_v1, signature)
    subtract = new_graph().wrap_function(subtract_v1, signature)
    different_variable_fn = new_graph().wrap_function(
        different_variable_fn_v1, signature)
    increment_variable = new_graph().wrap_function(
        increment_variable_v1, signature)

    self.assertEqual(10, add(constant_op.constant(7)).numpy())
    self.assertEqual(35, different_variable_fn(constant_op.constant(7)).numpy())

    # Because the variable in add_v1 was created first, its starting value is 3
    # instead of the values defined in subtract_v1 or increment_variable_v1.
    self.assertEqual(-4, subtract(constant_op.constant(7)).numpy())
    self.assertEqual(10, increment_variable(constant_op.constant(7)).numpy())

    # Check that variable updates
    self.assertEqual(17, add(constant_op.constant(7)).numpy())
    self.assertEqual(3, subtract(constant_op.constant(7)).numpy())

    # Sanity check - result from this function shouldn't change.
    self.assertEqual(35, different_variable_fn(constant_op.constant(7)).numpy())

    self.assertAllEqual({'v', 'different_scope/v'}, set(vh.variables.keys()))

  def testReturnOp(self):

    def update_var_v1(x):
      v = variables.Variable(3, name='v')
      update_op = state_ops.assign(v, x).op
      return update_op

    g = wrap_function.WrappedGraph()
    signature = [tensor_spec.TensorSpec([], dtypes.int32)]
    update_var = g.wrap_function(update_var_v1, signature)

    self.assertEqual(g.variables['v'].numpy(), 3)
    update_var(constant_op.constant(12))
    self.assertEqual(g.variables['v'].numpy(), 12)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
