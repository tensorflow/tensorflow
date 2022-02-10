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

import itertools

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
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_sendrecv_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import momentum


class AutomaticControlDependenciesTest(test.TestCase):

  def setUp(self):
    super().setUp()
    self.must_run_order_insensitive_stateful_ops = (
        acd.MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS)

  def tearDown(self):
    acd.MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS = (
        self.must_run_order_insensitive_stateful_ops)
    super().tearDown()

  def testBasic(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies() as c:
        v.assign(v + 1)
        v.assign(2 * v)
        val = v.read_value()
        val = c.mark_as_return(val)
      self.assertAllEqual(val, 4.0)

  def testUnorderedOpsRunInParallel(self):
    acd.MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS |= frozenset(("EagerPyFunc",))

    side_effects = []

    def side_effect_one(x):
      side_effects.append(1)
      return x

    def side_effect_two(x):
      side_effects.append(2)
      return x

    @def_function.function
    def f():
      script_ops.eager_py_func(side_effect_one, [1], [dtypes.int32])
      script_ops.eager_py_func(side_effect_two, [1], [dtypes.int32])
      return 1

    side_effects = []
    self.evaluate(f())

    self.assertSetEqual(set(side_effects), set((1, 2)))

  def testIndependentOpsRunInParallel(self):
    v = resource_variable_ops.ResourceVariable(1)
    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def f():
      gen_resource_variable_ops.assign_variable_op(v.handle, 1)
      ops.get_default_graph().experimental_acd_manager.run_independently(
          gen_resource_variable_ops.assign_variable_op(v.handle, 2))

    # A function with two identical ops, should cause a data race in most
    # conditions.
    var_values = set()
    for _ in range(1000):
      self.evaluate(f())
      var_values.add(
          self.evaluate(
              resource_variable_ops.read_variable_op(v.handle, dtypes.int32)))
    # With regular control dependencies, the function should always run the
    # first assign first, and the value 1 should never be seen.
    self.assertSetEqual(var_values, set((1, 2)))

  def testIndependentOpsInLoop(self):
    v = resource_variable_ops.ResourceVariable(0)
    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def f():
      for i in math_ops.range(3):
        ops.get_default_graph().experimental_acd_manager.run_independently(
            gen_resource_variable_ops.assign_variable_op(v.handle, i))

    self.evaluate(f())
    # TODO(mdan): Find a more robust way to test in loops.
    self.assertEqual(
        self.evaluate(
            resource_variable_ops.read_variable_op(v.handle, dtypes.int32)), 2)

  def testNoControlDepsBetweenVariableReads(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies():
        read_op1 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op2 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
      self.assertNotIn(read_op1, read_op2.control_inputs)
      self.assertNotIn(read_op2, read_op1.control_inputs)

  def testVariableReadThenWrite(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies():
        read_op1 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op2 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        assign_op = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
      # Writes should have control deps from "all" reads since last write
      # or start of the code block.
      self.assertIn(read_op1, assign_op.control_inputs)
      self.assertIn(read_op2, assign_op.control_inputs)
      # There should be no control deps between reads.
      self.assertNotIn(read_op1, read_op2.control_inputs)
      self.assertNotIn(read_op2, read_op1.control_inputs)

  def testVariableWriteThenRead(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies():
        assign_op = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        read_op1 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op2 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
      # Reads should have a control dep from the last write.
      self.assertIn(assign_op, read_op1.control_inputs)
      self.assertIn(assign_op, read_op2.control_inputs)
      # There should be no control deps between reads.
      self.assertNotIn(read_op1, read_op2.control_inputs)
      self.assertNotIn(read_op2, read_op1.control_inputs)

  def testIdentityPassThrough(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies():
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        identity_handle = gen_array_ops.identity(v.handle)
        assign_op2 = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        read_op = gen_resource_variable_ops.read_variable_op(
            identity_handle, v.dtype).op
      # Read should have a control dep from second last write even
      # with Identity applied to resource.
      self.assertIn(assign_op2, read_op.control_inputs)

  def testVariableReadsInOpsWithMustRun(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies() as c:
        read_op = gen_resource_variable_ops.read_variable_op(v.handle,
                                                             v.dtype).op
        # Read ops get added to control outputs only if they have consumers.
        c.mark_as_return(read_op.outputs[0])
      self.assertIn(read_op, c.ops_which_must_run)

  def testVariableMultipleReadsAndWrites(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies() as c:
        # 2 reads -> 2 writes -> 2 reads -> 2 writes.
        read_op1 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op2 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        assign_op1 = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        assign_op2 = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        read_op3 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        read_op4 = gen_resource_variable_ops.read_variable_op(
            v.handle, v.dtype).op
        assign_op3 = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        assign_op4 = gen_resource_variable_ops.assign_variable_op(
            v.handle, v + 1)
        # Read ops get added to control outputs only if they have consumers.
        c.mark_as_return(read_op1.outputs[0])
        c.mark_as_return(read_op2.outputs[0])
        c.mark_as_return(read_op3.outputs[0])
        c.mark_as_return(read_op4.outputs[0])

      # Verify the control edges.
      self.assertIn(read_op1, assign_op1.control_inputs)
      self.assertIn(read_op2, assign_op1.control_inputs)
      self.assertIn(assign_op1, assign_op2.control_inputs)
      self.assertIn(assign_op2, read_op3.control_inputs)
      self.assertIn(assign_op2, read_op4.control_inputs)
      self.assertIn(read_op3, assign_op3.control_inputs)
      self.assertIn(read_op4, assign_op3.control_inputs)
      self.assertIn(assign_op3, assign_op4.control_inputs)

      # There should be no control deps between reads.
      read_ops = [read_op1, read_op2, read_op3, read_op4]
      for src_op, tgt_op in itertools.product(read_ops, read_ops):
        self.assertNotIn(src_op, tgt_op.control_inputs)

      # Reads must be in `ops_which_must_run`.
      self.assertIn(read_op1, c.ops_which_must_run)
      self.assertIn(read_op2, c.ops_which_must_run)
      self.assertIn(read_op3, c.ops_which_must_run)
      self.assertIn(read_op4, c.ops_which_must_run)
      # Last write must be in `ops_which_must_run`.
      self.assertIn(assign_op4, c.ops_which_must_run)

  def testSendInOpsWithMustRun(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      with acd.AutomaticControlDependencies() as c:
        send_op = gen_sendrecv_ops.send(v, "x", "/", 0, "/")

      # Send must be in `ops_which_must_run`.
      self.assertIn(send_op, c.ops_which_must_run)

  def _testVariableReadInFunctionalOp(self, build_functional_op, op_type):
    v = resource_variable_ops.ResourceVariable(1.0)
    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def read_var_in_while():
      gen_resource_variable_ops.read_variable_op(
          v.handle, v.dtype, name="read1")

      result = build_functional_op(v)
      gen_resource_variable_ops.read_variable_op(
          v.handle, v.dtype, name="read2")
      gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
      return result

    func_graph = read_var_in_while.get_concrete_function().graph
    assert len(func_graph.inputs) == 1

    def get_op(op_type, sub_name):
      operations = [
          op for op in func_graph.get_operations()
          if op.type == op_type and sub_name in op.name
      ]
      assert len(operations) == 1
      return operations[0]

    read1 = get_op("ReadVariableOp", "read1")
    functional_op = get_op(op_type, "")
    read2 = get_op("ReadVariableOp", "read2")
    assign_op = get_op("AssignVariableOp", "")
    # Since the functional op only has reads, previous reads e.g. read1 do not\
    # have a control edge to it and next future reads e.g. read2 do not have a
    # control edge from it.
    self.assertNotIn(read1, functional_op.control_inputs)
    self.assertNotIn(functional_op, read2.control_inputs)
    self.assertIn(read1, assign_op.control_inputs)
    self.assertIn(read2, assign_op.control_inputs)
    self.assertIn(functional_op, assign_op.control_inputs)

  def testVariableReadInWhileLoop(self):

    def build_functional_op(v):

      def body(_):
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return control_flow_ops.while_loop(
          lambda i: True, body, [0.0], maximum_iterations=1)

    self._testVariableReadInFunctionalOp(build_functional_op, "While")

  def testVariableReadInCondTrueBranch(self):

    def build_functional_op(v):

      def then_branch():
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      def else_branch():
        return array_ops.zeros([], v.dtype)

      return control_flow_ops.cond(
          constant_op.constant(True), then_branch, else_branch)

    self._testVariableReadInFunctionalOp(build_functional_op, "If")

  def testVariableReadInCondFalseBranch(self):

    def build_functional_op(v):

      def then_branch():
        return array_ops.zeros([], v.dtype)

      def else_branch():
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return control_flow_ops.cond(
          constant_op.constant(False), then_branch, else_branch)

    self._testVariableReadInFunctionalOp(build_functional_op, "If")

  def testVariableReadInCaseBranch0(self):

    def build_functional_op(v):

      def branch0():
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      def branch1():
        return array_ops.zeros([], v.dtype)

      return control_flow_ops.switch_case(
          constant_op.constant(0), [branch0, branch1])

    self._testVariableReadInFunctionalOp(build_functional_op, "Case")

  def testVariableReadInCaseBranch1(self):

    def build_functional_op(v):

      def branch0():
        return array_ops.zeros([], v.dtype)

      def branch1():
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return control_flow_ops.switch_case(
          constant_op.constant(0), [branch0, branch1])

    self._testVariableReadInFunctionalOp(build_functional_op, "Case")

  def testVariableReadInFunction(self):

    def build_functional_op(v):

      @def_function.function
      def fn_with_read():
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return fn_with_read()

    self._testVariableReadInFunctionalOp(build_functional_op,
                                         "StatefulPartitionedCall")

  def testVariableReadInNestedFunction(self):

    def build_functional_op(v):

      @def_function.function
      def fn_with_read():

        @def_function.function
        def inner_fn():
          return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

        return inner_fn()

      return fn_with_read()

    self._testVariableReadInFunctionalOp(build_functional_op,
                                         "StatefulPartitionedCall")

  def testVariableReadInWhileInInnerFunc(self):

    def build_functional_op(v):

      @def_function.function
      def fn_with_read():

        @def_function.function
        def inner_fn():

          def body(_):
            return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

          return control_flow_ops.while_loop(
              lambda i: True, body, [0.0], maximum_iterations=1)

        return inner_fn()

      return fn_with_read()

    self._testVariableReadInFunctionalOp(build_functional_op,
                                         "StatefulPartitionedCall")

  def testVariableReadInCondInInnerFunc(self):

    def build_functional_op(v):

      @def_function.function
      def fn_with_read():

        @def_function.function
        def inner_fn():

          def then_branch():
            return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

          def else_branch():
            return array_ops.zeros([], v.dtype)

          return control_flow_ops.cond(
              constant_op.constant(True), then_branch, else_branch)

        return inner_fn()

      return fn_with_read()

    self._testVariableReadInFunctionalOp(build_functional_op,
                                         "StatefulPartitionedCall")

  def _testVariableWriteInFunctionalOp(self, build_functional_op, op_type):
    v = resource_variable_ops.ResourceVariable(1.0)
    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def write_var_in_while():
      gen_resource_variable_ops.read_variable_op(
          v.handle, v.dtype, name="read1")

      result = build_functional_op(v)
      gen_resource_variable_ops.read_variable_op(
          v.handle, v.dtype, name="read2")
      gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
      return result

    func_graph = write_var_in_while.get_concrete_function().graph
    assert len(func_graph.inputs) == 1

    def get_op(op_type, sub_name):
      operations = [
          op for op in func_graph.get_operations()
          if op.type == op_type and sub_name in op.name
      ]
      assert len(operations) == 1
      return operations[0]

    read1 = get_op("ReadVariableOp", "read1")
    functional_op = get_op(op_type, "")
    read2 = get_op("ReadVariableOp", "read2")
    assign_op = get_op("AssignVariableOp", "")
    # Since the While has writes, it has control edges from previous reads
    # e.g. `read1` and to future reads(`read2`) and writes(`assign_op`).
    self.assertIn(read1, functional_op.control_inputs)
    self.assertIn(functional_op, read2.control_inputs)
    self.assertIn(read2, assign_op.control_inputs)
    self.assertIn(functional_op, assign_op.control_inputs)

  def testVariableWriteInWhileLoop(self):

    def build_functional_op(v):

      def body(_):
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return control_flow_ops.while_loop(
          lambda i: True, body, [0.0], maximum_iterations=1)

    self._testVariableWriteInFunctionalOp(build_functional_op, "While")

  def testVariableWriteInCondTrueBranch(self):

    def build_functional_op(v):

      def then_branch():
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      def else_branch():
        return array_ops.zeros([], v.dtype)

      return control_flow_ops.cond(
          constant_op.constant(True), then_branch, else_branch)

    self._testVariableWriteInFunctionalOp(build_functional_op, "If")

  def testVariableWriteInCondFalseBranch(self):

    def build_functional_op(v):

      def then_branch():
        return array_ops.zeros([], v.dtype)

      def else_branch():
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return control_flow_ops.cond(
          constant_op.constant(False), then_branch, else_branch)

    self._testVariableWriteInFunctionalOp(build_functional_op, "If")

  def testVariableWriteInCaseBranch0(self):

    def build_functional_op(v):

      def branch0():
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      def branch1():
        return array_ops.zeros([], v.dtype)

      return control_flow_ops.switch_case(
          constant_op.constant(0), [branch0, branch1])

    self._testVariableWriteInFunctionalOp(build_functional_op, "Case")

  def testVariableWriteInCaseBranch1(self):

    def build_functional_op(v):

      def branch0():
        return array_ops.zeros([], v.dtype)

      def branch1():
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return control_flow_ops.switch_case(
          constant_op.constant(0), [branch0, branch1])

    self._testVariableWriteInFunctionalOp(build_functional_op, "Case")

  def testVariableWriteInFunction(self):

    def build_functional_op(v):

      @def_function.function
      def fn_with_write():
        gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
        return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

      return fn_with_write()

    self._testVariableWriteInFunctionalOp(build_functional_op,
                                          "StatefulPartitionedCall")

  def testVariableWriteInNestedFunction(self):

    def build_functional_op(v):

      @def_function.function
      def fn_with_write():

        @def_function.function
        def inner_fn():
          gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
          return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

        return inner_fn()

      return fn_with_write()

    self._testVariableWriteInFunctionalOp(build_functional_op,
                                          "StatefulPartitionedCall")

  def testVariableWriteInWhileInInnerFunc(self):

    def build_functional_op(v):

      @def_function.function
      def fn_with_write():

        @def_function.function
        def inner_fn():

          def body(_):
            gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
            return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

          return control_flow_ops.while_loop(
              lambda i: True, body, [0.0], maximum_iterations=1)

        return inner_fn()

      return fn_with_write()

    self._testVariableWriteInFunctionalOp(build_functional_op,
                                          "StatefulPartitionedCall")

  def testVariableWriteInCondInInnerFunc(self):

    def build_functional_op(v):

      @def_function.function
      def fn_with_write():

        @def_function.function
        def inner_fn():

          def then_branch():
            gen_resource_variable_ops.assign_variable_op(v.handle, v + 1)
            return gen_resource_variable_ops.read_variable_op(v.handle, v.dtype)

          def else_branch():
            return array_ops.zeros([], v.dtype)

          return control_flow_ops.cond(
              constant_op.constant(True), then_branch, else_branch)

        return inner_fn()

      return fn_with_write()

    self._testVariableWriteInFunctionalOp(build_functional_op,
                                          "StatefulPartitionedCall")

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
      self.assertAllEqual(v.read_value(), 5.0)
      one.eval(feed_dict={p: True})
      self.assertAllEqual(v.read_value(), 6.0)

  @test_util.run_v1_only("b/120545219")
  def testCondNested(self):
    with context.graph_mode(), self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())
      p = array_ops.placeholder(dtype=dtypes.bool)
      q = array_ops.placeholder(dtype=dtypes.bool)
      with acd.AutomaticControlDependencies() as c:

        def true_fn():
          v.assign(v + 1, name="true")
          return 1.0

        def false_fn():

          def inner_true_fn():
            v.assign(v * 2, name="false_true")
            return 2.0

          def inner_false_fn():
            v.assign(v * 3, name="false_false")
            return 3.0

          control_flow_ops.cond(q, inner_true_fn, inner_false_fn)
          return 1.0

        control_flow_ops.cond(p, true_fn, false_fn)
        with ops.name_scope("final"):
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

      self.assertAllEqual(f(), 4.0)

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

    with self.assertRaisesRegex(TypeError,
                                ".*must return zero or more Tensors.*"):
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


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
