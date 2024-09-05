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

"""Tests for tensorflow.python.ops.control_flow_util."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop as while_loop_tf
from tensorflow.python.platform import test


class ControlFlowUtilTest(test.TestCase):

  @test_util.run_v1_only("b/120545219")
  def testIsSwitch(self):
    switch_false, _ = control_flow_ops.switch(1, True)
    switch = switch_false.op
    self.assertTrue(control_flow_util.IsSwitch(switch))

    ref_switch_false, _ = control_flow_ops.ref_switch(test_ops.ref_output(),
                                                      True)
    ref_switch = ref_switch_false.op
    self.assertTrue(control_flow_util.IsSwitch(ref_switch))

    self.assertFalse(control_flow_util.IsSwitch(test_ops.int_output().op))

  @test_util.run_v1_only("b/120545219")
  def testIsLoopEnter(self):
    enter = gen_control_flow_ops.enter(1, frame_name="name").op
    self.assertTrue(control_flow_util.IsLoopEnter(enter))
    self.assertFalse(control_flow_util.IsLoopConstantEnter(enter))

    ref_enter = gen_control_flow_ops.ref_enter(test_ops.ref_output(),
                                               frame_name="name").op
    self.assertTrue(control_flow_util.IsLoopEnter(ref_enter))
    self.assertFalse(control_flow_util.IsLoopConstantEnter(ref_enter))

    const_enter = gen_control_flow_ops.enter(1, frame_name="name",
                                             is_constant=True).op
    self.assertTrue(control_flow_util.IsLoopEnter(const_enter))
    self.assertTrue(control_flow_util.IsLoopConstantEnter(const_enter))

    self.assertFalse(control_flow_util.IsLoopEnter(test_ops.int_output().op))

  @test_util.run_v1_only("b/120545219")
  def testIsLoopExit(self):
    exit_op = control_flow_ops.exit(1).op
    self.assertTrue(control_flow_util.IsLoopExit(exit_op))

    ref_exit = control_flow_ops.exit(test_ops.ref_output()).op
    self.assertTrue(control_flow_util.IsLoopExit(ref_exit))

    self.assertFalse(control_flow_util.IsLoopExit(test_ops.int_output().op))

  def build_test_graph(self) -> ops.Graph:
    g = ops.Graph()
    with g.as_default():

      def while_loop(x):

        def b(x):
          with ops.name_scope("NestedCond"):
            return cond.cond(
                math_ops.less(x, 100), lambda: math_ops.add(x, 1),
                lambda: math_ops.add(x, 2))

        c = lambda x: math_ops.less(x, 10000)
        with ops.name_scope("OuterWhile"):
          return while_loop_tf.while_loop(c, b, [x])

      x = array_ops.placeholder(dtypes.int32)
      with ops.name_scope("OuterCond"):
        cond.cond(
            math_ops.less(x, 1000), lambda: while_loop(x),
            lambda: math_ops.add(x, 2))
    return g

  def testIsCondSwitch(self):
    g = self.build_test_graph()

    cond_switch = [
        "OuterCond/cond/Switch",
        "OuterCond/cond/OuterWhile/while/Switch",
        "OuterCond/cond/OuterWhile/while/NestedCond/cond/Switch",
        "OuterCond/cond/OuterWhile/while/NestedCond/cond/Add/Switch",
        "OuterCond/cond/OuterWhile/while/NestedCond/cond/Add_1/Switch",
        "OuterCond/cond/Add/Switch",
    ]
    for n in g.get_operations():
      if control_flow_util.IsSwitch(n):
        self.assertTrue(
            control_flow_util.IsCondSwitch(n) != control_flow_util.IsLoopSwitch(
                n))
      if n.name in cond_switch:
        self.assertTrue(control_flow_util.IsSwitch(n))
        self.assertTrue(
            control_flow_util.IsCondSwitch(n),
            msg="Mismatch for {}".format(n.name))
        self.assertFalse(
            control_flow_util.IsLoopSwitch(n),
            msg="Mismatch for {}".format(n.name))
      else:
        self.assertFalse(
            control_flow_util.IsCondSwitch(n),
            msg="Mismatch for {}".format(n.name))

  def testIsLoopSwitch(self):
    g = self.build_test_graph()

    loop_switch = ["OuterCond/cond/OuterWhile/while/Switch_1"]
    for n in g.get_operations():
      if control_flow_util.IsSwitch(n):
        self.assertTrue(
            control_flow_util.IsCondSwitch(n) != control_flow_util.IsLoopSwitch(
                n))
      if n.name in loop_switch:
        self.assertTrue(control_flow_util.IsSwitch(n))
        self.assertFalse(
            control_flow_util.IsCondSwitch(n),
            msg="Mismatch for {}".format(n.name))
        self.assertTrue(
            control_flow_util.IsLoopSwitch(n),
            msg="Mismatch for {}".format(n.name))
      else:
        self.assertFalse(
            control_flow_util.IsLoopSwitch(n),
            msg="Mismatch for {}".format(n.name))

  def testIsCondMerge(self):
    g = self.build_test_graph()
    cond_merges = [
        "OuterCond/cond/OuterWhile/while/NestedCond/cond/Merge",
        "OuterCond/cond/Merge"
    ]
    for n in g.get_operations():
      if n.name in cond_merges:
        self.assertTrue(control_flow_util.IsMerge(n))
        self.assertTrue(control_flow_util.IsCondMerge(n))
        self.assertFalse(control_flow_util.IsLoopMerge(n))
      else:
        self.assertFalse(control_flow_util.IsCondMerge(n))
        self.assertTrue(not control_flow_util.IsMerge(n) or
                        control_flow_util.IsLoopMerge(n))

  def testIsLoopMerge(self):
    g = self.build_test_graph()
    loop_merges = [
        "OuterCond/cond/OuterWhile/while/Merge",
    ]
    for n in g.get_operations():
      if n.name in loop_merges:
        self.assertTrue(control_flow_util.IsMerge(n))
        self.assertFalse(control_flow_util.IsCondMerge(n))
        self.assertTrue(control_flow_util.IsLoopMerge(n))
      else:
        self.assertFalse(control_flow_util.IsLoopMerge(n))
        self.assertTrue(not control_flow_util.IsMerge(n) or
                        control_flow_util.IsCondMerge(n))


if __name__ == "__main__":
  test.main()
