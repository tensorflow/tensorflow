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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.platform import test


class ControlFlowUtilTest(test.TestCase):

  def testIsSwitch(self):
    switch_false, _ = control_flow_ops.switch(1, True)
    switch = switch_false.op
    self.assertTrue(control_flow_util.IsSwitch(switch))

    ref_switch_false, _ = control_flow_ops.ref_switch(test_ops.ref_output(),
                                                      True)
    ref_switch = ref_switch_false.op
    self.assertTrue(control_flow_util.IsSwitch(ref_switch))

    self.assertFalse(control_flow_util.IsSwitch(test_ops.int_output().op))

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

  def testIsLoopExit(self):
    exit_op = control_flow_ops.exit(1).op
    self.assertTrue(control_flow_util.IsLoopExit(exit_op))

    ref_exit = control_flow_ops.exit(test_ops.ref_output()).op
    self.assertTrue(control_flow_util.IsLoopExit(ref_exit))

    self.assertFalse(control_flow_util.IsLoopExit(test_ops.int_output().op))


if __name__ == "__main__":
  test.main()
