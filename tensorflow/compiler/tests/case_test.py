# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for case statements in XLA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test


class CaseTest(xla_test.XLATestCase):

  def testCaseBasic(self):

    @def_function.function(jit_compile=True)
    def switch_case_test(branch_index):

      def f1():
        return array_ops.constant(17)

      def f2():
        return array_ops.constant(31)

      def f3():
        return array_ops.constant(-1)

      return control_flow_ops.switch_case(
          branch_index, branch_fns={
              0: f1,
              1: f2
          }, default=f3)

    with ops.device(self.device):
      self.assertEqual(switch_case_test(array_ops.constant(0)).numpy(), 17)
      self.assertEqual(switch_case_test(array_ops.constant(1)).numpy(), 31)
      self.assertEqual(switch_case_test(array_ops.constant(2)).numpy(), -1)
      self.assertEqual(switch_case_test(array_ops.constant(3)).numpy(), -1)

  def testBranchIsPruned(self):

    @def_function.function(jit_compile=True)
    def switch_case_test():
      branch_index = array_ops.constant(0)

      def f1():
        return array_ops.constant(17)

      def f2():
        # Some operations that XLA cannot compile.
        image_ops.decode_image(io_ops.read_file('/tmp/bmp'))
        return array_ops.constant(31)

      # This tests that we do not try to compile all branches if the branch
      # index in trivially constant.
      return control_flow_ops.switch_case(
          branch_index, branch_fns={
              0: f1,
              1: f2
          }, default=f2)

    with ops.device(self.device):
      self.assertEqual(switch_case_test().numpy(), 17)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
