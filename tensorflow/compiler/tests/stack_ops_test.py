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
"""Tests for tensorflow.ops.stack_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.platform import test


class StackOpTest(xla_test.XLATestCase):

  def testStackPushPop(self):
    with self.cached_session(), self.test_scope():
      size = array_ops.placeholder(dtypes.int32)
      v = array_ops.placeholder(dtypes.float32)
      h = gen_data_flow_ops.stack_v2(size, dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push_v2(h, v)
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
      self.assertAllClose([[4.0, 5.0]], c1.eval({size: 5, v: [[4.0, 5.0]]}))

  def testStackPushPopSwap(self):
    with self.cached_session(), self.test_scope():
      a = np.arange(2000)
      x = array_ops.placeholder(dtypes.float32)
      h = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push_v2(h, x, swap_memory=True)
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
      self.assertAllClose(a, c1.eval({x: a}))

  def testMultiStack(self):
    with self.cached_session(), self.test_scope():
      v = array_ops.placeholder(dtypes.float32)
      h1 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_push_v2(h1, v)
      with ops.control_dependencies([c1]):
        c1 = gen_data_flow_ops.stack_pop_v2(h1, dtypes.float32)
      h2 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="bar")
      c2 = gen_data_flow_ops.stack_push_v2(h2, 5.0)
      with ops.control_dependencies([c2]):
        c2 = gen_data_flow_ops.stack_pop_v2(h2, dtypes.float32)
      r = c1 + c2
      self.assertAllClose(9.0, r.eval({v: 4.0}))

  def testSameNameStacks(self):
    """Different stacks with the same name do not interfere."""
    with self.cached_session() as sess, self.test_scope():
      v1 = array_ops.placeholder(dtypes.float32)
      v2 = array_ops.placeholder(dtypes.float32)
      h1 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")
      h2 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")

      c1 = gen_data_flow_ops.stack_push_v2(h1, v1)
      with ops.control_dependencies([c1]):
        c2 = gen_data_flow_ops.stack_push_v2(h2, v2)
      with ops.control_dependencies([c2]):
        pop1 = gen_data_flow_ops.stack_pop_v2(h1, dtypes.float32)
        pop2 = gen_data_flow_ops.stack_pop_v2(h2, dtypes.float32)

      out1, out2 = sess.run([pop1, pop2], {v1: 4.0, v2: 5.0})
      self.assertAllClose(out1, 4.0)
      self.assertAllClose(out2, 5.0)

  def testCloseStack(self):
    with self.cached_session() as sess, self.test_scope():
      size = array_ops.placeholder(dtypes.int32)
      h = gen_data_flow_ops.stack_v2(size, dtypes.float32, stack_name="foo")
      c1 = gen_data_flow_ops.stack_close_v2(h)
      sess.run(c1, {size: 5})

  def testPushCloseStack(self):
    with self.cached_session() as sess, self.test_scope():
      v = array_ops.placeholder(dtypes.float32)
      h = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name="foo")
      c = gen_data_flow_ops.stack_push_v2(h, v)
      with ops.control_dependencies([c]):
        c1 = gen_data_flow_ops.stack_close_v2(h)
      sess.run(c1, {v: [[4.0, 5.0]]})


if __name__ == "__main__":
  test.main()
