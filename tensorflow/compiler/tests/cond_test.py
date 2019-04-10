# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.cond in XLA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


@test_util.with_control_flow_v2
class CondTest(xla_test.XLATestCase):

  def testCondAndTensorArrayInDefun(self):
    with self.cached_session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      @function.defun
      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = control_flow_ops.cond(
            constant_op.constant(
                True), lambda: ta.write(0, 5.), lambda: ta.write(0, 10.))

        return output.stack()

      output_t = f()
      self.assertAllEqual(self.evaluate(output_t), [5.])

      xla_context.Exit()

  def testCondConstPropagation(self):
    with self.cached_session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()

      x = array_ops.placeholder(dtypes.float32)
      p = array_ops.placeholder(dtypes.int32)

      # TODO(b/129021699): Wrapping this in a tf.function does not work.
      def if_true():
        # This emits a StridedSlice op which expects the index to be a
        # compile-time const.
        return x[p]

      def if_false():
        return 5.

      output = control_flow_ops.cond(
          constant_op.constant(True), if_true, if_false)

      self.assertAllEqual(
          sess.run(output, feed_dict={
              x: [0., 1., 2.],
              p: 1
          }), 1.)

      xla_context.Exit()


if __name__ == '__main__':
  test.main()
