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
"""Tests for experimental capture-by-value feature in TF functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.framework import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class FunctionTest(test.TestCase):

  def testCaptureByValue(self):
    g = ops.Graph()
    with g.as_default():
      w = constant_op.constant([[1.0]])
      b = constant_op.constant([2.0])

      # Foo() captures w and b.
      @function.Defun(dtypes.float32, capture_by_value=True)
      def Foo(x):

        # Plus() captures b.
        @function.Defun(dtypes.float32, capture_by_value=True)
        def Plus(y):
          return y + b

        self.assertEqual(0, len(Plus.captured_inputs))

        return Plus(math_ops.matmul(w, x))

      y = Foo(constant_op.constant([[10.]]))

    self.assertEqual(0, len(Foo.captured_inputs))

    with self.test_session(graph=g):
      self.assertAllEqual(y.eval(), [[12.0]])


if __name__ == "__main__":
  test.main()
