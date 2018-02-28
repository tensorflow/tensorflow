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

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


@test_util.with_c_api
class SmartCondTest(test_util.TensorFlowTestCase):

  def testSmartCondTrue(self):
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(2)
        y = constant_op.constant(5)
        z = smart_cond.smart_cond(True, lambda: math_ops.multiply(x, 16),
                                  lambda: math_ops.multiply(y, 5))
        self.assertEqual(z.eval(), 32)

  def testSmartCondFalse(self):
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(4)
        y = constant_op.constant(3)
        z = smart_cond.smart_cond(False, lambda: math_ops.multiply(x, 16),
                                  lambda: math_ops.multiply(y, 3))
        self.assertEqual(z.eval(), 9)

  def testSmartCondMissingArg1(self):
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(1)
        with self.assertRaises(TypeError):
          smart_cond.smart_cond(True, false_fn=lambda: x)

  def testSmartCondMissingArg2(self):
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(1)
        with self.assertRaises(TypeError):
          smart_cond.smart_cond(True, lambda: x)


if __name__ == "__main__":
  googletest.main()
