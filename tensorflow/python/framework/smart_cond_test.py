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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


def raise_exception():
  raise RuntimeError("did not expect to be called")


class SmartCondTest(test_util.TensorFlowTestCase):

  def testTrue(self):
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(2)
        y = constant_op.constant(5)
        z = smart_cond.smart_cond(True, lambda: math_ops.multiply(x, 16),
                                  lambda: math_ops.multiply(y, 5))
        self.assertEqual(z.eval(), 32)

  def testFalse(self):
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(4)
        y = constant_op.constant(3)
        z = smart_cond.smart_cond(False, lambda: math_ops.multiply(x, 16),
                                  lambda: math_ops.multiply(y, 3))
        self.assertEqual(z.eval(), 9)

  def testUnknown(self):
    with ops.Graph().as_default():
      with session.Session():
        x = array_ops.placeholder(dtype=dtypes.int32)
        y = smart_cond.smart_cond(x > 0, lambda: constant_op.constant(1),
                                  lambda: constant_op.constant(2))
        self.assertEqual(y.eval(feed_dict={x: 1}), 1)
        self.assertEqual(y.eval(feed_dict={x: -1}), 2)

  def testEval(self):
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(1)
        y = constant_op.constant(2)
        # x * y > 0 can be evaluated at graph construction time, so the false
        # branch shouldn't be evaluated at all.
        z = smart_cond.smart_cond(x * y > 0, lambda: constant_op.constant(1),
                                  raise_exception)
        self.assertEqual(z.eval(feed_dict={x: 1}), 1)

  def testPlaceholderWithDefault(self):
    with ops.Graph().as_default():
      with session.Session():
        x = array_ops.placeholder_with_default(1, shape=())
        y = smart_cond.smart_cond(x > 0, lambda: constant_op.constant(1),
                                  lambda: constant_op.constant(2))
        self.assertEqual(y.eval(), 1)
        self.assertEqual(y.eval(feed_dict={x: -1}), 2)

  def testMissingArg1(self):
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(1)
        with self.assertRaises(TypeError):
          smart_cond.smart_cond(True, false_fn=lambda: x)

  def testMissingArg2(self):
    with ops.Graph().as_default():
      with session.Session():
        x = constant_op.constant(1)
        with self.assertRaises(TypeError):
          smart_cond.smart_cond(True, lambda: x)


class SmartCaseTest(test_util.TensorFlowTestCase):

  def testTrue(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(True, lambda: constant_op.constant(1)),
                  (x == 0, raise_exception)]
    y = smart_cond.smart_case(conditions, default=raise_exception,
                              exclusive=False)
    z = smart_cond.smart_case(conditions, default=raise_exception,
                              exclusive=True)
    with session.Session() as sess:
      # No feed_dict necessary
      self.assertEqual(sess.run(y), 1)
      self.assertEqual(sess.run(z), 1)

  def testFalse(self):
    conditions = [(False, raise_exception)]
    y = smart_cond.smart_case(conditions,
                              default=lambda: constant_op.constant(1),
                              exclusive=False)
    z = smart_cond.smart_case(conditions,
                              default=lambda: constant_op.constant(1),
                              exclusive=True)
    with session.Session() as sess:
      self.assertEqual(sess.run(y), 1)
      self.assertEqual(sess.run(z), 1)

  def testMix(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    y = constant_op.constant(10)
    conditions = [(x > 1, lambda: constant_op.constant(1)),
                  (y < 1, raise_exception),
                  (False, raise_exception),
                  (True, lambda: constant_op.constant(3))]
    z = smart_cond.smart_case(conditions, default=raise_exception)
    with session.Session() as sess:
      self.assertEqual(sess.run(z, feed_dict={x: 2}), 1)
      self.assertEqual(sess.run(z, feed_dict={x: 0}), 3)


class SmartConstantValueTest(test_util.TensorFlowTestCase):

  # TODO(skyewm): this is essentially a regression test for
  # TF_TryEvaluateConstant, and is not really a valid smart_constant_value test
  # (smart_constant_value is only supposed to return bools). Move the
  # TF_TryEvaluateConstant call to tensor_util.constant_value and make this a
  # constant_value test instead.
  def testCond(self):
    with ops.Graph().as_default():
      pred = array_ops.placeholder_with_default(True, shape=())
      x = control_flow_ops.cond(pred,
                                lambda: constant_op.constant(1),
                                lambda: constant_op.constant(2))
      self.assertIsNone(smart_cond.smart_constant_value(x))


if __name__ == "__main__":
  googletest.main()
