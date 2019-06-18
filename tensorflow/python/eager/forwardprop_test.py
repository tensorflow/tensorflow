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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import weakref

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import forwardprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ForwardpropTest(test.TestCase):

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testMultipleWatchesAdd(self):
    x = constant_op.constant(-2.)
    with forwardprop.ForwardGradientAccumulator() as acc:
      acc.watch(x, constant_op.constant(10.))
      self.assertAllClose(10., acc.jvp(x))
      acc.watch(x, constant_op.constant(11.))
      self.assertAllClose(21., acc.jvp(x))
      y = constant_op.constant(3.) * x
    self.assertAllClose(21., acc.jvp(x))
    self.assertAllClose(21. * 3., acc.jvp(y))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testDeadTensorsJVPCleared(self):
    x = array_ops.ones([100])
    x_weak = weakref.ref(x)
    grad_tensor = constant_op.constant(array_ops.zeros([100]))
    grad_tensor_weak = weakref.ref(grad_tensor)
    with forwardprop.ForwardGradientAccumulator() as acc:
      acc.watch(x, grad_tensor)
      derived_tensor = constant_op.constant(2.) * x
      del grad_tensor
      self.assertAllClose(array_ops.zeros([100]), acc.jvp(x))
      del x
      self.assertIsNone(x_weak())
      self.assertIsNone(grad_tensor_weak())
      derived_tensor_weak = weakref.ref(derived_tensor)
      derived_tensor_grad = acc.jvp(derived_tensor)
      derived_tensor_grad_weak = weakref.ref(derived_tensor_grad)
      del derived_tensor
      del derived_tensor_grad
      self.assertIsNone(derived_tensor_weak())
      self.assertIsNone(derived_tensor_grad_weak())

  def testAgainstExplicitJacobian(self):

    def f(x):
      return math_ops.reduce_sum(math_ops.sin(x) * math_ops.tan(x), axis=1)

    x = constant_op.constant([[2.0, 3.0], [1.0, 4.0]])

    def forward_accumulate():
      with forwardprop.ForwardGradientAccumulator() as acc:
        acc.watch(x, constant_op.constant([[5., 6.], [7., 8.]]))
        y = f(x)
      return acc.jvp(y)

    jvp_from_accumulator_eager = forward_accumulate()
    jvp_from_accumulator_function = def_function.function(forward_accumulate)()

    x_flat = array_ops.reshape(x, [-1])
    with backprop.GradientTape() as tape:
      tape.watch(x_flat)
      a = f(array_ops.reshape(x_flat, array_ops.shape(x)))
      a_flat = array_ops.reshape(a, [-1])
    jacobian = tape.jacobian(a_flat, x_flat)
    jvp_from_backprop = math_ops.tensordot(jacobian, [5., 6., 7., 8.], axes=1)

    self.assertAllClose(jvp_from_backprop, jvp_from_accumulator_eager)
    self.assertAllClose(jvp_from_backprop, jvp_from_accumulator_function)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
