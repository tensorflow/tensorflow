# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Functional test for moving_averages.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import types
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import moving_averages


class MovingAveragesTest(test_util.TensorFlowTestCase):

  def testAssignMovingAverage(self):
    with self.test_session():
      var = variables.Variable([10.0, 11.0])
      val = constant_op.constant([1.0, 2.0], types.float32)
      decay = 0.25
      assign = moving_averages.assign_moving_average(var, val, decay)
      variables.initialize_all_variables().run()
      self.assertAllClose([10.0, 11.0], var.eval())
      assign.op.run()
      self.assertAllClose([10.0 * 0.25 + 1.0 * (1.0 - 0.25),
                           11.0 * 0.25 + 2.0 * (1.0 - 0.25)],
                          var.eval())

def _Repeat(value, dim):
  if dim == 1:
    return value
  return [value for _ in xrange(dim)]

class ExponentialMovingAverageTest(test_util.TensorFlowTestCase):

  def _CheckDecay(self, ema, actual_decay, dim):
    tens = _Repeat(10.0, dim)
    thirties = _Repeat(30.0, dim)
    var0 = variables.Variable(tens, name="v0")
    var1 = variables.Variable(thirties, name="v1")
    variables.initialize_all_variables().run()
    # Note that tensor2 is not a Variable but just a plain Tensor resulting
    # from the sum operation.
    tensor2 = var0 + var1
    update = ema.apply([var0, var1, tensor2])
    avg0 = ema.average(var0)
    avg1 = ema.average(var1)
    avg2 = ema.average(tensor2)

    self.assertFalse(avg0 in variables.trainable_variables())
    self.assertFalse(avg1 in variables.trainable_variables())
    self.assertFalse(avg2 in variables.trainable_variables())
    variables.initialize_all_variables().run()

    self.assertEqual("v0/ExponentialMovingAverage:0", avg0.name)
    self.assertEqual("v1/ExponentialMovingAverage:0", avg1.name)
    self.assertEqual("add/ExponentialMovingAverage:0", avg2.name)

    # Check initial values.
    self.assertAllClose(tens, var0.eval())
    self.assertAllClose(thirties, var1.eval())
    self.assertAllClose(_Repeat(10.0 + 30.0, dim), tensor2.eval())

    # Check that averages are initialized correctly.
    self.assertAllClose(tens, avg0.eval())
    self.assertAllClose(thirties, avg1.eval())
    # Note that averages of Tensor's initialize to zeros_like since no value
    # of the Tensor is known because the Op has not been run (yet).
    self.assertAllClose(_Repeat(0.0, dim), avg2.eval())

    # Update the averages and check.
    update.run()
    dk = actual_decay

    expected = _Repeat(10.0 * dk + 10.0 * (1 - dk), dim)
    self.assertAllClose(expected, avg0.eval())
    expected = _Repeat(30.0 * dk + 30.0 * (1 - dk), dim)
    self.assertAllClose(expected, avg1.eval())
    expected = _Repeat(0.0 * dk + (10.0 + 30.0) * (1 - dk), dim)
    self.assertAllClose(expected, avg2.eval())

    # Again, update the averages and check.
    update.run()
    expected = _Repeat((10.0 * dk + 10.0 * (1 - dk)) * dk + 10.0 * (1 - dk),
                       dim)
    self.assertAllClose(expected, avg0.eval())
    expected = _Repeat((30.0 * dk + 30.0 * (1 - dk)) * dk + 30.0 * (1 - dk),
                       dim)
    self.assertAllClose(expected, avg1.eval())
    expected = _Repeat(((0.0 * dk + (10.0 + 30.0) * (1 - dk)) * dk +
                        (10.0 + 30.0) * (1 - dk)),
                       dim)
    self.assertAllClose(expected, avg2.eval())

  def testAverageVariablesNoNumUpdates_Scalar(self):
    with self.test_session():
      ema = moving_averages.ExponentialMovingAverage(0.25)
      self._CheckDecay(ema, actual_decay=0.25, dim=1)

  def testAverageVariablesNoNumUpdates_Vector(self):
    with self.test_session():
      ema = moving_averages.ExponentialMovingAverage(0.25)
      self._CheckDecay(ema, actual_decay=0.25, dim=5)

  def testAverageVariablesNumUpdates_Scalar(self):
    with self.test_session():
      # With num_updates 1, the decay applied is 0.1818
      ema = moving_averages.ExponentialMovingAverage(0.25, num_updates=1)
      self._CheckDecay(ema, actual_decay=0.181818, dim=1)

  def testAverageVariablesNumUpdates_Vector(self):
    with self.test_session():
      # With num_updates 1, the decay applied is 0.1818
      ema = moving_averages.ExponentialMovingAverage(0.25, num_updates=1)
      self._CheckDecay(ema, actual_decay=0.181818, dim=5)

  def testAverageVariablesNames(self):
    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(30.0, name="v1")
    tensor2 = v0 + v1
    ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
    self.assertEqual("v0/foo_avg", ema.average_name(v0))
    self.assertEqual("v1/foo_avg", ema.average_name(v1))
    self.assertEqual("add/foo_avg", ema.average_name(tensor2))
    ema.apply([v0, v1, tensor2])
    self.assertEqual(ema.average_name(v0), ema.average(v0).op.name)
    self.assertEqual(ema.average_name(v1), ema.average(v1).op.name)
    self.assertEqual(ema.average_name(tensor2), ema.average(tensor2).op.name)

  def testAverageVariablesDeviceAssignment(self):
    with ops.device("dev_v0"):
      v0 = variables.Variable(10.0, name="v0")
    with ops.device("dev_v1"):
      v1 = state_ops.variable_op(shape=[1], dtype=types.float32, name="v1")
    tensor2 = v0 + v1
    ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
    with ops.device("default"):
      ema.apply([v0, v1, tensor2])
    self.assertEqual("dev_v0", ema.average(v0).device)
    self.assertEqual("dev_v1", ema.average(v1).device)
    self.assertEqual("default", ema.average(tensor2).device)


if __name__ == "__main__":
  googletest.main()
