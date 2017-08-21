# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Functional test for learning rate decay."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_state_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import learning_rate_decay


class LRDecayTest(test_util.TensorFlowTestCase):

  def testContinuous(self):
    with self.test_session():
      step = 5
      decayed_lr = learning_rate_decay.exponential_decay(0.05, step, 10, 0.96)
      expected = .05 * 0.96 ** (5.0 / 10.0)
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testStaircase(self):
    with self.test_session():
      step = gen_state_ops._variable(shape=[], dtype=dtypes.int32,
          name="step", container="", shared_name="")
      assign_100 = state_ops.assign(step, 100)
      assign_1 = state_ops.assign(step, 1)
      assign_2 = state_ops.assign(step, 2)
      decayed_lr = learning_rate_decay.exponential_decay(.1, step, 3, 0.96,
                                                         staircase=True)
      # No change to learning rate
      assign_1.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      assign_2.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      # Decayed learning rate
      assign_100.op.run()
      expected = .1 * 0.96 ** (100 // 3)
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testVariables(self):
    with self.test_session():
      step = variables.Variable(1)
      assign_1 = step.assign(1)
      assign_2 = step.assign(2)
      assign_100 = step.assign(100)
      decayed_lr = learning_rate_decay.exponential_decay(.1, step, 3, 0.96,
                                                         staircase=True)
      variables.global_variables_initializer().run()
      # No change to learning rate
      assign_1.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      assign_2.op.run()
      self.assertAllClose(decayed_lr.eval(), .1, 1e-6)
      # Decayed learning rate
      assign_100.op.run()
      expected = .1 * 0.96 ** (100 // 3)
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testPiecewiseConstant(self):
    with self.test_session():
      x = variables.Variable(-999)
      assign_100 = x.assign(100)
      assign_105 = x.assign(105)
      assign_110 = x.assign(110)
      assign_120 = x.assign(120)
      assign_999 = x.assign(999)
      pc = learning_rate_decay.piecewise_constant(x, [100, 110, 120],
                                                  [1.0, 0.1, 0.01, 0.001])

      variables.global_variables_initializer().run()
      self.assertAllClose(pc.eval(), 1.0, 1e-6)
      assign_100.op.run()
      self.assertAllClose(pc.eval(), 1.0, 1e-6)
      assign_105.op.run()
      self.assertAllClose(pc.eval(), 0.1, 1e-6)
      assign_110.op.run()
      self.assertAllClose(pc.eval(), 0.1, 1e-6)
      assign_120.op.run()
      self.assertAllClose(pc.eval(), 0.01, 1e-6)
      assign_999.op.run()
      self.assertAllClose(pc.eval(), 0.001, 1e-6)

  def testPiecewiseConstantEdgeCases(self):
    with self.test_session():
      x_int = variables.Variable(0, dtype=variables.dtypes.int32)
      boundaries, values = [-1.0, 1.0], [1, 2, 3]
      with self.assertRaises(ValueError):
        learning_rate_decay.piecewise_constant(x_int, boundaries, values)
      x = variables.Variable(0.0)
      boundaries, values = [-1.0, 1.0], [1.0, 2, 3]
      with self.assertRaises(ValueError):
        learning_rate_decay.piecewise_constant(x, boundaries, values)

      # Test that ref types are valid.
      x_ref = x.op.outputs[0]   # float32_ref tensor should be accepted
      boundaries, values = [1.0, 2.0], [1, 2, 3]
      learning_rate_decay.piecewise_constant(x_ref, boundaries, values)

      # Test casting boundaries from int32 to int64.
      x_int64 = variables.Variable(0, dtype=variables.dtypes.int64)
      assign_1 = x_int64.assign(1)
      assign_2 = x_int64.assign(2)
      assign_3 = x_int64.assign(3)
      assign_4 = x_int64.assign(4)
      boundaries, values = [1, 2, 3], [0.4, 0.5, 0.6, 0.7]
      pc = learning_rate_decay.piecewise_constant(x_int64, boundaries, values)

      variables.global_variables_initializer().run()
      self.assertAllClose(pc.eval(), 0.4, 1e-6)
      assign_1.op.run()
      self.assertAllClose(pc.eval(), 0.4, 1e-6)
      assign_2.op.run()
      self.assertAllClose(pc.eval(), 0.5, 1e-6)
      assign_3.op.run()
      self.assertAllClose(pc.eval(), 0.6, 1e-6)
      assign_4.op.run()
      self.assertAllClose(pc.eval(), 0.7, 1e-6)


class LinearDecayTest(test_util.TensorFlowTestCase):

  def testHalfWay(self):
    with self.test_session():
      step = 5
      lr = 0.05
      end_lr = 0.0
      decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr)
      expected = lr * 0.5
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testEnd(self):
    with self.test_session():
      step = 10
      lr = 0.05
      end_lr = 0.001
      decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr)
      expected = end_lr
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testHalfWayWithEnd(self):
    with self.test_session():
      step = 5
      lr = 0.05
      end_lr = 0.001
      decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr)
      expected = (lr + end_lr) * 0.5
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testBeyondEnd(self):
    with self.test_session():
      step = 15
      lr = 0.05
      end_lr = 0.001
      decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr)
      expected = end_lr
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testBeyondEndWithCycle(self):
    with self.test_session():
      step = 15
      lr = 0.05
      end_lr = 0.001
      decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr,
                                                        cycle=True)
      expected = (lr - end_lr) * 0.25 + end_lr
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)


class SqrtDecayTest(test_util.TensorFlowTestCase):

  def testHalfWay(self):
    with self.test_session():
      step = 5
      lr = 0.05
      end_lr = 0.0
      power = 0.5
      decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr,
                                                        power=power)
      expected = lr * 0.5 ** power
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testEnd(self):
    with self.test_session():
      step = 10
      lr = 0.05
      end_lr = 0.001
      power = 0.5
      decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr,
                                                        power=power)
      expected = end_lr
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testHalfWayWithEnd(self):
    with self.test_session():
      step = 5
      lr = 0.05
      end_lr = 0.001
      power = 0.5
      decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr,
                                                        power=power)
      expected = (lr - end_lr) * 0.5 ** power + end_lr
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testBeyondEnd(self):
    with self.test_session():
      step = 15
      lr = 0.05
      end_lr = 0.001
      power = 0.5
      decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr,
                                                        power=power)
      expected = end_lr
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)

  def testBeyondEndWithCycle(self):
    with self.test_session():
      step = 15
      lr = 0.05
      end_lr = 0.001
      power = 0.5
      decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr,
                                                        power=power, cycle=True)
      expected = (lr - end_lr) * 0.25 ** power + end_lr
      self.assertAllClose(decayed_lr.eval(), expected, 1e-6)


class ExponentialDecayTest(test_util.TensorFlowTestCase):

  def testDecay(self):
    initial_lr = 0.1
    k = 10
    decay_rate = 0.96
    step = gen_state_ops._variable(shape=[], dtype=dtypes.int32,
        name="step", container="", shared_name="")
    assign_step = state_ops.assign(step, 0)
    increment_step = state_ops.assign_add(step, 1)
    decayed_lr = learning_rate_decay.natural_exp_decay(initial_lr, step,
                                                       k, decay_rate)
    with self.test_session():
      assign_step.op.run()
      for i in range(k+1):
        expected = initial_lr * math.exp(-i / k * decay_rate)
        self.assertAllClose(decayed_lr.eval(), expected, 1e-6)
        increment_step.op.run()

  def testStaircase(self):
    initial_lr = 0.1
    k = 10
    decay_rate = 0.96
    step = gen_state_ops._variable(shape=[], dtype=dtypes.int32,
        name="step", container="", shared_name="")
    assign_step = state_ops.assign(step, 0)
    increment_step = state_ops.assign_add(step, 1)
    decayed_lr = learning_rate_decay.natural_exp_decay(initial_lr,
                                                       step,
                                                       k,
                                                       decay_rate,
                                                       staircase=True)
    with self.test_session():
      assign_step.op.run()
      for i in range(k+1):
        expected = initial_lr * math.exp(-decay_rate * (i // k))
        self.assertAllClose(decayed_lr.eval(), expected, 1e-6)
        increment_step.op.run()


class InverseDecayTest(test_util.TensorFlowTestCase):

  def testDecay(self):
    initial_lr = 0.1
    k = 10
    decay_rate = 0.96
    step = gen_state_ops._variable(shape=[], dtype=dtypes.int32,
        name="step", container="", shared_name="")
    assign_step = state_ops.assign(step, 0)
    increment_step = state_ops.assign_add(step, 1)
    decayed_lr = learning_rate_decay.inverse_time_decay(initial_lr,
                                                        step,
                                                        k,
                                                        decay_rate)
    with self.test_session():
      assign_step.op.run()
      for i in range(k+1):
        expected = initial_lr / (1 + i / k * decay_rate)
        self.assertAllClose(decayed_lr.eval(), expected, 1e-6)
        increment_step.op.run()

  def testStaircase(self):
    initial_lr = 0.1
    k = 10
    decay_rate = 0.96
    step = gen_state_ops._variable(shape=[], dtype=dtypes.int32,
        name="step", container="", shared_name="")
    assign_step = state_ops.assign(step, 0)
    increment_step = state_ops.assign_add(step, 1)
    decayed_lr = learning_rate_decay.inverse_time_decay(initial_lr,
                                                        step,
                                                        k,
                                                        decay_rate,
                                                        staircase=True)
    with self.test_session():
      assign_step.op.run()
      for i in range(k+1):
        expected = initial_lr / (1 + decay_rate * (i // k))
        self.assertAllClose(decayed_lr.eval(), expected, 1e-6)
        increment_step.op.run()


if __name__ == "__main__":
  googletest.main()
