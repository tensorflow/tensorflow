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

from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import learning_rate_decay


class LRDecayTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testContinuous(self):
    self.evaluate(variables.global_variables_initializer())
    step = 5
    decayed_lr = learning_rate_decay.exponential_decay(0.05, step, 10, 0.96)
    expected = .05 * 0.96**(5.0 / 10.0)
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testStaircase(self):
    if context.executing_eagerly():
      step = resource_variable_ops.ResourceVariable(0)
      self.evaluate(variables.global_variables_initializer())
      decayed_lr = learning_rate_decay.exponential_decay(
          .1, step, 3, 0.96, staircase=True)

      # No change to learning rate due to staircase
      expected = .1
      self.evaluate(step.assign(1))
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

      expected = .1
      self.evaluate(step.assign(2))
      self.assertAllClose(self.evaluate(decayed_lr), .1, 1e-6)

      # Decayed learning rate
      expected = .1 * 0.96 ** (100 // 3)
      self.evaluate(step.assign(100))
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  def testVariables(self):
    step = variables.VariableV1(1)
    assign_1 = step.assign(1)
    assign_2 = step.assign(2)
    assign_100 = step.assign(100)
    decayed_lr = learning_rate_decay.exponential_decay(
        .1, step, 3, 0.96, staircase=True)
    self.evaluate(variables.global_variables_initializer())
    # No change to learning rate
    self.evaluate(assign_1.op)
    self.assertAllClose(self.evaluate(decayed_lr), .1, 1e-6)
    self.evaluate(assign_2.op)
    self.assertAllClose(self.evaluate(decayed_lr), .1, 1e-6)
    # Decayed learning rate
    self.evaluate(assign_100.op)
    expected = .1 * 0.96**(100 // 3)
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testPiecewiseConstant(self):
    x = resource_variable_ops.ResourceVariable(-999)
    decayed_lr = learning_rate_decay.piecewise_constant(
        x, [100, 110, 120], [1.0, 0.1, 0.01, 0.001])

    self.evaluate(variables.global_variables_initializer())

    self.assertAllClose(self.evaluate(decayed_lr), 1.0, 1e-6)
    self.evaluate(x.assign(100))
    self.assertAllClose(self.evaluate(decayed_lr), 1.0, 1e-6)
    self.evaluate(x.assign(105))
    self.assertAllClose(self.evaluate(decayed_lr), 0.1, 1e-6)
    self.evaluate(x.assign(110))
    self.assertAllClose(self.evaluate(decayed_lr), 0.1, 1e-6)
    self.evaluate(x.assign(120))
    self.assertAllClose(self.evaluate(decayed_lr), 0.01, 1e-6)
    self.evaluate(x.assign(999))
    self.assertAllClose(self.evaluate(decayed_lr), 0.001, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testPiecewiseConstantEdgeCases(self):
    x_int = resource_variable_ops.ResourceVariable(
        0, dtype=variables.dtypes.int32)
    boundaries, values = [-1.0, 1.0], [1, 2, 3]
    with self.assertRaises(ValueError):
      decayed_lr = learning_rate_decay.piecewise_constant(
          x_int, boundaries, values)
      if context.executing_eagerly():
        decayed_lr()

    x = resource_variable_ops.ResourceVariable(0.0)
    boundaries, values = [-1.0, 1.0], [1.0, 2, 3]
    with self.assertRaises(ValueError):
      decayed_lr = learning_rate_decay.piecewise_constant(
          x, boundaries, values)
      if context.executing_eagerly():
        decayed_lr()

    # Test that ref types are valid.
    if not context.executing_eagerly():
      x = variables.VariableV1(0.0)
      x_ref = x.op.outputs[0]   # float32_ref tensor should be accepted
      boundaries, values = [1.0, 2.0], [1, 2, 3]
      learning_rate_decay.piecewise_constant(x_ref, boundaries, values)

    # Test casting boundaries from int32 to int64.
    x_int64 = resource_variable_ops.ResourceVariable(
        0, dtype=variables.dtypes.int64)
    boundaries, values = [1, 2, 3], [0.4, 0.5, 0.6, 0.7]
    decayed_lr = learning_rate_decay.piecewise_constant(
        x_int64, boundaries, values)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(decayed_lr), 0.4, 1e-6)
    self.evaluate(x_int64.assign(1))
    self.assertAllClose(self.evaluate(decayed_lr), 0.4, 1e-6)
    self.evaluate(x_int64.assign(2))
    self.assertAllClose(self.evaluate(decayed_lr), 0.5, 1e-6)
    self.evaluate(x_int64.assign(3))
    self.assertAllClose(self.evaluate(decayed_lr), 0.6, 1e-6)
    self.evaluate(x_int64.assign(4))
    self.assertAllClose(self.evaluate(decayed_lr), 0.7, 1e-6)


class LinearDecayTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testHalfWay(self):
    step = 5
    lr = 0.05
    end_lr = 0.0
    decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr)
    expected = lr * 0.5
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testEnd(self):
    step = 10
    lr = 0.05
    end_lr = 0.001
    decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr)
    expected = end_lr
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testHalfWayWithEnd(self):
    step = 5
    lr = 0.05
    end_lr = 0.001
    decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr)
    expected = (lr + end_lr) * 0.5
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testBeyondEnd(self):
    step = 15
    lr = 0.05
    end_lr = 0.001
    decayed_lr = learning_rate_decay.polynomial_decay(lr, step, 10, end_lr)
    expected = end_lr
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testBeyondEndWithCycle(self):
    step = 15
    lr = 0.05
    end_lr = 0.001
    decayed_lr = learning_rate_decay.polynomial_decay(
        lr, step, 10, end_lr, cycle=True)
    expected = (lr - end_lr) * 0.25 + end_lr
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)


class SqrtDecayTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testHalfWay(self):
    step = 5
    lr = 0.05
    end_lr = 0.0
    power = 0.5
    decayed_lr = learning_rate_decay.polynomial_decay(
        lr, step, 10, end_lr, power=power)
    expected = lr * 0.5**power
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testEnd(self):
    step = 10
    lr = 0.05
    end_lr = 0.001
    power = 0.5
    decayed_lr = learning_rate_decay.polynomial_decay(
        lr, step, 10, end_lr, power=power)
    expected = end_lr
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testHalfWayWithEnd(self):
    step = 5
    lr = 0.05
    end_lr = 0.001
    power = 0.5
    decayed_lr = learning_rate_decay.polynomial_decay(
        lr, step, 10, end_lr, power=power)
    expected = (lr - end_lr) * 0.5**power + end_lr
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testBeyondEnd(self):
    step = 15
    lr = 0.05
    end_lr = 0.001
    power = 0.5
    decayed_lr = learning_rate_decay.polynomial_decay(
        lr, step, 10, end_lr, power=power)
    expected = end_lr
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testBeyondEndWithCycle(self):
    step = 15
    lr = 0.05
    end_lr = 0.001
    power = 0.5
    decayed_lr = learning_rate_decay.polynomial_decay(
        lr, step, 10, end_lr, power=power, cycle=True)
    expected = (lr - end_lr) * 0.25**power + end_lr
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)


class PolynomialDecayTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testBeginWithCycle(self):
    lr = 0.001
    decay_steps = 10
    step = 0
    decayed_lr = learning_rate_decay.polynomial_decay(
        lr, step, decay_steps, cycle=True)
    expected = lr
    self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)


class ExponentialDecayTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testDecay(self):
    initial_lr = 0.1
    k = 10
    decay_rate = 0.96
    step = resource_variable_ops.ResourceVariable(0)
    decayed_lr = learning_rate_decay.natural_exp_decay(initial_lr, step, k,
                                                       decay_rate)

    self.evaluate(variables.global_variables_initializer())
    for i in range(k + 1):
      expected = initial_lr * math.exp(-i / k * decay_rate)
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)
      self.evaluate(step.assign_add(1))

  @test_util.run_in_graph_and_eager_modes
  def testStaircase(self):
    initial_lr = 0.1
    k = 10
    decay_rate = 0.96
    step = resource_variable_ops.ResourceVariable(0)
    decayed_lr = learning_rate_decay.natural_exp_decay(
        initial_lr, step, k, decay_rate, staircase=True)

    self.evaluate(variables.global_variables_initializer())
    for i in range(k + 1):
      expected = initial_lr * math.exp(-decay_rate * (i // k))
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)
      self.evaluate(step.assign_add(1))


class InverseDecayTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testDecay(self):
    initial_lr = 0.1
    k = 10
    decay_rate = 0.96
    step = resource_variable_ops.ResourceVariable(0)
    decayed_lr = learning_rate_decay.inverse_time_decay(initial_lr, step, k,
                                                        decay_rate)

    self.evaluate(variables.global_variables_initializer())
    for i in range(k + 1):
      expected = initial_lr / (1 + i / k * decay_rate)
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)
      self.evaluate(step.assign_add(1))

  @test_util.run_in_graph_and_eager_modes
  def testStaircase(self):
    initial_lr = 0.1
    k = 10
    decay_rate = 0.96
    step = resource_variable_ops.ResourceVariable(0)
    decayed_lr = learning_rate_decay.inverse_time_decay(
        initial_lr, step, k, decay_rate, staircase=True)

    self.evaluate(variables.global_variables_initializer())
    for i in range(k + 1):
      expected = initial_lr / (1 + decay_rate * (i // k))
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)
      self.evaluate(step.assign_add(1))


class CosineDecayTest(test_util.TensorFlowTestCase):

  def np_cosine_decay(self, step, decay_steps, alpha=0.0):
    step = min(step, decay_steps)
    completed_fraction = step / decay_steps
    decay = 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
    return (1.0 - alpha) * decay + alpha

  @test_util.run_in_graph_and_eager_modes
  def testDecay(self):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_decay.cosine_decay(initial_lr, step,
                                                    num_training_steps)
      expected = self.np_cosine_decay(step, num_training_steps)
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testAlpha(self):
    num_training_steps = 1000
    initial_lr = 1.0
    alpha = 0.1
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_decay.cosine_decay(initial_lr, step,
                                                    num_training_steps, alpha)
      expected = self.np_cosine_decay(step, num_training_steps, alpha)
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)


class CosineDecayRestartsTest(test_util.TensorFlowTestCase):

  def np_cosine_decay_restarts(self, step, decay_steps, t_mul=2.0, m_mul=1.0,
                               alpha=0.0):
    fac = 1.0
    while step >= decay_steps:
      step -= decay_steps
      decay_steps *= t_mul
      fac *= m_mul

    completed_fraction = step / decay_steps
    decay = fac * 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
    return (1.0 - alpha) * decay + alpha

  @test_util.run_in_graph_and_eager_modes
  def testDecay(self):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_decay.cosine_decay_restarts(
          initial_lr, step, num_training_steps)
      expected = self.np_cosine_decay_restarts(step, num_training_steps)
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testAlpha(self):
    num_training_steps = 1000
    initial_lr = 1.0
    alpha = 0.1
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_decay.cosine_decay_restarts(
          initial_lr, step, num_training_steps, alpha=alpha)
      expected = self.np_cosine_decay_restarts(
          step, num_training_steps, alpha=alpha)
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testMMul(self):
    num_training_steps = 1000
    initial_lr = 1.0
    m_mul = 0.9
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_decay.cosine_decay_restarts(
          initial_lr, step, num_training_steps, m_mul=m_mul)
      expected = self.np_cosine_decay_restarts(
          step, num_training_steps, m_mul=m_mul)
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testTMul(self):
    num_training_steps = 1000
    initial_lr = 1.0
    t_mul = 1.0
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_decay.cosine_decay_restarts(
          initial_lr, step, num_training_steps, t_mul=t_mul)
      expected = self.np_cosine_decay_restarts(
          step, num_training_steps, t_mul=t_mul)
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)


class LinearCosineDecayTest(test_util.TensorFlowTestCase):

  def np_linear_cosine_decay(self,
                             step,
                             decay_steps,
                             alpha=0.0,
                             beta=0.001,
                             num_periods=0.5):
    step = min(step, decay_steps)
    linear_decayed = float(decay_steps - step) / decay_steps
    fraction = 2.0 * num_periods * step / float(decay_steps)
    cosine_decayed = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return (alpha + linear_decayed) * cosine_decayed + beta

  @test_util.run_in_graph_and_eager_modes
  def testDefaultDecay(self):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_decay.linear_cosine_decay(
          initial_lr, step, num_training_steps)
      expected = self.np_linear_cosine_decay(step, num_training_steps)
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)

  @test_util.run_in_graph_and_eager_modes
  def testNonDefaultDecay(self):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_decay.linear_cosine_decay(
          initial_lr,
          step,
          num_training_steps,
          alpha=0.1,
          beta=1e-4,
          num_periods=5)
      expected = self.np_linear_cosine_decay(
          step, num_training_steps, alpha=0.1, beta=1e-4, num_periods=5)
      self.assertAllClose(self.evaluate(decayed_lr), expected, 1e-6)


class NoisyLinearCosineDecayTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testDefaultNoisyLinearCosine(self):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      # No numerical check because of noise
      decayed_lr = learning_rate_decay.noisy_linear_cosine_decay(
          initial_lr, step, num_training_steps)
      # Cannot be deterministically tested
      self.evaluate(decayed_lr)

  @test_util.run_in_graph_and_eager_modes
  def testNonDefaultNoisyLinearCosine(self):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      # No numerical check because of noise
      decayed_lr = learning_rate_decay.noisy_linear_cosine_decay(
          initial_lr,
          step,
          num_training_steps,
          initial_variance=0.5,
          variance_decay=0.1,
          alpha=0.1,
          beta=1e-4,
          num_periods=5)
      # Cannot be deterministically tested
      self.evaluate(decayed_lr)


if __name__ == "__main__":
  googletest.main()
