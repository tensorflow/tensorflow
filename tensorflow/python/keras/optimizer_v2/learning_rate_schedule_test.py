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

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


def _maybe_serialized(lr_decay, serialize_and_deserialize):
  if serialize_and_deserialize:
    serialized = learning_rate_schedule.serialize(lr_decay)
    return learning_rate_schedule.deserialize(serialized)
  else:
    return lr_decay


# @parameterized.named_parameters(
#     ("NotSerialized", False),
#     ("Serialized", True))
@combinations.generate(combinations.combine(serialize=[False, True],
                                            mode=["graph", "eager"]))
class LRDecayTestV2(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testContinuous(self, serialize):
    self.evaluate(variables.global_variables_initializer())
    step = 5
    decayed_lr = learning_rate_schedule.ExponentialDecay(0.05, 10, 0.96)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = .05 * 0.96**(5.0 / 10.0)
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testStaircase(self, serialize):
    if context.executing_eagerly():
      step = resource_variable_ops.ResourceVariable(0)
      self.evaluate(variables.global_variables_initializer())
      decayed_lr = learning_rate_schedule.ExponentialDecay(
          .1, 3, 0.96, staircase=True)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)

      # No change to learning rate due to staircase
      expected = .1
      self.evaluate(step.assign(1))
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

      expected = .1
      self.evaluate(step.assign(2))
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

      # Decayed learning rate
      expected = .1 * 0.96 ** (100 // 3)
      self.evaluate(step.assign(100))
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testVariables(self, serialize):
    # TODO(tanzheny, omalleyt): Fix test in eager mode.
    with ops.Graph().as_default():
      step = variables.Variable(1)
      assign_1 = step.assign(1)
      assign_2 = step.assign(2)
      assign_100 = step.assign(100)
      decayed_lr = learning_rate_schedule.ExponentialDecay(
          .1, 3, 0.96, staircase=True)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)

      self.evaluate(variables.global_variables_initializer())
      # No change to learning rate
      self.evaluate(assign_1.op)
      self.assertAllClose(self.evaluate(decayed_lr(step)), .1, 1e-6)
      self.evaluate(assign_2.op)
      self.assertAllClose(self.evaluate(decayed_lr(step)), .1, 1e-6)
      # Decayed learning rate
      self.evaluate(assign_100.op)
      expected = .1 * 0.96**(100 // 3)
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testPiecewiseConstant(self, serialize):
    x = resource_variable_ops.ResourceVariable(-999)
    decayed_lr = learning_rate_schedule.PiecewiseConstantDecay(
        [100, 110, 120], [1.0, 0.1, 0.01, 0.001])
    decayed_lr = _maybe_serialized(decayed_lr, serialize)

    self.evaluate(variables.global_variables_initializer())

    self.assertAllClose(self.evaluate(decayed_lr(x)), 1.0, 1e-6)
    self.evaluate(x.assign(100))
    self.assertAllClose(self.evaluate(decayed_lr(x)), 1.0, 1e-6)
    self.evaluate(x.assign(105))
    self.assertAllClose(self.evaluate(decayed_lr(x)), 0.1, 1e-6)
    self.evaluate(x.assign(110))
    self.assertAllClose(self.evaluate(decayed_lr(x)), 0.1, 1e-6)
    self.evaluate(x.assign(120))
    self.assertAllClose(self.evaluate(decayed_lr(x)), 0.01, 1e-6)
    self.evaluate(x.assign(999))
    self.assertAllClose(self.evaluate(decayed_lr(x)), 0.001, 1e-6)

  def testPiecewiseFunction(self, serialize):
    del serialize
    with context.eager_mode():
      v = variables.Variable(1.)
      def loss_fn():
        return v * v
      learning_rate = learning_rate_schedule.PiecewiseConstantDecay(
          [1.], [1., 0.1])
      opt = gradient_descent.SGD(learning_rate=learning_rate)

      @def_function.function
      def minimize():
        with backprop.GradientTape() as tape:
          loss = loss_fn()
        g = tape.gradient(loss, [v])
        opt.apply_gradients(list(zip(g, [v])))

      minimize()
      self.assertAllEqual(v.read_value(), -1.0)

  def testPiecewiseConstantEdgeCases(self, serialize):
    # Test casting boundaries from int32 to int64.
    x_int64 = resource_variable_ops.ResourceVariable(
        0, dtype=variables.dtypes.int64)
    boundaries, values = [1, 2, 3], [0.4, 0.5, 0.6, 0.7]
    decayed_lr = learning_rate_schedule.PiecewiseConstantDecay(
        boundaries, values)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)

    self.evaluate(variables.global_variables_initializer())
    self.assertAllClose(self.evaluate(decayed_lr(x_int64)), 0.4, 1e-6)
    self.evaluate(x_int64.assign(1))
    self.assertAllClose(self.evaluate(decayed_lr(x_int64)), 0.4, 1e-6)
    self.evaluate(x_int64.assign(2))
    self.assertAllClose(self.evaluate(decayed_lr(x_int64)), 0.5, 1e-6)
    self.evaluate(x_int64.assign(3))
    self.assertAllClose(self.evaluate(decayed_lr(x_int64)), 0.6, 1e-6)
    self.evaluate(x_int64.assign(4))
    self.assertAllClose(self.evaluate(decayed_lr(x_int64)), 0.7, 1e-6)


# @parameterized.named_parameters(
#     ("NotSerialized", False),
#     ("Serialized", True))
@combinations.generate(combinations.combine(serialize=[False, True],
                                            mode=["graph", "eager"]))
class LinearDecayTestV2(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testHalfWay(self, serialize):
    step = 5
    lr = 0.05
    end_lr = 0.0
    decayed_lr = learning_rate_schedule.PolynomialDecay(lr, 10, end_lr)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = lr * 0.5
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testEnd(self, serialize):
    step = 10
    lr = 0.05
    end_lr = 0.001
    decayed_lr = learning_rate_schedule.PolynomialDecay(lr, 10, end_lr)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = end_lr
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testHalfWayWithEnd(self, serialize):
    step = 5
    lr = 0.05
    end_lr = 0.001
    decayed_lr = learning_rate_schedule.PolynomialDecay(lr, 10, end_lr)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = (lr + end_lr) * 0.5
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testBeyondEnd(self, serialize):
    step = 15
    lr = 0.05
    end_lr = 0.001
    decayed_lr = learning_rate_schedule.PolynomialDecay(lr, 10, end_lr)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = end_lr
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testBeyondEndWithCycle(self, serialize):
    step = 15
    lr = 0.05
    end_lr = 0.001
    decayed_lr = learning_rate_schedule.PolynomialDecay(
        lr, 10, end_lr, cycle=True)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = (lr - end_lr) * 0.25 + end_lr
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)


# @parameterized.named_parameters(
#     ("NotSerialized", False),
#     ("Serialized", True))
@combinations.generate(combinations.combine(serialize=[False, True],
                                            mode=["graph", "eager"]))
class SqrtDecayTestV2(test_util.TensorFlowTestCase,
                      parameterized.TestCase):

  def testHalfWay(self, serialize):
    step = 5
    lr = 0.05
    end_lr = 0.0
    power = 0.5
    decayed_lr = learning_rate_schedule.PolynomialDecay(
        lr, 10, end_lr, power=power)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = lr * 0.5**power
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testEnd(self, serialize):
    step = 10
    lr = 0.05
    end_lr = 0.001
    power = 0.5
    decayed_lr = learning_rate_schedule.PolynomialDecay(
        lr, 10, end_lr, power=power)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = end_lr
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testHalfWayWithEnd(self, serialize):
    step = 5
    lr = 0.05
    end_lr = 0.001
    power = 0.5
    decayed_lr = learning_rate_schedule.PolynomialDecay(
        lr, 10, end_lr, power=power)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = (lr - end_lr) * 0.5**power + end_lr
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testBeyondEnd(self, serialize):
    step = 15
    lr = 0.05
    end_lr = 0.001
    power = 0.5
    decayed_lr = learning_rate_schedule.PolynomialDecay(
        lr, 10, end_lr, power=power)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = end_lr
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testBeyondEndWithCycle(self, serialize):
    step = 15
    lr = 0.05
    end_lr = 0.001
    power = 0.5
    decayed_lr = learning_rate_schedule.PolynomialDecay(
        lr, 10, end_lr, power=power, cycle=True)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = (lr - end_lr) * 0.25**power + end_lr
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)


# @parameterized.named_parameters(
#     ("NotSerialized", False),
#     ("Serialized", True))
@combinations.generate(combinations.combine(serialize=[False, True],
                                            mode=["graph", "eager"]))
class PolynomialDecayTestV2(test_util.TensorFlowTestCase,
                            parameterized.TestCase):

  def testBeginWithCycle(self, serialize):
    lr = 0.001
    decay_steps = 10
    step = 0
    decayed_lr = learning_rate_schedule.PolynomialDecay(
        lr, decay_steps, cycle=True)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)
    expected = lr
    self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)


# @parameterized.named_parameters(
#     ("NotSerialized", False),
#     ("Serialized", True))
@combinations.generate(combinations.combine(serialize=[False, True],
                                            mode=["graph", "eager"]))
class InverseDecayTestV2(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testDecay(self, serialize):
    initial_lr = 0.1
    k = 10
    decay_rate = 0.96
    step = resource_variable_ops.ResourceVariable(0)
    decayed_lr = learning_rate_schedule.InverseTimeDecay(initial_lr, k,
                                                         decay_rate)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)

    self.evaluate(variables.global_variables_initializer())
    for i in range(k + 1):
      expected = initial_lr / (1 + i / k * decay_rate)
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)
      self.evaluate(step.assign_add(1))

  def testStaircase(self, serialize):
    initial_lr = 0.1
    k = 10
    decay_rate = 0.96
    step = resource_variable_ops.ResourceVariable(0)
    decayed_lr = learning_rate_schedule.InverseTimeDecay(
        initial_lr, k, decay_rate, staircase=True)
    decayed_lr = _maybe_serialized(decayed_lr, serialize)

    self.evaluate(variables.global_variables_initializer())
    for i in range(k + 1):
      expected = initial_lr / (1 + decay_rate * (i // k))
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)
      self.evaluate(step.assign_add(1))


@combinations.generate(combinations.combine(serialize=[False, True],
                                            mode=["graph", "eager"]))
class CosineDecayTestV2(test_util.TensorFlowTestCase, parameterized.TestCase):

  def np_cosine_decay(self, step, decay_steps, alpha=0.0):
    step = min(step, decay_steps)
    completed_fraction = step / decay_steps
    decay = 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
    return (1.0 - alpha) * decay + alpha

  def testDecay(self, serialize):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_schedule.CosineDecay(initial_lr,
                                                      num_training_steps)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)
      expected = self.np_cosine_decay(step, num_training_steps)
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testAlpha(self, serialize):
    num_training_steps = 1000
    initial_lr = 1.0
    alpha = 0.1
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_schedule.CosineDecay(initial_lr,
                                                      num_training_steps,
                                                      alpha)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)
      expected = self.np_cosine_decay(step, num_training_steps, alpha)
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)


@combinations.generate(combinations.combine(serialize=[False, True],
                                            mode=["graph", "eager"]))
class CosineDecayRestartsTestV2(test_util.TensorFlowTestCase,
                                parameterized.TestCase):

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

  def testDecay(self, serialize):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_schedule.CosineDecayRestarts(
          initial_lr, num_training_steps)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)
      expected = self.np_cosine_decay_restarts(step, num_training_steps)
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testAlpha(self, serialize):
    num_training_steps = 1000
    initial_lr = 1.0
    alpha = 0.1
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_schedule.CosineDecayRestarts(
          initial_lr, num_training_steps, alpha=alpha)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)
      expected = self.np_cosine_decay_restarts(
          step, num_training_steps, alpha=alpha)
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testMMul(self, serialize):
    num_training_steps = 1000
    initial_lr = 1.0
    m_mul = 0.9
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_schedule.CosineDecayRestarts(
          initial_lr, num_training_steps, m_mul=m_mul)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)
      expected = self.np_cosine_decay_restarts(
          step, num_training_steps, m_mul=m_mul)
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testTMul(self, serialize):
    num_training_steps = 1000
    initial_lr = 1.0
    t_mul = 1.0
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_schedule.CosineDecayRestarts(
          initial_lr, num_training_steps, t_mul=t_mul)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)
      expected = self.np_cosine_decay_restarts(
          step, num_training_steps, t_mul=t_mul)
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)


@combinations.generate(combinations.combine(serialize=[False, True],
                                            mode=["graph", "eager"]))
class LinearCosineDecayTestV2(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

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

  def testDefaultDecay(self, serialize):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_schedule.LinearCosineDecay(
          initial_lr, num_training_steps)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)
      expected = self.np_linear_cosine_decay(step, num_training_steps)
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)

  def testNonDefaultDecay(self, serialize):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      decayed_lr = learning_rate_schedule.LinearCosineDecay(
          initial_lr,
          num_training_steps,
          alpha=0.1,
          beta=1e-4,
          num_periods=5)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)
      expected = self.np_linear_cosine_decay(
          step, num_training_steps, alpha=0.1, beta=1e-4, num_periods=5)
      self.assertAllClose(self.evaluate(decayed_lr(step)), expected, 1e-6)


@combinations.generate(combinations.combine(serialize=[False, True],
                                            mode=["graph", "eager"]))
class NoisyLinearCosineDecayTestV2(test_util.TensorFlowTestCase,
                                   parameterized.TestCase):

  def testDefaultNoisyLinearCosine(self, serialize):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      # No numerical check because of noise
      decayed_lr = learning_rate_schedule.NoisyLinearCosineDecay(
          initial_lr, num_training_steps)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)
      # Cannot be deterministically tested
      self.evaluate(decayed_lr(step))

  def testNonDefaultNoisyLinearCosine(self, serialize):
    num_training_steps = 1000
    initial_lr = 1.0
    for step in range(0, 1500, 250):
      # No numerical check because of noise
      decayed_lr = learning_rate_schedule.NoisyLinearCosineDecay(
          initial_lr,
          num_training_steps,
          initial_variance=0.5,
          variance_decay=0.1,
          alpha=0.1,
          beta=1e-4,
          num_periods=5)
      decayed_lr = _maybe_serialized(decayed_lr, serialize)
      # Cannot be deterministically tested
      self.evaluate(decayed_lr(step))

if __name__ == "__main__":
  googletest.main()
