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
"""Test DistributionStrategy in the zero batch case."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent

all_combinations = combinations.combine(
    distribution=[
        strategy_combinations.one_device_strategy,
    ], mode=["graph"])


class NormalizationTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(all_combinations,
                         combinations.combine(fused=[True, False])))
  def testBNWithZeroBatchInput(self, distribution, fused):
    with distribution.scope(), self.cached_session() as sess:
      bn_list = []
      inputs = np.random.random((0, 4, 4, 3)) + 100
      targets = np.random.random((0, 4, 4, 3))
      inputs_placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 4, 4, 3])
      targets_placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 4, 4, 3])

      def step_fn(is_training, inputs, targets=None):
        bn = normalization.BatchNormalization(
            axis=3, epsilon=1e-3, momentum=0.9, fused=fused)
        bn_list.append(bn)
        outputs = bn.apply(inputs, training=is_training)
        if not is_training:
          return outputs

        loss = losses.mean_squared_error(targets, outputs)
        optimizer = gradient_descent.GradientDescentOptimizer(0.01)
        train_op = optimizer.minimize(loss)
        with ops.control_dependencies([train_op]):
          return array_ops.identity(loss)

      train_op = distribution.extended.call_for_each_replica(
          step_fn, args=(True, inputs_placeholder, targets_placeholder))
      predict_op = distribution.extended.call_for_each_replica(
          step_fn, args=(False, inputs_placeholder))
      bn = bn_list[0]

      self.evaluate(variables.global_variables_initializer())

      # Check for initial statistics and weights.
      moving_mean, moving_var = self.evaluate(
          [bn.moving_mean, bn.moving_variance])
      self.assertAllEqual([0, 0, 0], moving_mean)
      self.assertAllEqual([1, 1, 1], moving_var)

      np_gamma, np_beta = self.evaluate([bn.gamma, bn.beta])
      self.assertAllEqual([1, 1, 1], np_gamma)
      self.assertAllEqual([0, 0, 0], np_beta)

      for _ in range(100):
        np_output, _, _ = sess.run([train_op] + bn.updates, {
            inputs_placeholder: inputs,
            targets_placeholder: targets
        })
        self.assertEqual(0.0, np_output)

      # Verify that the statistics and weights are not changed after training.
      moving_mean, moving_var = self.evaluate(
          [bn.moving_mean, bn.moving_variance])
      self.assertAllEqual([0, 0, 0], moving_mean)
      self.assertAllEqual([1, 1, 1], moving_var)

      np_gamma, np_beta = self.evaluate([bn.gamma, bn.beta])
      self.assertAllEqual([1, 1, 1], np_gamma)
      self.assertAllEqual([0, 0, 0], np_beta)

      # Test inference.
      np_output = sess.run(predict_op, {inputs_placeholder: inputs})
      self.assertEqual([], np_output.tolist())


if __name__ == "__main__":
  test.main()
