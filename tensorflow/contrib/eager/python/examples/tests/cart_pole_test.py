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
"""Unit test for cart-pole reinforcement learning under eager exection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import glob
import os
import shutil
import tempfile
import time

import gym
import numpy as np

from tensorflow.contrib.eager.python.examples import cart_pole
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training


class CartPoleTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(CartPoleTest, self).setUp()
    self._tmp_logdir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self._tmp_logdir)
    super(CartPoleTest, self).tearDown()

  def testGetLogitsAndAction(self):
    hidden_size = 5
    policy_network = cart_pole.PolicyNetwork(hidden_size)

    dummy_inputs = np.array([[0.1, 0.3, 0.2, 0.5],
                             [0.0, -0.2, 0.6, -0.8]], dtype=np.float32)
    logits, actions = policy_network.forward(constant_op.constant(dummy_inputs))

    self.assertEqual((2, 1), logits.shape)
    self.assertEqual(dtypes.float32, logits.dtype)
    self.assertEqual((2, 1), actions.shape)
    self.assertEqual(dtypes.int64, actions.dtype)

  def testCrossEntropy(self):
    hidden_size = 5
    policy_network = cart_pole.PolicyNetwork(hidden_size)

    dummy_inputs = np.array([[0.1, 0.3, 0.2, 0.5],
                             [0.0, -0.2, 0.6, -0.8]], dtype=np.float32)
    cross_entropy = policy_network._get_cross_entropy_and_save_actions(
        constant_op.constant(dummy_inputs))

    self.assertEqual((2, 1), cross_entropy.shape)
    self.assertEqual(dtypes.float32, cross_entropy.dtype)

  def testPlayAGame(self):
    hidden_size = 5
    cart_pole_env = gym.make("CartPole-v0")
    cart_pole_env.seed(0)
    cart_pole_env.reset()

    device = "gpu:0" if context.context().num_gpus() > 0 else "cpu:0"
    logging.info("device = %s", device)
    with context.device(device):
      policy_network = cart_pole.PolicyNetwork(hidden_size)
      policy_network.play(cart_pole_env, max_steps=10, render=False)

  def testTrain(self):
    hidden_size = 5
    num_games_per_iteration = 5
    max_steps_per_game = 10
    discount_rate = 0.95
    learning_rate = 0.02

    cart_pole_env = gym.make("CartPole-v0")
    cart_pole_env.reset()

    device = "gpu:0" if context.context().num_gpus() > 0 else "cpu:0"
    logging.info("device = %s", device)
    with context.device(device):
      policy_network = cart_pole.PolicyNetwork(hidden_size,
                                               train_logdir=self._tmp_logdir)
      optimizer = training.AdamOptimizer(learning_rate)
      policy_network.train(
          cart_pole_env,
          optimizer,
          discount_rate,
          num_games_per_iteration,
          max_steps_per_game)
      self.assertTrue(glob.glob(os.path.join(self._tmp_logdir, "events.out.*")))


class EagerCartPoleTrainingBenchmark(test.Benchmark):

  def benchmarkEagerCartPolePolicyNetworkTraining(self):
    burn_in_iterations = 1
    benchmark_iterations = 2
    num_games_per_iteration = 10
    max_steps_per_game = 100
    discount_rate = 0.95
    learning_rate = 0.02

    cart_pole_env = gym.make("CartPole-v0")
    cart_pole_env.seed(0)
    random_seed.set_random_seed(0)
    cart_pole_env.reset()

    hidden_size = 5
    policy_network = cart_pole.PolicyNetwork(hidden_size)
    optimizer = training.AdamOptimizer(learning_rate)

    # Perform burn-in.
    for _ in xrange(burn_in_iterations):
      policy_network.train(
          cart_pole_env,
          optimizer,
          discount_rate,
          num_games_per_iteration,
          max_steps_per_game)

    gc.collect()
    start_time = time.time()
    for _ in xrange(benchmark_iterations):
      policy_network.train(
          cart_pole_env,
          optimizer,
          discount_rate,
          num_games_per_iteration,
          max_steps_per_game)
    wall_time = time.time() - start_time
    # Named "examples"_per_sec to conform with other benchmarks.
    extras = {"examples_per_sec": benchmark_iterations / wall_time}
    self.report_benchmark(
        name="EagerCartPoleReinforcementLearning",
        iters=benchmark_iterations,
        wall_time=wall_time,
        extras=extras)


if __name__ == "__main__":
  test.main()
