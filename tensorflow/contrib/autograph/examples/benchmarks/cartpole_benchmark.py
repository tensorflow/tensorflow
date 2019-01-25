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
"""A basic RL cartpole benchmark.

The RL model uses the OpenAI Gym environment to train a simple network using
the policy gradients method. The training scales the gradients for each step
by the episode's cumulative discounted reward and averages these gradients over
a fixed number of games before applying the optimization step.

For benchmarking purposes, we replace the OpenAI Gym environment to a fake
that returns random actions and rewards and never ends the episode. This way
the benchmarks compare the same amount of computation at each step.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf

from tensorflow.contrib import eager
from tensorflow.contrib.autograph.examples.benchmarks import benchmark_base
from tensorflow.python import autograph as ag
from tensorflow.python.eager import context

#
# AutoGraph implementation
#


@ag.convert()
def graph_append_discounted_rewards(destination, rewards, discount_rate):
  """Discounts episode rewards and appends them to destination."""
  ag.set_element_type(rewards, tf.float32)

  cdr = 0.0
  reverse_discounted = []
  ag.set_element_type(reverse_discounted, tf.float32)

  for i in range(len(rewards) - 1, -1, -1):
    cdr = cdr * discount_rate + rewards[i]
    cdr.set_shape(())
    reverse_discounted.append(cdr)

  retval = destination
  # Note: AutoGraph doesn't yet support reversed() so we use a loop instead.
  for i in range(len(reverse_discounted) - 1, -1, -1):
    retval.append(reverse_discounted[i])

  return retval


class GraphPolicyNetwork(tf.keras.Model):
  """Policy network for the cart-pole reinforcement learning problem.

  The forward path of the network takes an observation from the cart-pole
  environment (length-4 vector) and outputs an action.
  """

  def __init__(self, hidden_size):
    super(GraphPolicyNetwork, self).__init__()
    self._hidden_layer = tf.keras.layers.Dense(
        hidden_size, activation=tf.nn.elu)
    self._output_layer = tf.keras.layers.Dense(1)

  def call(self, inputs):
    """Calculates logits and action.

    Args:
      inputs: Observations from a step in the cart-pole environment, of shape
        `(batch_size, input_size)`

    Returns:
      logits: the logits output by the output layer. This can be viewed as the
        likelihood vales of choosing the left (0) action. Shape:
        `(batch_size, 1)`.
      actions: randomly selected actions ({0, 1}) based on the logits. Shape:
        `(batch_size, 1)`.
    """
    hidden = self._hidden_layer(inputs)
    logits = self._output_layer(hidden)

    left_prob = tf.nn.sigmoid(logits)
    action_probs = tf.concat([left_prob, 1.0 - left_prob], 1)

    actions = tf.multinomial(tf.log(action_probs), 1)
    return logits, actions

  # TODO(mdan): Move this method out of the class.
  @ag.convert()
  def train(self, cart_pole_env, optimizer, discount_rate, num_games,
            max_steps_per_game):
    var_list = tf.trainable_variables()
    grad_list = [
        tf.TensorArray(tf.float32, 0, dynamic_size=True) for _ in var_list
    ]

    step_counts = []
    discounted_rewards = []
    ag.set_element_type(discounted_rewards, tf.float32)
    ag.set_element_type(step_counts, tf.int32)

    # Note: we use a shared object, cart_pole_env here. Because calls to the
    # object's method are made through py_func, TensorFlow cannot detect its
    # data dependencies. Hence we must manually synchronize access to it
    # and ensure the control dependencies are set in such a way that
    # calls to reset(), take_one_step, etc. are made in the correct order.
    sync_counter = tf.constant(0)

    for _ in tf.range(num_games):
      with tf.control_dependencies([sync_counter]):
        obs = cart_pole_env.reset()
        with tf.control_dependencies([obs]):
          sync_counter += 1

        game_rewards = []
        ag.set_element_type(game_rewards, tf.float32)

        for step in tf.range(max_steps_per_game):
          logits, actions = self(obs)  # pylint:disable=not-callable
          logits = tf.reshape(logits, ())
          actions = tf.reshape(actions, ())

          labels = 1.0 - tf.cast(actions, tf.float32)
          loss = tf.nn.sigmoid_cross_entropy_with_logits(
              labels=labels, logits=logits)
          grads = tf.gradients(loss, var_list)

          for i in range(len(grads)):
            grad_list[i].append(grads[i])

          with tf.control_dependencies([sync_counter]):
            obs, reward, done = cart_pole_env.step(actions)
            with tf.control_dependencies([obs]):
              sync_counter += 1
            obs = tf.reshape(obs, (1, 4))

          game_rewards.append(reward)
          if reward < 0.1 or done:
            step_counts.append(step + 1)
            break

        discounted_rewards = graph_append_discounted_rewards(
            discounted_rewards, game_rewards, discount_rate)

    discounted_rewards = ag.stack(discounted_rewards)
    discounted_rewards.set_shape((None,))
    mean, variance = tf.nn.moments(discounted_rewards, [0])
    normalized_rewards = (discounted_rewards - mean) / tf.sqrt(variance)

    for i in range(len(grad_list)):
      g = ag.stack(grad_list[i])

      # This block just adjusts the shapes to match for multiplication.
      r = normalized_rewards
      if r.shape.ndims < g.shape.ndims:
        r = tf.expand_dims(r, -1)
      if r.shape.ndims < g.shape.ndims:
        r = tf.expand_dims(r, -1)

      grad_list[i] = tf.reduce_mean(g * r, axis=0)

    optimizer.apply_gradients(
        zip(grad_list, var_list), global_step=tf.train.get_global_step())

    return ag.stack(step_counts)


@ag.convert()
def graph_train_model(policy_network, cart_pole_env, optimizer, iterations):
  """Trains the policy network for a given number of iterations."""
  i = tf.constant(0)
  mean_steps_per_iteration = []
  ag.set_element_type(mean_steps_per_iteration, tf.int32)

  while i < iterations:
    steps_per_game = policy_network.train(
        cart_pole_env,
        optimizer,
        discount_rate=0.95,
        num_games=20,
        max_steps_per_game=200)
    mean_steps_per_iteration.append(tf.reduce_mean(steps_per_game))
    i += 1

  return ag.stack(mean_steps_per_iteration)


class GraphGymCartpoleEnv(object):
  """An env backed by OpenAI Gym's CartPole environment.

  Used to confirm a functional model only.
  """

  def __init__(self):
    cart_pole_env = gym.make('CartPole-v1')
    cart_pole_env.seed(0)
    cart_pole_env.reset()
    self.env = cart_pole_env

  def reset(self):
    obs = ag.utils.wrap_py_func(self.env.reset, tf.float64, ())
    obs = tf.reshape(obs, (1, 4))
    obs = tf.cast(obs, tf.float32)
    return obs

  def step(self, actions):

    def take_one_step(actions):
      obs, reward, done, _ = self.env.step(actions)
      obs = obs.astype(np.float32)
      reward = np.float32(reward)
      return obs, reward, done

    return ag.utils.wrap_py_func(take_one_step,
                                 (tf.float32, tf.float32, tf.bool), (actions,))


class GraphRandomCartpoleEnv(object):
  """An environment that returns random actions and never finishes.

  Used during benchmarking, it will cause training to run a constant number of
  steps.
  """

  def reset(self):
    return tf.random.normal((1, 4))

  def step(self, actions):
    with tf.control_dependencies([actions]):
      random_obs = tf.random.normal((1, 4))
      fixed_reward = tf.constant(0.001)
      done = tf.constant(False)
      return random_obs, fixed_reward, done


#
# Eager implementation
#


def eager_append_discounted_rewards(discounted_rewards, rewards, discount_rate):
  cdr = 0.0
  reverse_discounted = []

  for i in range(len(rewards) - 1, -1, -1):
    cdr = cdr * discount_rate + rewards[i]
    reverse_discounted.append(cdr)

  discounted_rewards.extend(reversed(reverse_discounted))
  return discounted_rewards


class EagerPolicyNetwork(tf.keras.Model):
  """Policy network for the cart-pole reinforcement learning problem.

  The forward path of the network takes an observation from the cart-pole
  environment (length-4 vector) and outputs an action.
  """

  def __init__(self, hidden_size):
    super(EagerPolicyNetwork, self).__init__()
    self._hidden_layer = tf.keras.layers.Dense(
        hidden_size, activation=tf.nn.elu)
    self._output_layer = tf.keras.layers.Dense(1)

  def call(self, inputs):
    """Calculates logits and action.

    Args:
      inputs: Observations from a step in the cart-pole environment, of shape
        `(batch_size, input_size)`

    Returns:
      logits: the logits output by the output layer. This can be viewed as the
        likelihood vales of choosing the left (0) action. Shape:
        `(batch_size, 1)`.
      actions: randomly selected actions ({0, 1}) based on the logits. Shape:
        `(batch_size, 1)`.
    """
    hidden = self._hidden_layer(inputs)
    logits = self._output_layer(hidden)

    left_prob = tf.nn.sigmoid(logits)
    action_probs = tf.concat([left_prob, 1.0 - left_prob], 1)

    self._grad_fn = eager.implicit_gradients(
        self._get_cross_entropy_and_save_actions)

    actions = tf.multinomial(tf.log(action_probs), 1)
    return logits, actions

  def _get_cross_entropy_and_save_actions(self, inputs):
    logits, actions = self(inputs)  # pylint:disable=not-callable
    self._current_actions = actions
    labels = 1.0 - tf.cast(actions, tf.float32)
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

  def train(self, cart_pole_env, optimizer, discount_rate, num_games,
            max_steps_per_game):
    grad_list = None

    step_counts = []
    discounted_rewards = []

    for _ in range(num_games):
      obs = cart_pole_env.reset()

      game_rewards = []

      for step in range(max_steps_per_game):
        grads_and_vars = self._grad_fn(tf.constant([obs], dtype=tf.float32))
        grads, var_list = zip(*grads_and_vars)
        actions = self._current_actions.numpy()[0][0]

        if grad_list is None:
          grad_list = [[g] for g in grads]
        else:
          for i in range(len(grads)):
            grad_list[i].append(grads[i])

        obs, reward, done = cart_pole_env.step(actions)

        game_rewards.append(reward)
        if reward < 0.1 or done:
          step_counts.append(step + 1)
          break

      discounted_rewards = eager_append_discounted_rewards(
          discounted_rewards, game_rewards, discount_rate)

    discounted_rewards = tf.stack(discounted_rewards)
    mean, variance = tf.nn.moments(discounted_rewards, [0])
    normalized_rewards = (discounted_rewards - mean) / tf.sqrt(variance)

    for i in range(len(grad_list)):
      g = tf.stack(grad_list[i])

      r = normalized_rewards
      while r.shape.ndims < g.shape.ndims:
        r = tf.expand_dims(r, -1)

      grad_list[i] = tf.reduce_mean(g * r, axis=0)

    optimizer.apply_gradients(
        zip(grad_list, var_list), global_step=tf.train.get_global_step())

    return tf.stack(step_counts)


def eager_train_model(policy_network, cart_pole_env, optimizer, iterations):
  """Trains the policy network for a given number of iterations."""
  mean_steps_per_iteration = []

  for _ in range(iterations):
    steps_per_game = policy_network.train(
        cart_pole_env,
        optimizer,
        discount_rate=0.95,
        num_games=20,
        max_steps_per_game=200)
    mean_steps_per_iteration.append(tf.reduce_mean(steps_per_game))

  return mean_steps_per_iteration


class EagerGymCartpoleEnv(object):
  """An env backed by OpenAI Gym's CartPole environment.

  Used to confirm a functional model only.
  """

  def __init__(self):
    cart_pole_env = gym.make('CartPole-v1')
    cart_pole_env.seed(0)
    cart_pole_env.reset()
    self.env = cart_pole_env

  def reset(self):
    return self.env.reset()

  def step(self, actions):
    obs, reward, done, _ = self.env.step(actions)
    return obs, reward, done


class EagerRandomCartpoleEnv(object):
  """An environment that returns random actions and never finishes.

  Used during benchmarking, it will cause training to run a constant number of
  steps.
  """

  def reset(self):
    return np.random.normal(size=(4,))

  def step(self, actions):
    with tf.control_dependencies([actions]):
      random_obs = np.random.normal(size=(4,))
      fixed_reward = 0.001
      done = False
      return random_obs, fixed_reward, done


def graph_demo_training():
  """Not used in the benchmark. Used to confirm a functional model."""
  with tf.Graph().as_default():
    tf.set_random_seed(0)

    network = GraphPolicyNetwork(hidden_size=5)
    network.build((1, 4))
    env = GraphGymCartpoleEnv()
    opt = tf.train.AdamOptimizer(0.05)

    train_ops = graph_train_model(network, env, opt, iterations=5)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      steps_per_iteration = sess.run(train_ops)
      for i, steps in enumerate(steps_per_iteration):
        print('Step {} iterations: {}'.format(i, steps))


def eager_demo_training():
  with context.eager_mode():
    network = EagerPolicyNetwork(hidden_size=5)
    network.build((1, 4))
    env = EagerGymCartpoleEnv()
    opt = tf.train.AdamOptimizer(0.05)

    steps_per_iteration = eager_train_model(network, env, opt, iterations=5)
    for i, steps in enumerate(steps_per_iteration):
      print('Step {} iterations: {}'.format(i, steps))


class RLCartPoleBenchmark(benchmark_base.ReportingBenchmark):
  """Actual benchmark.

  Trains the RL agent a fixed number of times, on random environments that
  result in constant number of steps.
  """

  def benchmark_cartpole(self):

    def train_session(sess, ops):
      return lambda: sess.run(ops)

    def train_eager(network, env, opt):
      return lambda: eager_train_model(network, env, opt, iterations=10)

    for model_size in (10, 100, 1000):
      with tf.Graph().as_default():
        network = GraphPolicyNetwork(hidden_size=model_size)
        network.build((1, 4))
        env = GraphRandomCartpoleEnv()
        opt = tf.train.AdamOptimizer(0.05)
        train_ops = graph_train_model(network, env, opt, iterations=10)

        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.local_variables_initializer())

          self.time_execution(('cartpole', 'autograph', model_size),
                              train_session(sess, train_ops), 20)

      with context.eager_mode():
        network = EagerPolicyNetwork(hidden_size=model_size)
        network.build((1, 4))
        env = EagerRandomCartpoleEnv()
        opt = tf.train.AdamOptimizer(0.05)

        self.time_execution(('cartpole', 'eager', model_size),
                            train_eager(network, env, opt), 20)


if __name__ == '__main__':
  tf.test.main()
