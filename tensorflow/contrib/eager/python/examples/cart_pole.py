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
r"""TensorFlow Eager Execution Example: OpenAI Gym CartPole.

Solves the cart-pole problem with policy gradient-based reinforcement learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import gym
import numpy as np
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.eager.python import tfe
from tensorflow.contrib.eager.python.examples import cart_pole_helper


class PolicyNetwork(object):
  """Policy network for the cart-pole reinforcement learning problem.

  The forward path of the network takes an observation from the cart-pole
  environment (length-4 vector) and outputs an action.
  """

  def __init__(self, hidden_size, train_logdir=None):
    """Constructor of PolicyNetwork.

    Args:
      hidden_size: Size of the hidden layer, as an `int`.
      train_logdir: The directory in which summaries will be written for
        TensorBoard during training (optional).
    """
    self._hidden_layer = tf.layers.Dense(hidden_size, activation=tf.nn.elu)
    self._output_layer = tf.layers.Dense(1)

    # Gradient function.
    self._grad_fn = tfe.implicit_gradients(
        self._get_cross_entropy_and_save_actions)

    # Support for TensorBoard summaries. Once training has started, use:
    #   tensorboard --logdir=<train_logdir>
    self._summary_writer = (tfe.SummaryWriter(train_logdir) if train_logdir
                            else None)

  def forward(self, inputs):
    """Given inputs, calculate logits and action.

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

    # Probability of selecting the left action.
    left_p = tf.nn.sigmoid(logits)
    # Probabilities of selecting the left and right actions.
    left_right_ps = tf.concat([left_p, 1.0 - left_p], 1)
    # Randomly-generated actions based on the probabilities.
    actions = tf.multinomial(tf.log(left_right_ps), 1)
    return logits, actions

  def _get_cross_entropy_and_save_actions(self, inputs):
    """Given inputs, get the sigmoid cross entropy and save selection action.

    Args:
      inputs: Observation from a step in the cart-pole environment.

    Returns:
      The sigmoid cross-entropy loss given the selected action and logits, based
        on the assumption that the selected action was rewarded by the
        environment.
    """
    logits, actions = self.forward(inputs)

    # N.B.: This is an important step. We save the value of the `actions` in a
    # member variable for use with the RL environment. In classic TensorFlow
    # (non-eager execution), it is less straightfoward to access intermediate
    # computation results in this manner (c.f., `tf.Session.partial_run()`).
    self._current_actions = actions

    labels = 1.0 - tf.cast(actions, tf.float32)
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

  def train(self,
            cart_pole_env,
            optimizer,
            discount_rate,
            num_games,
            max_steps_per_game):
    """Train the PolicyNetwork by playing `num_games` games in `cart_pole_env`.

    Arguments:
      cart_pole_env: The cart-pole gym environment object.
      optimizer: A TensorFlow `Optimizer` object to be used in this training
        (e.g., `tf.train.AdamOptimizer`).
      discount_rate: Reward discounting rate.
      num_games: Number of games to run per parameter update.
      max_steps_per_game: Maximum number of steps to run in each game.

    Returns:
      Step counts from all games, as a `list` of `int`.
    """
    all_gradient_lists = []
    all_rewards = []
    for _ in xrange(num_games):
      obs = cart_pole_env.reset()
      game_rewards = []
      game_gradient_lists = []
      for _ in xrange(max_steps_per_game):
        # TODO(cais): Can we save the tf.constant() call?
        grad_list, var_list = zip(*self._grad_fn(tf.constant([obs])))
        game_gradient_lists.append(grad_list)

        action = self._current_actions.numpy()[0][0]
        obs, reward, done, _ = cart_pole_env.step(action)
        game_rewards.append(reward)
        if reward != 1.0 or done:
          break

      all_gradient_lists.append(game_gradient_lists)
      all_rewards.append(game_rewards)

    normalized_rewards = cart_pole_helper.discount_and_normalize_rewards(
        all_rewards, discount_rate)
    all_grads_and_vars = self._scale_and_average_gradients(var_list,
                                                           all_gradient_lists,
                                                           normalized_rewards)
    optimizer.apply_gradients(all_grads_and_vars)
    step_counts = [len(rewards) for rewards in all_rewards]

    if self._summary_writer:
      self._summary_writer.scalar("mean_step_count", np.mean(step_counts))
      self._summary_writer.step()

    return step_counts

  def _scale_and_average_gradients(self,
                                   variable_list,
                                   all_gradient_lists,
                                   normalized_rewards):
    """Scale gradient tensors with normalized rewards."""
    num_games = len(all_gradient_lists)
    grads_and_vars = []
    for j, var in enumerate(variable_list):
      scaled_gradients = []
      for g in xrange(int(num_games)):
        num_steps = len(all_gradient_lists[g])
        for s in xrange(num_steps):
          scaled_gradients.append(
              all_gradient_lists[g][s][j] * normalized_rewards[g][s])
      mean_scaled_gradients = sum(scaled_gradients) / len(scaled_gradients)
      grads_and_vars.append((mean_scaled_gradients, var))
    return grads_and_vars

  def play(self, cart_pole_env, max_steps=None, render=False):
    """Play a game in the cart-pole gym environment.

    Args:
      cart_pole_env: The cart-pole gym environment object.
      max_steps: Maximum number of steps to run in the game.
      render: Whether the game state is to be rendered on the screen.
    """
    if render:
      input("\nAbout to play a game with rendering. Press Enter to continue: ")

    steps = 0
    obs = cart_pole_env.reset()
    while True:
      # TODO(cais): Can we save the tf.constant() call?
      _, actions = self.forward(tf.constant([obs]))
      if render:
        cart_pole_env.render()
      obs, reward, done, _ = cart_pole_env.step(actions.numpy()[0][0])
      steps += 1
      if done or reward != 1.0 or max_steps is not None and steps >= max_steps:
        break


def main(_):
  tf.set_random_seed(0)

  cart_pole_env = gym.make("CartPole-v0")
  cart_pole_env.seed(0)
  cart_pole_env.reset()

  device = "gpu:0" if tfe.num_gpus() else "cpu:0"
  print("Using device: %s" % device)

  with tf.device(device):
    policy_network = PolicyNetwork(FLAGS.hidden_size, train_logdir=FLAGS.logdir)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    # Training loop.
    for i in xrange(FLAGS.num_iterations):
      step_counts = policy_network.train(
          cart_pole_env,
          optimizer,
          FLAGS.discount_rate,
          FLAGS.num_games_per_iteration,
          FLAGS.max_steps_per_game)
      print("Iteration %d: step counts = %s; mean = %g" % (
          i, step_counts, np.mean(step_counts)))
      sys.stdout.flush()

    # Optional playing after training, with rendering.
    if FLAGS.play_after_training:
      policy_network.play(cart_pole_env,
                          max_steps=FLAGS.max_steps_per_game,
                          render=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--hidden_size",
      type=int,
      default=5,
      help="Size of the hidden layer of the policy network.")
  parser.add_argument(
      "--discount_rate",
      type=float,
      default=0.95,
      help="Reward discounting rate.")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.05,
      help="Learning rate to be used during training.")
  parser.add_argument(
      "--num_iterations",
      type=int,
      default=100,
      help="Number of training iterations.")
  parser.add_argument(
      "--num_games_per_iteration",
      type=int,
      default=20,
      help="Number of games to run in each training iteration.")
  parser.add_argument(
      "--max_steps_per_game",
      type=int,
      default=1000,
      help="Maximum number of steps to run in each game.")
  parser.add_argument(
      "--logdir",
      type=str,
      default=None,
      help="logdir in which TensorBoard summaries will be written (optional).")
  parser.add_argument(
      "--play_after_training",
      action="store_true",
      help="Play a game after training (with rendering).")

  FLAGS, unparsed = parser.parse_known_args()
  tfe.run(main=main, argv=[sys.argv[0]] + unparsed)
