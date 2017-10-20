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
"""Helper functions for reinforcement learning in the cart-pole problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def discount_rewards(rewards, discount_rate):
  """Discout reward values with discount rate.

  Args:
    rewards: A sequence of reward values in time.
    discount_rate: (`float`) reward discounting rate (e.g., 0.95).

  Returns:
    Discounted reward values.
  """
  discounted = []
  for reward in reversed(rewards):
    discounted.append(
        (discounted[-1] if discounted else 0.0) * discount_rate + reward)
  return list(reversed(discounted))


def discount_and_normalize_rewards(reward_sequences, discount_rate):
  """Perform discounting on a number of reward sequences; then normalize values.

  Args:
    reward_sequences: an `iterable` of reward sequences.
    discount_rate: reward discounting rate (e.g., 0.95).

  Returns:
    A `list` of reward value `list`s, discounted and normalized.
  """
  discounted = []
  for sequence in reward_sequences:
    discounted.append(discount_rewards(sequence, discount_rate))
  discounted = np.array(discounted)

  # Compute overall mean and stddev.
  flattened = np.concatenate(discounted)
  mean = np.mean(flattened)
  std = np.std(flattened)
  return [((d - mean) / std) for d in discounted]
