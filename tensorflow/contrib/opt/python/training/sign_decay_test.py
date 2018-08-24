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
"""Tests for sign_decay."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.contrib.opt.python.training import sign_decay
from tensorflow.python.platform import test


def py_linear_decay_fn(decay_steps):

  def linear_decay(step):
    step = min(step, decay_steps)
    return float(decay_steps - step) / decay_steps

  return linear_decay


def py_cosine_decay_fn(decay_steps, num_periods=0.5, zero_after=None):

  def cosine_decay(step):
    step = min(step, decay_steps)
    fraction = 2.0 * num_periods * step / float(decay_steps)
    if zero_after is not None and fraction >= 2 * zero_after:
      return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * fraction))

  return cosine_decay


def py_restart_decay_fn(decay_steps, num_periods=1, zero_after=None):

  def restart_decay(step):
    step = min(step, decay_steps)
    tmp = num_periods * step / float(decay_steps)
    fraction = (
        num_periods * step % decay_steps) / float(decay_steps)
    if zero_after is not None and tmp >= zero_after:
      return 0
    return 0.5 * (1.0 + math.cos(math.pi * fraction))

  return restart_decay


class SignDecaysTest(test.TestCase):

  def testLinearDecay(self):
    num_training_steps = 1000
    linear_decay_fn = sign_decay.get_linear_decay_fn(num_training_steps)

    for step in range(0, 1000, 100):
      with self.cached_session():
        tf_decayed = linear_decay_fn(step).eval()
        py_decayed = py_linear_decay_fn(num_training_steps)(step)
        self.assertAlmostEqual(tf_decayed, py_decayed, places=4)

  def testCosineDecay(self):
    num_training_steps = 1000
    cosine_decay_fn = sign_decay.get_cosine_decay_fn(num_training_steps)
    cosine_decay_2_fn = sign_decay.get_cosine_decay_fn(
        num_training_steps, num_periods=5, zero_after=2)

    for step in range(0, 1000, 100):
      with self.cached_session():
        tf_decayed = cosine_decay_fn(step).eval()
        py_decayed = py_cosine_decay_fn(num_training_steps)(step)
        self.assertAlmostEqual(tf_decayed, py_decayed, places=4)

        tf_decayed = cosine_decay_2_fn(step).eval()
        py_decayed = py_cosine_decay_fn(
            num_training_steps, num_periods=5, zero_after=2)(step)
        self.assertAlmostEqual(tf_decayed, py_decayed, places=4)

  def testRestartDecay(self):
    num_training_steps = 1000
    restart_decay_fn = sign_decay.get_restart_decay_fn(num_training_steps)
    restart_decay_2_fn = sign_decay.get_restart_decay_fn(
        num_training_steps, num_periods=5, zero_after=2)

    for step in range(0, 1000, 100):
      with self.cached_session():
        tf_decayed = restart_decay_fn(step).eval()
        py_decayed = py_restart_decay_fn(num_training_steps)(step)
        self.assertAlmostEqual(tf_decayed, py_decayed, places=4)

        tf_decayed = restart_decay_2_fn(step).eval()
        py_decayed = py_restart_decay_fn(
            num_training_steps, num_periods=5, zero_after=2)(step)
        self.assertAlmostEqual(tf_decayed, py_decayed, places=4)


if __name__ == "__main__":
  test.main()
