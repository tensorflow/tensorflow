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
"""Tests for enabling and disabling TF2 behavior."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized

from tensorflow.python import tf2
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


def set_environ():
  os.environ['TF2_BEHAVIOR'] = '1'


def unset_environ():
  os.environ['TF2_BEHAVIOR'] = '0'


class EnablingTF2Behavior(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(EnablingTF2Behavior, self).setUp()
    tf2._force_enable = None
    if 'TF2_BEHAVIOR' in os.environ:
      del os.environ['TF2_BEHAVIOR']

  actions = [tf2.enable, tf2.disable, set_environ, unset_environ]

  @combinations.generate(
      combinations.combine(
          action_0=actions, action_1=actions,
          action_2=actions, action_3=actions))
  def test_scenarios(self, action_0, action_1, action_2, action_3):

    def state(action, enabled, disabled):
      """Returns bool tuple (tf2_enabled, force_enabled, force_disabled)."""
      if action is tf2.enable:
        return True, True, False
      elif action is tf2.disable:
        return False, False, True
      elif action is set_environ:
        return not disabled, enabled, disabled
      elif action is unset_environ:
        return enabled, enabled, disabled
      else:
        raise ValueError('Unexpected action {}. {} are supported'.format(
            action, EnablingTF2Behavior.actions))

    action_0()
    expected, enabled, disabled = state(action_0, False, False)
    self.assertEqual(tf2.enabled(), expected)

    action_1()
    expected, enabled, disabled = state(action_1, enabled, disabled)
    self.assertEqual(tf2.enabled(), expected)

    action_2()
    expected, enabled, disabled = state(action_2, enabled, disabled)
    self.assertEqual(tf2.enabled(), expected)

    action_3()
    expected, enabled, disabled = state(action_3, enabled, disabled)
    self.assertEqual(tf2.enabled(), expected)


if __name__ == '__main__':
  test.main()
