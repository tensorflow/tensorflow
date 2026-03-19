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

from absl.testing import parameterized

from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.platform import _pywrap_tf2
from tensorflow.python.platform import test


class EnablingTF2Behavior(test.TestCase, parameterized.TestCase):

  def __init__(self, methodName):
    super().__init__(methodName)
    self._set_default_seed = False

  @combinations.generate(test_base.v1_only_combinations())
  def test_tf1_enable_tf2_behaviour(self):
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

    v2_compat.enable_v2_behavior()
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())

    v2_compat.disable_v2_behavior()
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

  @combinations.generate(test_base.v1_only_combinations())
  def test_tf1_disable_tf2_behaviour(self):
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

    v2_compat.disable_v2_behavior()
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

    v2_compat.enable_v2_behavior()
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())

  @combinations.generate(test_base.v2_only_combinations())
  def test_tf2_enable_tf2_behaviour(self):
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())

    v2_compat.enable_v2_behavior()
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())

    v2_compat.disable_v2_behavior()
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

  @combinations.generate(test_base.v2_only_combinations())
  def test_tf2_disable_tf2_behaviour(self):
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())

    v2_compat.disable_v2_behavior()
    self.assertFalse(tf2.enabled())
    self.assertFalse(_pywrap_tf2.is_enabled())

    v2_compat.enable_v2_behavior()
    self.assertTrue(tf2.enabled())
    self.assertTrue(_pywrap_tf2.is_enabled())


if __name__ == '__main__':
  test.main()
