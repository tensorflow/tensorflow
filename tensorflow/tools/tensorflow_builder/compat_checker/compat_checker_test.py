# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""Tests for version compatibility checker for TensorFlow Builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import unittest
from tensorflow.tools.tensorflow_builder.compat_checker import compat_checker

PATH_TO_DIR = "tensorflow/tools/tensorflow_builder/compat_checker"

USER_CONFIG_IN_RANGE = {
    "apple": ["1.0"],
    "banana": ["3"],
    "kiwi": ["2.0"],
    "watermelon": ["2.0.0"],
    "orange": ["4.1"],
    "cherry": ["1.5"],
    "cranberry": ["1.0"],
    "raspberry": ["3.0"],
    "tangerine": ["2.0.0"],
    "jackfruit": ["1.0"],
    "grapefruit": ["2.0"],
    "apricot": ["wind", "flower"],
    "grape": ["7.1"],
    "blueberry": ["3.0"]
}
USER_CONFIG_NOT_IN_RANGE = {
    "apple": ["4.0"],
    "banana": ["5"],
    "kiwi": ["3.5"],
    "watermelon": ["5.0"],
    "orange": ["3.5"],
    "cherry": ["2.0"],
    "raspberry": ["-1"],
    "cranberry": ["4.5"],
    "tangerine": ["0"],
    "jackfruit": ["5.0"],
    "grapefruit": ["2.5"],
    "apricot": ["hello", "world"],
    "blueberry": ["11.0"],
    "grape": ["7.0"],
    "cantaloupe": ["11.0"]
}
USER_CONFIG_MISSING = {
    "avocado": ["3.0"],
    "apple": [],
    "banana": ""
}


class CompatCheckerTest(unittest.TestCase):

  def setUp(self):
    """Set up test."""
    super(CompatCheckerTest, self).setUp()
    self.test_file = os.path.join(PATH_TO_DIR, "test_config.ini")

  def testWithUserConfigInRange(self):
    """Test a set of configs that are supported.

    Testing with the following combination should always return `success`:
      [1] A set of configurations that are supported and/or compatible.
      [2] `.ini` config file with proper formatting.
    """
    # Initialize compatibility checker.
    self.compat_checker = compat_checker.ConfigCompatChecker(
        USER_CONFIG_IN_RANGE, self.test_file)
    # Compatibility check should succeed.
    self.assertTrue(self.compat_checker.check_compatibility())
    # Make sure no warning or error messages are recorded.
    self.assertFalse(len(self.compat_checker.error_msg))
    # Make sure total # of successes match total # of configs.
    cnt = len(list(USER_CONFIG_IN_RANGE.keys()))
    self.assertEqual(len(self.compat_checker.successes), cnt)

  def testWithUserConfigNotInRange(self):
    """Test a set of configs that are NOT supported.

    Testing with the following combination should always return `failure`:
      [1] A set of configurations that are NOT supported and/or compatible.
      [2] `.ini` config file with proper formatting.
    """
    self.compat_checker = compat_checker.ConfigCompatChecker(
        USER_CONFIG_NOT_IN_RANGE, self.test_file)
    # Compatibility check should fail.
    self.assertFalse(self.compat_checker.check_compatibility())
    # Check error and warning messages.
    err_msg_list = self.compat_checker.failures
    self.assertTrue(len(err_msg_list))
    # Make sure total # of failures match total # of configs.
    cnt = len(list(USER_CONFIG_NOT_IN_RANGE.keys()))
    self.assertEqual(len(err_msg_list), cnt)

  def testWithUserConfigMissing(self):
    """Test a set of configs that are empty or missing specification."""
    self.compat_checker = compat_checker.ConfigCompatChecker(
        USER_CONFIG_MISSING, self.test_file)
    # With missing specification in config file, the check should
    # always fail.
    self.assertFalse(self.compat_checker.check_compatibility())


if __name__ == "__main__":
  unittest.main()
