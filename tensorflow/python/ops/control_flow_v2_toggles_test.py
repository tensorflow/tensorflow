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
# ==============================================================================
"""Tests for control_flow_v2_toggles.py."""

from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class ControlFlowV2TogglesTest(test.TestCase):

  def testOutputAllIntermediates(self):
    self.assertIsNone(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)
    control_flow_v2_toggles.output_all_intermediates(True)
    self.assertTrue(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)
    control_flow_v2_toggles.output_all_intermediates(False)
    self.assertFalse(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)
    control_flow_v2_toggles.output_all_intermediates(None)
    self.assertIsNone(
        control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE)


if __name__ == '__main__':
  googletest.main()
