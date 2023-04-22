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
"""Tests that TF2_BEHAVIOR=1 enables cfv2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["TF2_BEHAVIOR"] = "1"

from tensorflow.python import tf2  # pylint: disable=g-import-not-at-top
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class ControlFlowV2EnableTest(test.TestCase):

  def testIsEnabled(self):
    self.assertTrue(tf2.enabled())
    self.assertTrue(control_flow_util.ENABLE_CONTROL_FLOW_V2)


if __name__ == "__main__":
  googletest.main()
