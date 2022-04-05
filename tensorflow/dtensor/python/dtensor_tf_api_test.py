# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the dtensor API under tensorflow namespace."""

from tensorflow.dtensor import python as dtensor
from tensorflow.python.platform import test


class VerifyDTensorAPITest(test.TestCase):

  def testDTensorAPI(self):

    self.assertTrue(hasattr(dtensor, "call_with_layout"))

if __name__ == "__main__":
  test.main()
