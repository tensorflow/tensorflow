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
"""Tests for all_renames_v2."""

import six

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import all_renames_v2


class AllRenamesV2Test(test_util.TensorFlowTestCase):

  def test_no_identity_renames(self):
    identity_renames = [
        old_name
        for old_name, new_name in six.iteritems(all_renames_v2.symbol_renames)
        if old_name == new_name
    ]
    self.assertEmpty(identity_renames)


if __name__ == "__main__":
  test_lib.main()
