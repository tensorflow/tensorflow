# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.lite.experimental.support.metadata.metadata_parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.lite.experimental.support.metadata import metadata_parser
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class MetadataParserTest(test_util.TensorFlowTestCase):

  def test_version_wellFormedSemanticVersion(self):
    # Validates that the version is well-formed (x.y.z).
    self.assertTrue(
        re.match('[0-9]+\\.[0-9]+\\.[0-9]+',
                 metadata_parser.MetadataParser.VERSION))


if __name__ == '__main__':
  test.main()
