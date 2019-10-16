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
"""Tests for the XLATestCase test fixture base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tests import xla_test
from tensorflow.python.platform import test


class XlaTestCaseTestCase(test.TestCase):

  def testManifestEmptyLineDoesNotCatchAll(self):
    manifest = """
testCaseOne
"""
    disabled_regex, _ = xla_test.parse_disabled_manifest(manifest)
    self.assertEqual(disabled_regex, "testCaseOne")

  def testManifestWholeLineCommentDoesNotCatchAll(self):
    manifest = """# I am a comment
testCaseOne
testCaseTwo
"""
    disabled_regex, _ = xla_test.parse_disabled_manifest(manifest)
    self.assertEqual(disabled_regex, "testCaseOne|testCaseTwo")


if __name__ == "__main__":
  test.main()
