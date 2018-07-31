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
"""Tests for tf 2.0 upgrader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tempfile
import six
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import tf_upgrade_v2


class TestUpgrade(test_util.TensorFlowTestCase):
  """Test various APIs that have been changed in 2.0.

  We also test whether a converted file is executable. test_file_v1_10.py
  aims to exhaustively test that API changes are convertible and actually
  work when run with current TensorFlow.
  """

  def _upgrade(self, old_file_text):
    in_file = six.StringIO(old_file_text)
    out_file = six.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return count, report, errors, out_file.getvalue()

  def testParseError(self):
    _, report, unused_errors, unused_new_text = self._upgrade(
        "import tensorflow as tf\na + \n")
    self.assertTrue(report.find("Failed to parse") != -1)

  def testReport(self):
    text = "tf.acos(a)\n"
    _, report, unused_errors, unused_new_text = self._upgrade(text)
    # This is not a complete test, but it is a sanity test that a report
    # is generating information.
    self.assertTrue(report.find("Renamed function `tf.acos` to `tf.math.acos`"))

  def testRename(self):
    text = "tf.acos(a)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.math.acos(a)\n")
    text = "tf.rsqrt(tf.log(3.8))\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.math.rsqrt(tf.math.log(3.8))\n")


class TestUpgradeFiles(test_util.TensorFlowTestCase):

  def testInplace(self):
    """Check to make sure we don't have a file system race."""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "tf.acos(a, b)\n"
    upgraded = "tf.math.acos(a, b)\n"
    temp_file.write(original)
    temp_file.close()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    upgrader.process_file(temp_file.name, temp_file.name)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)


if __name__ == "__main__":
  test_lib.main()
