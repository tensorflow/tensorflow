# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf upgrader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tempfile
import six
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import tf_upgrade


class TestUpgrade(test_util.TensorFlowTestCase):
  """Test various APIs that have been changed in 1.0.

  We also test whether a converted file is executable. test_file_v0_11.py
  aims to exhaustively test that API changes are convertible and actually
  work when run with current TensorFlow.
  """

  def _upgrade(self, old_file_text):
    in_file = six.StringIO(old_file_text)
    out_file = six.StringIO()
    upgrader = tf_upgrade.TensorFlowCodeUpgrader()
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return count, report, errors, out_file.getvalue()

  def testParseError(self):
    _, report, unused_errors, unused_new_text = self._upgrade(
        "import tensorflow as tf\na + \n")
    self.assertTrue(report.find("Failed to parse") != -1)

  def testReport(self):
    text = "tf.mul(a, b)\n"
    _, report, unused_errors, unused_new_text = self._upgrade(text)
    # This is not a complete test, but it is a sanity test that a report
    # is generating information.
    self.assertTrue(report.find("Renamed function `tf.mul` to `tf.multiply`"))

  def testRename(self):
    text = "tf.mul(a, tf.sub(b, c))\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.multiply(a, tf.subtract(b, c))\n")

  def testReorder(self):
    text = "tf.concat(a, b)\ntf.split(a, b, c)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.concat(concat_dim=a, values=b)\n"
                     "tf.split(axis=a, num_or_size_splits=b, value=c)\n")

  def testKeyword(self):
    text = "tf.reduce_any(a, reduction_indices=[1, 2])\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.reduce_any(a, axis=[1, 2])\n")

  def testComplexExpression(self):
    text = "(foo + bar)[a].word()"
    _ = self._upgrade(text)

  def testReverse(self):
    text = "tf.reverse(a, b)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, new_text)
    self.assertEqual(errors, ["test.py:1: tf.reverse requires manual check."])

  # TODO(aselle): Explicitly not testing command line interface and process_tree
  # for now, since this is a one off utility.


class TestUpgradeFiles(test_util.TensorFlowTestCase):

  def testInplace(self):
    """Check to make sure we don't have a file system race."""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "tf.mul(a, b)\n"
    upgraded = "tf.multiply(a, b)\n"
    temp_file.write(original)
    temp_file.close()
    upgrader = tf_upgrade.TensorFlowCodeUpgrader()
    upgrader.process_file(temp_file.name, temp_file.name)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)


if __name__ == "__main__":
  test_lib.main()
