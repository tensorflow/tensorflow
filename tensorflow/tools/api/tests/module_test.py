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
"""Smoke tests for tensorflow module."""

import pkgutil

import tensorflow as tf

from tensorflow.python import tf2
from tensorflow.python.platform import test


class ModuleTest(test.TestCase):

  def testCanLoadWithPkgutil(self):
    out = pkgutil.find_loader('tensorflow')
    self.assertIsNotNone(out)

  def testDocString(self):
    self.assertIn('TensorFlow', tf.__doc__)
    self.assertNotIn('Wrapper', tf.__doc__)

  def testDict(self):
    # Check that a few modules are in __dict__.
    # pylint: disable=pointless-statement
    tf.nn
    tf.keras
    tf.image
    # pylint: enable=pointless-statement
    self.assertIn('nn', tf.__dict__)
    self.assertIn('keras', tf.__dict__)
    self.assertIn('image', tf.__dict__)

  def testName(self):
    self.assertEqual('tensorflow', tf.__name__)

  def testBuiltInName(self):
    # range is a built-in name in Python. Just checking that
    # tf.range works fine.
    if tf2.enabled():
      self.assertEqual(
          'tf.Tensor([1 2 3 4 5 6 7 8 9], shape=(9,), dtype=int32)',
          str(tf.range(1, 10)))
    else:
      self.assertEqual('Tensor("range:0", shape=(9,), dtype=int32)',
                       str(tf.range(1, 10)))

  def testCompatV2HasCompatV1(self):
    # pylint: disable=pointless-statement
    tf.compat.v2.compat.v1.keras
    # pylint: enable=pointless-statement

  def testSummaryMerged(self):
    # pylint: disable=pointless-statement
    tf.summary.image
    # If we use v2 API, check for create_file_writer,
    # otherwise check for FileWriter.
    if hasattr(tf, '_major_api_version') and tf._major_api_version == 2:
      tf.summary.create_file_writer
    else:
      tf.compat.v1.summary.FileWriter
    # pylint: enable=pointless-statement

  def testPythonModuleIsHidden(self):
    self.assertNotIn('python', dir(tf))


if __name__ == '__main__':
  test.main()
