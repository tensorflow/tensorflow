# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for exposed tensorflow versions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class VersionTest(tf.test.TestCase):

  def testVersion(self):
    self.assertEqual(type(tf.__version__), str)
    self.assertEqual(type(tf.VERSION), str)
    # This pattern will need to grow as we include alpha, builds, etc.
    self.assertRegexpMatches(tf.__version__, r'^\d+\.\d+\.\w+$')
    self.assertRegexpMatches(tf.VERSION, r'^\d+\.\d+\.\w+$')

  def testGraphDefVersion(self):
    version = tf.GRAPH_DEF_VERSION
    min_consumer = tf.GRAPH_DEF_VERSION_MIN_CONSUMER
    min_producer = tf.GRAPH_DEF_VERSION_MIN_PRODUCER
    for v in version, min_consumer, min_producer:
      self.assertEqual(type(v), int)
    self.assertLessEqual(0, min_consumer)
    self.assertLessEqual(0, min_producer)
    self.assertLessEqual(min_producer, version)

  def testGitAndCompilerVersion(self):
    self.assertEqual(type(tf.__git_version__), str)
    self.assertEqual(type(tf.__compiler_version__), str)
    self.assertEqual(type(tf.GIT_VERSION), str)
    self.assertEqual(type(tf.COMPILER_VERSION), str)

if __name__ == "__main__":
  tf.test.main()
