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
"""Tests for the deprecated summary ops in tf.contrib.deprecated."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class DeprecatedSummariesTest(tf.test.TestCase):

  def testScalarSummary(self):
    with self.test_session():
      c = tf.constant(3)
      s = tf.contrib.deprecated.scalar_summary('tag', c)
      self.assertEqual(s.op.type, u'ScalarSummary')

  def testHistogramSummary(self):
    with self.test_session():
      c = tf.constant(3)
      s = tf.contrib.deprecated.histogram_summary('tag', c)
      self.assertEqual(s.op.type, u'HistogramSummary')

  def testImageSummary(self):
    with self.test_session():
      i = tf.ones((5, 4, 4, 3))
      s = tf.contrib.deprecated.image_summary('tag', i)
      self.assertEqual(s.op.type, u'ImageSummary')

  def testAudioSummary(self):
    with self.test_session():
      c = tf.constant(3.0)
      s = tf.contrib.deprecated.audio_summary('tag', c, sample_rate=8000)
      self.assertEqual(s.op.type, u'AudioSummaryV2')

  def testMergeSummary(self):
    with self.test_session():
      c = tf.constant(3)
      a = tf.contrib.deprecated.scalar_summary('a', c)
      b = tf.contrib.deprecated.scalar_summary('b', c)
      s = tf.contrib.deprecated.merge_summary([a, b])
      self.assertEqual(s.op.type, u'MergeSummary')


if __name__ == '__main__':
  tf.test.main()
