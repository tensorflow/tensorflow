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

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.platform import test


class DeprecatedSummariesTest(test.TestCase):

  def testScalarSummary(self):
    with self.cached_session():
      c = constant_op.constant(3)
      s = logging_ops.scalar_summary('tag', c)
      self.assertEqual(s.op.type, u'ScalarSummary')

  def testHistogramSummary(self):
    with self.cached_session():
      c = constant_op.constant(3)
      s = logging_ops.histogram_summary('tag', c)
      self.assertEqual(s.op.type, u'HistogramSummary')

  def testImageSummary(self):
    with self.cached_session():
      i = array_ops.ones((5, 4, 4, 3))
      s = logging_ops.image_summary('tag', i)
      self.assertEqual(s.op.type, u'ImageSummary')

  def testAudioSummary(self):
    with self.cached_session():
      c = constant_op.constant(3.0)
      s = logging_ops.audio_summary('tag', c, sample_rate=8000)
      self.assertEqual(s.op.type, u'AudioSummaryV2')

  def testMergeSummary(self):
    with self.cached_session():
      c = constant_op.constant(3)
      a = logging_ops.scalar_summary('a', c)
      b = logging_ops.scalar_summary('b', c)
      s = logging_ops.merge_summary([a, b])
      self.assertEqual(s.op.type, u'MergeSummary')


if __name__ == '__main__':
  test.main()
