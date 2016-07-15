# pylint: disable=g-bad-file-header
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
"""Tests for Runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import summary_writer_cache


class SummaryWriterCacheTest(tf.test.TestCase):
  """SummaryWriterCache tests."""

  def _test_dir(self, test_name):
    """Create an empty dir to use for tests.

    Args:
      test_name: Name of the test.

    Returns:
      Absolute path to the test directory.
    """
    test_dir = os.path.join(self.get_temp_dir(), test_name)
    if os.path.isdir(test_dir):
      for f in glob.glob('%s/*' % test_dir):
        os.remove(f)
    else:
      os.makedirs(test_dir)
    return test_dir

  def test_cache(self):
    with tf.Graph().as_default():
      dir1 = self._test_dir('test_cache_1')
      dir2 = self._test_dir('test_cache_2')
      sw1 = summary_writer_cache.SummaryWriterCache.get(dir1)
      sw2 = summary_writer_cache.SummaryWriterCache.get(dir2)
      sw3 = summary_writer_cache.SummaryWriterCache.get(dir1)
      self.assertEqual(sw1, sw3)
      self.assertFalse(sw1 == sw2)
      sw1.close()
      sw2.close()
      events1 = glob.glob(os.path.join(dir1, 'event*'))
      self.assertTrue(events1)
      events2 = glob.glob(os.path.join(dir2, 'event*'))
      self.assertTrue(events2)
      events3 = glob.glob(os.path.join('nowriter', 'event*'))
      self.assertFalse(events3)

  def test_clear(self):
    with tf.Graph().as_default():
      dir1 = self._test_dir('test_clear')
      sw1 = summary_writer_cache.SummaryWriterCache.get(dir1)
      summary_writer_cache.SummaryWriterCache.clear()
      sw2 = summary_writer_cache.SummaryWriterCache.get(dir1)
      self.assertFalse(sw1 == sw2)


if __name__ == '__main__':
  tf.test.main()
