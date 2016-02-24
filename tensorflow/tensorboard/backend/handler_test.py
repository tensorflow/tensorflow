# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Tests for utility functions in handler.py.

We import and test the utility functions directly because it's easier than
starting up a server.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from tensorflow.python.platform import googletest
from tensorflow.tensorboard.backend import handler


class UniformSampleTest(googletest.TestCase):

  def testNotEnoughValues(self):
    self.assertEqual(handler._uniform_sample([1, 2, 3], 10), [1, 2, 3])

  def includesProperNumberOfValues(self):
    values = range(10)
    for count in xrange(2, len(values)):
      self.assertEqual(
          len(handler._uniform_sample(values, count)), count,
          'Sampling %d values from 10 produced the wrong number of values' %
          count)

  def testIncludesBeginningAndEnd(self):
    values = range(10)
    for count in xrange(2, len(values)):
      sampled = handler._uniform_sample(values, count)
      self.assertEqual(
          sampled[0], values[0],
          'Sampling %d values from 10 didn\'t start with the first value' %
          count)
      self.assertEqual(
          sampled[-1], values[-1],
          'Sampling %d values from 10 didn\'t end with the last value' % count)

  def testNonIntegerCountFails(self):
    with self.assertRaises(TypeError):
      handler._uniform_sample([1, 2, 3, 4], 3.14159)

    with self.assertRaises(TypeError):
      handler._uniform_sample([1, 2, 3, 4], 3.0)


if __name__ == '__main__':
  googletest.main()
