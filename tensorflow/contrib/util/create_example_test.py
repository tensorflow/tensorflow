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
"""Tests for the Example creation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import util
from tensorflow.core.example import example_pb2
from tensorflow.python.platform import googletest


class CreateExampleTest(googletest.TestCase):

  def testCreateExample_empty(self):
    self.assertEqual(util.create_example(), example_pb2.Example())

    # np.asarray([]) == np.array([], dtype=np.float64), but the dtype should not
    # matter here.
    actual = util.create_example(foo=[], bar=())
    expected = example_pb2.Example()
    expected.features.feature['foo'].float_list.value.extend([])
    expected.features.feature['bar'].float_list.value.extend([])
    self.assertEqual(actual, expected)

  def testCreateExample_scalars(self):
    actual = util.create_example(foo=3, bar=4.2, baz='x', qux=b'y')
    expected = example_pb2.Example()
    expected.features.feature['foo'].int64_list.value.append(3)
    # 4.2 cannot be represented exactly in floating point.
    expected.features.feature['bar'].float_list.value.append(np.float32(4.2))
    expected.features.feature['baz'].bytes_list.value.append(b'x')
    expected.features.feature['qux'].bytes_list.value.append(b'y')
    self.assertEqual(actual, expected)

  def testCreateExample_listContainingString(self):
    actual = util.create_example(foo=[3, 4.2, 'foo'])
    # np.asarray([3, 4.2, 'foo']) == np.array(['3', '4.2', 'foo'])
    expected = example_pb2.Example()
    expected.features.feature['foo'].bytes_list.value.extend(
        [b'3', b'4.2', b'foo'])
    self.assertEqual(actual, expected)

  def testCreateExample_lists_tuples_ranges(self):
    actual = util.create_example(
        foo=[1, 2, 3, 4, 5], bar=(0.5, 0.25, 0.125), baz=range(3))
    expected = example_pb2.Example()
    expected.features.feature['foo'].int64_list.value.extend([1, 2, 3, 4, 5])
    expected.features.feature['bar'].float_list.value.extend([0.5, 0.25, 0.125])
    expected.features.feature['baz'].int64_list.value.extend([0, 1, 2])
    self.assertEqual(actual, expected)

  def testCreateExample_ndarrays(self):
    a = np.random.random((3, 4, 5)).astype(np.float32)
    b = np.random.randint(low=1, high=10, size=(6, 5, 4))
    actual = util.create_example(A=a, B=b)
    expected = example_pb2.Example()
    expected.features.feature['A'].float_list.value.extend(a.ravel())
    expected.features.feature['B'].int64_list.value.extend(b.ravel())
    self.assertEqual(actual, expected)

  def testCreateExample_unicode(self):
    actual = util.create_example(A=[u'\u4242', u'\u5555'])
    expected = example_pb2.Example()
    expected.features.feature['A'].bytes_list.value.extend(
        [u'\u4242'.encode('utf-8'), u'\u5555'.encode('utf-8')])
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  googletest.main()
