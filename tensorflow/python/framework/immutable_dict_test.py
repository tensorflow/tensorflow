# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.framework.immutable_dict."""

from absl.testing import parameterized

from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class ImmutableDictTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testGetItem(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(d['x'], 1)
    self.assertEqual(d['y'], 2)
    with self.assertRaises(KeyError):
      d['z']  # pylint: disable=pointless-statement

  def testIter(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(set(iter(d)), set(['x', 'y']))

  def testContains(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertIn('x', d)
    self.assertIn('y', d)
    self.assertNotIn('z', d)

  def testLen(self):
    d1 = immutable_dict.ImmutableDict({})
    self.assertLen(d1, 0)  # pylint: disable=g-generic-assert
    d2 = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertLen(d2, 2)

  def testRepr(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    s = repr(d)
    self.assertTrue(s == "ImmutableDict({'x': 1, 'y': 2})" or
                    s == "ImmutableDict({'y': 1, 'x': 2})")

  def testGet(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(d.get('x'), 1)
    self.assertEqual(d.get('y'), 2)
    self.assertIsNone(d.get('z'))
    self.assertEqual(d.get('z', 'Foo'), 'Foo')

  def testKeys(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(set(d.keys()), set(['x', 'y']))

  def testValues(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(set(d.values()), set([1, 2]))

  def testItems(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(set(d.items()), set([('x', 1), ('y', 2)]))

  def testEqual(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(d, {'x': 1, 'y': 2})

  def testNotEqual(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertNotEqual(d, {'x': 1})

  def testSetItemFails(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    with self.assertRaises(TypeError):
      d['x'] = 5  # pylint: disable=unsupported-assignment-operation
    with self.assertRaises(TypeError):
      d['z'] = 5  # pylint: disable=unsupported-assignment-operation

  def testDelItemFails(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    with self.assertRaises(TypeError):
      del d['x']  # pylint: disable=unsupported-delete-operation
    with self.assertRaises(TypeError):
      del d['z']  # pylint: disable=unsupported-delete-operation


if __name__ == '__main__':
  googletest.main()
