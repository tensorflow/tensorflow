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
# ==============================================================================
"""Unit tests for object_identity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity


class ObjectIdentityWrapperTest(test.TestCase):

  def testWrapperNotEqualToWrapped(self):
    class SettableHash(object):

      def __init__(self):
        self.hash_value = 8675309

      def __hash__(self):
        return self.hash_value

    o = SettableHash()
    wrap1 = object_identity._ObjectIdentityWrapper(o)
    wrap2 = object_identity._ObjectIdentityWrapper(o)

    self.assertEqual(wrap1, wrap1)
    self.assertEqual(wrap1, wrap2)
    self.assertEqual(o, wrap1.unwrapped)
    self.assertEqual(o, wrap2.unwrapped)
    with self.assertRaises(TypeError):
      bool(o == wrap1)
    with self.assertRaises(TypeError):
      bool(wrap1 != o)

    self.assertNotIn(o, set([wrap1]))
    o.hash_value = id(o)
    # Since there is now a hash collision we raise an exception
    with self.assertRaises(TypeError):
      bool(o in set([wrap1]))

  def testNestFlatten(self):
    a = object_identity._ObjectIdentityWrapper('a')
    b = object_identity._ObjectIdentityWrapper('b')
    c = object_identity._ObjectIdentityWrapper('c')
    flat = nest.flatten([[[(a, b)]], c])
    self.assertEqual(flat, [a, b, c])

  def testNestMapStructure(self):
    k = object_identity._ObjectIdentityWrapper('k')
    v1 = object_identity._ObjectIdentityWrapper('v1')
    v2 = object_identity._ObjectIdentityWrapper('v2')
    struct = nest.map_structure(lambda a, b: (a, b), {k: v1}, {k: v2})
    self.assertEqual(struct, {k: (v1, v2)})


class ObjectIdentitySetTest(test.TestCase):

  def testDifference(self):

    class Element(object):
      pass

    a = Element()
    b = Element()
    c = Element()
    set1 = object_identity.ObjectIdentitySet([a, b])
    set2 = object_identity.ObjectIdentitySet([b, c])
    diff_set = set1.difference(set2)
    self.assertIn(a, diff_set)
    self.assertNotIn(b, diff_set)
    self.assertNotIn(c, diff_set)


if __name__ == '__main__':
  test.main()
