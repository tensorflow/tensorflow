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
"""Tests for cache module."""

from tensorflow.python.autograph.pyct import cache
from tensorflow.python.platform import test


class CacheTest(test.TestCase):

  def test_code_object_cache(self):

    def factory(x):
      def test_fn():
        return x + 1
      return test_fn

    c = cache.CodeObjectCache()

    f1 = factory(1)
    dummy = object()

    c[f1][1] = dummy

    self.assertTrue(c.has(f1, 1))
    self.assertFalse(c.has(f1, 2))
    self.assertIs(c[f1][1], dummy)
    self.assertEqual(len(c), 1)

    f2 = factory(2)

    self.assertTrue(c.has(f2, 1))
    self.assertIs(c[f2][1], dummy)
    self.assertEqual(len(c), 1)

  def test_unbound_instance_cache(self):

    class TestClass(object):

      def method(self):
        pass

    c = cache.UnboundInstanceCache()

    o1 = TestClass()
    dummy = object()

    c[o1.method][1] = dummy

    self.assertTrue(c.has(o1.method, 1))
    self.assertFalse(c.has(o1.method, 2))
    self.assertIs(c[o1.method][1], dummy)
    self.assertEqual(len(c), 1)

    o2 = TestClass()

    self.assertTrue(c.has(o2.method, 1))
    self.assertIs(c[o2.method][1], dummy)
    self.assertEqual(len(c), 1)


if __name__ == '__main__':
  test.main()
