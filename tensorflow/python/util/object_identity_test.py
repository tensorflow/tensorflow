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
from tensorflow.python.util import object_identity


class ObjectIdentityWrapperTest(test.TestCase):

  def testWrapperNotEqualToWrapped(self):
    o = object()
    self.assertNotEqual(o, object_identity._ObjectIdentityWrapper(o))
    self.assertNotEqual(object_identity._ObjectIdentityWrapper(o), o)


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
