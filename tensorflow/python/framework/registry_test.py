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

"""Tests for tensorflow.ops.registry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import registry
from tensorflow.python.platform import test


def bar():
  pass

class RegistryTest(test.TestCase, parameterized.TestCase):

  class Foo(object):
    pass

  @parameterized.parameters([Foo, bar])
  def testRegistryBasics(self, candidate):
    myreg = registry.Registry('testRegistry')
    with self.assertRaises(LookupError):
      myreg.lookup('testKey')
    myreg.register(candidate)
    self.assertEqual(myreg.lookup(candidate.__name__), candidate)
    myreg.register(candidate, "testKey")
    self.assertEqual(myreg.lookup('testKey'), candidate)
    self.assertEqual(sorted(myreg.list()),
                     sorted(["testKey", candidate.__name__]))

  def testDuplicate(self):
    myreg = registry.Registry('testbar')
    myreg.register(bar, 'Bar')
    with self.assertRaisesRegexp(
        KeyError, r'Registering two testbar with name \'Bar\'! '
        r'\(Previous registration was in [^ ]+ .*.py:[0-9]+\)'):
      myreg.register(bar, 'Bar')


if __name__ == '__main__':
  test.main()
