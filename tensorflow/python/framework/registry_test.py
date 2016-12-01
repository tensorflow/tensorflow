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

from tensorflow.python.framework import registry
from tensorflow.python.platform import googletest


class RegistryTest(googletest.TestCase):

  class Foo(object):
    pass

  def testRegisterClass(self):
    myreg = registry.Registry('testfoo')
    with self.assertRaises(LookupError):
      myreg.lookup('Foo')
    myreg.register(RegistryTest.Foo, 'Foo')
    assert myreg.lookup('Foo') == RegistryTest.Foo

  def testRegisterFunction(self):
    myreg = registry.Registry('testbar')
    with self.assertRaises(LookupError):
      myreg.lookup('Bar')
    myreg.register(bar, 'Bar')
    assert myreg.lookup('Bar') == bar

  def testDuplicate(self):
    myreg = registry.Registry('testbar')
    myreg.register(bar, 'Bar')
    with self.assertRaises(KeyError):
      myreg.register(bar, 'Bar')


def bar():
  pass


if __name__ == '__main__':
  googletest.main()
