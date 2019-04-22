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
"""Tests for tensorflow.python.util.deprecation_wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import types

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation_wrapper
from tensorflow.python.util import tf_inspect


class MockModule(types.ModuleType):
  __file__ = 'test.py'


class DeprecationWrapperTest(test.TestCase):

  def testWrapperIsAModule(self):
    module = MockModule('test')
    wrapped_module = deprecation_wrapper.DeprecationWrapper(
        module, 'test', {'bar': 'bar2', 'baz': 'compat.v1.baz'})
    self.assertTrue(tf_inspect.ismodule(wrapped_module))

  @test.mock.patch.object(logging, 'warning', autospec=True)
  def testDeprecationWarnings(self, mock_warning):
    module = MockModule('test')
    module.foo = 1
    module.bar = 2
    module.baz = 3

    wrapped_module = deprecation_wrapper.DeprecationWrapper(
        module, 'test', {'bar': 'bar2', 'baz': 'compat.v1.baz'})
    self.assertTrue(tf_inspect.ismodule(wrapped_module))

    self.assertEqual(0, mock_warning.call_count)
    bar = wrapped_module.bar
    self.assertEqual(1, mock_warning.call_count)
    foo = wrapped_module.foo
    self.assertEqual(1, mock_warning.call_count)
    baz = wrapped_module.baz
    self.assertEqual(2, mock_warning.call_count)
    baz = wrapped_module.baz
    self.assertEqual(2, mock_warning.call_count)

    # Check that values stayed the same
    self.assertEqual(module.foo, foo)
    self.assertEqual(module.bar, bar)
    self.assertEqual(module.baz, baz)


if __name__ == '__main__':
  test.main()
