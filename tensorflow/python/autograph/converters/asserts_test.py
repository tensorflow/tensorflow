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
"""Tests for asserts module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.converters import asserts
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class AssertsTest(converter_testing.TestCase):

  def test_transform(self):

    def test_fn(a):
      assert a > 0

    node, ctx = self.prepare(test_fn, {})
    node = asserts.transform(node, ctx)

    self.assertTrue(isinstance(node.body[0].value, gast.Call))


if __name__ == '__main__':
  test.main()
