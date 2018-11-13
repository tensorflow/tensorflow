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
"""Tests for error_handlers module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.converters import error_handlers
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.core import errors
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.platform import test


class ErrorHandlersTest(converter_testing.TestCase):

  def test_basic(self):

    def test_fn():
      raise ValueError()

    with self.converted(test_fn, error_handlers, {}) as result:
      with self.assertRaises(errors.GraphConstructionError):
        # Here we just assert that the handler works.
        result.test_fn()

  def test_no_origin_annotation(self):

    def test_fn(x):
      return x + 1

    node, ctx = self.prepare(test_fn, {})
    anno.delanno(node, anno.Basic.ORIGIN)
    node = error_handlers.transform(node, ctx)
    self.assertIsInstance(node.body[0], gast.Return)


if __name__ == '__main__':
  test.main()
