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

from tensorflow.contrib.autograph.converters import error_handlers
from tensorflow.contrib.autograph.core import converter_testing
from tensorflow.contrib.autograph.core import errors
from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import origin_info
from tensorflow.python.platform import test


class ErrorHandlersTest(converter_testing.TestCase):

  def test_basic(self):

    def test_fn():
      raise ValueError()

    node, ctx = self.prepare(test_fn, {})
    anno.setanno(
        node, anno.Basic.ORIGIN,
        origin_info.OriginInfo(None, 'test_function_name', 'test_code',
                               'test_comment'))
    node = error_handlers.transform(node, ctx)
    with self.compiled(node, {}) as result:
      with self.assertRaises(errors.GraphConstructionError):
        # Here we just assert that the handler works. Its correctness is
        # verified by errors_test.py.
        result.test_fn()

  def test_no_origin_annotation(self):

    def test_fn():
      raise ValueError()

    with self.converted(test_fn, error_handlers, {}) as result:
      with self.assertRaises(ValueError):
        result.test_fn()


if __name__ == '__main__':
  test.main()
