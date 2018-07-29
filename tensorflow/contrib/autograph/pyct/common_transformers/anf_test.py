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
"""Tests for anf module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.pyct import compiler
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.contrib.autograph.pyct.common_transformers import anf
from tensorflow.python.platform import test


class AnfTransformerTest(test.TestCase):

  def _simple_source_info(self):
    return transformer.EntityInfo(
        source_code=None,
        source_file=None,
        namespace=None,
        arg_values=None,
        arg_types=None,
        owner_type=None)

  def test_basic(self):

    def test_function():
      a = 0
      return a

    node, _ = parser.parse_entity(test_function)
    node = anf.transform(node.body[0], self._simple_source_info())
    result, _ = compiler.ast_to_object(node)

    self.assertEqual(test_function(), result.test_function())


if __name__ == '__main__':
  test.main()
