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
"""Tests for for_canonicalization module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.converters import control_flow
from tensorflow.contrib.py2tf.converters import converter_test_base
from tensorflow.contrib.py2tf.converters import for_canonicalization
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.python.platform import test


class TestNamer(control_flow.SymbolNamer):

  def new_symbol(self, name_root, _):
    return name_root


class ControlFlowTest(converter_test_base.TestCase):

  def test_basic_for(self):

    def test_fn(l):
      s = 0
      for e in l:
        s += e
      return s

    node = self.parse_and_analyze(test_fn, {})
    node = for_canonicalization.transform(node, TestNamer())
    result = compiler.ast_to_object(node)

    l = [1, 2, 3]
    self.assertEqual(test_fn(l), result.test_fn(l))
    l = []
    self.assertEqual(test_fn(l), result.test_fn(l))


if __name__ == '__main__':
  test.main()
