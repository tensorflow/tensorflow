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
"""Tests for print_functions module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.converters import converter_test_base
from tensorflow.contrib.py2tf.converters import print_functions
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.python.platform import test


class PrintFunctionsTest(converter_test_base.TestCase):

  def test_transform(self):

    def test_fn(a):
      print(a)

    node = self.parse_and_analyze(test_fn, {'print': print})
    node = print_functions.transform(node)
    result = compiler.ast_to_object(node)

    result.test_fn('a')
    self.assertTrue(isinstance(node.body[0].body[0].value, gast.Call))


if __name__ == '__main__':
  test.main()
