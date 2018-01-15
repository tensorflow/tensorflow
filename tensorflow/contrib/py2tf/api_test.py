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
"""Tests for api module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf import api
from tensorflow.contrib.py2tf import config
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class ApiTest(test.TestCase):

  def test_to_graph_basic(self):
    def test_fn(x, s):
      while math_ops.reduce_sum(x) > s:
        x //= 2
      return x

    config.DEFAULT_UNCOMPILED_MODULES.add((math_ops.__name__,))
    config.COMPILED_IMPORT_STATEMENTS = (
        'from tensorflow.python.ops '
        'import control_flow_ops as tf',
    )
    compiled_fn = api.to_graph(test_fn)

    with self.test_session() as sess:
      x = compiled_fn(constant_op.constant([4, 8]), 4)
      self.assertListEqual([1, 2], sess.run(x).tolist())

  def test_to_code_basic(self):
    def test_fn(x, s):
      while math_ops.reduce_sum(x) > s:
        x /= 2
      return x

    config.DEFAULT_UNCOMPILED_MODULES.add((math_ops.__name__,))
    compiled_code = api.to_code(test_fn)

    # Just check for some key words and that it is parseable Python code.
    self.assertRegexpMatches(compiled_code, 'tf\\.while_loop')
    self.assertIsNotNone(parser.parse_str(compiled_code))


if __name__ == '__main__':
  test.main()
