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
"""Tests for builtin_functions module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import six

from tensorflow.contrib.py2tf.converters import builtin_functions
from tensorflow.contrib.py2tf.converters import converter_test_base
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class BuiltinFunctionsTest(converter_test_base.TestCase):

  def test_len(self):

    def test_fn(a):
      return len(a)

    node = self.parse_and_analyze(test_fn, {'len': len})
    node = builtin_functions.transform(node, self.ctx)

    with self.compiled(node, array_ops.shape) as result:
      with self.test_session() as sess:
        self.assertEqual(3,
                         sess.run(
                             result.test_fn(constant_op.constant([0, 0, 0]))))

  def test_print_with_op(self):

    def test_fn(a):
      print(a)

    node = self.parse_and_analyze(test_fn, {'print': print})
    node = builtin_functions.transform(node, self.ctx)

    # Note: it's relevant not to include script_ops.py_func here, to verify
    # that tf.Print is used.
    with self.compiled(node, logging_ops.Print) as result:
      with self.test_session() as sess:
        try:
          out_capturer = six.StringIO()
          sys.stdout = out_capturer
          result.test_fn('a')
          sess.run(sess.graph.get_operations())
          self.assertEqual(out_capturer.getvalue(), 'a\n')
        finally:
          sys.stdout = sys.__stdout__

  def test_print_with_op_multiple_values(self):

    def test_fn(a, b):
      print(a, b)

    node = self.parse_and_analyze(test_fn, {'print': print})
    node = builtin_functions.transform(node, self.ctx)

    # Note: it's relevant not to include script_ops.py_func here, to verify
    # that tf.Print is used.
    with self.compiled(node, logging_ops.Print) as result:
      with self.test_session() as sess:
        try:
          out_capturer = six.StringIO()
          sys.stdout = out_capturer
          result.test_fn('a', 1)
          sess.run(sess.graph.get_operations())
          self.assertEqual(out_capturer.getvalue(), 'a 1\n')
        finally:
          sys.stdout = sys.__stdout__

  def test_print_with_py_func(self):

    def test_fn(a, b, c):
      print(a, b, c)

    node = self.parse_and_analyze(test_fn, {'print': print})
    node = builtin_functions.transform(node, self.ctx)

    # Note: it's relevant not to include logging_ops.Print here, to verify
    # that py_func is used.
    with self.compiled(node, script_ops.py_func) as result:
      with self.test_session() as sess:
        try:
          out_capturer = six.StringIO()
          sys.stdout = out_capturer
          result.test_fn('a', 1, [2, 3])
          sess.run(sess.graph.get_operations())
          self.assertEqual(out_capturer.getvalue(), 'a 1 [2, 3]\n')
        finally:
          sys.stdout = sys.__stdout__


if __name__ == '__main__':
  test.main()
