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
"""Tests for printing module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import six

from tensorflow.contrib.py2tf.utils import printing
from tensorflow.python.platform import test


class ContextManagersTest(test.TestCase):

  def test_call_print_tf(self):
    try:
      out_capturer = six.StringIO()
      sys.stdout = out_capturer
      with self.test_session() as sess:
        sess.run(printing.call_print('test message', 1))
        self.assertEqual(out_capturer.getvalue(), 'test message 1\n')
    finally:
      sys.stdout = sys.__stdout__

  def test_call_print_py_func(self):
    try:
      out_capturer = six.StringIO()
      sys.stdout = out_capturer
      with self.test_session() as sess:
        sess.run(printing.call_print('test message', [1, 2]))
        self.assertEqual(out_capturer.getvalue(), 'test message [1, 2]\n')
    finally:
      sys.stdout = sys.__stdout__


if __name__ == '__main__':
  test.main()
