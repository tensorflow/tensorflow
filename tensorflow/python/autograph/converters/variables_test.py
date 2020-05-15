# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for variables module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.autograph.converters import variables
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.platform import test


class VariablesTest(converter_testing.TestCase):

  @contextlib.contextmanager
  def apply_add_one_conversion(self, fn):
    """Generates code which adds 1 to all variable reads."""
    with self.converted(fn, variables, {}) as result:
      result.ag__.__dict__['ld'] = lambda x: x + 1
      yield result

  def test_read(self):

    def test_fn(l):
      return l

    with self.apply_add_one_conversion(test_fn) as result:
      self.assertEqual(result.test_fn(1), 2)

  def test_aug_assign(self):

    def test_fn(l):
      l *= 10
      return l

    with self.apply_add_one_conversion(test_fn) as result:
      self.assertEqual(result.test_fn(1), (1 + 1) * 10 + 1)  # two reads

  def test_attribute(self):

    class TestClass(object):

      def __init__(self):
        self.v = 1

      def __add__(self, other):
        self.v += other
        return self

    def test_fn(l):
      return l.v

    tc = TestClass()
    with self.apply_add_one_conversion(test_fn) as result:
      self.assertEqual(result.test_fn(tc), 2)

  def test_subscript(self):

    class TestClass(object):

      def __init__(self):
        self.v = 1

      def __add__(self, other):
        self.v += other
        return self

      def __getitem__(self, _):
        return self.v

    def test_fn(l):
      return l[0]

    tc = TestClass()
    with self.apply_add_one_conversion(test_fn) as result:
      self.assertEqual(result.test_fn(tc), 2)

  def test_call(self):

    class TestClass(object):

      def __init__(self):
        self.v = 1

      def __add__(self, other):
        self.v += other
        return self

      def __call__(self):
        return self.v

    def test_fn(l):
      return l()

    tc = TestClass()
    with self.apply_add_one_conversion(test_fn) as result:
      self.assertEqual(result.test_fn(tc), 2)


if __name__ == '__main__':
  test.main()
