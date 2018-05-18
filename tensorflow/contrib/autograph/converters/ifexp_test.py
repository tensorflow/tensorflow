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
"""Tests for ifexp module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph import utils
from tensorflow.contrib.autograph.converters import converter_test_base
from tensorflow.contrib.autograph.converters import ifexp
from tensorflow.python.platform import test


class IfExpTest(converter_test_base.TestCase):

  def compiled_fn(self, test_fn, *args):
    node = self.parse_and_analyze(test_fn, {})
    node = ifexp.transform(node, self.ctx)
    module = self.compiled(node, *args)
    return module

  def test_simple(self):

    def test_fn(x):
      return 1 if x else 0

    with self.compiled_fn(test_fn) as result:
      result.autograph_util = utils
      for x in [0, 1]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_fn(self):

    def f(x):
      return 3 * x

    def test_fn(x):
      y = f(x * x if x > 0 else x)
      return y

    with self.compiled_fn(test_fn) as result:
      result.autograph_util = utils
      result.f = f
      for x in [-2, 2]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_exp(self):

    def test_fn(x):
      return x * x if x > 0 else x

    with self.compiled_fn(test_fn) as result:
      result.autograph_util = utils
      for x in [-2, 2]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_nested(self):

    def test_fn(x):
      return x * x if x > 0 else x if x else 1

    with self.compiled_fn(test_fn) as result:
      result.autograph_util = utils
      for x in [-2, 0, 2]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_in_cond(self):

    def test_fn(x):
      if x > 0:
        return x * x if x < 5 else x * x * x
      return -x

    with self.compiled_fn(test_fn) as result:
      result.autograph_util = utils
      for x in [-2, 2, 5]:
        self.assertEqual(test_fn(x), result.test_fn(x))

  def test_assign_in_cond(self):

    def test_fn(x):
      if x > 0:
        x = -x if x < 5 else x
      return x

    with self.compiled_fn(test_fn) as result:
      result.autograph_util = utils
      for x in [-2, 2, 5]:
        self.assertEqual(test_fn(x), result.test_fn(x))


if __name__ == '__main__':
  test.main()
