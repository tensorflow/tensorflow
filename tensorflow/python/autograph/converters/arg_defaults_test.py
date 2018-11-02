# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for arg_defaults module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import arg_defaults
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.platform import test


class ArgDefaultsTransformerTest(converter_testing.TestCase):

  def assertTransformedFirstLineIs(self, node, expected):
    self.assertEqual(compiler.ast_to_source(node).split('\n')[0], expected)

  def test_no_args(self):

    def test_fn():
      pass

    node, ctx = self.prepare(test_fn, {})
    node = arg_defaults.transform(node, ctx)
    self.assertTransformedFirstLineIs(node, 'def test_fn():')

  def test_no_defaults(self):

    def test_fn(a, b, *c, **e):
      return a, b, c, e

    node, ctx = self.prepare(test_fn, {})
    node = arg_defaults.transform(node, ctx)
    self.assertTransformedFirstLineIs(node, 'def test_fn(a, b, *c, **e):')

  # TODO(mdan): Add kwonly-arg tests when PY2 is no longer supported.

  def test_arg_defaults(self):

    def test_fn(a, b=1, c=2):
      return a, b, c

    node, ctx = self.prepare(test_fn, {})
    node = arg_defaults.transform(node, ctx)
    self.assertTransformedFirstLineIs(node, 'def test_fn(a, b=None, c=None):')

  def test_arg_defaults_with_vararg(self):

    def test_fn(a, b=1, *c):  # pylint: disable=keyword-arg-before-vararg
      return a, b, c

    node, ctx = self.prepare(test_fn, {})
    node = arg_defaults.transform(node, ctx)
    self.assertTransformedFirstLineIs(node, 'def test_fn(a, b=None, *c):')


if __name__ == '__main__':
  test.main()
