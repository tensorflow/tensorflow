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
"""Tests for dataset options utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.util import options
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class _TestOptions(options.OptionsBase):
  x = options.create_option(
      name="x",
      ty=int,
      docstring="the answer to everything",
      default_factory=lambda: 42)
  y = options.create_option(
      name="y", ty=float, docstring="a tasty pie", default_factory=lambda: 3.14)


class _NestedTestOptions(options.OptionsBase):
  opts = options.create_option(
      name="opts", ty=_TestOptions, docstring="nested options")


class OptionsTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testDocumentation(self):
    self.assertEqual(_TestOptions.x.__doc__, "the answer to everything")
    self.assertEqual(_TestOptions.y.__doc__, "a tasty pie")

  @combinations.generate(test_base.default_test_combinations())
  def testCreateOption(self):
    opts = _TestOptions()
    self.assertEqual(opts.x, 42)
    self.assertEqual(opts.y, 3.14)
    self.assertIsInstance(opts.x, int)
    self.assertIsInstance(opts.y, float)
    opts.x = 0
    self.assertEqual(opts.x, 0)
    with self.assertRaises(TypeError):
      opts.x = 3.14
    opts.y = 0.0
    self.assertEqual(opts.y, 0.0)
    with self.assertRaises(TypeError):
      opts.y = 42

  @combinations.generate(test_base.default_test_combinations())
  def testMergeOptions(self):
    options1, options2 = _TestOptions(), _TestOptions()
    with self.assertRaises(ValueError):
      options.merge_options()
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.x, 42)
    self.assertEqual(merged_options.y, 3.14)
    options1.x = 0
    options2.y = 0.0
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.x, 0)
    self.assertEqual(merged_options.y, 0.0)

  @combinations.generate(test_base.default_test_combinations())
  def testMergeNestedOptions(self):
    options1, options2 = _NestedTestOptions(), _NestedTestOptions()
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.opts, None)
    options1.opts = _TestOptions()
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.opts, _TestOptions())
    options2.opts = _TestOptions()
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.opts, _TestOptions())
    options1.opts.x = 0
    options2.opts.y = 0.0
    merged_options = options.merge_options(options1, options2)
    self.assertEqual(merged_options.opts.x, 0)
    self.assertEqual(merged_options.opts.y, 0.0)

  @combinations.generate(test_base.default_test_combinations())
  def testMergeOptionsInvalid(self):
    with self.assertRaises(TypeError):
      options.merge_options(0)
    options1, options2 = _TestOptions(), _NestedTestOptions()
    with self.assertRaises(TypeError):
      options.merge_options(options1, options2)

  @combinations.generate(test_base.default_test_combinations())
  def testNoSpuriousAttrs(self):
    test_options = _TestOptions()
    with self.assertRaises(AttributeError):
      test_options.wrong_attr = True
    with self.assertRaises(AttributeError):
      _ = test_options.wrong_attr


if __name__ == "__main__":
  test.main()
