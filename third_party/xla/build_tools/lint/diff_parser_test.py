# Copyright 2023 The OpenXLA Authors.
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
# ============================================================================
from absl.testing import absltest

from xla.build_tools import test_utils
from xla.build_tools.lint import diff_parser


class ParseDiffTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    testdata = test_utils.xla_src_root() / "build_tools" / "lint" / "testdata"

    with (testdata / "bad_cc.diff").open() as f:
      cls.bad_cc_diff = f.read()

    with (testdata / "important_cc.diff").open() as f:
      cls.important_cc_diff = f.read()

    with (testdata / "crosstool.diff").open() as f:
      cls.crosstool_diff = f.read()

  def test_parse_important_cc_diff(self):
    hunks = diff_parser.parse_hunks(self.important_cc_diff)
    self.assertLen(hunks, 1)
    [hunk] = hunks
    self.assertEqual(hunk.file, "src/important.cc")
    self.assertEqual(hunk.start, 1)
    self.assertEqual(hunk.length, 3)

    expected_lines = [
        "+// Here we care if we find prohibited regexes.",
        "+std::unique_ptr<int> add(int a, int b) {",
        "+  return std::make_unique<int>(a + b);",
        "+}",
    ]

    self.assertEqual(hunk.lines, expected_lines)

  def test_parse_bad_cc_diff(self):
    hunks = diff_parser.parse_hunks(self.bad_cc_diff)
    self.assertLen(hunks, 2)

    bad_cc_hunk, important_cc_hunk = hunks

    # check bad_cc_hunk
    self.assertEqual(bad_cc_hunk.file, "src/dir/bad.cc")
    self.assertEqual(bad_cc_hunk.start, 1)
    self.assertEqual(bad_cc_hunk.length, 7)

    expected_lines = [
        "+// This code is bad!",
        "+",
        "+using Make_Unique = std::make_unique; // OK",
        "+",
        "+std::unique_ptr<int> add(int a, int b) {",
        "+  return Make_Unique<int>(a + b); // OK. Fixed now!",
        "+}",
    ]

    self.assertEqual(bad_cc_hunk.lines, expected_lines)

    # check important_cc_hunk
    self.assertEqual(important_cc_hunk.file, "src/important.cc")
    self.assertEqual(important_cc_hunk.start, 1)
    self.assertEqual(important_cc_hunk.length, 5)

    expected_lines = [
        "+// Here we care if we find prohibited regexes.",
        "+",
        "+std::unique_ptr<int> add(int a, int b) {",
        "+  return std::make_unique<int>(a + b);",
        "+}",
    ]

    self.assertEqual(important_cc_hunk.lines, expected_lines)

  def test_parse_crosstool_diff(self):
    hunks = diff_parser.parse_hunks(self.crosstool_diff)

    self.assertLen(hunks, 3)

    small_hunk, big_hunk, literal_cc_hunk = hunks

    self.assertEqual(
        small_hunk.file,
        "third_party/gpus/crosstool/cc_toolchain_config.bzl.tpl",
    )
    self.assertEqual(small_hunk.start, 24)
    self.assertEqual(small_hunk.length, 7)

    self.assertEqual(
        big_hunk.file, "third_party/gpus/crosstool/cc_toolchain_config.bzl.tpl"
    )
    self.assertEqual(big_hunk.start, 300)
    self.assertEqual(big_hunk.length, 45)

    self.assertEqual(literal_cc_hunk.file, "xla/literal.cc")
    self.assertEqual(literal_cc_hunk.start, 47)
    self.assertEqual(literal_cc_hunk.length, 7)

  def test_added_lines(self):
    hunks = diff_parser.parse_hunks(self.crosstool_diff)
    small_hunk, big_hunk, literal_cc_hunk = hunks

    line_numbers = lambda hunk: [line_no for line_no, _ in hunk.added_lines()]

    self.assertEqual(line_numbers(small_hunk), [27])
    self.assertEqual(line_numbers(big_hunk), list(range(303, 342)))
    self.assertEqual(line_numbers(literal_cc_hunk), [50])


if __name__ == "__main__":
  absltest.main()
