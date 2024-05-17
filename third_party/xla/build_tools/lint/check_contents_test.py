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
from xla.build_tools.lint import check_contents
from xla.build_tools.lint import diff_parser


class CheckDiffsTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    testdata = test_utils.xla_src_root() / "build_tools" / "lint" / "testdata"
    with (testdata / "bad_cc.diff").open() as f:
      cls.bad_cc_hunks = diff_parser.parse_hunks(f.read())

    with (testdata / "important_cc.diff").open() as f:
      cls.important_cc_hunks = diff_parser.parse_hunks(f.read())

  def test_check_good_diff(self):

    locs = check_contents.check_diffs(
        self.bad_cc_hunks,
        prohibited_regex="Make_Unique",
        suppression_regex="OK",
    )
    self.assertEmpty(locs, 0)

  def test_check_suppressed_diff_without_suppressions(self):

    locs = check_contents.check_diffs(
        self.bad_cc_hunks, prohibited_regex="Make_Unique"
    )

    expected_locs = [
        check_contents.RegexLocation(
            path="src/dir/bad.cc",
            line_number=3,
            line_contents="using Make_Unique = std::make_unique; // OK",
            matched_text="Make_Unique",
        ),
        check_contents.RegexLocation(
            path="src/dir/bad.cc",
            line_number=6,
            line_contents="  return Make_Unique<int>(a + b); // OK. Fixed now!",
            matched_text="Make_Unique",
        ),
    ]

    self.assertEqual(locs, expected_locs)

  def test_check_suppressed_diff_with_path_regexes(self):
    filtered_hunks = check_contents.filter_hunks_by_path(
        self.bad_cc_hunks,
        path_regexes=["src/important\\..*"],
        path_regex_exclusions=[],
    )

    self.assertLen(filtered_hunks, 1)

    locs = check_contents.check_diffs(
        filtered_hunks, prohibited_regex="Make_Unique"
    )

    self.assertEmpty(locs)

  def test_check_suppressed_diff_with_exclusions(self):
    filtered_hunks = check_contents.filter_hunks_by_path(
        self.bad_cc_hunks,
        path_regexes=[],
        path_regex_exclusions=["src/dir/.*"],
    )

    self.assertLen(filtered_hunks, 1)

    locs = check_contents.check_diffs(
        filtered_hunks, prohibited_regex="Make_Unique"
    )

    self.assertEmpty(locs)

  def test_check_suppressed_diff_with_suppression(self):

    filtered_hunks = check_contents.filter_hunks_by_path(
        self.bad_cc_hunks, path_regexes=[], path_regex_exclusions=[]
    )

    # filtering without path_regex(_exclusions) is a noop
    self.assertEqual(self.bad_cc_hunks, filtered_hunks)

    locs = check_contents.check_diffs(
        filtered_hunks, prohibited_regex="Make_Unique", suppression_regex="OK"
    )

    self.assertEmpty(locs)


if __name__ == "__main__":
  absltest.main()
