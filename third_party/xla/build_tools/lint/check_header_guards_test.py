# Copyright 2026 The OpenXLA Authors.
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

from unittest import mock

from absl.testing import absltest

from build_tools.lint import check_header_guards


class CheckHeaderGuardsTest(absltest.TestCase):

  def _create_file(self, content):
    return self.create_tempfile("test.h", content).full_path

  def test_valid_guard(self):
    content = """
#ifndef XLA_TEST_H_
#define XLA_TEST_H_

void foo();

#endif  // XLA_TEST_H_
"""
    path = self._create_file(content)
    is_valid, _ = check_header_guards.check_file(path)
    self.assertTrue(is_valid)

  def test_missing_ifndef(self):
    content = """
#define XLA_TEST_H_

void foo();

#endif  // XLA_TEST_H_
"""
    path = self._create_file(content)
    is_valid, msg = check_header_guards.check_file(path)
    self.assertFalse(is_valid)
    self.assertIn("Missing or malformed #ifndef", msg)

  def test_mismatched_guard_name(self):
    content = """
#ifndef XLA_TEST_H_
#define XLA_WRONG_H_

void foo();

#endif  // XLA_TEST_H_
"""
    path = self._create_file(content)
    is_valid, msg = check_header_guards.check_file(path)
    self.assertFalse(is_valid)
    self.assertIn("Mismatched guard name", msg)

  def test_missing_endif_comment(self):
    content = """
#ifndef XLA_TEST_H_
#define XLA_TEST_H_

void foo();

#endif
"""
    path = self._create_file(content)
    is_valid, msg = check_header_guards.check_file(path)
    self.assertFalse(is_valid)
    self.assertIn("Missing or malformed #endif", msg)

  def test_guard_not_at_start_of_line(self):
    content = """
 // #ifndef XLA_TEST_H_
 // #define XLA_TEST_H_

void foo();

#endif  // XLA_TEST_H_
"""
    path = self._create_file(content)
    is_valid, msg = check_header_guards.check_file(path)
    self.assertFalse(is_valid)
    self.assertIn("Missing or malformed #ifndef", msg)

  def test_endif_not_at_start_of_line(self):
    content = """
#ifndef XLA_TEST_H_
#define XLA_TEST_H_

void foo();

 // #endif  // XLA_TEST_H_
"""
    path = self._create_file(content)
    is_valid, msg = check_header_guards.check_file(path)
    self.assertFalse(is_valid)
    self.assertIn("Missing or malformed #endif", msg)

  def test_get_added_header_files(self):
    with mock.patch.object(check_header_guards.subprocess, "run") as mock_run:
      mock_run.return_value.stdout = "foo.h\nbar.cc\nbaz.h\n"
      added_files = check_header_guards.get_added_header_files()
      self.assertEqual(added_files, ["foo.h", "baz.h"])
      mock_run.assert_called_once_with(
          [
              "git",
              "diff",
              "--name-only",
              "--diff-filter=ARC",
              "origin/main",
              "HEAD",
          ],
          capture_output=True,
          text=True,
          check=True,
      )


if __name__ == "__main__":
  absltest.main()
