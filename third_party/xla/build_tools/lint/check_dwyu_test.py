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
import unittest

from build_tools.lint.check_dwyu import extract_targets

ALLOWED_RULES = {"cc_library", "xla_test", "xla_cc_test"}


class ExtractTargetsTest(unittest.TestCase):

  def test_cc_library(self):
    build = """\
cc_library(
    name = "foo",
    srcs = ["foo.cc"],
)
"""
    self.assertEqual(extract_targets(build, ALLOWED_RULES), ["foo"])

  def test_xla_test(self):
    build = """\
xla_test(
    name = "bar_test",
    srcs = ["bar_test.cc"],
)
"""
    self.assertEqual(extract_targets(build, ALLOWED_RULES), ["bar_test"])

  def test_xla_cc_test(self):
    build = """\
xla_cc_test(
    name = "baz_test",
    srcs = ["baz_test.cc"],
)
"""
    self.assertEqual(extract_targets(build, ALLOWED_RULES), ["baz_test"])

  def test_skips_non_allowed_rules(self):
    build = """\
cc_binary(
    name = "my_binary",
    srcs = ["main.cc"],
)

py_library(
    name = "my_lib",
    srcs = ["lib.py"],
)
"""
    self.assertEqual(extract_targets(build, ALLOWED_RULES), [])

  def test_mixed_rules(self):
    build = """\
cc_library(
    name = "lib",
    srcs = ["lib.cc"],
)

cc_binary(
    name = "bin",
    srcs = ["main.cc"],
)

xla_test(
    name = "lib_test",
    srcs = ["lib_test.cc"],
)
"""
    self.assertEqual(extract_targets(build, ALLOWED_RULES), ["lib", "lib_test"])

  def test_empty_build_file(self):
    self.assertEqual(extract_targets("", ALLOWED_RULES), [])

  def test_name_on_same_line(self):
    build = 'cc_library(name = "inline_lib", srcs = ["a.cc"])\n'
    self.assertEqual(extract_targets(build, ALLOWED_RULES), ["inline_lib"])


if __name__ == "__main__":
  unittest.main()
