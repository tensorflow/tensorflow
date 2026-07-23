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
    result = extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "foo")
    self.assertEqual(result[0][1], {"foo.cc"})

  def test_xla_test(self):
    build = """\
xla_test(
    name = "bar_test",
    srcs = ["bar_test.cc"],
)
"""
    result = extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "bar_test")
    self.assertEqual(result[0][1], {"bar_test.cc"})

  def test_xla_cc_test(self):
    build = """\
xla_cc_test(
    name = "baz_test",
    srcs = ["baz_test.cc"],
)
"""
    result = extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "baz_test")
    self.assertEqual(result[0][1], {"baz_test.cc"})

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
    result = extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0][0], "lib")
    self.assertEqual(result[1][0], "lib_test")

  def test_empty_build_file(self):
    self.assertEqual(extract_targets("", ALLOWED_RULES), [])

  def test_name_on_same_line(self):
    build = 'cc_library(name = "inline_lib", srcs = ["a.cc"])\n'
    result = extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "inline_lib")
    self.assertEqual(result[0][1], {"a.cc"})

  def test_srcs_and_hdrs(self):
    build = """\
cc_library(
    name = "mylib",
    srcs = ["mylib.cc"],
    hdrs = ["mylib.h"],
)
"""
    result = extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "mylib")
    self.assertEqual(result[0][1], {"mylib.cc", "mylib.h"})

  def test_multiple_srcs(self):
    build = """\
cc_library(
    name = "multi",
    srcs = [
        "a.cc",
        "b.cc",
    ],
    hdrs = [
        "a.h",
        "b.h",
    ],
)
"""
    result = extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "multi")
    self.assertEqual(result[0][1], {"a.cc", "b.cc", "a.h", "b.h"})


if __name__ == "__main__":
  unittest.main()
