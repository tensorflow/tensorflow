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
import contextlib
import io
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from build_tools.lint import check_dwyu

ALLOWED_RULES = set(check_dwyu.DEFAULT_ALLOWED_RULES)


class ExtractTargetsTest(unittest.TestCase):

  def test_cc_library(self):
    build = """\
cc_library(
    name = "foo",
    srcs = ["foo.cc"],
)
"""
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
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
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
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
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "baz_test")
    self.assertEqual(result[0][1], {"baz_test.cc"})

  def test_cc_binary(self):
    build = """\
cc_binary(
    name = "my_binary",
    srcs = ["main.cc"],
)
"""
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "my_binary")
    self.assertEqual(result[0][1], {"main.cc"})

  def test_cuda_library(self):
    build = """\
cuda_library(
    name = "my_cuda_lib",
    srcs = ["kernel.cu.cc"],
    hdrs = ["kernel.h"],
)
"""
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "my_cuda_lib")
    self.assertEqual(result[0][1], {"kernel.cu.cc", "kernel.h"})

  def test_skips_non_allowed_rules(self):
    build = """\
py_binary(
    name = "my_binary",
    srcs = ["main.py"],
)

py_library(
    name = "my_lib",
    srcs = ["lib.py"],
)
"""
    self.assertEqual(check_dwyu.extract_targets(build, ALLOWED_RULES), [])

  def test_mixed_rules(self):
    build = """\
cc_library(
    name = "lib",
    srcs = ["lib.cc"],
)

py_binary(
    name = "bin",
    srcs = ["main.py"],
)

xla_test(
    name = "lib_test",
    srcs = ["lib_test.cc"],
)
"""
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0][0], "lib")
    self.assertEqual(result[1][0], "lib_test")

  def test_empty_build_file(self):
    self.assertEqual(check_dwyu.extract_targets("", ALLOWED_RULES), [])

  def test_name_on_same_line(self):
    build = 'cc_library(name = "inline_lib", srcs = ["a.cc"])\n'
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
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
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
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
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "multi")
    self.assertEqual(result[0][1], {"a.cc", "b.cc", "a.h", "b.h"})

  def test_concatenated_srcs(self):
    build = """\
cc_library(
    name = "concat",
    srcs = [
        "a.cc",
    ] + if_cuda([
        "b.cu",
    ]),
    hdrs = ["a.h"],
)
"""
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "concat")
    self.assertEqual(result[0][1], {"a.cc", "b.cu", "a.h"})

  def test_conditional_srcs(self):
    build = """\
cc_library(
    name = "conditional",
    srcs = if_cuda_is_configured(["cuda.cc"]),
)
"""
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "conditional")
    self.assertEqual(result[0][1], {"cuda.cc"})

  def test_name_after_other_attributes(self):
    build = """\
cc_library(
    # Long descriptive comments
    # explaining the rationale of the target
    # that exceed standard 5 lines
    # before reaching the name attribute
    tags = ["manual"],
    visibility = ["//visibility:public"],
    srcs = ["a.cc"],
    name = "late_name",
)
"""
    result = check_dwyu.extract_targets(build, ALLOWED_RULES)
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0][0], "late_name")
    self.assertEqual(result[0][1], {"a.cc"})


class WorkspaceTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    workspace = tempfile.TemporaryDirectory()
    self.addCleanup(workspace.cleanup)
    self.addCleanup(os.chdir, os.getcwd())
    os.chdir(workspace.name)

  def write_file(self, path, content):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)

  def affected_targets(self, changed_files):
    packages = check_dwyu.find_packages(changed_files)
    grouped = check_dwyu._group_changed_files_by_package(
        changed_files, packages
    )
    return check_dwyu.find_affected_targets(packages, ALLOWED_RULES, grouped)


class FindAffectedTargetsTest(WorkspaceTest):

  def setUp(self):
    super().setUp()
    self.write_file(
        "xla/pkg/BUILD",
        """\
cc_library(
    name = "foo",
    srcs = ["foo.cc"],
    hdrs = ["foo.h"],
)
cc_library(
    name = "bar",
    srcs = ["bar.cc"],
    deps = [":foo"],
)
xla_test(
    name = "foo_test",
    srcs = ["foo_test.cc"],
    deps = [":foo"],
)
""",
    )

  def test_only_direct_source_owner(self):
    self.assertEqual(
        self.affected_targets(["xla/pkg/foo.cc"]), ["//xla/pkg:foo"]
    )

  def test_header_does_not_select_dependents(self):
    self.assertEqual(
        self.affected_targets(["xla/pkg/foo.h"]), ["//xla/pkg:foo"]
    )

  def test_build_only_change_selects_nothing(self):
    self.assertEqual(self.affected_targets(["xla/pkg/BUILD"]), [])

  def test_build_and_source_change_selects_only_source_owner(self):
    self.assertEqual(
        self.affected_targets(["xla/pkg/BUILD", "xla/pkg/foo.cc"]),
        ["//xla/pkg:foo"],
    )

  def test_unowned_file_selects_nothing(self):
    self.assertEqual(self.affected_targets(["xla/pkg/README.md"]), [])

  def test_subdirectory_does_not_match_same_basename(self):
    self.assertEqual(self.affected_targets(["xla/pkg/subdir/foo.cc"]), [])

  def test_subdirectory_source_paths_and_labels(self):
    self.write_file(
        "xla/pkg/BUILD.bazel",
        """\
cc_library(name = "relative", srcs = ["subdir/foo.cc"])
cc_library(name = "local_label", srcs = [":subdir/foo.cc"])
cc_library(name = "full_label", srcs = ["//xla/pkg:subdir/foo.cc"])
cc_library(name = "other", srcs = ["elsewhere/foo.cc"])
""",
    )
    self.assertEqual(
        self.affected_targets(["xla/pkg/subdir/foo.cc"]),
        ["//xla/pkg:relative", "//xla/pkg:local_label", "//xla/pkg:full_label"],
    )

  def test_build_bazel_takes_precedence(self):
    self.write_file(
        "xla/pkg/BUILD.bazel",
        """\
cc_library(name = "bazel_foo", srcs = ["foo.cc"])
""",
    )
    self.assertEqual(
        self.affected_targets(["xla/pkg/foo.cc"]), ["//xla/pkg:bazel_foo"]
    )
    self.assertEqual(self.affected_targets(["xla/pkg/BUILD.bazel"]), [])

  def test_nested_package_is_not_grouped_with_parent(self):
    self.write_file(
        "xla/pkg/nested/BUILD",
        """\
cc_library(name = "nested_foo", srcs = ["foo.cc"])
""",
    )
    self.assertEqual(
        self.affected_targets(["xla/pkg/nested/foo.cc", "xla/pkg/bar.cc"]),
        ["//xla/pkg:bar", "//xla/pkg/nested:nested_foo"],
    )

  def test_root_package(self):
    self.write_file(
        "BUILD.bazel",
        """\
cc_library(name = "root", srcs = ["root.cc", "subdir/root.cc"])
""",
    )
    self.assertEqual(self.affected_targets(["root.cc"]), ["//:root"])
    self.assertEqual(self.affected_targets(["subdir/root.cc"]), ["//:root"])
    self.assertEqual(self.affected_targets(["BUILD.bazel"]), [])

  def test_no_package(self):
    self.assertEqual(self.affected_targets(["unowned/foo.cc"]), [])

  def test_include_tsl_and_other_third_party_packages(self):
    for package in ("third_party/tsl", "third_party/tsl_extra"):
      self.write_file(
          f"{package}/BUILD",
          """\
cc_library(name = "foo", srcs = ["foo.cc"])
""",
      )
    self.assertEqual(
        self.affected_targets([
            "third_party/tsl/BUILD",
            "third_party/tsl/foo.cc",
            "third_party/tsl_extra/foo.cc",
            "xla/pkg/foo.cc",
        ]),
        ["@tsl//:foo", "//third_party/tsl_extra:foo", "//xla/pkg:foo"],
    )

  def test_tsl_package_source_labels_and_test_rules(self):
    self.write_file(
        "third_party/tsl/tsl/profiler/BUILD",
        """\
cc_library(name = "relative", srcs = ["//tsl/profiler:controller.cc"])
cc_library(name = "qualified", srcs = ["@tsl//tsl/profiler:controller.cc"])
tsl_cc_test(name = "controller_test", srcs = ["controller_test.cc"])
""",
    )
    self.assertEqual(
        self.affected_targets([
            "third_party/tsl/tsl/profiler/controller.cc",
            "third_party/tsl/tsl/profiler/controller_test.cc",
        ]),
        [
            "@tsl//tsl/profiler:relative",
            "@tsl//tsl/profiler:qualified",
            "@tsl//tsl/profiler:controller_test",
        ],
    )


@unittest.skipUnless(shutil.which("git"), "requires git in PATH")
class ChangedFilesTest(WorkspaceTest):

  def git(self, *args):
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=DWYU test",
            "-c",
            "user.email=dwyu-test@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()

  def commit(self):
    self.git("add", ".")
    self.git("commit", "-qm", "Test change")

  def setUp(self):
    super().setUp()
    self.git("init", "-q", "-b", "main")
    self.write_file(
        "xla/pkg/BUILD",
        """\
cc_library(name = "foo", srcs = ["foo.cc"])
cc_library(name = "bar", srcs = ["bar.cc"])
""",
    )
    self.write_file("xla/pkg/foo.cc", "// Original foo\n")
    self.write_file("xla/pkg/bar.cc", "// Original bar\n")
    self.commit()

  def test_base_branch_changes_are_excluded(self):
    self.git("branch", "topic")
    self.write_file("xla/pkg/bar.cc", "// Unrelated upstream change\n")
    self.commit()
    self.git("checkout", "-q", "topic")
    self.write_file("xla/pkg/foo.cc", "// PR change\n")
    self.commit()
    self.assertEqual(check_dwyu.get_changed_files("main"), ["xla/pkg/foo.cc"])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      check_dwyu.main(["check_dwyu.py", "--base_ref", "main"])
    self.assertEqual(output.getvalue(), "//xla/pkg:foo\n")

  def test_push_compares_with_previous_commit(self):
    self.write_file("xla/pkg/foo.cc", "// Pushed change\n")
    self.commit()
    self.assertEqual(check_dwyu.get_changed_files("HEAD^"), ["xla/pkg/foo.cc"])

  def test_rename_includes_old_and_new_paths(self):
    self.git("mv", "xla/pkg/foo.cc", "xla/pkg/new foo.cc")
    self.commit()
    self.assertEqual(
        check_dwyu.get_changed_files("HEAD^"),
        ["xla/pkg/foo.cc", "xla/pkg/new foo.cc"],
    )

  def test_empty_and_binary_files_do_not_require_diff_hunks(self):
    self.write_file("xla/pkg/empty.cc", "")
    self.write_file("xla/pkg/binary.cc", "\0data")
    self.commit()
    self.assertEqual(
        check_dwyu.get_changed_files("HEAD^"),
        ["xla/pkg/binary.cc", "xla/pkg/empty.cc"],
    )

  def test_deleted_file(self):
    self.git("rm", "xla/pkg/foo.cc")
    self.commit()
    self.assertEqual(check_dwyu.get_changed_files("HEAD^"), ["xla/pkg/foo.cc"])

  def test_no_changes(self):
    self.assertEqual(check_dwyu.get_changed_files("HEAD"), [])


if __name__ == "__main__":
  unittest.main()
