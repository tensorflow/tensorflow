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
"""Regression tests using BANT, without requiring a Bazel build or downloads."""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

from build_tools.lint import check_dwyu
from build_tools.lint import run_dwyu

_BANT = shutil.which(os.environ.get("BANT_BIN", "bant"))
_BANT_MACROS = pathlib.Path(run_dwyu.__file__).parents[2] / ".bant-macros"


class BantWorkspaceTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = pathlib.Path(temporary.name) / "checkout"
    self.root.mkdir()
    # Match a clean checkout after `bazel build --nobuild`: dependencies are
    # fetched, but there is no bazel-out symlink or execution output directory.
    self.output_base = pathlib.Path(temporary.name) / "bazel cache"
    self.external = self.output_base / "external"
    self.external.mkdir(parents=True)
    (self.external / "_main").symlink_to(self.root, target_is_directory=True)
    self.write_file("MODULE.bazel", 'module(name = "xla")\n')
    self.write_file(".bant-macros", _BANT_MACROS.read_text())
    self.write_file("third_party/tsl/WORKSPACE", "")

  def write_file(self, path, contents):
    path = self.root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)

  def test_uses_current_checkout_without_changing_cached_repositories(self):
    cached_tsl = self.external / "tsl"
    cached_tsl.mkdir()
    (self.external / "dependency+").mkdir()
    with run_dwyu.bant_workspace(self.root, self.output_base) as workspace:
      external = (workspace / "bazel-out/../../../external").resolve()
      self.assertEqual((external / "xla").resolve(), self.root)
      self.assertFalse((external / "_main").is_symlink())
      self.assertEqual(
          (external / "tsl").resolve(), self.root / "third_party/tsl"
      )
      self.assertEqual(
          (external / "dependency+").resolve(), self.external / "dependency+"
      )
    self.assertFalse(workspace.exists())
    self.assertTrue(cached_tsl.is_dir())
    self.assertFalse(cached_tsl.is_symlink())
    self.assertFalse((self.external / "xla").exists())
    self.assertEqual((self.external / "_main").resolve(), self.root)
    self.assertFalse((self.root / "bazel-out").is_symlink())
    self.assertFalse((self.output_base / "execroot").exists())

  def test_ignores_stale_bazel_out_symlink(self):
    (self.root / "bazel-out").symlink_to(self.root / "missing-output")
    with run_dwyu.bant_workspace(self.root, self.output_base) as workspace:
      self.assertTrue((workspace / "external/xla").is_dir())
    self.assertEqual(
        (self.root / "bazel-out").readlink(), self.root / "missing-output"
    )

  def test_empty_selection_does_not_run_bant(self):
    self.assertEqual(
        run_dwyu.run_dwyu(
            [], output_base=self.root / "missing-output", bant="missing-bant"
        ),
        0,
    )


@unittest.skipUnless(
    _BANT, "requires the BANT binary installed by the workflow"
)
class DwyuIntegrationTest(BantWorkspaceTest):

  def setUp(self):
    super().setUp()
    self.write_file(
        "xla/platform/BUILD",
        """\
cc_library(name = "errors", hdrs = ["errors.h"], visibility = ["//visibility:public"])
cc_library(name = "unused", hdrs = ["unused.h"], visibility = ["//visibility:public"])
""",
    )
    self.write_file("xla/platform/errors.h", "// Error declarations.\n")
    self.write_file("xla/platform/unused.h", "// Unused header.\n")
    self.write_file(
        "xla/consumer/BUILD",
        """\
cc_library(name = "consumer", srcs = ["consumer.cc"], deps = ["//xla/platform:errors"])
""",
    )
    self.write_file(
        "xla/consumer/consumer.cc", '#include "xla/platform/errors.h"\n'
    )
    self.write_file(
        "third_party/tsl/tsl/profiler/BUILD",
        """\
cc_library(
    name = "controller",
    srcs = ["controller.cc"],
    deps = ["@xla//xla/platform:errors"],
)
cc_library(name = "unrelated", srcs = ["unrelated.cc"])
""",
    )
    self.write_file(
        "third_party/tsl/tsl/profiler/controller.cc",
        '// A benign source change.\n#include "xla/platform/errors.h"\n',
    )
    # Deliberately broken but unselected: it must never produce a finding.
    self.write_file(
        "third_party/tsl/tsl/profiler/unrelated.cc",
        '#include "xla/platform/errors.h"\n',
    )

  def check_targets(self, *targets):
    return subprocess.run(
        [
            sys.executable or shutil.which("python3"),
            run_dwyu.__file__,
            "--bant",
            _BANT,
            "--output-base",
            str(self.output_base),
            *targets,
        ],
        cwd=self.root,
        capture_output=True,
        text=True,
        check=False,
    )

  def test_benign_tsl_change_has_no_false_positive(self):
    result = self.check_targets("@tsl//tsl/profiler:controller")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(result.stdout, "")
    self.assertIn("Checked DWYU on 1 targets.", result.stderr)

  def test_reports_real_missing_root_dependency(self):
    self.write_file(
        "third_party/tsl/tsl/profiler/BUILD",
        """\
cc_library(name = "controller", srcs = ["controller.cc"])
""",
    )
    result = self.check_targets("@tsl//tsl/profiler:controller")
    self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
    self.assertEqual(
        result.stdout,
        "buildozer 'add deps @xla//xla/platform:errors'"
        " @tsl//tsl/profiler:controller\n",
    )

  def test_reports_real_unused_dependency(self):
    self.write_file(
        "third_party/tsl/tsl/profiler/BUILD",
        """\
cc_library(
    name = "controller",
    srcs = ["controller.cc"],
    deps = ["@xla//xla/platform:errors", "@xla//xla/platform:unused"],
)
""",
    )
    result = self.check_targets("@tsl//tsl/profiler:controller")
    self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
    self.assertEqual(
        result.stdout,
        "buildozer 'remove deps @xla//xla/platform:unused'"
        " @tsl//tsl/profiler:controller\n",
    )

  def test_mixed_xla_and_tsl_targets_use_one_root_identity(self):
    result = self.check_targets(
        "//xla/consumer:consumer", "@tsl//tsl/profiler:controller"
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(result.stdout, "")
    self.assertIn("Checked DWYU on 2 targets.", result.stderr)

  def test_keeps_implementation_edge_but_reports_other_unused_dependency(self):
    self.write_file(
        "third_party/tsl/tsl/profiler/BUILD",
        """\
cc_library(name = "implementation", hdrs = ["api.h"], srcs = ["api.cc"])
cc_library(
    name = "api",
    hdrs = ["api.h"],
    deps = [
        ":implementation",  # keep: implements api.h
        "@xla//xla/platform:unused",
    ],
)
""",
    )
    self.write_file("third_party/tsl/tsl/profiler/api.h", "void Api();\n")
    self.write_file(
        "third_party/tsl/tsl/profiler/api.cc",
        '#include "tsl/profiler/api.h"\nvoid Api() {}\n',
    )
    result = self.check_targets("@tsl//tsl/profiler:api")
    self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
    self.assertEqual(
        result.stdout,
        "buildozer 'remove deps @xla//xla/platform:unused'"
        " @tsl//tsl/profiler:api\n",
    )

  def test_conditional_deps_macro_resolves_wrapped_dependency(self):
    self.write_file(
        "xla/consumer/BUILD",
        """\
cc_library(
    name = "consumer",
    srcs = ["consumer.cc"],
    deps = if_cuda_is_configured(["//xla/platform:errors"]),
)
""",
    )
    result = self.check_targets("//xla/consumer:consumer")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(result.stdout, "")
    self.assertIn("Checked DWYU on 1 targets.", result.stderr)

  def test_all_allowed_rules_supported_by_bant_macros(self):
    rules = []
    labels = []
    for index, rule in enumerate(check_dwyu.DEFAULT_ALLOWED_RULES):
      target = f"target_{index}_{rule}"
      rules.append(
          f'{rule}(name = "{target}", srcs = ["consumer.cc"],'
          ' deps = ["//xla/platform:errors"])'
      )
      labels.append(f"//xla/consumer:{target}")
    self.write_file("xla/consumer/BUILD", "\n".join(rules) + "\n")
    result = self.check_targets(*labels)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(result.stdout, "")
    self.assertIn(
        f"Checked DWYU on {len(labels)} targets.",
        result.stderr,
    )


if __name__ == "__main__":
  unittest.main()
