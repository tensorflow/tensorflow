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
"""Tests for bazel_diff module."""

import hashlib
import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from build_tools.ci import bazel_diff


class BazelDiffTest(parameterized.TestCase):

  @parameterized.parameters(
      (["MODULE.bazel"], True),
      (["REPO.bazel"], True),
      (["WORKSPACE"], True),
      (["WORKSPACE.bzlmod"], True),
      ([".bazelrc"], True),
      ([".bazelversion"], True),
      (["xla/flags.bzl"], True),
      (["third_party/tsl/tsl/platform/default/rules.bzl"], True),
      (["third_party/tsl/tsl/BUILD"], True),
      (["xla/service/cpu/cpu_compiler.cc"], False),
      (["docs/overview.md", "xla/BUILD"], False),
  )
  def test_is_global_config_changed(self, changed_files, expected):
    self.assertEqual(
        bazel_diff.is_global_config_changed(changed_files), expected
    )

  @parameterized.parameters(
      (["README.md", "docs/index.md"], True),
      (["OWNERS", "LICENSE"], True),
      ([".gitignore", ".vscode/settings.json"], True),
      (["logo.png", "diagram.svg"], True),
      (["README.md", "xla/service/hlo_parser.cc"], False),
      ([], False),
  )
  def test_is_docs_or_metadata_only(self, changed_files, expected):
    self.assertEqual(
        bazel_diff.is_docs_or_metadata_only(changed_files), expected
    )

  @parameterized.parameters(
      ("//xla/service:cpu_compiler", "//xla/service:cpu_compiler"),
      ("@@//xla/python:types", "//xla/python:types"),
      ("@//xla/python:types", "//xla/python:types"),
      ("@@+tsl+tsl//tsl/platform:logging", "@tsl//tsl/platform:logging"),
      ("@@tsl//tsl/platform:logging", "@tsl//tsl/platform:logging"),
      ("@tsl//tsl/platform:logging", "@tsl//tsl/platform:logging"),
      ("@@llvm-project//llvm:Support", "@llvm-project//llvm:Support"),
      ("", ""),
  )
  def test_normalize_target_label(self, input_label, expected):
    self.assertEqual(bazel_diff.normalize_target_label(input_label), expected)

  def test_filter_and_normalize_targets(self):
    raw_targets = [
        "//xla/service:hlo_parser",
        "@@//xla/tools:runner",
        "@@+tsl+tsl//tsl/platform:logging",
        "//build_tools/ci:build",
        "//external:tsl",
        "@external:llvm",
        "@unrelated//foo:bar",
        "//xla/service:hlo_parser",  # duplicate
        "# comment line",
        "   ",
    ]
    expected = [
        "//build_tools/ci:build",
        "//xla/service:hlo_parser",
        "//xla/tools:runner",
        "@tsl//tsl/platform:logging",
    ]
    self.assertEqual(
        bazel_diff.filter_and_normalize_targets(raw_targets), expected
    )

  def test_verify_sha256(self):
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
      f.write(b"hello world")
      temp_path = f.name

    try:
      expected_hash = hashlib.sha256(b"hello world").hexdigest()
      self.assertTrue(bazel_diff.verify_sha256(temp_path, expected_hash))
      self.assertFalse(bazel_diff.verify_sha256(temp_path, "0" * 64))
      self.assertFalse(
          bazel_diff.verify_sha256("/non/existent/file", expected_hash)
      )
    finally:
      os.unlink(temp_path)

  def test_report_decision(self):
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
      summary_file = f.name

    try:
      with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": summary_file}):
        decision = bazel_diff.BazelDiffDecision(
            decision=bazel_diff.BazelDiffDecisionType.IMPACTED,
            impacted_targets=("//xla/service:hlo_parser",),
            reason="Found 1 impacted target",
            changed_files_count=2,
            elapsed_seconds=12.3,
        )
        bazel_diff.report_decision(decision, "shadow", "xla_linux_x86_gpu_l4")

      with open(summary_file, "r") as f:
        content = f.read()

      self.assertIn("bazel-diff Impact Analysis", content)
      self.assertIn("xla_linux_x86_gpu_l4", content)
      self.assertIn("IMPACTED", content)
      self.assertIn("//xla/service:hlo_parser", content)
    finally:
      os.unlink(summary_file)

  @mock.patch.object(bazel_diff, "get_changed_files")
  def test_compute_impacted_targets_global_config_fallback(self, mock_changed):
    mock_changed.return_value = ["xla/flags.bzl", "xla/service/cpu_compiler.cc"]
    build_mock = mock.MagicMock()
    decision = bazel_diff.compute_impacted_targets(
        build_mock, "base_sha", "head_sha"
    )
    self.assertEqual(decision.decision, bazel_diff.BazelDiffDecisionType.FULL)
    self.assertIn("Global Bazel configuration", decision.reason)

  @mock.patch.object(bazel_diff, "get_changed_files")
  def test_compute_impacted_targets_docs_only_skip(self, mock_changed):
    mock_changed.return_value = ["README.md", "docs/architecture.md"]
    build_mock = mock.MagicMock()
    decision = bazel_diff.compute_impacted_targets(
        build_mock, "base_sha", "head_sha"
    )
    self.assertEqual(decision.decision, bazel_diff.BazelDiffDecisionType.SKIP)
    self.assertIn("Only documentation", decision.reason)

  @mock.patch.object(bazel_diff, "get_changed_files")
  def test_compute_impacted_targets_git_fail_open(self, mock_changed):
    mock_changed.side_effect = RuntimeError("git failure")
    build_mock = mock.MagicMock()
    decision = bazel_diff.compute_impacted_targets(
        build_mock, "base_sha", "head_sha"
    )
    self.assertEqual(decision.decision, bazel_diff.BazelDiffDecisionType.FULL)
    self.assertIn("Failed to get git changed files", decision.reason)


if __name__ == "__main__":
  absltest.main()
