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
      (["xla/flags.bzl"], False),
      (["third_party/tsl/tsl/platform/default/rules.bzl"], False),
      (["third_party/tsl/REPO.bazel"], True),
      (["third_party/tsl/tsl/BUILD"], False),
      (["third_party/tsl/tsl/platform/denormal.cc"], False),
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

  @parameterized.parameters(
      ("//xla/service/gpu:gpu_compiler", "//xla/service/gpu/...", True),
      ("//xla/service/gpu/sub:target", "//xla/service/gpu/...", True),
      ("//xla/service/cpu:cpu_compiler", "//xla/service/gpu/...", False),
      ("//xla/service/gpu_other:target", "//xla/service/gpu/...", False),
      ("//xla/backends/cpu:test", "//xla/backends/cpu:all", True),
      ("//xla/backends/cpu:test", "//xla/backends/cpu:*", True),
      ("//xla/backends/cpu:test", "//xla/backends/cpu:other", False),
      ("//xla/backends/cpu:test", "//xla/backends/cpu:test", True),
  )
  def test_target_matches_pattern(self, target, pattern, expected):
    self.assertEqual(
        bazel_diff.target_matches_pattern(target, pattern), expected
    )

  def test_target_allowed_by_patterns(self):
    patterns = (
        "//xla/...",
        "@tsl//tsl/...",
        "-//xla/backends/gpu/...",
        "-//xla/service/gpu/...",
        "-//xla/tpu/...",
        "-//xla/backends/cpu/collectives:gloo_collectives_test",
    )
    self.assertTrue(
        bazel_diff.target_allowed_by_patterns(
            "//xla/service/cpu:cpu_compiler", patterns
        )
    )
    self.assertTrue(
        bazel_diff.target_allowed_by_patterns(
            "@tsl//tsl/platform:default/denormal.cc", patterns
        )
    )
    self.assertFalse(
        bazel_diff.target_allowed_by_patterns(
            "//xla/service/gpu:gpu_compiler", patterns
        )
    )
    self.assertFalse(
        bazel_diff.target_allowed_by_patterns(
            "//xla/backends/gpu/tests:gpu_atomic_test", patterns
        )
    )
    self.assertFalse(
        bazel_diff.target_allowed_by_patterns(
            "//xla/backends/cpu/collectives:gloo_collectives_test", patterns
        )
    )
    self.assertTrue(
        bazel_diff.target_allowed_by_patterns(
            "//xla/backends/cpu/collectives:mpi_test", patterns
        )
    )
    self.assertFalse(
        bazel_diff.target_allowed_by_patterns(
            "@unrelated//pkg:target", patterns
        )
    )

  def test_filter_and_normalize_targets_with_patterns(self):
    raw_targets = [
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/cpu:cpu_compiler",
        "//xla/backends/gpu/tests:gpu_atomic_test",
        "@tsl//tsl/platform:default/denormal.cc",
    ]
    patterns = (
        "//xla/...",
        "@tsl//tsl/...",
        "-//xla/service/gpu/...",
        "-//xla/backends/gpu/...",
    )
    expected = [
        "//xla/service/cpu:cpu_compiler",
        "@tsl//tsl/platform:default/denormal.cc",
    ]
    self.assertEqual(
        bazel_diff.filter_and_normalize_targets(
            raw_targets, target_patterns=patterns
        ),
        expected,
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
        bazel_diff.report_decision(decision, "xla_linux_x86_gpu_l4")

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
    mock_changed.return_value = ["MODULE.bazel", "xla/service/cpu_compiler.cc"]
    build_mock = mock.MagicMock()
    decision = bazel_diff.compute_impacted_targets(
        build_mock, "base_sha", "head_sha"
    )
    self.assertEqual(decision.decision, bazel_diff.BazelDiffDecisionType.FULL)
    self.assertIn("Global Bazel configuration", decision.reason)

  @mock.patch.object(bazel_diff.subprocess, "run")
  @mock.patch.object(bazel_diff, "get_diff_base")
  def test_touches_build_keywords_sensitive(self, mock_base, mock_run):
    mock_base.return_value = "base_sha"
    mock_run.return_value = mock.MagicMock(
        returncode=0,
        stdout=(
            "+ label_flag(name = 'custom_flag', default = '//xla/pkg:target')\n"
        ),
    )
    self.assertTrue(
        bazel_diff.touches_build_keywords(
            ["xla/flags.bzl"], "base_sha", "head_sha"
        )
    )

  @mock.patch.object(bazel_diff.subprocess, "run")
  @mock.patch.object(bazel_diff, "get_diff_base")
  def test_touches_build_keywords_benign_bzl(self, mock_base, mock_run):
    mock_base.return_value = "base_sha"
    mock_run.return_value = mock.MagicMock(
        returncode=0,
        stdout=(
            "+def my_macro(name, deps=[]):\n"
            "+  native.cc_library(name='hlo_alias_analysis')\n"
        ),
    )
    self.assertFalse(
        bazel_diff.touches_build_keywords(
            ["xla/service/custom.bzl"], "base_sha", "head_sha"
        )
    )
    mock_run.assert_called_once_with(
        [
            "git",
            "diff",
            "-U0",
            "base_sha",
            "head_sha",
            "--",
            "xla/service/custom.bzl",
        ],
        cwd=".",
        capture_output=True,
        text=True,
        check=True,
    )

  def test_get_cquery_command_options(self):
    build_mock = mock.MagicMock()
    build_mock.configs = (
        "warnings",
        "rbe_linux_cuda_nvcc",
        "hermetic_cuda_umd",
    )
    build_mock.repo_env = {"TF_CUDA_COMPUTE_CAPABILITIES": "7.5"}
    build_mock.options = {
        "//xla/tsl:ci_build": True,
        "@local_config_cuda//cuda:override_include_cuda_libs": False,
        "flaky_test_attempts": 3,
        "test_output": "errors",
    }
    self.assertEqual(
        bazel_diff.get_cquery_command_options(build_mock),
        [
            "--config=warnings",
            "--config=rbe_linux_cuda_nvcc",
            "--config=hermetic_cuda_umd",
            "--repo_env=TF_CUDA_COMPUTE_CAPABILITIES=7.5",
            "--//xla/tsl:ci_build",
            "--@local_config_cuda//cuda:override_include_cuda_libs=False",
        ],
    )

  @mock.patch.object(bazel_diff.subprocess, "run")
  def test_run_generate_hashes_cquery_options(self, mock_run):
    bazel_diff.run_generate_hashes(
        "/tmp/bazel-diff.jar",
        "/workspace",
        "/tmp/hashes.json",
        cquery_options=[
            "--config=warnings",
            "--config=rbe_linux_cuda_nvcc",
            "--repo_env=TF_CUDA_COMPUTE_CAPABILITIES=7.5",
            "--//xla/tsl:ci_build",
        ],
    )
    args = mock_run.call_args[0][0]
    self.assertIn(
        "--cqueryCommandOptions=--config=warnings --config=rbe_linux_cuda_nvcc"
        " --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=7.5 --//xla/tsl:ci_build",
        args,
    )

  @mock.patch.object(bazel_diff, "touches_build_keywords")
  @mock.patch.object(bazel_diff, "get_changed_files")
  def test_compute_impacted_targets_build_keywords_fallback(
      self, mock_changed, mock_keywords
  ):
    mock_changed.return_value = ["xla/flags.bzl"]
    mock_keywords.return_value = True
    build_mock = mock.MagicMock()
    decision = bazel_diff.compute_impacted_targets(
        build_mock, "base_sha", "head_sha"
    )
    self.assertEqual(decision.decision, bazel_diff.BazelDiffDecisionType.FULL)
    self.assertIn("label_flag, config_setting, or alias", decision.reason)

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

  @mock.patch.object(bazel_diff.subprocess, "run")
  def test_get_merge_base_success(self, mock_run):
    mock_run.return_value = mock.MagicMock(returncode=0, stdout="base_commit\n")
    self.assertEqual(bazel_diff.get_merge_base("base", "head"), "base_commit")

  @mock.patch.object(bazel_diff.subprocess, "run")
  def test_get_merge_base_failure(self, mock_run):
    mock_run.return_value = mock.MagicMock(returncode=128, stdout="")
    self.assertIsNone(bazel_diff.get_merge_base("base", "head"))

  @mock.patch.object(bazel_diff, "get_merge_base")
  def test_get_diff_base_with_merge_base(self, mock_mb):
    mock_mb.return_value = "merge_base_sha"
    self.assertEqual(
        bazel_diff.get_diff_base("base_sha", "head_sha"), "merge_base_sha"
    )

  @mock.patch.object(bazel_diff, "get_merge_base")
  def test_get_diff_base_fallback(self, mock_mb):
    mock_mb.return_value = None
    self.assertEqual(
        bazel_diff.get_diff_base("base_sha", "head_sha"), "base_sha"
    )


if __name__ == "__main__":
  absltest.main()
