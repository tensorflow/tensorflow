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

import dataclasses
import hashlib
import os
import subprocess
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from build_tools.ci import bazel_diff
from build_tools.ci import change_detector


@dataclasses.dataclass(frozen=True)
class _FakeBuildConfig:
  target_patterns: tuple[str, ...] = ("//xla/...",)
  configs: tuple[str, ...] = ()
  repo_env: dict[str, str] = dataclasses.field(default_factory=dict)
  options: dict[str, object] = dataclasses.field(default_factory=dict)
  startup_options: dict[str, object] = dataclasses.field(default_factory=dict)
  bazel_diff_use_cquery: bool = True


def _write_impacted_targets(
    jar_path: str,
    workspace_dir: str,
    base_hashes: str,
    head_hashes: str,
    output_path: str,
) -> None:
  del jar_path, workspace_dir, base_hashes, head_hashes
  with open(output_path, "w", encoding="utf-8") as f:
    f.write("//xla/service/cpu:cpu_compiler_test\n")


class BazelDiffTest(parameterized.TestCase):

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

  @mock.patch.object(bazel_diff.subprocess, "run", autospec=True, spec_set=True)
  @mock.patch.object(
      bazel_diff.change_detector, "get_diff_base", autospec=True, spec_set=True
  )
  def test_touches_build_keywords_sensitive(self, mock_base, mock_run):
    mock_base.return_value = "base_sha"
    mock_run.return_value = subprocess.CompletedProcess(
        args=["git", "diff"],
        returncode=0,
        stdout=(
            "+ label_flag(name = 'custom_flag', default = '//xla/pkg:target')\n"
        ),
    )
    self.assertTrue(
        bazel_diff._touches_build_keywords(
            ["xla/flags.bzl"], "base_sha", "head_sha"
        )
    )

  @mock.patch.object(bazel_diff.subprocess, "run", autospec=True, spec_set=True)
  @mock.patch.object(
      bazel_diff.change_detector, "get_diff_base", autospec=True, spec_set=True
  )
  def test_touches_build_keywords_benign_bzl(self, mock_base, mock_run):
    mock_base.return_value = "base_sha"
    mock_run.return_value = subprocess.CompletedProcess(
        args=["git", "diff"],
        returncode=0,
        stdout=(
            "+def my_macro(name, deps=[]):\n"
            "+  native.cc_library(name='hlo_alias_analysis')\n"
        ),
    )
    self.assertFalse(
        bazel_diff._touches_build_keywords(
            ["xla/service/custom.bzl"], "base_sha", "head_sha"
        )
    )
    mock_run.assert_called_once_with(
        [
            "git",
            "diff",
            "-U0",
            "--end-of-options",
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
      self.assertNotIn("more targets omitted", content)
    finally:
      os.unlink(summary_file)

  def test_report_decision_truncates_large_step_summary(self):
    summary_file = self.create_tempfile("step_summary.md")
    targets = tuple(
        f"//xla/service/cpu:generated_target_{i}" for i in range(25000)
    )
    with mock.patch.dict(
        os.environ, {"GITHUB_STEP_SUMMARY": summary_file.full_path}
    ):
      decision = bazel_diff.BazelDiffDecision(
          decision=bazel_diff.BazelDiffDecisionType.IMPACTED,
          impacted_targets=targets,
          reason="Found 25000 impacted targets",
          changed_files_count=10,
          elapsed_seconds=45.0,
      )
      bazel_diff.report_decision(decision, "xla_linux_x86_cpu")

    content = summary_file.read_text()
    self.assertLess(len(content.encode("utf-8")), 1024 * 1024)
    self.assertIn("(showing first 500 of 25000)", content)
    self.assertIn("//xla/service/cpu:generated_target_0\n", content)
    self.assertIn("//xla/service/cpu:generated_target_499\n", content)
    self.assertNotIn("//xla/service/cpu:generated_target_500\n", content)
    self.assertIn("... (24500 more targets omitted)", content)

  @mock.patch.object(
      change_detector, "get_changed_files", autospec=True, spec_set=True
  )
  def test_compute_impacted_targets_global_config_fallback(self, mock_changed):
    mock_changed.return_value = ["MODULE.bazel", "xla/service/cpu_compiler.cc"]
    build_mock = mock.MagicMock()
    decision = bazel_diff.compute_impacted_targets(
        build_mock, "base_sha", "head_sha"
    )
    self.assertEqual(decision.decision, bazel_diff.BazelDiffDecisionType.FULL)
    self.assertIn("Global Bazel configuration", decision.reason)

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
        "--fineGrainedHashExternalRepos=@tsl,@@+tsl_extension+tsl",
        args,
    )
    self.assertIn(
        "--cqueryCommandOptions=--config=warnings --config=rbe_linux_cuda_nvcc"
        " --repo_env=TF_CUDA_COMPUTE_CAPABILITIES=7.5 --//xla/tsl:ci_build",
        args,
    )

  @mock.patch.object(
      bazel_diff, "_touches_build_keywords", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_changed_files", autospec=True, spec_set=True
  )
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

  @mock.patch.object(
      change_detector, "get_changed_files", autospec=True, spec_set=True
  )
  def test_compute_impacted_targets_git_fail_open(self, mock_changed):
    mock_changed.side_effect = OSError("git failure")
    build_mock = mock.MagicMock()
    decision = bazel_diff.compute_impacted_targets(
        build_mock, "base_sha", "head_sha"
    )
    self.assertEqual(decision.decision, bazel_diff.BazelDiffDecisionType.FULL)
    self.assertIn("Failed to get git changed files", decision.reason)

  @mock.patch.object(
      bazel_diff, "run_get_impacted_targets", autospec=True, spec_set=True
  )
  @mock.patch.object(
      bazel_diff, "run_generate_hashes", autospec=True, spec_set=True
  )
  @mock.patch.object(
      bazel_diff, "download_and_verify_jar", autospec=True, spec_set=True
  )
  @mock.patch.object(bazel_diff.subprocess, "run", autospec=True, spec_set=True)
  @mock.patch.object(
      change_detector, "get_diff_base", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_changed_files", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "ensure_base_fetched", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "run_git_best_effort", autospec=True, spec_set=True
  )
  def test_compute_impacted_targets_checkout_and_restore(
      self,
      mock_git_best_effort,
      mock_ensure_fetched,
      mock_changed_files,
      mock_diff_base,
      mock_subprocess_run,
      mock_download_jar,
      mock_generate_hashes,
      mock_get_impacted,
  ):
    del mock_ensure_fetched, mock_generate_hashes
    mock_git_best_effort.side_effect = [
        subprocess.CompletedProcess(
            args=["git", "rev-parse", "HEAD"],
            returncode=0,
            stdout="orig_head_sha\n",
        ),
        subprocess.CompletedProcess(
            args=["git", "checkout", "-f", "orig_head_sha"],
            returncode=0,
            stdout="",
        ),
    ]
    mock_changed_files.return_value = ["xla/service/cpu/cpu_compiler.cc"]
    mock_diff_base.return_value = "merge_base_sha"
    mock_download_jar.return_value = "/tmp/bazel-diff.jar"
    mock_get_impacted.side_effect = _write_impacted_targets

    decision = bazel_diff.compute_impacted_targets(
        _FakeBuildConfig(), "base_sha", "HEAD", workspace_dir="/workspace"
    )

    self.assertEqual(
        decision.decision, bazel_diff.BazelDiffDecisionType.IMPACTED
    )
    self.assertEqual(
        decision.impacted_targets, ("//xla/service/cpu:cpu_compiler_test",)
    )
    self.assertEqual(
        mock_subprocess_run.call_args_list,
        [
            mock.call(
                ["git", "checkout", "-f", "merge_base_sha"],
                cwd="/workspace",
                check=True,
                capture_output=True,
            ),
            mock.call(
                ["git", "checkout", "-f", "orig_head_sha"],
                cwd="/workspace",
                check=True,
                capture_output=True,
            ),
        ],
    )
    self.assertEqual(
        mock_git_best_effort.call_args_list,
        [
            mock.call(["rev-parse", "HEAD"], cwd="/workspace"),
            mock.call(["checkout", "-f", "orig_head_sha"], cwd="/workspace"),
        ],
    )

  @mock.patch.object(
      bazel_diff, "compute_impacted_targets", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "evaluate_skip", autospec=True, spec_set=True
  )
  def test_main_impacted_writes_github_output(
      self, mock_eval_skip, mock_compute
  ):
    output_file = self.create_tempfile("github_output.txt")
    targets_file = self.create_tempfile("impacted.txt", content="//xla:test\n")
    dest_targets_file = self.create_tempfile("dest_impacted.txt")
    mock_eval_skip.return_value = change_detector.SkipDecision(
        should_skip=False, reason="Non-doc change", changed_files_count=1
    )
    mock_compute.return_value = bazel_diff.BazelDiffDecision(
        decision=bazel_diff.BazelDiffDecisionType.IMPACTED,
        impacted_targets=("//xla:test",),
        impacted_targets_file=targets_file.full_path,
        reason="Found 1 impacted target",
        changed_files_count=1,
    )
    rc = bazel_diff.main([
        "--base_sha=base_sha",
        "--head_sha=head_sha",
        "--build_name=XLA ROCm",
        "--target_pattern=//xla/...",
        f"--output_target_pattern_file={dest_targets_file.full_path}",
        f"--github_output={output_file.full_path}",
    ])
    self.assertEqual(rc, 0)
    self.assertEqual(
        output_file.read_text(),
        "decision=IMPACTED\n"
        "should_skip=false\n"
        f"target_pattern_file={os.path.abspath(dest_targets_file.full_path)}\n",
    )
    self.assertEqual(dest_targets_file.read_text(), "//xla:test\n")

  @mock.patch.object(
      change_detector, "evaluate_skip", autospec=True, spec_set=True
  )
  def test_main_docs_skip_writes_github_output(self, mock_eval_skip):
    output_file = self.create_tempfile("github_output.txt")
    mock_eval_skip.return_value = change_detector.SkipDecision(
        should_skip=True, reason="Docs only", changed_files_count=1
    )
    rc = bazel_diff.main([
        "--base_sha=base_sha",
        "--head_sha=head_sha",
        f"--github_output={output_file.full_path}",
    ])
    self.assertEqual(rc, 0)
    self.assertEqual(
        output_file.read_text(),
        "decision=SKIP\nshould_skip=true\ntarget_pattern_file=\n",
    )

  @mock.patch.object(bazel_diff.subprocess, "run", autospec=True, spec_set=True)
  @mock.patch.object(bazel_diff.shutil, "which", autospec=True, spec_set=True)
  def test_resolve_java_bin_falls_back_to_bazel_java_home(
      self, mock_which, mock_run
  ):
    mock_which.return_value = None
    fake_jdk_dir = self.create_tempdir("embedded_jdk")
    fake_bin_dir = os.path.join(fake_jdk_dir.full_path, "bin")
    os.makedirs(fake_bin_dir, exist_ok=True)
    fake_java = os.path.join(fake_bin_dir, "java")
    with open(fake_java, "w", encoding="utf-8") as f:
      f.write("#!/bin/sh\n")
    os.chmod(fake_java, 0o755)
    mock_run.return_value = subprocess.CompletedProcess(
        args=["bazel", "--bazelrc=rocm.bazelrc", "info", "java-home"],
        returncode=0,
        stdout=f"{fake_jdk_dir.full_path}\n",
    )
    with mock.patch.dict(os.environ, {"JAVA_HOME": ""}, clear=False):
      resolved = bazel_diff.resolve_java_bin(
          workspace_dir="/workspace",
          startup_options=["--bazelrc=rocm.bazelrc"],
      )
    self.assertEqual(resolved, fake_java)
    mock_run.assert_called_once_with(
        ["bazel", "--bazelrc=rocm.bazelrc", "info", "java-home"],
        cwd="/workspace",
        capture_output=True,
        text=True,
        check=True,
    )


if __name__ == "__main__":
  absltest.main()
