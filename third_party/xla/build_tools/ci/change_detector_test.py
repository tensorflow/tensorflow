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
"""Tests for change_detector module."""

import os
import subprocess
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from build_tools.ci import change_detector


class ChangeDetectorTest(parameterized.TestCase):

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
        change_detector.is_global_config_changed(changed_files), expected
    )

  @parameterized.parameters(
      (["README.md", "docs/index.md"], True),
      (["OWNERS", "LICENSE", "LICENSE.txt"], True),
      ([".gitignore", ".clang-format", ".clang-tidy", ".clangd"], True),
      (["logo.png", "diagram.svg"], True),
      (["README.md", "xla/service/hlo_parser.cc"], False),
      (["LICENSE.py"], False),
      (["LICENSE.sh"], False),
      ([".clang_hook.sh"], False),
      ([".gitignore.py"], False),
      (["docs/script.py"], False),
      ([], False),
  )
  def test_is_docs_or_metadata_only(self, changed_files, expected):
    self.assertEqual(
        change_detector.is_docs_or_metadata_only(changed_files), expected
    )

  @parameterized.parameters(
      ("26672b856b96dfb0a893f160ee61525526366663",),
      ("HEAD",),
      ("HEAD~1",),
      ("origin/main",),
      ("base_sha",),
  )
  def test_validate_git_ref_valid(self, ref):
    self.assertEqual(change_detector.validate_git_ref(ref), ref)

  @parameterized.parameters(
      ("",),
      ("-f",),
      ("--upload-pack=evil",),
      ("--output=/tmp/pwn",),
      ("HEAD; rm -rf /",),
      ("ref with spaces",),
  )
  def test_validate_git_ref_invalid(self, ref):
    with self.assertRaises(ValueError):
      change_detector.validate_git_ref(ref)

  @mock.patch.object(
      change_detector.subprocess, "run", autospec=True, spec_set=True
  )
  def test_get_merge_base_success(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["git", "merge-base"], returncode=0, stdout="base_commit\n"
    )
    self.assertEqual(
        change_detector.get_merge_base("base", "head"), "base_commit"
    )
    mock_run.assert_called_once_with(
        ["git", "merge-base", "--", "base", "head"],
        cwd=".",
        capture_output=True,
        text=True,
        check=False,
    )

  @mock.patch.object(
      change_detector.subprocess, "run", autospec=True, spec_set=True
  )
  def test_get_merge_base_failure(self, mock_run):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["git", "merge-base"], returncode=128, stdout=""
    )
    self.assertIsNone(change_detector.get_merge_base("base", "head"))

  @mock.patch.object(
      change_detector.subprocess, "run", autospec=True, spec_set=True
  )
  def test_get_merge_base_rejects_flag_injection(self, mock_run):
    self.assertIsNone(
        change_detector.get_merge_base("--upload-pack=evil", "HEAD")
    )
    mock_run.assert_not_called()

  @mock.patch.object(
      change_detector, "get_merge_base", autospec=True, spec_set=True
  )
  def test_get_diff_base_with_merge_base(self, mock_mb):
    mock_mb.return_value = "merge_base_sha"
    self.assertEqual(
        change_detector.get_diff_base("base_sha", "head_sha"), "merge_base_sha"
    )

  @mock.patch.object(
      change_detector, "get_merge_base", autospec=True, spec_set=True
  )
  def test_get_diff_base_fallback(self, mock_mb):
    mock_mb.return_value = None
    self.assertEqual(
        change_detector.get_diff_base("base_sha", "head_sha"), "base_sha"
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="docs_only_xla_build",
          changed_files=["README.md", "docs/architecture.md"],
          is_jax_build=False,
          expected_skip=True,
      ),
      dict(
          testcase_name="docs_only_jax_build",
          changed_files=["README.md", "OWNERS"],
          is_jax_build=True,
          expected_skip=True,
      ),
      dict(
          testcase_name="non_docs_xla_build_defers_to_bazel_diff",
          changed_files=["xla/service/gpu/fusion_merger_test.cc"],
          is_jax_build=False,
          expected_skip=False,
      ),
      dict(
          testcase_name="xla_tests_and_benchmarks_skip_jax",
          changed_files=[
              "xla/service/gpu/fusion_merger_test.cc",
              "xla/hlo/ir/hlo_instruction_benchmark.cc",
              "xla/service/gpu/tests/gemm_rewrite.hlo",
          ],
          is_jax_build=True,
          expected_skip=True,
      ),
      dict(
          testcase_name="mosaic_test_does_not_skip_jax",
          changed_files=["xla/mosaic/gpu/mosaic_gpu_test.py"],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="workflow_and_build_tools_do_not_skip_jax",
          changed_files=[
              ".github/workflows/benchmark_presubmit.yml",
              "build_tools/rocm/run_xla.sh",
              "build_tools/lint/diff_parser.py",
          ],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="ci_yml_does_not_skip_jax",
          changed_files=[".github/workflows/ci.yml"],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="build_tools_ci_does_not_skip_jax",
          changed_files=["build_tools/ci/build.py"],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="gpu_sources_do_not_skip_jax",
          changed_files=["xla/service/gpu/nvptx_compiler.cc"],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="build_file_does_not_skip_jax",
          changed_files=[
              "xla/service/gpu/fusion_merger_test.cc",
              "xla/service/gpu/BUILD",
          ],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="proto_file_does_not_skip_jax",
          changed_files=["xla/service/gpu/model/hlo_op_profile.proto"],
          is_jax_build=True,
          expected_skip=False,
      ),
      dict(
          testcase_name="core_hlo_change_does_not_skip_jax",
          changed_files=["xla/hlo/ir/hlo_instruction.cc"],
          is_jax_build=True,
          expected_skip=False,
      ),
  )
  @mock.patch.object(
      change_detector, "ensure_base_fetched", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_changed_files", autospec=True, spec_set=True
  )
  def test_evaluate_skip(
      self,
      mock_changed,
      mock_fetch,
      changed_files,
      is_jax_build,
      expected_skip,
  ):
    mock_changed.return_value = changed_files
    decision = change_detector.evaluate_skip(
        "base_sha",
        "head_sha",
        cwd="/workspace/openxla/xla",
        is_jax_build=is_jax_build,
    )
    self.assertEqual(decision.should_skip, expected_skip)
    self.assertLen(changed_files, decision.changed_files_count)
    mock_fetch.assert_called_once_with(
        "base_sha", "head_sha", cwd="/workspace/openxla/xla"
    )

  @mock.patch.object(
      change_detector, "ensure_base_fetched", autospec=True, spec_set=True
  )
  @mock.patch.object(
      change_detector, "get_changed_files", autospec=True, spec_set=True
  )
  def test_evaluate_skip_git_fail_open(self, mock_changed, mock_fetch):
    mock_changed.side_effect = OSError("git failure")
    decision = change_detector.evaluate_skip(
        "base_sha", "head_sha", is_jax_build=True
    )
    self.assertFalse(decision.should_skip)
    self.assertIn("Failed to get git changed files", decision.reason)
    mock_fetch.assert_called_once()

  def test_report_skip(self):
    summary_file = self.create_tempfile("step_summary.md")
    with mock.patch.dict(
        os.environ, {"GITHUB_STEP_SUMMARY": summary_file.full_path}
    ):
      decision = change_detector.SkipDecision(
          should_skip=True,
          reason="Modified files do not affect JAX builds",
          changed_files_count=3,
          elapsed_seconds=0.04,
      )
      change_detector.report_skip(
          decision, "BuildType.JAX_LINUX_X86_CPU_BZLMOD_GITHUB_ACTIONS"
      )

    content = summary_file.read_text()
    self.assertIn("Change Detector Analysis", content)
    self.assertIn("JAX_LINUX_X86_CPU_BZLMOD_GITHUB_ACTIONS", content)
    self.assertIn("SKIP", content)
    self.assertIn("Modified files do not affect JAX builds", content)


if __name__ == "__main__":
  absltest.main()
