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
"""Tests for build module."""

from collections.abc import Sequence
import subprocess
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from build_tools.ci import build


class BuildTest(parameterized.TestCase):

  def test_windows_builds_configure_command_retries(self) -> None:
    builds = build.Build.all_builds()
    self.assertEqual(
        builds[
            build.BuildType.XLA_WINDOWS_X86_CPU_GITHUB_ACTIONS
        ].command_retries,
        2,
    )
    self.assertEqual(
        builds[
            build.BuildType.JAX_WINDOWS_X86_CPU_GITHUB_ACTIONS
        ].command_retries,
        2,
    )
    self.assertEqual(
        builds[
            build.BuildType.XLA_LINUX_X86_CPU_GITHUB_ACTIONS
        ].command_retries,
        0,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="xla_windows_succeeds_on_first_attempt",
          build_type=build.BuildType.XLA_WINDOWS_X86_CPU_GITHUB_ACTIONS,
          returncodes=[0],
      ),
      dict(
          testcase_name="xla_windows_succeeds_on_first_retry",
          build_type=build.BuildType.XLA_WINDOWS_X86_CPU_GITHUB_ACTIONS,
          returncodes=[1, 0],
      ),
      dict(
          testcase_name="xla_windows_succeeds_on_second_retry",
          build_type=build.BuildType.XLA_WINDOWS_X86_CPU_GITHUB_ACTIONS,
          returncodes=[1, 1, 0],
      ),
      dict(
          testcase_name="jax_windows_succeeds_on_first_retry",
          build_type=build.BuildType.JAX_WINDOWS_X86_CPU_GITHUB_ACTIONS,
          returncodes=[1, 0],
      ),
  )
  @mock.patch.object(build, "sh", autospec=True, spec_set=True)
  def test_execute_build_commands_windows_retries_and_succeeds(
      self,
      mock_sh: mock.MagicMock,
      build_type: build.BuildType,
      returncodes: Sequence[int],
  ) -> None:
    mock_sh.side_effect = [
        subprocess.CompletedProcess(args=["bazel"], returncode=return_code)
        for return_code in returncodes
    ]
    target_build = build.Build.all_builds()[build_type]
    build.execute_build_commands(target_build)
    self.assertLen(returncodes, mock_sh.call_count)
    mock_sh.assert_called_with(
        target_build.commands(target_pattern_file=None)[0], check=False
    )

  @mock.patch.object(build, "sh", autospec=True, spec_set=True)
  def test_execute_build_commands_windows_exits_after_exhausting_retries(
      self, mock_sh: mock.MagicMock
  ) -> None:
    mock_sh.return_value = subprocess.CompletedProcess(
        args=["bazel"], returncode=1
    )
    target_build = build.Build.all_builds()[
        build.BuildType.XLA_WINDOWS_X86_CPU_GITHUB_ACTIONS
    ]
    with self.assertLogs(level="WARNING") as logs:
      with self.assertRaisesRegex(SystemExit, "^1$"):
        build.execute_build_commands(target_build)
    self.assertEqual(mock_sh.call_count, 3)
    self.assertLen(logs.output, 2)
    self.assertIn("attempt 1/3", logs.output[0])
    self.assertIn("attempt 2/3", logs.output[1])

  @mock.patch.object(build, "sh", autospec=True, spec_set=True)
  def test_execute_build_commands_non_windows_does_not_retry(
      self, mock_sh: mock.MagicMock
  ) -> None:
    mock_sh.side_effect = [
        subprocess.CompletedProcess(args=["parallel"], returncode=0),
        subprocess.CompletedProcess(args=["bazel"], returncode=2),
    ]
    target_build = build.Build.all_builds()[
        build.BuildType.XLA_LINUX_X86_CPU_GITHUB_ACTIONS
    ]
    with self.assertRaisesRegex(SystemExit, "^2$"):
      build.execute_build_commands(target_build)
    self.assertEqual(mock_sh.call_count, 2)

  @mock.patch.object(build, "sh", autospec=True, spec_set=True)
  def test_execute_build_commands_exit_code_4_with_target_pattern_file(
      self, mock_sh: mock.MagicMock
  ) -> None:
    mock_sh.return_value = subprocess.CompletedProcess(
        args=["bazel"], returncode=4
    )
    target_build = build.Build.all_builds()[
        build.BuildType.XLA_WINDOWS_X86_CPU_GITHUB_ACTIONS
    ]
    build.execute_build_commands(
        target_build, target_pattern_file="/tmp/targets.txt"
    )
    self.assertEqual(mock_sh.call_count, 1)

  @mock.patch.object(build, "sh", autospec=True, spec_set=True)
  def test_execute_build_commands_exit_code_4_without_target_pattern_file(
      self, mock_sh: mock.MagicMock
  ) -> None:
    mock_sh.return_value = subprocess.CompletedProcess(
        args=["bazel"], returncode=4
    )
    target_build = build.Build.all_builds()[
        build.BuildType.XLA_WINDOWS_X86_CPU_GITHUB_ACTIONS
    ]
    with self.assertRaisesRegex(SystemExit, "^4$"):
      build.execute_build_commands(target_build, target_pattern_file=None)
    self.assertEqual(mock_sh.call_count, 3)


if __name__ == "__main__":
  absltest.main()
