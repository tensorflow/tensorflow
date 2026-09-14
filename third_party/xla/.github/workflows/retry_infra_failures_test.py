# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
"""Unit tests verifying detection and retrying of CI infrastructure failures."""

import tempfile
import unittest
from unittest import mock

import retry_infra_failures


class RetryInfraFailuresTest(unittest.TestCase):

  def test_check_text_for_infra_errors_matches(self):
    cases = [
        (
            (
                "FetchError: request to"
                " https://34.118.224.1/apis/authorization.k8s.io/v1/"
                "selfsubjectaccessreviews failed, reason: connect ETIMEDOUT"
            ),
            "GKE control plane SelfSubjectAccessReview authorization timeout",
        ),
        (
            (
                "Executing the custom container implementation failed. Please"
                " contact your self hosted runner administrator."
            ),
            "Runner container hook execution failure",
        ),
        (
            (
                "The runner has received a shutdown signal. This can happen"
                " when a RunnerPod is evicted"
            ),
            "Runner pod shutdown / eviction signal received",
        ),
        (
            (
                "Runner lost communication with the server. Verify the machine"
                " is running and connected."
            ),
            "Runner lost communication with GitHub Actions backend",
        ),
        (
            (
                "Cannot connect to the Docker daemon at"
                " unix:///var/run/docker.sock. Is the docker daemon running?"
            ),
            "Runner container runtime daemon connection failure",
        ),
        (
            "A container failed to start. Error: CreateContainerConfigError",
            "Kubernetes container creation error",
        ),
    ]
    for text, expected_category in cases:
      match = retry_infra_failures.check_text_for_infra_errors(
          text=text,
          job_id=123,
          job_name="test-job",
          source="test",
      )
      self.assertIsNotNone(match, f"Expected match for: {text}")
      self.assertEqual(match.category, expected_category)
      self.assertEqual(match.job_id, 123)
      self.assertEqual(match.job_name, "test-job")

  def test_check_text_for_infra_errors_ignores_normal_failures(self):
    normal_failures = [
        "FAILED: //xla/tests:gpu_test (see sponge/12345)",
        "ERROR: /workspace/xla/BUILD:10:1: Target '//xla:foo' has syntax error",
        "AssertionError: expected 42 to equal 0",
        (
            "Aspect"
            " build_tools/dependencies/aspects.bzl%validate_gpu_tag failed:"
            " Violation found"
        ),
    ]
    for text in normal_failures:
      match = retry_infra_failures.check_text_for_infra_errors(
          text=text,
          job_id=456,
          job_name="test-job",
          source="test",
      )
      self.assertIsNone(match, f"Should not match code failure: {text}")

  def test_inspect_job_annotations_finds_match(self):
    mock_api = mock.MagicMock()
    mock_api.get_check_run_annotations.return_value = [
        {
            "message": "Process completed with exit code 1.",
            "title": "Failure",
        },
        {
            "message": (
                "Executing the custom container implementation failed. Please"
                " contact your self hosted runner administrator."
            ),
            "title": "Error",
        },
    ]

    match = retry_infra_failures.inspect_job_annotations(
        mock_api, repo="openxla/xla", job_id=101, job_name="job-1"
    )
    self.assertIsNotNone(match)
    self.assertEqual(match.job_id, 101)
    self.assertEqual(
        match.category, "Runner container hook execution failure"
    )

  def test_inspect_job_logs_finds_match(self):
    mock_api = mock.MagicMock()
    log_text = "\n".join([
        "2026-09-14T11:00:00Z INFO Starting build",
        "2026-09-14T11:05:00Z The runner has received a shutdown signal",
        "2026-09-14T11:05:01Z Exiting process",
    ])
    mock_api.get_job_logs.return_value = log_text

    match = retry_infra_failures.inspect_job_logs(
        mock_api, repo="openxla/xla", job_id=102, job_name="job-2"
    )
    self.assertIsNotNone(match)
    self.assertEqual(
        match.category, "Runner pod shutdown / eviction signal received"
    )

  def test_evaluate_and_retry_skips_when_max_attempts_reached(self):
    mock_api = mock.MagicMock()
    mock_api.get_workflow_run.return_value = {
        "status": "completed",
        "conclusion": "failure",
        "run_attempt": 2,
    }

    result = retry_infra_failures.evaluate_and_retry(
        mock_api, repo="openxla/xla", run_id=1000, max_attempts=2
    )
    self.assertFalse(result)
    mock_api.rerun_failed_jobs.assert_not_called()

  def test_evaluate_and_retry_skips_when_not_completed(self):
    mock_api = mock.MagicMock()
    mock_api.get_workflow_run.return_value = {
        "status": "in_progress",
        "conclusion": None,
        "run_attempt": 1,
    }

    result = retry_infra_failures.evaluate_and_retry(
        mock_api, repo="openxla/xla", run_id=1000, max_attempts=2
    )
    self.assertFalse(result)
    mock_api.rerun_failed_jobs.assert_not_called()

  def test_evaluate_and_retry_skips_when_success(self):
    mock_api = mock.MagicMock()
    mock_api.get_workflow_run.return_value = {
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
    }

    result = retry_infra_failures.evaluate_and_retry(
        mock_api, repo="openxla/xla", run_id=1000, max_attempts=2
    )
    self.assertFalse(result)
    mock_api.rerun_failed_jobs.assert_not_called()

  def test_evaluate_and_retry_skips_on_genuine_test_failure(self):
    mock_api = mock.MagicMock()
    mock_api.get_workflow_run.return_value = {
        "status": "completed",
        "conclusion": "failure",
        "run_attempt": 1,
    }
    mock_api.get_workflow_run_jobs.return_value = [
        {"id": 201, "name": "build-cpu", "conclusion": "success"},
        {"id": 202, "name": "test-gpu", "conclusion": "failure"},
    ]
    mock_api.get_check_run_annotations.return_value = [
        {"message": "Test target //xla:foo_test failed with exit code 1"}
    ]
    mock_api.get_job_logs.return_value = "FAIL: //xla:foo_test failed"

    result = retry_infra_failures.evaluate_and_retry(
        mock_api, repo="openxla/xla", run_id=1000, max_attempts=2
    )
    self.assertFalse(result)
    mock_api.rerun_failed_jobs.assert_not_called()

  def test_evaluate_and_retry_triggers_rerun_on_infra_failure(self):
    mock_api = mock.MagicMock()
    mock_api.get_workflow_run.return_value = {
        "status": "completed",
        "conclusion": "failure",
        "run_attempt": 1,
    }
    mock_api.get_workflow_run_jobs.return_value = [
        {"id": 301, "name": "no-gpu-targets", "conclusion": "failure"},
    ]
    mock_api.get_check_run_annotations.return_value = [
        {
            "message": (
                "Executing the custom container implementation failed. Please"
                " contact your self hosted runner administrator."
            )
        }
    ]

    result = retry_infra_failures.evaluate_and_retry(
        mock_api,
        repo="openxla/xla",
        run_id=1000,
        max_attempts=2,
        dry_run=False,
        cooldown_seconds=0,
    )
    self.assertTrue(result)
    mock_api.rerun_failed_jobs.assert_called_once_with("openxla/xla", 1000)

  def test_evaluate_and_retry_dry_run_does_not_trigger_rerun(self):
    mock_api = mock.MagicMock()
    mock_api.get_workflow_run.return_value = {
        "status": "completed",
        "conclusion": "failure",
        "run_attempt": 1,
    }
    mock_api.get_workflow_run_jobs.return_value = [
        {"id": 301, "name": "no-gpu-targets", "conclusion": "failure"},
    ]
    mock_api.get_check_run_annotations.return_value = [
        {
            "message": (
                "Executing the custom container implementation failed. Please"
                " contact your self hosted runner administrator."
            )
        }
    ]

    result = retry_infra_failures.evaluate_and_retry(
        mock_api,
        repo="openxla/xla",
        run_id=1000,
        max_attempts=2,
        dry_run=True,
        cooldown_seconds=0,
    )
    self.assertTrue(result)
    mock_api.rerun_failed_jobs.assert_not_called()

  def test_write_step_summary(self):
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as temp_file:
      with mock.patch.dict(
          "os.environ", {"GITHUB_STEP_SUMMARY": temp_file.name}
      ):
        hook_pattern = (
            r"Executing the custom container implementation failed"
        )
        matches = [
            retry_infra_failures.InfraFailureMatch(
                job_id=401,
                job_name="test-job",
                category="Runner container hook execution failure",
                matched_pattern=hook_pattern,
                source="annotation",
                sample="Executing the custom container implementation failed",
            )
        ]
        retry_infra_failures.write_step_summary(
            matches=matches,
            repo="openxla/xla",
            run_id=5000,
            run_attempt=1,
            max_attempts=2,
            dry_run=False,
            rerun_succeeded=True,
        )

        temp_file.seek(0)
        content = temp_file.read()
        self.assertIn("Infrastructure Auto-Retry Evaluator", content)
        self.assertIn("`test-job`", content)
        self.assertIn("Runner container hook execution failure", content)
        self.assertIn("Rerun Dispatched", content)


if __name__ == "__main__":
  unittest.main()
