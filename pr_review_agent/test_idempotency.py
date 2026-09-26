# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

# pylint: disable=bad-indentation,line-too-long

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import requests

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Provide lightweight mocks for optional external SDKs if running in a bare
# Python test environment
for mod_name in [
    "dotenv",
    "google",
    "google.adk",
    "google.adk.agents",
    "google.adk.agents.run_config",
    "google.adk.cli",
    "google.adk.cli.utils",
    "google.adk.runners",
    "google.genai",
    "google.genai.errors",
    "google.genai.types",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


class _MockAPIError(Exception):
    pass


class _MockClientError(_MockAPIError):
    pass


class _MockServerError(_MockAPIError):
    pass


sys.modules["google.genai.errors"].APIError = _MockAPIError
sys.modules["google.genai.errors"].ClientError = _MockClientError
sys.modules["google.genai.errors"].ServerError = _MockServerError

os.environ["GITHUB_TOKEN"] = "test-token"
os.environ["OWNER"] = "tensorflow"
os.environ["REPO"] = "tensorflow"
os.environ["PULL_REQUEST_NUMBER"] = "12345"

from agent import agent, main, utils  # pylint: disable=wrong-import-position


class TestCommitIdempotency(unittest.TestCase):

    def setUp(self):
        agent._PREFETCHED_PR_DETAILS = None

    @patch("agent.utils.get_request")
    def test_1_agent_marker_and_matching_commit_returns_true(
        self, mock_get_request
    ):
        """1. matching agent marker + matching commit -> True"""
        sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e"
        body_text = (
            f"Category: Bug fix\n\nSummary...\n\n"
            f"<!-- tensorflow-pr-review-agent: commit_sha={sha} -->"
        )
        mock_get_request.return_value = [
            {
                "id": 999,
                "commit_id": sha,
                "body": body_text,
            }
        ]
        self.assertTrue(
            utils.has_agent_reviewed_commit(
                "https://api.github.com/repos/tf/tf/pulls/1/reviews", sha
            )
        )

    @patch("agent.utils.get_request")
    def test_2_agent_marker_and_different_commit_returns_false(
        self, mock_get_request
    ):
        """2. matching agent marker + different commit -> False"""
        old_sha = "1111111111111111111111111111111111111111"
        new_sha = "2222222222222222222222222222222222222222"
        body_text = (
            f"Category: Bug fix\n\nSummary...\n\n"
            f"<!-- tensorflow-pr-review-agent: commit_sha={old_sha} -->"
        )
        mock_get_request.return_value = [
            {
                "id": 999,
                "commit_id": old_sha,
                "body": body_text,
            }
        ]
        self.assertFalse(
            utils.has_agent_reviewed_commit(
                "https://api.github.com/repos/tf/tf/pulls/1/reviews", new_sha
            )
        )

    @patch("agent.utils.get_request")
    def test_3_same_commit_but_unrelated_review_body_returns_false(
        self, mock_get_request
    ):
        """3. same commit + unrelated review -> False"""
        sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e"
        mock_get_request.return_value = [
            {
                "id": 1001,
                "commit_id": sha,
                "body": "LGTM! Looks great to me.",
            }
        ]
        self.assertFalse(
            utils.has_agent_reviewed_commit(
                "https://api.github.com/repos/tf/tf/pulls/1/reviews", sha
            )
        )

    @patch("agent.utils.get_request")
    def test_4_same_commit_with_category_prefix_no_marker_returns_false(
        self, mock_get_request
    ):
        """4. same commit + Category: but no agent marker -> False"""
        sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e"
        body_text = (
            "Category: Bug fix\n"
            "Reason: Human review using similar header without agent marker."
        )
        mock_get_request.return_value = [
            {
                "id": 1002,
                "commit_id": sha,
                "body": body_text,
            }
        ]
        self.assertFalse(
            utils.has_agent_reviewed_commit(
                "https://api.github.com/repos/tf/tf/pulls/1/reviews", sha
            )
        )

    @patch("agent.utils.get_request")
    @patch("agent.utils.requests.post")
    def test_5_failed_review_submission_leaves_commit_eligible_for_retry(
        self, mock_post, mock_get_request
    ):
        """5. failed review submission -> commit remains eligible for retry"""
        sha = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        agent._PREFETCHED_PR_DETAILS = {
            "status": "success",
            "pull_request": {
                "headRefOid": sha,
            },
        }
        # Simulate GitHub HTTP 500 failure during review POST
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = (
            requests.exceptions.HTTPError("500 Server Error")
        )
        mock_post.return_value = mock_resp

        res = agent.submit_pr_code_review(
            pr_number=12345,
            overall_assessment="No actionable review comments identified.",
            summary_comment="Summary",
            inline_comments=[],
        )
        self.assertEqual(res["status"], "error")

        # Because POST failed, GitHub's reviews endpoint has no review for sha
        mock_get_request.return_value = []
        self.assertFalse(
            utils.has_agent_reviewed_commit(
                "https://api.github.com/repos/tf/tf/pulls/12345/reviews", sha
            )
        )

    @patch("agent.main.clear_and_set_reaction")
    @patch("agent.main.has_agent_reviewed_commit")
    @patch("agent.agent.get_pull_request_details")
    @patch("agent.agent.run_pr_review", new_callable=AsyncMock)
    def test_6_successful_review_repeated_label_event_skips_llm(
        self, mock_run_pr_review, mock_get_details, mock_has_reviewed, mock_react
    ):
        """6. successful review + repeated Needs Review event -> skip LLM"""
        sha = "abcdef1234567890abcdef1234567890abcdef12"
        mock_get_details.return_value = {
            "status": "success",
            "pull_request": {
                "headRefOid": sha,
                "title": "Fix bug",
                "body": "Body",
                "diff": "",
                "files": {"nodes": []},
            },
        }
        # Simulate repeated Needs Review label event where commit sha already
        # has a persisted agent review on GitHub
        mock_has_reviewed.return_value = True

        asyncio.run(main.main())

        # Verify LLM review was NOT invoked
        mock_run_pr_review.assert_not_called()
        # Verify final rocket reaction was set
        mock_react.assert_called_with(12345, add_content="rocket")


class TestPylintIntegration(unittest.TestCase):

    def setUp(self):
        agent._PREFETCHED_PR_DETAILS = None

    def test_1_extract_modified_lines_from_unified_diff(self):
        """1. modified-line extraction from raw and annotated unified diffs"""
        raw_diff = (
            "--- a/tensorflow/python/ops/math_ops.py\n"
            "+++ b/tensorflow/python/ops/math_ops.py\n"
            "@@ -10,4 +10,5 @@\n"
            " context_line\n"
            "-deleted_line\n"
            "+added_line_11\n"
            "+added_line_12\n"
            " trailing_context\n"
        )
        mod_raw = utils.extract_modified_lines_by_file(raw_diff)
        self.assertEqual(
            mod_raw,
            {"tensorflow/python/ops/math_ops.py": {11, 12}},
        )

        annotated_diff = utils.annotate_diff_with_line_numbers(raw_diff)
        mod_annotated = utils.extract_modified_lines_by_file(annotated_diff)
        self.assertEqual(
            mod_annotated,
            {"tensorflow/python/ops/math_ops.py": {11, 12}},
        )

    @patch("agent.utils.subprocess.run")
    def test_2_skip_pylint_when_no_python_files_changed(self, mock_subproc_run):
        """2. skipping Pylint when no Python files changed (or only deleted)"""
        files = [
            {"path": "tensorflow/core/kernels/matmul_op.cc", "changeType": "MODIFIED"},
            {"path": "README.md", "changeType": "ADDED"},
            {"path": "tensorflow/python/deprecated.py", "changeType": "DELETED"},
        ]
        output = utils.run_pylint_on_changed_files(files, raw_diff="", head_sha="abc1234")
        self.assertEqual(
            output, "No Python files were modified in this pull request."
        )
        mock_subproc_run.assert_not_called()

    @patch("agent.utils._fetch_file_content_at_commit")
    @patch("agent.utils.subprocess.run")
    def test_3_filter_diagnostics_to_changed_lines(
        self, mock_subproc_run, mock_fetch_content
    ):
        """3. filtering diagnostics to changed lines and PR-caused import issues"""
        files = [{"path": "tensorflow/python/ops/math_ops.py", "changeType": "MODIFIED"}]
        raw_diff = (
            "--- a/tensorflow/python/ops/math_ops.py\n"
            "+++ b/tensorflow/python/ops/math_ops.py\n"
            "@@ -14,3 +14,3 @@\n"
            " def my_func():\n"
            "-  return unused_helper.compute()\n"
            "+   return 42\n"
        )
        mock_fetch_content.return_value = (
            "import legacy_os\n"
            "import unused_helper\n"
            "def my_func():\n"
            "   return 42\n"
        )
        mock_proc_res = MagicMock()
        mock_proc_res.returncode = 4
        mock_proc_res.stdout = (
            "************* Module math_ops\n"
            "tensorflow/python/ops/math_ops.py:15:0: W0311 (bad-indentation): Bad indentation. Found 3 spaces, expected 2\n"
            "tensorflow/python/ops/math_ops.py:5:0: W0611 (unused-import): Unused import unused_helper\n"
            "tensorflow/python/ops/math_ops.py:3:0: W0611 (unused-import): Unused import legacy_os\n"
            "tensorflow/python/ops/math_ops.py:200:0: C0301 (line-too-long): Line too long (120/80)\n"
        )
        mock_subproc_run.return_value = mock_proc_res

        output = utils.run_pylint_on_changed_files(
            files, raw_diff=raw_diff, head_sha="a1b2c3d4"
        )
        self.assertIn("math_ops.py:15:0: W0311 (bad-indentation)", output)
        self.assertIn("math_ops.py:5:0: W0611 (unused-import): Unused import unused_helper", output)
        self.assertNotIn("legacy_os", output)
        self.assertNotIn("line-too-long", output)

    @patch("agent.utils._fetch_file_content_at_commit")
    @patch("agent.utils.subprocess.run")
    def test_4_graceful_handling_of_pylint_timeout_and_failure(
        self, mock_subproc_run, mock_fetch_content
    ):
        """4. graceful handling of Pylint timeout and execution failure"""
        import subprocess as sp

        files = [{"path": "tensorflow/python/ops/math_ops.py", "changeType": "MODIFIED"}]
        mock_fetch_content.return_value = "x = 1\n"

        mock_subproc_run.side_effect = sp.TimeoutExpired(cmd="pylint", timeout=120)
        timeout_out = utils.run_pylint_on_changed_files(
            files, raw_diff="", head_sha="a1b2c3d4"
        )
        self.assertEqual(
            timeout_out, "Pylint static analysis timed out and was skipped."
        )

        mock_subproc_run.side_effect = OSError("Subprocess execution error")
        err_out = utils.run_pylint_on_changed_files(
            files, raw_diff="", head_sha="a1b2c3d4"
        )
        self.assertIn("Pylint static analysis could not be completed:", err_out)

    @patch("agent.main.clear_and_set_reaction")
    @patch("agent.main.run_pylint_on_changed_files")
    @patch("agent.main.has_agent_reviewed_commit")
    @patch("agent.agent.get_pull_request_details")
    @patch("agent.agent.run_pr_review", new_callable=AsyncMock)
    def test_5_idempotency_prevents_pylint_execution(
        self,
        mock_run_pr_review,
        mock_get_details,
        mock_has_reviewed,
        mock_run_pylint,
        mock_react,
    ):
        """5. idempotency check prevents Pylint execution on already-reviewed commit"""
        sha = "9999999999999999999999999999999999999999"
        mock_get_details.return_value = {
            "status": "success",
            "pull_request": {
                "headRefOid": sha,
                "title": "Fix issue",
                "body": "Body",
                "diff": "",
                "files": {
                    "nodes": [
                        {"path": "tensorflow/python/foo.py", "changeType": "MODIFIED"}
                    ]
                },
            },
        }
        mock_has_reviewed.return_value = True

        asyncio.run(main.main())

        mock_run_pylint.assert_not_called()
        mock_run_pr_review.assert_not_called()
        mock_react.assert_called_with(12345, add_content="rocket")

    @patch("agent.main.clear_and_set_reaction")
    @patch("agent.main.run_pylint_on_changed_files")
    @patch("agent.main.has_agent_reviewed_commit")
    @patch("agent.agent.get_pull_request_details")
    @patch("agent.agent.run_pr_review", new_callable=AsyncMock)
    def test_6_pylint_runs_once_across_model_fallback(
        self,
        mock_run_pr_review,
        mock_get_details,
        mock_has_reviewed,
        mock_run_pylint,
        mock_react,
    ):
        """6. Pylint runs exactly once before model fallback loop and evidence is reused"""
        sha = "7777777777777777777777777777777777777777"
        mock_get_details.return_value = {
            "status": "success",
            "pull_request": {
                "headRefOid": sha,
                "title": "Add python op",
                "body": "Body",
                "diff": "@@ -1,1 +1,1 @@\n+x=1",
                "files": {
                    "nodes": [
                        {"path": "tensorflow/python/foo.py", "changeType": "MODIFIED"}
                    ]
                },
            },
        }
        # First check (startup idempotency check): False
        # Second check (after model 1 raises 503): False
        # Third check (after model 2 succeeds): True
        mock_has_reviewed.side_effect = [False, False, True]
        pylint_diag = (
            "tensorflow/python/foo.py:1:0: W0311 (bad-indentation): Bad indentation"
        )
        mock_run_pylint.return_value = pylint_diag

        mock_run_pr_review.side_effect = [
            _MockServerError("503 UNAVAILABLE"),
            "Review submitted successfully",
        ]

        asyncio.run(main.main())

        self.assertEqual(mock_run_pylint.call_count, 1)
        self.assertEqual(mock_run_pr_review.call_count, 2)
        for call_args in mock_run_pr_review.call_args_list:
            self.assertEqual(call_args.kwargs.get("pylint_output"), pylint_diag)
        mock_react.assert_called_with(12345, add_content="rocket")

    @patch("agent.utils._fetch_file_content_at_commit")
    @patch("agent.utils.subprocess.run")
    def test_7_pylint_path_and_option_injection_hardening(
        self, mock_subproc_run, mock_fetch_content
    ):
        """7. option-like paths are rejected and '--' precedes materialized paths in Pylint cmd"""
        self.assertFalse(utils._is_safe_relative_path("--init-hook=evil.py"))
        self.assertFalse(utils._is_safe_relative_path("tensorflow/--init-hook=evil.py"))
        self.assertFalse(utils._is_safe_relative_path("-rcfile=evil.py"))
        self.assertTrue(utils._is_safe_relative_path("tensorflow/foo.py"))

        files = [
            {"path": "--init-hook=evil.py", "changeType": "ADDED"},
            {"path": "tensorflow/--rcfile=evil.py", "changeType": "ADDED"},
            {"path": "tensorflow/foo.py", "changeType": "MODIFIED"},
        ]
        mock_fetch_content.return_value = "x = 1\n"
        mock_proc_res = MagicMock()
        mock_proc_res.returncode = 0
        mock_proc_res.stdout = ""
        mock_subproc_run.return_value = mock_proc_res

        utils.run_pylint_on_changed_files(
            files, raw_diff="", head_sha="a1b2c3d4"
        )

        self.assertEqual(mock_subproc_run.call_count, 1)
        cmd = mock_subproc_run.call_args[0][0]
        self.assertEqual(cmd[:4], [sys.executable, "-P", "-m", "pylint"])
        self.assertIn("--", cmd)
        sep_idx = cmd.index("--")
        self.assertEqual(cmd[sep_idx + 1 :], ["tensorflow/foo.py"])
        self.assertNotIn("--init-hook=evil.py", cmd)


class TestModelFallback(unittest.TestCase):

    def setUp(self):
        agent._PREFETCHED_PR_DETAILS = None

    def _mock_pr_payload(self, sha: str) -> dict:
        return {
            "status": "success",
            "pull_request": {
                "headRefOid": sha,
                "title": "Fix tensor op",
                "body": "Body",
                "diff": "@@ -1,1 +1,1 @@\n+x = 1",
                "files": {
                    "nodes": [
                        {"path": "tensorflow/python/foo.py", "changeType": "MODIFIED"}
                    ]
                },
            },
        }

    @patch("agent.main.clear_and_set_reaction")
    @patch("agent.main.run_pylint_on_changed_files")
    @patch("agent.main.has_agent_reviewed_commit")
    @patch("agent.agent.get_pull_request_details")
    @patch("agent.agent.run_pr_review", new_callable=AsyncMock)
    def test_1_retired_model_404_causes_fallback_to_next_model(
        self,
        mock_run_pr_review,
        mock_get_details,
        mock_has_reviewed,
        mock_run_pylint,
        mock_react,
    ):
        """1. retired/unavailable model (404 NOT_FOUND) triggers fallback to next model"""
        sha = "4040404040404040404040404040404040404040"
        mock_get_details.return_value = self._mock_pr_payload(sha)
        mock_has_reviewed.side_effect = [False, False, True]
        mock_run_pylint.return_value = "No Pylint issues detected on modified lines."

        mock_run_pr_review.side_effect = [
            _MockClientError(
                "404 NOT_FOUND: This model models/gemini-3-pro-preview is no longer available."
            ),
            "Review submitted successfully",
        ]

        asyncio.run(main.main())

        self.assertEqual(mock_run_pylint.call_count, 1)
        self.assertEqual(mock_run_pr_review.call_count, 2)
        self.assertEqual(
            mock_run_pr_review.call_args_list[0].kwargs.get("model_name"),
            agent.MODELS_POOL[0],
        )
        self.assertEqual(
            mock_run_pr_review.call_args_list[1].kwargs.get("model_name"),
            agent.MODELS_POOL[1],
        )
        mock_react.assert_called_with(12345, add_content="rocket")

    @patch("agent.main.clear_and_set_reaction")
    @patch("agent.main.run_pylint_on_changed_files")
    @patch("agent.main.has_agent_reviewed_commit")
    @patch("agent.agent.get_pull_request_details")
    @patch("agent.agent.run_pr_review", new_callable=AsyncMock)
    def test_2_overloaded_model_503_causes_fallback_to_next_model(
        self,
        mock_run_pr_review,
        mock_get_details,
        mock_has_reviewed,
        mock_run_pylint,
        mock_react,
    ):
        """2. temporary service failure (503 UNAVAILABLE) triggers fallback to next model"""
        sha = "5030503050305030503050305030503050305030"
        mock_get_details.return_value = self._mock_pr_payload(sha)
        mock_has_reviewed.side_effect = [False, False, True]
        mock_run_pylint.return_value = "No Pylint issues detected on modified lines."

        mock_run_pr_review.side_effect = [
            _MockServerError("503 UNAVAILABLE: Model overloaded"),
            "Review submitted successfully",
        ]

        asyncio.run(main.main())

        self.assertEqual(mock_run_pylint.call_count, 1)
        self.assertEqual(mock_run_pr_review.call_count, 2)
        mock_react.assert_called_with(12345, add_content="rocket")

    @patch("agent.main.clear_and_set_reaction")
    @patch("agent.main.run_pylint_on_changed_files")
    @patch("agent.main.has_agent_reviewed_commit")
    @patch("agent.agent.get_pull_request_details")
    @patch("agent.agent.run_pr_review", new_callable=AsyncMock)
    def test_3_successful_first_model_stops_fallback(
        self,
        mock_run_pr_review,
        mock_get_details,
        mock_has_reviewed,
        mock_run_pylint,
        mock_react,
    ):
        """3. successful first model completes immediately and stops fallback"""
        sha = "2000200020002000200020002000200020002000"
        mock_get_details.return_value = self._mock_pr_payload(sha)
        mock_has_reviewed.side_effect = [False, True]
        mock_run_pylint.return_value = "No Pylint issues detected on modified lines."
        mock_run_pr_review.return_value = "Review submitted successfully"

        asyncio.run(main.main())

        self.assertEqual(mock_run_pylint.call_count, 1)
        self.assertEqual(mock_run_pr_review.call_count, 1)
        self.assertEqual(
            mock_run_pr_review.call_args_list[0].kwargs.get("model_name"),
            agent.MODELS_POOL[0],
        )
        mock_react.assert_called_with(12345, add_content="rocket")

    @patch("agent.main.clear_and_set_reaction")
    @patch("agent.main.run_pylint_on_changed_files")
    @patch("agent.main.has_agent_reviewed_commit")
    @patch("agent.agent.get_pull_request_details")
    @patch("agent.agent.run_pr_review", new_callable=AsyncMock)
    def test_4_all_models_failing_produces_clear_final_failure(
        self,
        mock_run_pr_review,
        mock_get_details,
        mock_has_reviewed,
        mock_run_pylint,
        mock_react,
    ):
        """4. all models failing produces a clear final RuntimeError failure"""
        sha = "fa11fa11fa11fa11fa11fa11fa11fa11fa11fa11"
        mock_get_details.return_value = self._mock_pr_payload(sha)
        mock_has_reviewed.return_value = False
        mock_run_pylint.return_value = "No Pylint issues detected on modified lines."

        mock_run_pr_review.side_effect = [
            _MockClientError("404 NOT_FOUND"),
            *[_MockServerError("503 UNAVAILABLE") for _ in range(len(agent.MODELS_POOL) - 1)],
        ]

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(main.main())

        self.assertIn("All models in MODELS_POOL failed", str(ctx.exception))
        self.assertEqual(mock_run_pylint.call_count, 1)
        self.assertEqual(mock_run_pr_review.call_count, len(agent.MODELS_POOL))

    @patch("agent.main.clear_and_set_reaction")
    @patch("agent.main.run_pylint_on_changed_files")
    @patch("agent.main.has_agent_reviewed_commit")
    @patch("agent.agent.get_pull_request_details")
    @patch("agent.agent.run_pr_review", new_callable=AsyncMock)
    def test_5_non_fallback_error_does_not_silently_continue(
        self,
        mock_run_pr_review,
        mock_get_details,
        mock_has_reviewed,
        mock_run_pylint,
        mock_react,
    ):
        """5. non-fallback error (401 UNAUTHENTICATED) raises immediately without fallback"""
        sha = "4010401040104010401040104010401040104010"
        mock_get_details.return_value = self._mock_pr_payload(sha)
        mock_has_reviewed.return_value = False
        mock_run_pylint.return_value = "No Pylint issues detected on modified lines."

        mock_run_pr_review.side_effect = _MockClientError(
            "401 UNAUTHENTICATED: API key not valid."
        )

        with self.assertRaises(_MockClientError):
            asyncio.run(main.main())

        self.assertEqual(mock_run_pr_review.call_count, 1)


if __name__ == "__main__":
    unittest.main()
