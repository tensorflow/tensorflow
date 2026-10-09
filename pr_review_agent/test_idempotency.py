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

import ast
import asyncio
import importlib
import inspect
import logging
import os
from pathlib import Path
import subprocess
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
        agent._VERIFIED_HEAD_SHA = None

    @patch("agent.utils.get_request")
    def test_1_agent_marker_and_matching_commit_returns_true(
        self, mock_get_request
    ):
        """1. matching github-actions[bot] author + agent marker + matching commit -> True"""
        sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e"
        body_text = (
            f"Category: Bug fix\n\nSummary...\n\n"
            f"<!-- tensorflow-pr-review-agent: commit_sha={sha} -->"
        )
        mock_get_request.return_value = [
            {
                "id": 999,
                "user": {"login": "github-actions[bot]"},
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
                "user": {"login": "github-actions[bot]"},
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
                "user": {"login": "github-actions[bot]"},
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
                "user": {"login": "github-actions[bot]"},
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
        """6. successful review + repeated Agent Review event -> skip LLM"""
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
        # Simulate repeated Agent Review label event where commit sha already
        # has a persisted agent review on GitHub
        mock_has_reviewed.return_value = True

        with patch.dict(os.environ, {"PR_HEAD_SHA": sha}):
            asyncio.run(main.main())

        # Verify LLM review was NOT invoked
        mock_run_pr_review.assert_not_called()
        # Verify final rocket reaction was set
        mock_react.assert_called_with(12345, add_content="rocket")


class TestPylintIntegration(unittest.TestCase):

    def setUp(self):
        agent._PREFETCHED_PR_DETAILS = None
        agent._VERIFIED_HEAD_SHA = None

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
        files = [{"path": "tensorflow/python/ops/math_ops.py", "changeType": "MODIFIED"}]
        mock_fetch_content.return_value = "x = 1\n"

        mock_subproc_run.side_effect = subprocess.TimeoutExpired(cmd="pylint", timeout=120)
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

        with patch.dict(os.environ, {"PR_HEAD_SHA": sha}):
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

        with patch.dict(os.environ, {"PR_HEAD_SHA": sha}):
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
        agent._VERIFIED_HEAD_SHA = None

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

        with patch.dict(os.environ, {"PR_HEAD_SHA": sha}):
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

        with patch.dict(os.environ, {"PR_HEAD_SHA": sha}):
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

        with patch.dict(os.environ, {"PR_HEAD_SHA": sha}):
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

        with patch.dict(os.environ, {"PR_HEAD_SHA": sha}):
            with self.assertRaisesRegex(
                RuntimeError, "All models in MODELS_POOL failed"
            ):
                asyncio.run(main.main())

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

        with patch.dict(os.environ, {"PR_HEAD_SHA": sha}):
            with self.assertRaisesRegex(_MockClientError, "401 UNAUTHENTICATED"):
                asyncio.run(main.main())

        self.assertEqual(mock_run_pr_review.call_count, 1)


class TestSecurityHardening(unittest.TestCase):

    def setUp(self):
        agent._PREFETCHED_PR_DETAILS = None
        agent._VERIFIED_HEAD_SHA = None

    @patch("agent.utils.post_pull_request_review")
    def test_finding_1_review_event_always_comment_across_all_assessments(
        self, mock_post_review
    ):
        """Finding 1: submit_pr_code_review always submits event='COMMENT' and preserves inline comments."""
        sha = "1234567890abcdef1234567890abcdef12345678"
        agent._VERIFIED_HEAD_SHA = sha
        mock_post_review.return_value = {"id": 1}

        assessments = [
            "No actionable review comments identified.",
            "Minor improvements suggested.",
            "Changes required.",
        ]
        inline_input = [
            {
                "path": "tensorflow/python/foo.py",
                "line": 10,
                "side": "RIGHT",
                "body": "**[Priority 1: Correctness]** Fix value.",
                "suggestion_code": "x = 2",
            }
        ]
        for assessment in assessments:
            mock_post_review.reset_mock()
            res = agent.submit_pr_code_review(
                overall_assessment=assessment,
                summary_comment="## Summary\nReview summary.",
                inline_comments=inline_input,
            )
            self.assertEqual(res["status"], "success")
            mock_post_review.assert_called_once()
            url, payload = mock_post_review.call_args[0]
            self.assertEqual(
                url,
                "https://api.github.com/repos/tensorflow/tensorflow/pulls/12345/reviews",
            )
            self.assertEqual(payload["event"], "COMMENT")
            self.assertEqual(payload["commit_id"], sha)
            self.assertEqual(len(payload["comments"]), 1)
            self.assertIn("```suggestion\nx = 2\n```", payload["comments"][0]["body"])

    @patch("agent.utils.requests.post")
    def test_finding_1_422_fallback_review_event_is_explicitly_comment(
        self, mock_post
    ):
        """Finding 1: 422 fallback in post_pull_request_review explicitly uses event='COMMENT'."""
        resp_422 = MagicMock()
        resp_422.status_code = 422
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"id": 42}
        mock_post.side_effect = [resp_422, resp_200]

        payload = {
            "body": "Summary",
            "event": "APPROVE",
            "comments": [
                {"path": "tensorflow/foo.py", "line": 5, "body": "Inline note"}
            ],
        }
        utils.post_pull_request_review(
            "https://api.github.com/repos/tensorflow/tensorflow/pulls/12345/reviews",
            payload,
        )
        self.assertEqual(mock_post.call_count, 2)
        fallback_payload = mock_post.call_args_list[1].kwargs["json"]
        self.assertEqual(fallback_payload["event"], "COMMENT")

    @patch("agent.agent.run_graphql_query")
    @patch("agent.agent.get_diff")
    def test_finding_2_llm_tools_reject_pr_number_and_bind_configured_pr(
        self, mock_get_diff, mock_graphql
    ):
        """Finding 2: LLM-facing tools do not accept pr_number and bind to PULL_REQUEST_NUMBER."""
        self.assertNotIn(
            "pr_number",
            inspect.signature(agent.get_pull_request_details).parameters,
        )
        self.assertNotIn(
            "pr_number",
            inspect.signature(agent.submit_pr_code_review).parameters,
        )
        with self.assertRaisesRegex(
            TypeError, "unexpected keyword argument 'pr_number'"
        ):
            agent.get_pull_request_details(pr_number=99999)  # pylint: disable=unexpected-keyword-arg
        with self.assertRaisesRegex(
            TypeError, "unexpected keyword argument 'pr_number'"
        ):
            agent.submit_pr_code_review(  # pylint: disable=unexpected-keyword-arg
                pr_number=99999,
                overall_assessment="Minor improvements suggested.",
                summary_comment="Summary",
                inline_comments=[],
            )
        with self.assertRaisesRegex(
            TypeError, "unexpected keyword argument 'pr_number'"
        ):
            asyncio.run(
                agent.run_pr_review(  # pylint: disable=unexpected-keyword-arg
                    model_name="gemini-3.1-pro-preview",
                    pr_number=99999,
                )
            )

        captured_tools = []
        with patch("agent.agent.LlmAgent") as mock_llm_agent:
            agent.make_review_agent(
                model_name="gemini-3.1-pro-preview",
                category="Bug fix",
                reason="Test",
                focus_areas="Correctness",
                skip_areas="Docs",
            )
            captured_tools = mock_llm_agent.call_args.kwargs["tools"]

        self.assertEqual(len(captured_tools), 2)
        for tool_fn in captured_tools:
            self.assertNotIn(
                "pr_number", inspect.signature(tool_fn).parameters
            )

        mock_get_diff.return_value = ""
        mock_graphql.return_value = {
            "data": {
                "repository": {
                    "pullRequest": {
                        "id": "PR_1",
                        "number": 12345,
                        "title": "Title",
                        "body": "Body",
                        "state": "OPEN",
                        "headRefOid": "abcdef1234567890",
                        "author": {"login": "dev"},
                        "files": {"nodes": []},
                        "comments": {"nodes": []},
                        "commits": {"nodes": []},
                    }
                }
            }
        }
        res = agent.get_pull_request_details()
        self.assertEqual(res["status"], "success")
        self.assertEqual(
            mock_graphql.call_args[0][1]["prNumber"], 12345
        )

    @patch("agent.utils.get_request")
    def test_finding_3_idempotency_author_validation(self, mock_get_request):
        """Finding 3: has_agent_reviewed_commit requires github-actions[bot] author + commit_id + marker."""
        sha = "abcdef1234567890abcdef1234567890abcdef12"
        marker = f"<!-- tensorflow-pr-review-agent: commit_sha={sha} -->"
        reviews_url = "https://api.github.com/repos/tensorflow/tensorflow/pulls/12345/reviews"

        # 1. Spoofed marker by another user -> False
        mock_get_request.return_value = [
            {
                "id": 1,
                "user": {"login": "malicious-contributor"},
                "commit_id": sha,
                "body": f"Spoofed review\n{marker}",
            }
        ]
        self.assertFalse(utils.has_agent_reviewed_commit(reviews_url, sha))

        # 2. Missing user field -> False
        mock_get_request.return_value = [
            {
                "id": 2,
                "user": None,
                "commit_id": sha,
                "body": f"Missing user\n{marker}",
            }
        ]
        self.assertFalse(utils.has_agent_reviewed_commit(reviews_url, sha))

        # 3. Legitimate bot user but wrong commit_id -> False
        mock_get_request.return_value = [
            {
                "id": 3,
                "user": {"login": "github-actions[bot]"},
                "commit_id": "0000000000000000000000000000000000000000",
                "body": f"Wrong commit\n{marker}",
            }
        ]
        self.assertFalse(utils.has_agent_reviewed_commit(reviews_url, sha))

        # 4. Legitimate github-actions[bot] + matching commit_id + marker -> True
        mock_get_request.return_value = [
            {
                "id": 4,
                "user": {"login": "github-actions[bot]"},
                "commit_id": sha,
                "body": f"Valid review\n{marker}",
            }
        ]
        self.assertTrue(utils.has_agent_reviewed_commit(reviews_url, sha))

    @patch("agent.utils._fetch_file_content_at_commit")
    @patch("agent.utils.subprocess.run")
    def test_finding_4_pylint_subprocess_minimal_environment(
        self, mock_subproc_run, mock_fetch_content
    ):
        """Finding 4: Pylint subprocess receives only PATH and HOME in env and no secrets."""
        files = [{"path": "tensorflow/python/foo.py", "changeType": "MODIFIED"}]
        mock_fetch_content.return_value = "x = 1\n"
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_subproc_run.return_value = mock_proc

        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "secret-gh-token",
                "GEMINI_API_KEY": "secret-gemini-key",
                "OWNER": "tensorflow",
                "REPO": "tensorflow",
                "PULL_REQUEST_NUMBER": "12345",
            },
        ):
            utils.run_pylint_on_changed_files(
                files, raw_diff="", head_sha="abcdef12"
            )

        mock_subproc_run.assert_called_once()
        passed_env = mock_subproc_run.call_args.kwargs.get("env")
        self.assertIsNotNone(passed_env)
        self.assertEqual(set(passed_env.keys()), {"PATH", "HOME"})
        for forbidden_key in (
            "GITHUB_TOKEN",
            "GEMINI_API_KEY",
            "OWNER",
            "REPO",
            "PULL_REQUEST_NUMBER",
        ):
            self.assertNotIn(forbidden_key, passed_env)

    @patch("agent.main.clear_and_set_reaction")
    @patch("agent.main.run_pylint_on_changed_files")
    @patch("agent.main.has_agent_reviewed_commit")
    @patch("agent.agent.get_pull_request_details")
    @patch("agent.agent.run_pr_review", new_callable=AsyncMock)
    def test_finding_5_head_sha_mismatch_aborts_before_pylint_and_gemini(
        self,
        mock_run_pr_review,
        mock_get_details,
        mock_has_reviewed,
        mock_run_pylint,
        mock_react,
    ):
        """Finding 5: expected_head_sha != current_head_sha aborts before Pylint, Gemini, or review."""
        expected_sha = "1111111111111111111111111111111111111111"
        current_sha = "2222222222222222222222222222222222222222"
        mock_get_details.return_value = {
            "status": "success",
            "pull_request": {
                "headRefOid": current_sha,
                "title": "PR Title",
                "body": "PR Body",
                "diff": "@@ -1,1 +1,1 @@\n+x = 1",
                "files": {
                    "nodes": [
                        {"path": "tensorflow/python/foo.py", "changeType": "MODIFIED"}
                    ]
                },
            },
        }

        with patch.dict(os.environ, {"PR_HEAD_SHA": expected_sha}):
            asyncio.run(main.main())

        mock_has_reviewed.assert_not_called()
        mock_run_pylint.assert_not_called()
        mock_run_pr_review.assert_not_called()
        mock_react.assert_called_once_with(12345, add_content="eyes")

    @patch("agent.main.clear_and_set_reaction")
    @patch("agent.main.run_pylint_on_changed_files")
    @patch("agent.main.has_agent_reviewed_commit")
    @patch("agent.agent.get_pull_request_details")
    @patch("agent.agent.run_pr_review", new_callable=AsyncMock)
    def test_finding_5_matching_head_sha_propagates_to_all_downstream_steps(
        self,
        mock_run_pr_review,
        mock_get_details,
        mock_has_reviewed,
        mock_run_pylint,
        mock_react,
    ):
        """Finding 5: expected_head_sha == current_head_sha passes verified SHA to all downstream calls."""
        verified_sha = "3333333333333333333333333333333333333333"
        mock_get_details.return_value = {
            "status": "success",
            "pull_request": {
                "headRefOid": verified_sha,
                "title": "PR Title",
                "body": "PR Body",
                "diff": "@@ -1,1 +1,1 @@\n+x = 1",
                "files": {
                    "nodes": [
                        {"path": "tensorflow/python/foo.py", "changeType": "MODIFIED"}
                    ]
                },
            },
        }
        mock_has_reviewed.side_effect = [False, True]
        mock_run_pylint.return_value = "No Pylint issues detected on modified lines."
        mock_run_pr_review.return_value = "Done"

        with patch.dict(os.environ, {"PR_HEAD_SHA": verified_sha}):
            asyncio.run(main.main())

        self.assertEqual(agent._VERIFIED_HEAD_SHA, verified_sha)
        mock_run_pylint.assert_called_once()
        self.assertEqual(
            mock_run_pylint.call_args.kwargs.get("head_sha"), verified_sha
        )
        mock_run_pr_review.assert_called_once()
        self.assertEqual(
            mock_run_pr_review.call_args.kwargs.get("commit_sha"), verified_sha
        )
        mock_react.assert_called_with(12345, add_content="rocket")

    @patch("agent.utils.get_request")
    @patch("agent.utils.subprocess.run")
    def test_finding_5_supplied_head_sha_failure_does_not_read_local_base_files(
        self, mock_subproc_run, mock_get_request
    ):
        """Finding 5: when head_sha is supplied and git/API fail, _fetch_file_content_at_commit returns None."""
        mock_proc = MagicMock()
        mock_proc.returncode = 128
        mock_subproc_run.return_value = mock_proc
        mock_get_request.side_effect = RuntimeError("GitHub API unavailable")

        with patch("pathlib.Path.read_text") as mock_read_text:
            content = utils._fetch_file_content_at_commit(
                Path("."),
                "abcdef1234567890abcdef1234567890abcdef12",
                "tensorflow/python/foo.py",
            )
            self.assertIsNone(content)
            mock_read_text.assert_not_called()

    @patch("agent.utils.subprocess.run")
    def test_low_1_git_show_subprocess_minimal_environment(
        self, mock_subproc_run
    ):
        """LOW-1: git show subprocess in _fetch_file_content_at_commit receives only PATH and HOME."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "x = 1\n"
        mock_subproc_run.return_value = mock_proc

        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "secret-gh-token",
                "GEMINI_API_KEY": "secret-gemini-key",
                "OWNER": "tensorflow",
                "REPO": "tensorflow",
                "PULL_REQUEST_NUMBER": "12345",
                "PR_HEAD_SHA": "abcdef1234567890abcdef1234567890abcdef12",
            },
        ):
            content = utils._fetch_file_content_at_commit(
                Path("."),
                "abcdef1234567890abcdef1234567890abcdef12",
                "tensorflow/python/foo.py",
            )

        self.assertEqual(content, "x = 1\n")
        mock_subproc_run.assert_called_once()
        passed_env = mock_subproc_run.call_args.kwargs.get("env")
        self.assertIsNotNone(passed_env)
        self.assertEqual(set(passed_env.keys()), {"PATH", "HOME"})
        for forbidden_key in (
            "GITHUB_TOKEN",
            "GEMINI_API_KEY",
            "OWNER",
            "REPO",
            "PULL_REQUEST_NUMBER",
            "PR_HEAD_SHA",
        ):
            self.assertNotIn(forbidden_key, passed_env)

    def test_low_2_adk_logger_not_configured_at_debug_level(self):
        """LOW-2: ADK logger is configured at logging.WARNING in both mock and real google-adk environments."""
        if hasattr(main.logs.setup_adk_logger, "assert_called_with"):
            main.logs.setup_adk_logger.assert_called_with(level=logging.WARNING)
        else:
            adk_logger = logging.getLogger("google.adk")
            self.assertEqual(adk_logger.getEffectiveLevel(), logging.WARNING)

        # Also verify the real-function path when setup_adk_logger is not a MagicMock
        def _real_setup_adk_logger(level=logging.INFO):
            logging.getLogger("google.adk").setLevel(level)

        with patch.object(main.logs, "setup_adk_logger", side_effect=_real_setup_adk_logger):
            main.logs.setup_adk_logger(level=logging.WARNING)
            self.assertEqual(
                logging.getLogger("google.adk").getEffectiveLevel(),
                logging.WARNING,
            )
        # Restore mock call state if running under MagicMock
        if hasattr(main.logs.setup_adk_logger, "assert_called_with"):
            importlib.reload(main)
            main.logs.setup_adk_logger.assert_called_with(level=logging.WARNING)

    @patch("agent.utils._fetch_file_content_at_commit")
    @patch("agent.utils.subprocess.run")
    @patch("agent.utils.get_request")
    @patch("agent.agent.run_graphql_query")
    @patch("agent.agent.get_diff")
    def test_engineer_review_full_diff_preserved_for_pylint_beyond_30k(
        self,
        mock_get_diff,
        mock_graphql,
        mock_get_request,
        mock_subproc_run,
        mock_fetch_content,
    ):
        """Finding 2: Full diff >30,000 chars is preserved for Pylint in GraphQL & REST paths while LLM diff is capped at <=30,000 on a complete line boundary."""
        padding_lines = "\n".join(
            f"+// padding line {i} " + ("x" * 60) for i in range(450)
        )
        raw_large_diff = (
            "diff --git a/tensorflow/core/kernels/large_op.cc b/tensorflow/core/kernels/large_op.cc\n"
            "--- a/tensorflow/core/kernels/large_op.cc\n"
            "+++ b/tensorflow/core/kernels/large_op.cc\n"
            "@@ -1,1 +1,450 @@\n"
            f"{padding_lines}\n"
            "diff --git a/tensorflow/python/ops/late_file.py b/tensorflow/python/ops/late_file.py\n"
            "--- a/tensorflow/python/ops/late_file.py\n"
            "+++ b/tensorflow/python/ops/late_file.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def compute():\n"
            "+   return 42\n"
        )
        self.assertGreater(
            raw_large_diff.index("tensorflow/python/ops/late_file.py"), 30000
        )
        mock_get_diff.return_value = raw_large_diff

        files_nodes = [
            {
                "path": "tensorflow/core/kernels/large_op.cc",
                "additions": 450,
                "deletions": 0,
                "changeType": "MODIFIED",
            },
            {
                "path": "tensorflow/python/ops/late_file.py",
                "additions": 1,
                "deletions": 0,
                "changeType": "MODIFIED",
            },
        ]

        # 1. Verify GraphQL path preserves full diff >30,000 chars for Pylint
        agent._PREFETCHED_PR_DETAILS = None
        mock_graphql.return_value = {
            "data": {
                "repository": {
                    "pullRequest": {
                        "id": "PR_1",
                        "number": 12345,
                        "title": "Large PR",
                        "body": "Body",
                        "state": "OPEN",
                        "headRefOid": "abcdef1234567890abcdef1234567890abcdef12",
                        "author": {"login": "dev"},
                        "files": {"nodes": files_nodes},
                        "comments": {"nodes": []},
                        "commits": {"nodes": []},
                    }
                }
            }
        }
        gql_details = agent.get_pull_request_details()
        full_diff_gql = gql_details["pull_request"]["diff"]
        self.assertGreater(len(full_diff_gql), 30000)
        self.assertIn("tensorflow/python/ops/late_file.py", full_diff_gql)

        # Verify LLM-facing prefetched return is capped at <=30,000 chars at a complete line boundary without mutating stored full diff
        agent._PREFETCHED_PR_DETAILS = gql_details
        llm_details = agent.get_pull_request_details()
        llm_diff = llm_details["pull_request"]["diff"]
        self.assertLessEqual(len(llm_diff), agent.MAX_LLM_DIFF_CHARS)
        self.assertGreater(len(llm_diff), 0)
        self.assertTrue(full_diff_gql.startswith(llm_diff + "\n"))
        self.assertEqual(
            llm_diff.splitlines()[-1],
            full_diff_gql.splitlines()[len(llm_diff.splitlines()) - 1],
        )
        self.assertGreater(len(gql_details["pull_request"]["diff"]), 30000)

        # Verify Pylint retains diagnostic on the Python file after the 30,000-char boundary
        mock_fetch_content.return_value = "def compute():\n   return 42\n"
        mock_proc = MagicMock()
        mock_proc.returncode = 4
        mock_proc.stdout = (
            "tensorflow/python/ops/late_file.py:2:0: W0311 (bad-indentation): "
            "Bad indentation. Found 3 spaces, expected 2\n"
        )
        mock_subproc_run.return_value = mock_proc

        pylint_out = utils.run_pylint_on_changed_files(
            files_nodes,
            raw_diff=full_diff_gql,
            head_sha="abcdef1234567890abcdef1234567890abcdef12",
        )
        self.assertIn(
            "tensorflow/python/ops/late_file.py:2:0: W0311 (bad-indentation)",
            pylint_out,
        )

        # 2. Verify REST fallback path also preserves full diff >30,000 chars
        agent._PREFETCHED_PR_DETAILS = None
        mock_graphql.side_effect = requests.exceptions.RequestException("GQL down")
        mock_get_request.side_effect = [
            {
                "id": 1,
                "number": 12345,
                "title": "Large PR",
                "body": "Body",
                "state": "open",
                "head": {"sha": "abcdef1234567890abcdef1234567890abcdef12"},
                "user": {"login": "dev"},
            },
            [
                {
                    "filename": "tensorflow/core/kernels/large_op.cc",
                    "additions": 450,
                    "deletions": 0,
                    "status": "modified",
                },
                {
                    "filename": "tensorflow/python/ops/late_file.py",
                    "additions": 1,
                    "deletions": 0,
                    "status": "modified",
                },
            ],
        ]
        rest_details = agent.get_pull_request_details()
        self.assertGreater(len(rest_details["pull_request"]["diff"]), 30000)
        self.assertIn(
            "tensorflow/python/ops/late_file.py",
            rest_details["pull_request"]["diff"],
        )

    def test_engineer_review_api_change_classification_per_file(self):
        """Finding 4: API signature-change detection is associated per file rather than over-counting every Python file."""
        # Case 1: 1 public Python file has a signature change (+40 pts), 2 other public Python files have only body edits (0 pts),
        # and 2 TFLite files are modified (2 * 50 = 100 pts). Without per-file association, API change would get 3 * 40 = 120 pts and win.
        files = [
            {"path": "tensorflow/python/ops/math_ops.py", "additions": 2, "deletions": 0, "changeType": "MODIFIED"},
            {"path": "tensorflow/python/ops/array_ops.py", "additions": 1, "deletions": 1, "changeType": "MODIFIED"},
            {"path": "tensorflow/python/ops/nn_ops.py", "additions": 1, "deletions": 1, "changeType": "MODIFIED"},
            {"path": "tensorflow/lite/kernels/add.cc", "additions": 5, "deletions": 2, "changeType": "MODIFIED"},
            {"path": "tensorflow/lite/kernels/sub.cc", "additions": 3, "deletions": 1, "changeType": "MODIFIED"},
        ]
        diff = (
            "diff --git a/tensorflow/python/ops/math_ops.py b/tensorflow/python/ops/math_ops.py\n"
            "--- a/tensorflow/python/ops/math_ops.py\n"
            "+++ b/tensorflow/python/ops/math_ops.py\n"
            "@@ -10,2 +10,4 @@\n"
            "+def new_public_math_op(x):\n"
            "+  return x\n"
            "diff --git a/tensorflow/python/ops/array_ops.py b/tensorflow/python/ops/array_ops.py\n"
            "--- a/tensorflow/python/ops/array_ops.py\n"
            "+++ b/tensorflow/python/ops/array_ops.py\n"
            "@@ -20,2 +20,2 @@\n"
            "-  val = 1\n"
            "+  val = 2\n"
            "diff --git a/tensorflow/python/ops/nn_ops.py b/tensorflow/python/ops/nn_ops.py\n"
            "--- a/tensorflow/python/ops/nn_ops.py\n"
            "+++ b/tensorflow/python/ops/nn_ops.py\n"
            "@@ -30,2 +30,2 @@\n"
            "-  out = x\n"
            "+  out = x + 1\n"
        )
        cat, _ = agent.classify_pr_with_scoring(files, title="Update ops", body="", diff=diff)
        self.assertEqual(cat, "TensorFlow Lite")

        # Case 2: Only math_ops.py has an API signature change -> classified as API change
        single_api_files = [
            {"path": "tensorflow/python/ops/math_ops.py", "additions": 2, "deletions": 0, "changeType": "MODIFIED"},
            {"path": "tensorflow/python/ops/array_ops.py", "additions": 1, "deletions": 1, "changeType": "MODIFIED"},
        ]
        cat_api, _ = agent.classify_pr_with_scoring(
            single_api_files, title="Update math op", body="", diff=diff
        )
        self.assertEqual(cat_api, "API change")

        # Case 3: A signature change in a test file must not cause an unrelated public Python file to be counted as API change
        test_sig_files = [
            {"path": "tensorflow/python/ops/array_ops.py", "additions": 1, "deletions": 1, "changeType": "MODIFIED"},
            {"path": "tensorflow/python/ops/array_ops_test.py", "additions": 2, "deletions": 0, "changeType": "MODIFIED"},
        ]
        test_sig_diff = (
            "diff --git a/tensorflow/python/ops/array_ops.py b/tensorflow/python/ops/array_ops.py\n"
            "--- a/tensorflow/python/ops/array_ops.py\n"
            "+++ b/tensorflow/python/ops/array_ops.py\n"
            "@@ -20,2 +20,2 @@\n"
            "-  val = 1\n"
            "+  val = 2\n"
            "diff --git a/tensorflow/python/ops/array_ops_test.py b/tensorflow/python/ops/array_ops_test.py\n"
            "--- a/tensorflow/python/ops/array_ops_test.py\n"
            "+++ b/tensorflow/python/ops/array_ops_test.py\n"
            "@@ -10,2 +10,4 @@\n"
            "+def test_new_helper():\n"
            "+  pass\n"
        )
        cat_non_api, _ = agent.classify_pr_with_scoring(
            test_sig_files, title="Minor tweak", body="", diff=test_sig_diff
        )
        self.assertEqual(cat_non_api, "General TensorFlow")

    @patch("agent.main.requests.post")
    @patch("agent.main.requests.delete")
    @patch("agent.main.requests.get")
    def test_engineer_review_clear_and_set_reaction_deletes_only_bot_eyes(
        self, mock_get, mock_delete, mock_post
    ):
        """Finding 5: clear_and_set_reaction deletes only github-actions[bot] eyes reactions and preserves human eyes reactions."""
        resp_get = MagicMock()
        resp_get.status_code = 200
        resp_get.json.return_value = [
            {
                "id": 101,
                "content": "eyes",
                "user": {"login": "human-maintainer"},
            },
            {
                "id": 102,
                "content": "eyes",
                "user": {"login": "github-actions[bot]"},
            },
            {
                "id": 103,
                "content": "eyes",
                "user": None,
            },
            {
                "id": 104,
                "content": "rocket",
                "user": {"login": "github-actions[bot]"},
            },
        ]
        mock_get.return_value = resp_get

        main.clear_and_set_reaction(12345, add_content="rocket")

        mock_get.assert_called_once()
        self.assertEqual(mock_get.call_args.kwargs.get("params"), {"per_page": 100})
        mock_delete.assert_called_once()
        deleted_url = mock_delete.call_args[0][0]
        self.assertTrue(deleted_url.endswith("/issues/reactions/102"))
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args.kwargs.get("json"), {"content": "rocket"})

    @patch("agent.utils.get_request")
    @patch("agent.agent.run_graphql_query")
    @patch("agent.agent.get_diff")
    def test_engineer_review_rest_fallback_requests_per_page_100(
        self, mock_get_diff, mock_graphql, mock_get_request
    ):
        """Finding 6: REST fallback in get_pull_request_details requests /files with params={'per_page': 100}."""
        agent._PREFETCHED_PR_DETAILS = None
        mock_graphql.side_effect = requests.exceptions.RequestException("GraphQL unavailable")
        mock_get_diff.return_value = ""
        mock_get_request.side_effect = [
            {
                "id": 1,
                "number": 12345,
                "title": "PR Title",
                "body": "PR Body",
                "state": "open",
                "head": {"sha": "abcdef1234567890"},
                "user": {"login": "dev"},
            },
            [
                {
                    "filename": "tensorflow/python/foo.py",
                    "additions": 2,
                    "deletions": 1,
                    "status": "modified",
                }
            ],
        ]

        res = agent.get_pull_request_details()
        self.assertEqual(res["status"], "success")
        self.assertEqual(mock_get_request.call_count, 2)
        files_call = mock_get_request.call_args_list[1]
        self.assertEqual(
            files_call[0][0],
            "https://api.github.com/repos/tensorflow/tensorflow/pulls/12345/files",
        )
        self.assertEqual(files_call.kwargs.get("params"), {"per_page": 100})

    def test_engineer_review_pylint_preserves_hidden_filenames(self):
        """Hidden filenames like .hidden.py and ./.hidden.py are preserved during Pylint diagnostic filtering."""
        raw_diff = (
            "--- a/.hidden.py\n"
            "+++ b/.hidden.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def f():\n"
            "+   return 1\n"
            "--- a/tensorflow/python/.hidden_mod.py\n"
            "+++ b/tensorflow/python/.hidden_mod.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def g():\n"
            "+   return 2\n"
        )
        mod_lines = utils.extract_modified_lines_by_file(raw_diff)
        self.assertEqual(
            mod_lines,
            {
                ".hidden.py": {2},
                "tensorflow/python/.hidden_mod.py": {2},
            },
        )
        pylint_stdout = (
            ".hidden.py:2:0: W0311 (bad-indentation): Bad indentation\n"
            "./.hidden.py:2:0: C0116 (missing-function-docstring): Missing docstring\n"
            "./tensorflow/python/.hidden_mod.py:2:0: W0311 (bad-indentation): Bad indentation\n"
        )
        retained = utils.filter_pylint_output_by_diff(
            pylint_stdout, mod_lines, raw_diff=raw_diff
        )
        self.assertEqual(
            retained,
            [
                ".hidden.py:2:0: W0311 (bad-indentation): Bad indentation",
                ".hidden.py:2:0: C0116 (missing-function-docstring): Missing docstring",
                "tensorflow/python/.hidden_mod.py:2:0: W0311 (bad-indentation): Bad indentation",
            ],
        )

    def test_engineer_review_docs_dir_source_files_receive_code_focus_areas(self):
        """Source files under /docs/ are categorized as code rather than Documentation and receive code-review focus areas."""
        # 1. Python source file under /docs/ -> not classified as Documentation
        py_docs_files = [
            {
                "path": "tensorflow/tools/docs/generate_lib.py",
                "additions": 10,
                "deletions": 2,
                "changeType": "MODIFIED",
            }
        ]
        cat_py, _ = agent.classify_pr_with_scoring(
            py_docs_files,
            title="Fix crash in doc parser",
            body="",
            diff="@@ -10,2 +10,3 @@\n+  return val\n",
        )
        self.assertEqual(cat_py, "Bug fix")
        focus_py, skip_py = agent.get_focus_skip_areas(cat_py)
        self.assertIn("Correctness", focus_py)
        self.assertNotIn("Testing requirements", skip_py)

        # 2. Python source file under /docs/ with neutral title -> General TensorFlow (not Documentation)
        cat_general, _ = agent.classify_pr_with_scoring(
            py_docs_files,
            title="Update generate_lib",
            body="",
            diff="@@ -10,2 +10,3 @@\n+  return val\n",
        )
        self.assertEqual(cat_general, "General TensorFlow")
        focus_gen, _ = agent.get_focus_skip_areas(cat_general)
        self.assertIn("code quality", focus_gen)

        # 3. Test file under /docs/ -> Test-only (not Documentation)
        test_docs_files = [
            {
                "path": "tensorflow/tools/docs/generate_lib_test.py",
                "additions": 8,
                "deletions": 1,
                "changeType": "MODIFIED",
            }
        ]
        cat_test, _ = agent.classify_pr_with_scoring(
            test_docs_files, title="Add unit test", body="", diff=""
        )
        self.assertEqual(cat_test, "Test-only")

        # 4. Pure documentation files (.md, .rst) under /docs/ still classify as Documentation
        md_docs_files = [
            {
                "path": "tensorflow/docs/guide.md",
                "additions": 5,
                "deletions": 1,
                "changeType": "MODIFIED",
            },
            {
                "path": "tensorflow/docs/overview.rst",
                "additions": 3,
                "deletions": 0,
                "changeType": "MODIFIED",
            },
        ]
        cat_doc, _ = agent.classify_pr_with_scoring(
            md_docs_files, title="Update guide", body="", diff=""
        )
        self.assertEqual(cat_doc, "Documentation")

    def test_engineer_review_crlf_diff_line_synchronization(self):
        """Finding 1: CRLF and bare \\r context lines in diffs keep left/right line counters synchronized."""
        # 1. Pure LF vs CRLF diff with blank context lines, added lines, deleted lines, and post-blank context
        lf_diff = (
            "--- a/tensorflow/python/foo.py\n"
            "+++ b/tensorflow/python/foo.py\n"
            "@@ -10,5 +10,5 @@\n"
            " def alpha():\n"
            "\n"
            "-  old_call()\n"
            "+  new_call()\n"
            "   return 1\n"
            "+\n"
            "+  unreachable = 2\n"
        )
        crlf_diff = lf_diff.replace("\n", "\r\n")
        # Mixed line endings with bare '\r' context line before an addition
        mixed_diff = (
            "--- a/tensorflow/python/foo.py\r\n"
            "+++ b/tensorflow/python/foo.py\n"
            "@@ -10,5 +10,5 @@\r\n"
            " def alpha():\r\n"
            "\r\n"
            "-  old_call()\n"
            "+  new_call()\r\n"
            "   return 1\n"
            "+\r\n"
            "+  unreachable = 2\r\n"
        )

        ann_lf = utils.annotate_diff_with_line_numbers(lf_diff)
        ann_crlf = utils.annotate_diff_with_line_numbers(crlf_diff)
        ann_mixed = utils.annotate_diff_with_line_numbers(mixed_diff)

        # Stripping \r from annotated output must yield identical line-number tags
        self.assertEqual(ann_crlf.replace("\r", ""), ann_lf)
        self.assertEqual(ann_mixed.replace("\r", ""), ann_lf)
        self.assertIn("[LEFT L12] -  old_call()", ann_crlf)
        self.assertIn("[L12] +  new_call()", ann_crlf)
        self.assertIn("[L13]    return 1", ann_crlf)
        self.assertIn("[L14] +", ann_crlf)
        self.assertIn("[L15] +  unreachable = 2", ann_crlf)

        # Verify extract_modified_lines_by_file matches between raw and annotated diffs across LF, CRLF, and mixed
        expected_mod = {"tensorflow/python/foo.py": {12, 14, 15}}
        self.assertEqual(utils.extract_modified_lines_by_file(lf_diff), expected_mod)
        self.assertEqual(utils.extract_modified_lines_by_file(crlf_diff), expected_mod)
        self.assertEqual(utils.extract_modified_lines_by_file(mixed_diff), expected_mod)
        self.assertEqual(utils.extract_modified_lines_by_file(ann_crlf), expected_mod)
        self.assertEqual(utils.extract_modified_lines_by_file(ann_mixed), expected_mod)

        # Verify Pylint filtering retains diagnostics on modified lines after CRLF blank context lines
        pylint_out = (
            "tensorflow/python/foo.py:11:0: C0303 (trailing-whitespace): Trailing whitespace\n"
            "tensorflow/python/foo.py:12:0: W0311 (bad-indentation): Bad indentation\n"
            "tensorflow/python/foo.py:15:0: W0101 (unreachable): Unreachable code\n"
        )
        retained = utils.filter_pylint_output_by_diff(
            pylint_out,
            utils.extract_modified_lines_by_file(ann_crlf),
            raw_diff=ann_crlf,
        )
        self.assertEqual(
            retained,
            [
                "tensorflow/python/foo.py:12:0: W0311 (bad-indentation): Bad indentation",
                "tensorflow/python/foo.py:15:0: W0101 (unreachable): Unreachable code",
            ],
        )

    @patch("agent.utils.requests.get")
    @patch("agent.utils.subprocess.run")
    def test_engineer_review_path_normalization_blocks_agent_dir_variants(
        self, mock_subproc_run, mock_http_get
    ):
        """Finding 2: Normalized path checks block ./pr_review_agent/... and related variants while allowing legitimate files."""
        blocked_paths = [
            "pr_review_agent/agent/utils.py",
            "./pr_review_agent/agent/utils.py",
            "././pr_review_agent/agent/main.py",
            "pr_review_agent/../pr_review_agent/agent/utils.py",
            "foo/pr_review_agent/bar.py",
            ".",
            "./",
            "",
        ]
        for bad_path in blocked_paths:
            self.assertFalse(
                utils._is_safe_relative_path(bad_path),
                f"Expected {bad_path!r} to be rejected by _is_safe_relative_path",
            )

        allowed_paths = [
            "tensorflow/python/foo.py",
            "./tensorflow/python/foo.py",
            ".hidden.py",
            "./.hidden.py",
            "tensorflow/python/pr_review_agent_helper.py",
        ]
        for good_path in allowed_paths:
            self.assertTrue(
                utils._is_safe_relative_path(good_path),
                f"Expected {good_path!r} to be allowed by _is_safe_relative_path",
            )

        # Ensure _fetch_file_content_at_commit refuses ./pr_review_agent/... without making git or HTTP calls
        self.assertIsNone(
            utils._fetch_file_content_at_commit(
                "/tmp/repo",
                "17f28d8fb7e47177e0d340b7908144e9a98fe939",
                "./pr_review_agent/agent/utils.py",
            )
        )
        mock_http_get.assert_not_called()
        mock_subproc_run.assert_not_called()

        # Ensure run_pylint_on_changed_files skips ./pr_review_agent/... even if present in diff
        bypass_diff = (
            "--- a/./pr_review_agent/agent/utils.py\n"
            "+++ b/./pr_review_agent/agent/utils.py\n"
            "@@ -1,1 +1,2 @@\n"
            " import os\n"
            "+import sys\n"
        )
        res = utils.run_pylint_on_changed_files(
            [{"path": "./pr_review_agent/agent/utils.py", "changeType": "MODIFIED"}],
            raw_diff=bypass_diff,
            head_sha="17f28d8fb7e47177e0d340b7908144e9a98fe939",
        )
        self.assertEqual(res, "No Python files were modified in this pull request.")
        mock_http_get.assert_not_called()
        mock_subproc_run.assert_not_called()

    @patch("agent.main.requests.post")
    @patch("agent.main.requests.delete")
    @patch("agent.main.requests.get")
    def test_engineer_review_reactions_pagination_and_error_handling(
        self, mock_get, mock_delete, mock_post
    ):
        """Finding 3: clear_and_set_reaction paginates through all reaction pages and handles API errors safely."""
        # 1. Multi-page pagination: page 1 via Link header -> page 2 via response.links -> page 3 (last)
        page1 = MagicMock()
        page1.status_code = 200
        page1.links = {}
        page1.headers = {
            "Link": (
                '<https://api.github.com/repositories/1/issues/99/reactions?per_page=100&page=2>; rel="next", '
                '<https://api.github.com/repositories/1/issues/99/reactions?per_page=100&page=3>; rel="last"'
            )
        }
        page1.json.return_value = [
            {"id": 1, "content": "+1", "user": {"login": "user-a"}},
            {"id": 2, "content": "eyes", "user": {"login": "human-reviewer"}},
        ]

        page2 = MagicMock()
        page2.status_code = 200
        page2.links = {
            "next": {
                "url": "https://api.github.com/repositories/1/issues/99/reactions?per_page=100&page=3"
            }
        }
        page2.headers = {}
        page2.json.return_value = [
            {"id": 105, "content": "eyes", "user": {"login": "github-actions[bot]"}},
            {"id": 106, "content": "heart", "user": {"login": "user-b"}},
        ]

        page3 = MagicMock()
        page3.status_code = 200
        page3.links = {}
        page3.headers = {}
        page3.json.return_value = [
            {"id": 205, "content": "eyes", "user": {"login": "github-actions[bot]"}},
        ]

        mock_get.side_effect = [page1, page2, page3]

        main.clear_and_set_reaction(99, add_content="rocket")

        self.assertEqual(mock_get.call_count, 3)
        # First page passes params={'per_page': 100}; subsequent Link URLs already include query params
        self.assertEqual(
            mock_get.call_args_list[0].kwargs.get("params"), {"per_page": 100}
        )
        self.assertIsNone(mock_get.call_args_list[1].kwargs.get("params"))
        self.assertIsNone(mock_get.call_args_list[2].kwargs.get("params"))

        # Both bot eyes reactions from page 2 (id 105) and page 3 (id 205) are deleted; human eyes (id 2) is preserved
        deleted_urls = [c.args[0] for c in mock_delete.call_args_list]
        self.assertEqual(
            deleted_urls,
            [
                "https://api.github.com/repos/tensorflow/tensorflow/issues/reactions/105",
                "https://api.github.com/repos/tensorflow/tensorflow/issues/reactions/205",
            ],
        )
        mock_post.assert_called_once()

        # 2. Non-200 response or RequestException stops pagination gracefully and still posts new reaction if requested
        mock_get.reset_mock()
        mock_delete.reset_mock()
        mock_post.reset_mock()

        err_page = MagicMock()
        err_page.status_code = 502
        mock_get.side_effect = [err_page]
        main.clear_and_set_reaction(99, add_content="eyes")
        mock_get.assert_called_once()
        mock_delete.assert_not_called()
        mock_post.assert_called_once()

    def test_engineer_review_agent_import_grouping_order(self):
        """Finding 4: pr_review_agent/agent/agent.py groups stdlib, third-party, and local imports in standard order."""
        agent_py_path = Path(__file__).resolve().parent / "agent" / "agent.py"
        source = agent_py_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(agent_py_path))

        stdlib_modules = {"__future__", "os", "pathlib", "re", "typing"}
        third_party_roots = {"requests", "google"}
        local_roots = {"agent"}

        group_sequence = []
        seen_imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    key = f"import:{alias.name}"
                    self.assertNotIn(key, seen_imports, f"Duplicate import {key}")
                    seen_imports.add(key)
                    if root_pkg in stdlib_modules:
                        group_sequence.append((1, alias.name, node.lineno))
                    elif root_pkg in third_party_roots:
                        group_sequence.append((2, alias.name, node.lineno))
                    elif root_pkg in local_roots:
                        group_sequence.append((3, alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_pkg = node.module.split(".")[0]
                for alias in node.names:
                    key = f"from:{node.module}:{alias.name}"
                    self.assertNotIn(key, seen_imports, f"Duplicate import {key}")
                    seen_imports.add(key)
                if root_pkg in stdlib_modules:
                    group_sequence.append((1, node.module, node.lineno))
                elif root_pkg in third_party_roots:
                    group_sequence.append((2, node.module, node.lineno))
                elif root_pkg in local_roots:
                    group_sequence.append((3, node.module, node.lineno))

        group_ids = [g[0] for g in group_sequence]
        self.assertEqual(
            group_ids,
            sorted(group_ids),
            f"Expected stdlib (1) -> third-party (2) -> local (3) import order, got {group_sequence}",
        )
        # Ensure all three groups are present and requests is grouped with third-party before local agent imports
        self.assertIn(1, group_ids)
        self.assertIn(2, group_ids)
        self.assertIn(3, group_ids)

    @patch("agent.utils.requests.get")
    def test_engineer_review_centralized_get_request_pagination_and_idempotency(
        self, mock_http_get
    ):
        """Finding 1: get_request follows pagination links for list responses and preserves idempotency across >100 reviews."""
        reviews_url = "https://api.github.com/repos/tensorflow/tensorflow/pulls/128063/reviews"
        commit_sha = "45e821c3df6e49c0ac3db3c06c62ff3f6469ce7e"
        marker = utils.format_commit_review_marker(commit_sha)

        # 1. Multi-page response where existing agent review is on page 2
        page1 = MagicMock()
        page1.status_code = 200
        page1.links = {}
        page1.headers = {
            "Link": (
                '<https://api.github.com/repositories/45717250/pulls/128063/reviews?per_page=100&page=2>; rel="next", '
                '<https://api.github.com/repositories/45717250/pulls/128063/reviews?per_page=100&page=2>; rel="last"'
            )
        }
        page1.json.return_value = [
            {
                "id": i,
                "user": {"login": f"reviewer-{i}"},
                "commit_id": commit_sha,
                "body": "Human review comment",
            }
            for i in range(1, 101)
        ]

        page2 = MagicMock()
        page2.status_code = 200
        page2.links = {}
        page2.headers = {}
        page2.json.return_value = [
            {
                "id": 101,
                "user": {"login": "github-actions[bot]"},
                "commit_id": commit_sha,
                "body": f"### Summary\nAutomated review.{marker}",
            }
        ]

        mock_http_get.side_effect = [page1, page2]
        self.assertTrue(
            utils.has_agent_reviewed_commit(reviews_url, commit_sha)
        )
        self.assertEqual(mock_http_get.call_count, 2)
        self.assertEqual(mock_http_get.call_args_list[0].args[0], reviews_url)
        self.assertEqual(
            mock_http_get.call_args_list[0].kwargs.get("params"),
            {"per_page": 100},
        )
        self.assertEqual(
            mock_http_get.call_args_list[0].kwargs.get("timeout"), 60
        )
        self.assertEqual(
            mock_http_get.call_args_list[1].args[0],
            "https://api.github.com/repositories/45717250/pulls/128063/reviews?per_page=100&page=2",
        )
        self.assertIsNone(mock_http_get.call_args_list[1].kwargs.get("params"))

        # 2. Single-page list response with no Link header and 401 auth fallback
        mock_http_get.reset_mock()
        unauth_resp = MagicMock()
        unauth_resp.status_code = 401
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.links = {}
        ok_resp.headers = {}
        ok_resp.json.return_value = [{"filename": "tensorflow/python/foo.py"}]
        mock_http_get.side_effect = [unauth_resp, ok_resp]

        single_page_res = utils.get_request(
            "https://api.github.com/repos/tensorflow/tensorflow/pulls/128063/files",
            params={"per_page": 100},
        )
        self.assertEqual(
            single_page_res, [{"filename": "tensorflow/python/foo.py"}]
        )
        self.assertEqual(mock_http_get.call_count, 2)
        self.assertNotIn(
            "Authorization", mock_http_get.call_args_list[1].kwargs["headers"]
        )

        # 3. Non-list JSON response (dict) returned unchanged even if Link header is present
        mock_http_get.reset_mock()
        dict_resp = MagicMock()
        dict_resp.status_code = 200
        dict_resp.headers = {
            "Link": '<https://api.github.com/repos/tensorflow/tensorflow/pulls/128063?page=2>; rel="next"'
        }
        dict_resp.json.return_value = {"id": 128063, "state": "open"}
        mock_http_get.side_effect = [dict_resp]

        dict_data = utils.get_request(
            "https://api.github.com/repos/tensorflow/tensorflow/pulls/128063"
        )
        self.assertEqual(dict_data, {"id": 128063, "state": "open"})
        mock_http_get.assert_called_once()

        # 4. Malformed / untrusted Link headers and pagination loop protection
        mock_http_get.reset_mock()
        loop_page1 = MagicMock()
        loop_page1.status_code = 200
        loop_page1.links = {}
        loop_page1.headers = {
            "Link": f'<{reviews_url}>; rel="next"'
        }
        loop_page1.json.return_value = [{"id": 1}]
        mock_http_get.side_effect = [loop_page1]

        loop_res = utils.get_request(reviews_url)
        self.assertEqual(loop_res, [{"id": 1}])
        mock_http_get.assert_called_once()

        mock_http_get.reset_mock()
        bad_link_page = MagicMock()
        bad_link_page.status_code = 200
        bad_link_page.links = {}
        bad_link_page.headers = {
            "Link": '<https://evil.example.com/reviews?page=2>; rel="next"'
        }
        bad_link_page.json.return_value = [{"id": 2}]
        mock_http_get.side_effect = [bad_link_page]

        bad_link_res = utils.get_request(reviews_url)
        self.assertEqual(bad_link_res, [{"id": 2}])
        mock_http_get.assert_called_once()

        # 5. API / HTTP error on later page raises RequestException in get_request and is handled safely by has_agent_reviewed_commit
        mock_http_get.reset_mock()
        err_page2 = MagicMock()
        err_page2.status_code = 500
        err_page2.raise_for_status.side_effect = (
            requests.exceptions.HTTPError("500 Server Error")
        )
        mock_http_get.side_effect = [page1, err_page2]
        with self.assertRaisesRegex(
            requests.exceptions.HTTPError, r"500 Server Error"
        ):
            utils.get_request(reviews_url, params={"per_page": 100})

        mock_http_get.reset_mock()
        mock_http_get.side_effect = [page1, err_page2]
        self.assertFalse(
            utils.has_agent_reviewed_commit(reviews_url, commit_sha)
        )

    @patch("agent.utils.requests.get")
    def test_engineer_review_get_request_later_page_non_list_raises_error(
        self, mock_http_get
    ):
        """Later paginated page returning non-list JSON raises RequestException instead of returning partial results."""
        reviews_url = "https://api.github.com/repos/tensorflow/tensorflow/pulls/128063/reviews"
        commit_sha = "45e821c3df6e49c0ac3db3c06c62ff3f6469ce7e"
        marker = utils.format_commit_review_marker(commit_sha)

        page1_without_match = MagicMock()
        page1_without_match.status_code = 200
        page1_without_match.links = {}
        page1_without_match.headers = {
            "Link": (
                '<https://api.github.com/repositories/45717250/pulls/128063/reviews?per_page=100&page=2>; rel="next"'
            )
        }
        page1_without_match.json.return_value = [
            {
                "id": 1,
                "user": {"login": "reviewer-1"},
                "commit_id": commit_sha,
                "body": "Human review comment",
            }
        ]

        page1_with_match = MagicMock()
        page1_with_match.status_code = 200
        page1_with_match.links = {}
        page1_with_match.headers = {
            "Link": (
                '<https://api.github.com/repositories/45717250/pulls/128063/reviews?per_page=100&page=2>; rel="next"'
            )
        }
        page1_with_match.json.return_value = [
            {
                "id": 1,
                "user": {"login": "github-actions[bot]"},
                "commit_id": commit_sha,
                "body": f"### Summary\nAutomated review.{marker}",
            }
        ]

        non_list_page2 = MagicMock()
        non_list_page2.status_code = 200
        non_list_page2.links = {}
        non_list_page2.headers = {}
        non_list_page2.json.return_value = {"message": "Unexpected object on page 2"}

        # 1. Direct get_request call raises RequestException rather than returning partial list
        mock_http_get.side_effect = [page1_without_match, non_list_page2]
        with self.assertRaisesRegex(
            requests.exceptions.RequestException,
            r"Expected list response for paginated request.*got dict",
        ):
            utils.get_request(reviews_url, params={"per_page": 100})

        # 2. When page 1 has no match and page 2 is non-list, has_agent_reviewed_commit
        # explicitly catches RequestException and logs a warning instead of silently
        # iterating over partial results.
        mock_http_get.reset_mock()
        mock_http_get.side_effect = [page1_without_match, non_list_page2]
        with patch("builtins.print") as mock_print:
            self.assertFalse(
                utils.has_agent_reviewed_commit(reviews_url, commit_sha)
            )
            mock_print.assert_called_once()
            self.assertIn(
                "Warning: Failed to check PR reviews for commit",
                mock_print.call_args.args[0],
            )

        # 3. Even if page 1 contained a matching review, an incomplete paginated history
        # must fail explicitly and not pass has_agent_reviewed_commit.
        mock_http_get.reset_mock()
        mock_http_get.side_effect = [page1_with_match, non_list_page2]
        with patch("builtins.print") as mock_print:
            self.assertFalse(
                utils.has_agent_reviewed_commit(reviews_url, commit_sha)
            )
            mock_print.assert_called_once()
            self.assertIn(
                "Warning: Failed to check PR reviews for commit",
                mock_print.call_args.args[0],
            )

    @patch("agent.utils._fetch_file_content_at_commit")
    @patch("agent.utils.subprocess.run")
    def test_engineer_review_pylint_diagnostic_truncation_line_boundaries(
        self, mock_subproc_run, mock_fetch_content
    ):
        """Finding 2: Pylint diagnostic truncation respects the 5,000-char limit and cuts only on complete line boundaries."""
        mock_fetch_content.return_value = "x = 1\n"
        files = [{"path": "tensorflow/python/foo.py", "changeType": "MODIFIED"}]
        raw_diff = (
            "--- a/tensorflow/python/foo.py\n"
            "+++ b/tensorflow/python/foo.py\n"
            "@@ -1,0 +1,50 @@\n"
            + "".join(f"+x_{i} = {i}\n" for i in range(1, 51))
        )

        # 1. Output below the 5,000-character limit -> preserved unchanged
        short_diags = [
            f"tensorflow/python/foo.py:{i}:0: W0311 (bad-indentation): Bad indentation"
            for i in range(1, 6)
        ]
        mock_proc = MagicMock(returncode=4, stdout="\n".join(short_diags) + "\n")
        mock_subproc_run.return_value = mock_proc
        out_short = utils.run_pylint_on_changed_files(
            files, raw_diff=raw_diff, head_sha="deadbeef"
        )
        self.assertEqual(out_short, "\n".join(short_diags))

        # 2. Output exceeding the 5,000-character limit -> cut at the last newline within 5,000 chars
        many_diags = [
            f"tensorflow/python/foo.py:{i}:0: C0301 (line-too-long): Line too long ({120 + i}/80) "
            + ("x" * 60)
            for i in range(1, 51)
        ]
        full_joined = "\n".join(many_diags)
        self.assertGreater(len(full_joined), 5000)
        mock_proc.stdout = full_joined + "\n"
        out_exceeding = utils.run_pylint_on_changed_files(
            files, raw_diff=raw_diff, head_sha="deadbeef"
        )
        self.assertLessEqual(len(out_exceeding), 5000)
        self.assertGreater(len(out_exceeding), 0)
        out_lines = out_exceeding.split("\n")
        self.assertEqual(out_lines, many_diags[: len(out_lines)])
        # Adding the next diagnostic line would exceed 5,000 characters
        self.assertGreater(
            len(out_exceeding) + 1 + len(many_diags[len(out_lines)]), 5000
        )

        # 3. Newline near / right at the 5,000-character boundary (newline at index 4999 and at index 5000)
        prefix_4999 = "tensorflow/python/foo.py:1:0: C0301 (line-too-long): "
        line_4999 = prefix_4999 + ("a" * (4999 - len(prefix_4999)))
        self.assertEqual(len(line_4999), 4999)
        line2 = "tensorflow/python/foo.py:2:0: W0311 (bad-indentation): Bad indentation"
        mock_proc.stdout = f"{line_4999}\n{line2}\n"
        out_boundary_4999 = utils.run_pylint_on_changed_files(
            files, raw_diff=raw_diff, head_sha="deadbeef"
        )
        self.assertEqual(out_boundary_4999, line_4999)

        prefix_5000 = "tensorflow/python/foo.py:1:0: C0301 (line-too-long): "
        line_5000 = prefix_5000 + ("b" * (5000 - len(prefix_5000)))
        self.assertEqual(len(line_5000), 5000)
        mock_proc.stdout = f"{line_5000}\n{line2}\n"
        out_boundary_5000 = utils.run_pylint_on_changed_files(
            files, raw_diff=raw_diff, head_sha="deadbeef"
        )
        self.assertEqual(out_boundary_5000, line_5000)

        # 4. Single long line with no newline within the 5,000-character limit -> does not return partial diagnostic line
        long_single_line = (
            "tensorflow/python/foo.py:1:0: C0301 (line-too-long): "
            + ("z" * 5200)
        )
        self.assertGreater(len(long_single_line), 5000)
        mock_proc.stdout = long_single_line + "\n"
        out_no_newline = utils.run_pylint_on_changed_files(
            files, raw_diff=raw_diff, head_sha="deadbeef"
        )
        self.assertEqual(out_no_newline, "")
        self.assertLessEqual(len(out_no_newline), 5000)


if __name__ == "__main__":
    unittest.main()
