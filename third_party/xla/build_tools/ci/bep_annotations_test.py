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
# ==============================================================================

import base64
import io
import json
import os
import re
from unittest import mock

from absl.testing import absltest

from build_tools.ci import bep_annotations


def _write_bep(
    directory: str, name: str, events: list[dict[str, object]]
) -> str:
  path = os.path.join(directory, name)
  with open(path, "w", encoding="utf-8") as f:
    for event in events:
      f.write(json.dumps(event) + "\n")
  return path


class BepAnnotationsTest(absltest.TestCase):

  def test_failed_test_reads_local_log_tail_and_skips_passing_flake(self):
    tmpdir = self.create_tempdir().full_path
    flaky_log = os.path.join(tmpdir, "flaky.log")
    fail_log = os.path.join(tmpdir, "test.log")
    with open(flaky_log, "w", encoding="utf-8") as f:
      f.write("flaky failure attempt\n")
    with open(fail_log, "w", encoding="utf-8") as f:
      f.write(
          "\n".join(f"line {i}" for i in range(80)) + "\nASSERT 100% failed\n"
      )

    bep = _write_bep(
        tmpdir,
        "events.json",
        [
            {"id": {"started": {}}, "started": {"uuid": "inv-123"}},
            {
                "id": {"optionsParsed": {}},
                "optionsParsed": {
                    "cmdLine": [
                        "--bes_results_url=https://source.cloud.google.com/results/invocations/"
                    ]
                },
            },
            {
                "id": {"testResult": {"label": "//xla:flaky_pass"}},
                "testResult": {
                    "status": "FAILED",
                    "testActionOutput": [
                        {"name": "test.log", "uri": f"file://{flaky_log}"}
                    ],
                },
            },
            {
                "id": {"testSummary": {"label": "//xla:flaky_pass"}},
                "testSummary": {"overallStatus": "FLAKY"},
            },
            {
                "id": {"testResult": {"label": "//xla:real_fail"}},
                "testResult": {
                    "status": "FAILED",
                    "testActionOutput": [
                        {"name": "test.log", "uri": f"file://{fail_log}"}
                    ],
                },
            },
            {
                "id": {"testSummary": {"label": "//xla:real_fail"}},
                "testSummary": {"overallStatus": "FAILED"},
            },
            {
                "id": {"buildFinished": {}},
                "finished": {"exitCode": {"code": 3}},
            },
        ],
    )

    buf = io.StringIO()
    with mock.patch.object(
        bep_annotations,
        "_read_log_tail",
        wraps=bep_annotations._read_log_tail,
    ) as mock_read:
      count = bep_annotations.emit_github_annotations(
          [bep], label="CPU_CI", stream=buf
      )
    self.assertEqual(mock_read.call_count, 1)
    lines = buf.getvalue().splitlines()
    self.assertEqual(count, 1)
    self.assertLen(lines, 1)
    self.assertIn("title=//xla%3Areal_fail (FAILED)::", lines[0])
    self.assertIn(
        "CPU_CI | Invocation:"
        " https://source.cloud.google.com/results/invocations/inv-123",
        lines[0],
    )
    self.assertIn("ASSERT 100%25 failed", lines[0])
    self.assertIn("line 79", lines[0])
    self.assertNotIn("line 0%0A", lines[0])
    self.assertNotIn("flaky_pass", lines[0])

  def test_action_failure_uses_id_label_stdout_fallback_and_aborted(self):
    tmpdir = self.create_tempdir().full_path
    stdout_b64 = base64.b64encode(b"error: undeclared identifier\n").decode()
    bep = _write_bep(
        tmpdir,
        "events.json",
        [
            {"id": None, "started": None, "action": None, "aborted": None},
            {
                "id": {"buildMetadata": {}},
                "buildMetadata": {"metadata": {"INVOCATION_ID": "meta-9"}},
            },
            {
                "id": {"actionCompleted": {"label": "//xla:compile_target"}},
                "action": {
                    "success": False,
                    "stderr": {"contents": "!!!invalid-base64!!!"},
                    "stdout": {"contents": stdout_b64},
                },
            },
            {
                "id": {"targetConfigured": {"label": "//xla:skipped_target"}},
                "aborted": {"reason": "SKIPPED", "description": "ignored"},
            },
            {
                "id": {"targetConfigured": {"label": "//xla:bad_target"}},
                "aborted": {
                    "reason": "ANALYSIS_FAILURE",
                    "description": "missing dep",
                },
            },
        ],
    )

    buf = io.StringIO()
    count = bep_annotations.emit_github_annotations([bep], stream=buf)
    output = buf.getvalue()
    self.assertEqual(count, 2)
    self.assertIn("title=//xla%3Acompile_target (BUILD_FAILED)::", output)
    self.assertIn("error: undeclared identifier", output)
    self.assertIn("title=//xla%3Abad_target (ANALYSIS_FAILURE)::", output)
    self.assertIn("missing dep", output)
    self.assertIn("Invocation: meta-9", output)
    self.assertNotIn("skipped_target", output)

  def test_caps_annotations_and_truncates_without_splitting_escapes(self):
    tmpdir = self.create_tempdir().full_path
    long_log = base64.b64encode(("100% error line\n" * 400).encode()).decode()
    events: list[dict[str, object]] = [
        {"id": {"started": {}}, "started": {"uuid": "inv-cap"}}
    ]
    for i in range(15):
      long_label = f"//xla/{'sub:' * 40}:test_{i}"
      events.append({
          "id": {"testResult": {"label": long_label}},
          "testResult": {
              "status": "FAILED",
              "testActionOutput": [{"name": "test.log", "contents": long_log}],
          },
      })
    bep = _write_bep(tmpdir, "events.json", events)

    buf = io.StringIO()
    count = bep_annotations.emit_github_annotations(
        [bep],
        label="XLA",
        max_annotations=10,
        max_message_chars=400,
        stream=buf,
    )
    lines = buf.getvalue().splitlines()
    self.assertEqual(count, 10)
    self.assertLen(lines, 10)
    self.assertIn("and 6 more failing targets", lines[-1])
    bad_escape = re.compile(r"%(?![0-9A-Fa-f]{2})")
    for line in lines:
      title_part, msg = line.split("::", 2)[1:]
      title_val = title_part.removeprefix("error title=")
      self.assertLessEqual(len(title_val), 120)
      self.assertLessEqual(len(msg), 400)
      self.assertIsNone(bad_escape.search(title_val))
      self.assertIsNone(bad_escape.search(msg))

  def test_best_effort_on_success_missing_corrupt_and_cli(self):
    tmpdir = self.create_tempdir().full_path
    corrupt = os.path.join(tmpdir, "corrupt.json")
    with open(corrupt, "w", encoding="utf-8") as f:
      f.write("{not valid json\n")

    buf = io.StringIO()
    self.assertEqual(
        bep_annotations.emit_github_annotations(
            [os.path.join(tmpdir, "missing.json"), corrupt], stream=buf
        ),
        0,
    )
    self.assertEqual(buf.getvalue(), "")

    self.assertEqual(
        bep_annotations.emit_github_annotations(
            [corrupt], label="ROCm", exit_code=130, stream=buf
        ),
        1,
    )
    self.assertIn("ROCm failed (exit 130)", buf.getvalue())
    self.assertEqual(bep_annotations.main(["--invalid-flag"]), 0)
    self.assertEqual(bep_annotations.main([corrupt]), 0)


if __name__ == "__main__":
  absltest.main()
