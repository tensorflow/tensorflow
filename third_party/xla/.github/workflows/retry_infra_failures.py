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
"""Detects transient CI infrastructure failures and triggers job reruns."""

import argparse
import dataclasses
import logging
import os
import re
import subprocess
import time
from typing import Optional, Sequence

import github_api

# Known infrastructure error patterns indicating transient runner failures.
# Tuple format: (regex_pattern, human_readable_description)
INFRA_ERROR_PATTERNS: Sequence[tuple[str, str]] = (
    # GKE / K8s runner container hook authorization timeout
    (
        (
            r"FetchError: request to"
            r" https://.*/apis/authorization\.k8s\.io/.* failed"
        ),
        "GKE control plane SelfSubjectAccessReview authorization timeout",
    ),
    # Runner container hook execution failure (generic wrapper)
    (
        r"Executing the custom container implementation failed",
        "Runner container hook execution failure",
    ),
    # Runner node shutdown or eviction signal
    (
        r"The runner has received a shutdown signal",
        "Runner pod shutdown / eviction signal received",
    ),
    # Runner connection drop with GitHub Actions backend
    (
        r"Runner lost communication with the server",
        "Runner lost communication with GitHub Actions backend",
    ),
    # Docker / container runtime daemon unreachable
    (
        r"Cannot connect to the Docker daemon",
        "Runner container runtime daemon connection failure",
    ),
    # Kubernetes container startup error
    (
        r"A container failed to start\. Error: CreateContainerConfigError",
        "Kubernetes container creation error",
    ),
)


@dataclasses.dataclass(frozen=True)
class InfraFailureMatch:
  """Record of a failed job matching an infrastructure error signature."""

  job_id: int
  job_name: str
  category: str
  matched_pattern: str
  source: str  # "annotations" or "logs"
  sample: str


def check_text_for_infra_errors(
    text: str,
    job_id: int,
    job_name: str,
    source: str,
) -> Optional[InfraFailureMatch]:
  """Checks text content against known infrastructure error patterns."""
  for pattern, description in INFRA_ERROR_PATTERNS:
    if match := re.search(pattern, text):
      matched_sample = match.group(0).strip()
      return InfraFailureMatch(
          job_id=job_id,
          job_name=job_name,
          category=description,
          matched_pattern=pattern,
          source=source,
          sample=matched_sample,
      )
  return None


def inspect_job_annotations(
    api: github_api.GitHubAPI,
    repo: str,
    job_id: int,
    job_name: str,
) -> Optional[InfraFailureMatch]:
  """Queries check run annotations for known infrastructure error signatures."""
  try:
    annotations = api.get_check_run_annotations(repo, job_id)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(
        "Failed to fetch annotations for job %d (%s): %s", job_id, job_name, e
    )
    return None

  for annotation in annotations:
    # Check all relevant text fields in the annotation
    for field in ("message", "title", "raw_details"):
      content = annotation.get(field)
      if content:
        if match := check_text_for_infra_errors(
            content, job_id, job_name, source="annotation"
        ):
          return match
  return None


def inspect_job_logs(
    api: github_api.GitHubAPI,
    repo: str,
    job_id: int,
    job_name: str,
    max_tail_lines: int = 250,
) -> Optional[InfraFailureMatch]:
  """Fetches and inspects the tail of a job log for infra error signatures."""
  try:
    log_content = api.get_job_logs(repo, job_id)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(
        "Failed to fetch logs for job %d (%s): %s", job_id, job_name, e
    )
    return None

  lines = log_content.splitlines()
  tail_lines = lines[-max_tail_lines:] if len(lines) > max_tail_lines else lines
  tail_text = "\n".join(tail_lines)

  return check_text_for_infra_errors(tail_text, job_id, job_name, source="logs")


def write_step_summary(
    matches: Sequence[InfraFailureMatch],
    repo: str,
    run_id: int,
    run_attempt: int,
    max_attempts: int,
    dry_run: bool,
    rerun_succeeded: bool,
) -> None:
  """Appends a markdown summary of detection and rerun to the step summary."""
  summary_path = os.getenv("GITHUB_STEP_SUMMARY")
  if not summary_path:
    return

  run_url = f"https://github.com/{repo}/actions/runs/{run_id}"
  if rerun_succeeded:
    status_badge = "✅ **Rerun Dispatched**"
    action_text = (
        f"Triggered partial rerun of failed jobs for attempt {run_attempt + 1}"
        f" of {max_attempts}."
    )
  elif dry_run:
    status_badge = "🧪 **Dry-Run Mode**"
    action_text = (
        "Matches detected. Rerun was not triggered because `--dry-run` was set."
    )
  else:
    status_badge = "❌ **Rerun Failed**"
    action_text = "Failed to dispatch rerun via GitHub API / CLI."

  lines = [
      "## 🔄 Infrastructure Auto-Retry Evaluator",
      "",
      (
          f"Evaluated workflow run [#{run_id}]({run_url}) "
          f"(attempt {run_attempt} of {max_attempts}). Status: {status_badge}"
      ),
      "",
      (
          f"Detected **{len(matches)}** job(s) matching known infrastructure"
          " errors:"
      ),
      "",
      "| Job Name | Job ID | Source | Failure Category | Matched Detail |",
      "| :--- | :--- | :--- | :--- | :--- |",
  ]
  for m in matches:
    lines.append(
        f"| `{m.job_name}` | `{m.job_id}` | {m.source} | {m.category} |"
        f" `{m.sample[:80]}` |"
    )

  lines.extend(["", action_text, ""])

  try:
    with open(summary_path, "a", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")
  except IOError as e:
    logging.warning("Failed to write to GITHUB_STEP_SUMMARY: %s", e)


def trigger_partial_rerun(
    api: github_api.GitHubAPI, repo: str, run_id: int
) -> bool:
  """Dispatches a rerun for only the failed jobs in a workflow run."""
  try:
    logging.info(
        "Triggering rerun for failed jobs in run %d via REST API...", run_id
    )
    api.rerun_failed_jobs(repo, run_id)
    logging.info("Successfully requested rerun via GitHub API.")
    return True
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(
        "REST API rerun request failed: %s. Falling back to gh CLI...", e
    )

  # Fallback to gh CLI
  try:
    cmd = ["gh", "run", "rerun", str(run_id), "--failed", "--repo", repo]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    logging.info(
        "Successfully requested rerun via gh CLI: %s", result.stdout.strip()
    )
    return True
  except subprocess.CalledProcessError as err:
    logging.error("gh CLI rerun request failed: %s", err.stderr.strip())
    return False


def evaluate_and_retry(
    api: github_api.GitHubAPI,
    repo: str,
    run_id: int,
    max_attempts: int = 2,
    dry_run: bool = False,
    cooldown_seconds: int = 15,
) -> bool:
  """Main evaluation routine: inspects failures and triggers rerun."""
  logging.info(
      "Evaluating workflow run %d in %s (max attempts: %d)...",
      run_id,
      repo,
      max_attempts,
  )

  try:
    run_data = api.get_workflow_run(repo, run_id)
  except Exception as e:  # pylint: disable=broad-except
    logging.error("Error fetching workflow run %d: %s", run_id, e)
    return False

  run_attempt = run_data.get("run_attempt", 1)
  status = run_data.get("status")
  conclusion = run_data.get("conclusion")

  logging.info(
      "Run status: %s, conclusion: %s, attempt: %d/%d",
      status,
      conclusion,
      run_attempt,
      max_attempts,
  )

  if run_attempt >= max_attempts:
    logging.info(
        "Run attempt %d has reached or exceeded max allowed attempts (%d)."
        " Will not retry.",
        run_attempt,
        max_attempts,
    )
    return False

  if status != "completed":
    logging.info(
        "Run is not completed yet (status=%s). Skipping evaluation.", status
    )
    return False

  if conclusion != "failure":
    logging.info(
        "Run conclusion is '%s' (not 'failure'). Skipping evaluation.",
        conclusion,
    )
    return False

  # Fetch jobs for this workflow run
  try:
    jobs = api.get_workflow_run_jobs(repo, run_id)
  except Exception as e:  # pylint: disable=broad-except
    logging.error("Error fetching jobs for run %d: %s", run_id, e)
    return False

  failed_jobs = [j for j in jobs if j.get("conclusion") == "failure"]
  if not failed_jobs:
    logging.info("No failed jobs found in this workflow run. Nothing to retry.")
    return False

  logging.info(
      "Found %d failed job(s). Checking for infra error signatures...",
      len(failed_jobs),
  )

  infra_matches: list[InfraFailureMatch] = []
  for job in failed_jobs:
    job_id = job["id"]
    job_name = job.get("name", f"job-{job_id}")
    logging.info("Scanning failed job: '%s' (ID: %d)...", job_name, job_id)

    # Check 1: Check run annotations
    match = inspect_job_annotations(api, repo, job_id, job_name)
    if match:
      logging.info(
          "  -> Matched via annotation: %s ('%s')", match.category, match.sample
      )
      infra_matches.append(match)
      continue

    # Check 2: Check run logs tail
    match = inspect_job_logs(api, repo, job_id, job_name)
    if match:
      logging.info(
          "  -> Matched via logs: %s ('%s')", match.category, match.sample
      )
      infra_matches.append(match)
      continue

    logging.info("  -> No infrastructure error signature detected.")

  if not infra_matches:
    logging.info(
        "None of the failed jobs matched known infrastructure error patterns."
    )
    logging.info("Treating as legitimate code / test failure. Skipping retry.")
    return False

  logging.info(
      "Matched %d of %d failed job(s) to transient infrastructure errors.",
      len(infra_matches),
      len(failed_jobs),
  )

  if dry_run:
    logging.info("[DRY-RUN] `--dry-run` set: skipping rerun dispatch.")
    write_step_summary(
        infra_matches,
        repo=repo,
        run_id=run_id,
        run_attempt=run_attempt,
        max_attempts=max_attempts,
        dry_run=True,
        rerun_succeeded=False,
    )
    return True

  if cooldown_seconds > 0:
    logging.info(
        "Waiting %ds cooldown before rerun to let transient settle...",
        cooldown_seconds,
    )
    time.sleep(cooldown_seconds)

  success = trigger_partial_rerun(api, repo, run_id)
  write_step_summary(
      infra_matches,
      repo=repo,
      run_id=run_id,
      run_attempt=run_attempt,
      max_attempts=max_attempts,
      dry_run=False,
      rerun_succeeded=success,
  )
  return success


def parse_args() -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Inspects a failed workflow run for infrastructure errors and"
          " triggers a partial rerun."
      )
  )
  parser.add_argument(
      "--repo",
      type=str,
      default=os.getenv("GITHUB_REPOSITORY", "openxla/xla"),
      help="GitHub repository in 'owner/repo' format (default: openxla/xla)",
  )
  parser.add_argument(
      "--run-id",
      type=int,
      default=int(os.getenv("WORKFLOW_RUN_ID", "0"))
      if os.getenv("WORKFLOW_RUN_ID")
      else None,
      required=not bool(os.getenv("WORKFLOW_RUN_ID")),
      help="ID of the workflow run to evaluate (default: $WORKFLOW_RUN_ID)",
  )
  parser.add_argument(
      "--max-attempts",
      type=int,
      default=2,
      help="Maximum allowed attempts before skipping retries (default: 2)",
  )
  parser.add_argument(
      "--cooldown-seconds",
      type=int,
      default=15,
      help="Cooldown seconds to wait before triggering rerun (default: 15)",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Evaluate and report matches without triggering an actual rerun",
  )
  parser.add_argument(
      "--token",
      type=str,
      default=os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN"),
      help="GitHub personal access token or action token",
  )
  return parser.parse_args()


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args = parse_args()
  api = github_api.GitHubAPI(args.token)
  evaluate_and_retry(
      api=api,
      repo=args.repo,
      run_id=args.run_id,
      max_attempts=args.max_attempts,
      dry_run=args.dry_run,
      cooldown_seconds=args.cooldown_seconds,
  )


if __name__ == "__main__":
  main()
