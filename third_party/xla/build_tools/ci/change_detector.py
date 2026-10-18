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
"""Git diff inspection and file-based fast-skip detection for XLA and JAX CI.

Provides generic git diff utilities operating on a caller-supplied directory
and evaluates whether a CI build can be skipped based on the modified files.
"""

from collections.abc import Sequence
import dataclasses
import logging
import os
import re
import subprocess
import sys
import time
from typing import List, Optional, Tuple

# Global configuration files whose modification invalidates incremental diffs
# and necessitates a full test suite run.
GLOBAL_BAZEL_CONFIG_PATTERNS: Tuple[str, ...] = (
    r"(^|/)MODULE\.bazel$",
    r"(^|/)REPO\.bazel$",
    r"(^|/)WORKSPACE(\.bzlmod)?$",
    r"\.bazelrc$",
    r"\.bazelversion$",
)

# Strict allowlist pattern for Git revision arguments (hexadecimal commit SHAs
# or alphanumeric ref names not starting with '-') to prevent flag injection.
SAFE_GIT_REF_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./~^-]*$")

# Regex matching files that only contain documentation or repository metadata.
DOCS_OR_METADATA_PATTERN = re.compile(
    r"(\.(md|rst|png|jpg|jpeg|svg|webp|gif)$"
    r"|(^|/)OWNERS$"
    r"|(^|/)LICENSE(\.(txt|md))?$"
    r"|(^|/)\.clang-(format|tidy)$"
    r"|(^|/)\.clangd$"
    r"|(^|/)\.gitignore$)"
)

# Regex matching BUILD, .bzl, or .proto files that could alter build graphs.
BUILD_OR_PROTO_PATTERN = re.compile(r"((^|/)BUILD(\.bazel)?$|\.bzl$|\.proto$)")

# Regex matching XLA unit test, benchmark, and HLO snapshot files.
XLA_TEST_OR_BENCHMARK_PATTERN = re.compile(
    r"((_test|_tests|_test_gpu|_test_cpu|_benchmark)\.(cc|h|cu\.cc|py)$"
    r"|\.hlo$)"
)


@dataclasses.dataclass(frozen=True)
class SkipDecision:
  """Result of evaluating whether a CI build can be skipped.

  Attributes:
    should_skip: Whether the CI build can be safely skipped.
    reason: Human-readable explanation for the decision.
    changed_files_count: Number of git changed files between base and head.
    elapsed_seconds: Wall-clock duration of the skip evaluation in seconds.
  """

  should_skip: bool
  reason: str = ""
  changed_files_count: int = 0
  elapsed_seconds: float = 0.0


def validate_git_ref(ref: str) -> str:
  """Validates that ref is a safe Git revision string and cannot inject flags.

  Args:
    ref: Git commit SHA or ref name to validate.

  Returns:
    The validated ref string.

  Raises:
    ValueError: If ref is empty, starts with '-', or contains disallowed chars.
  """
  if not ref or ref.startswith("-") or not SAFE_GIT_REF_PATTERN.fullmatch(ref):
    raise ValueError(f"Invalid or unsafe git revision parameter: {ref!r}")
  return ref


def matches_any_pattern(path: str, patterns: Sequence[str]) -> bool:
  """Returns True if path matches any regex pattern in patterns."""
  return any(re.search(pattern, path) for pattern in patterns)


def is_global_config_changed(changed_files: Sequence[str]) -> bool:
  """Returns True if any changed file matches global bazel config patterns."""
  return any(
      matches_any_pattern(file_path, GLOBAL_BAZEL_CONFIG_PATTERNS)
      for file_path in changed_files
  )


def is_docs_or_metadata_only(changed_files: Sequence[str]) -> bool:
  """Returns True if all changed files are docs or non-build metadata."""
  if not changed_files:
    return False
  return all(
      bool(DOCS_OR_METADATA_PATTERN.search(file_path))
      for file_path in changed_files
  )


def run_git_best_effort(
    args: Sequence[str], cwd: str = "."
) -> Optional[subprocess.CompletedProcess[str]]:
  """Runs a git command best-effort, returning None on OS/subprocess error."""
  try:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
  except (OSError, subprocess.SubprocessError):
    return None


def get_merge_base(
    base_sha: str, head_sha: str = "HEAD", cwd: str = "."
) -> Optional[str]:
  """Computes git merge-base between base_sha and head_sha.

  Args:
    base_sha: Base git commit SHA.
    head_sha: Head git commit SHA or ref.
    cwd: Directory where the git command should be executed.

  Returns:
    The merge base commit SHA as a string, or None if not found.
  """
  try:
    safe_base = validate_git_ref(base_sha)
    safe_head = validate_git_ref(head_sha)
  except ValueError:
    return None
  result = run_git_best_effort(
      ["merge-base", "--", safe_base, safe_head], cwd=cwd
  )
  if result and result.returncode == 0 and result.stdout.strip():
    return result.stdout.strip()
  return None


def get_diff_base(base_sha: str, head_sha: str = "HEAD", cwd: str = ".") -> str:
  """Returns the base commit to diff against head_sha.

  Uses git merge-base when available so that changes that landed on the base
  branch after the feature branch diverged are not falsely attributed to the PR.

  Args:
    base_sha: Base git commit SHA.
    head_sha: Head git commit SHA or ref.
    cwd: Directory where the git command should be executed.

  Returns:
    The merge-base SHA if available, otherwise validated base_sha.
  """
  safe_base = validate_git_ref(base_sha)
  safe_head = validate_git_ref(head_sha)
  merge_base = get_merge_base(safe_base, safe_head, cwd=cwd)
  return validate_git_ref(merge_base) if merge_base else safe_base


def ensure_base_fetched(
    base_sha: str, head_sha: str = "HEAD", cwd: str = "."
) -> None:
  """Best-effort fetches base_sha and unshallows if merge-base is missing."""
  try:
    safe_base = validate_git_ref(base_sha)
    safe_head = validate_git_ref(head_sha)
  except ValueError:
    return
  run_git_best_effort(
      ["fetch", "--depth=1", "--", "origin", safe_base], cwd=cwd
  )
  if not get_merge_base(safe_base, safe_head, cwd=cwd):
    run_git_best_effort(["fetch", "--unshallow", "origin"], cwd=cwd)


def get_changed_files(
    base_sha: str, head_sha: str = "HEAD", cwd: str = "."
) -> List[str]:
  """Returns list of changed filepaths between base_sha and head_sha.

  Args:
    base_sha: Base git commit SHA.
    head_sha: Head git commit SHA or ref.
    cwd: Directory where the git command should be executed.

  Returns:
    List of relative paths of changed files.
  """
  diff_base = get_diff_base(base_sha, head_sha, cwd=cwd)
  safe_head = validate_git_ref(head_sha)
  command = [
      "git",
      "diff",
      "--name-only",
      "--end-of-options",
      diff_base,
      safe_head,
      "--",
  ]
  result = subprocess.run(
      command, cwd=cwd, capture_output=True, text=True, check=True
  )
  return [
      line.strip().replace("\\", "/")
      for line in result.stdout.splitlines()
      if line.strip()
  ]


def is_build_or_proto_or_ci_core_file(file_path: str) -> bool:
  """Returns True if file_path requires a full run on JAX presubmits."""
  normalized = file_path.replace("\\", "/")
  if matches_any_pattern(normalized, GLOBAL_BAZEL_CONFIG_PATTERNS):
    return True
  if BUILD_OR_PROTO_PATTERN.search(normalized):
    return True
  if normalized.startswith((".github/", "build_tools/")) and not (
      DOCS_OR_METADATA_PATTERN.search(normalized)
  ):
    return True
  return False


def is_unimpacted_for_all_jax(file_path: str) -> bool:
  """Returns True if file_path is a test-only or non-code file for JAX CI."""
  normalized = file_path.replace("\\", "/")
  if is_build_or_proto_or_ci_core_file(normalized):
    return False
  if DOCS_OR_METADATA_PATTERN.search(normalized):
    return True
  if (
      normalized.startswith("xla/")
      and not normalized.startswith("xla/mosaic/")
      and XLA_TEST_OR_BENCHMARK_PATTERN.search(normalized)
  ):
    return True
  return False


def evaluate_skip(
    base_sha: str,
    head_sha: str = "HEAD",
    cwd: str = ".",
    *,
    is_jax_build: bool = False,
) -> SkipDecision:
  """Evaluates whether a CI build can be skipped based on git changed files.

  Guaranteed fail-open: any git, OS, or validation failure returns
  should_skip=False.

  Args:
    base_sha: Base git commit SHA to diff against.
    head_sha: Head git commit SHA or ref.
    cwd: Path to the git repository root to inspect.
    is_jax_build: Whether the build under evaluation is a JAX CI job.

  Returns:
    A SkipDecision indicating whether the build can be skipped and why.
  """
  start_time = time.time()
  cwd = os.path.abspath(cwd)

  try:
    base_sha = validate_git_ref(base_sha)
    head_sha = validate_git_ref(head_sha or "HEAD")
  except ValueError as e:
    return SkipDecision(
        should_skip=False,
        reason=f"Invalid git revision parameter: {e}",
        elapsed_seconds=time.time() - start_time,
    )

  ensure_base_fetched(base_sha, head_sha, cwd=cwd)

  try:
    changed_files = get_changed_files(base_sha, head_sha, cwd=cwd)
  except (OSError, subprocess.SubprocessError, ValueError) as e:
    return SkipDecision(
        should_skip=False,
        reason=f"Failed to get git changed files: {e}",
        elapsed_seconds=time.time() - start_time,
    )

  changed_count = len(changed_files)
  if not changed_files:
    return SkipDecision(
        should_skip=False,
        reason="No changed files detected",
        changed_files_count=0,
        elapsed_seconds=time.time() - start_time,
    )

  if is_docs_or_metadata_only(changed_files):
    return SkipDecision(
        should_skip=True,
        reason="Only documentation or repository metadata modified",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  if not is_jax_build:
    return SkipDecision(
        should_skip=False,
        reason="Non-documentation changes require target analysis",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  if any(is_build_or_proto_or_ci_core_file(f) for f in changed_files):
    return SkipDecision(
        should_skip=False,
        reason="Build configuration, proto, or CI workflow modified",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  if all(is_unimpacted_for_all_jax(f) for f in changed_files):
    return SkipDecision(
        should_skip=True,
        reason="Modified files do not affect JAX builds",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  return SkipDecision(
      should_skip=False,
      reason="Changed files may impact JAX build targets",
      changed_files_count=changed_count,
      elapsed_seconds=time.time() - start_time,
  )


def report_skip(decision: SkipDecision, build_name: str) -> None:
  """Emits skip decision summary to stdout and appends to $GITHUB_STEP_SUMMARY."""
  status = "SKIP" if decision.should_skip else "RUN"
  summary_line = (
      f"change-detector Build='{build_name}': "
      f"Decision={status}, "
      f"ChangedFiles={decision.changed_files_count}, "
      f"Reason='{decision.reason}', "
      f"Duration={decision.elapsed_seconds:.2f}s"
  )
  separator = "=" * 55
  sys.stdout.write(f"\n{separator}\n")
  sys.stdout.write(f"{summary_line}\n")
  sys.stdout.write(f"{separator}\n\n")
  sys.stdout.flush()

  step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
  if step_summary_file:
    try:
      with open(step_summary_file, "a", encoding="utf-8") as f:
        f.write("### Change Detector Analysis\n")
        f.write(
            "| Parameter | Value |\n"
            "|---|---|\n"
            f"| **Build** | `{build_name}` |\n"
            f"| **Decision** | **`{status}`** |\n"
            f"| **Changed files** | {decision.changed_files_count} |\n"
            f"| **Reason** | {decision.reason} |\n"
            f"| **Duration** | {decision.elapsed_seconds:.2f}s |\n\n"
        )
    except OSError as e:
      logging.warning("Failed to write to GITHUB_STEP_SUMMARY: %s", e)
