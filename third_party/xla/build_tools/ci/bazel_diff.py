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
"""bazel-diff integration module for XLA CI.

Analyzes changed files between a base SHA and head commit, determines impacted
Bazel targets using bazel-diff (with cquery by default), filters and normalizes
labels for XLA, and produces a decision (FULL, SKIP, or IMPACTED targets).
"""

import argparse
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import enum
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, List, Optional, Protocol, Tuple

from build_tools.ci import change_detector

BAZEL_DIFF_VERSION = "v46.1.0"
BAZEL_DIFF_JAR_URL = "https://github.com/Tinder/bazel-diff/releases/download/v46.1.0/bazel-diff_deploy.jar"
BAZEL_DIFF_JAR_SHA256 = (
    "1026c66304d262e26065fc0ccd0281766c4c8b63abd18ea1fc4366fe05453c1e"
)

# Target prefixes to keep in XLA's CI.
KEEP_TARGET_PREFIXES: Tuple[str, ...] = (
    "//xla/",
    "//xla:",
    "//build_tools/",
    "@tsl//tsl/",
)

# Cap targets rendered in $GITHUB_STEP_SUMMARY so large diffs stay well below
# GitHub Actions' 1024 KB step summary upload limit.
_MAX_STEP_SUMMARY_TARGETS = 500


class BuildConfig(Protocol):
  """Protocol describing the Build attributes used by bazel-diff.

  Attributes:
    target_patterns: Sequence of Bazel target patterns to build or test.
    configs: Sequence of Bazel --config names.
    repo_env: Mapping of --repo_env variables.
    options: Mapping of additional Bazel flags.
    startup_options: Mapping of Bazel startup options.
    bazel_diff_use_cquery: Whether bazel-diff should use cquery (--useCquery).
  """

  target_patterns: Sequence[str]
  configs: Sequence[str]
  repo_env: Mapping[str, Any]
  options: Mapping[str, Any]
  startup_options: Mapping[str, Any]
  bazel_diff_use_cquery: bool


class BazelDiffDecisionType(enum.Enum):
  FULL = "FULL"
  SKIP = "SKIP"
  IMPACTED = "IMPACTED"


@dataclasses.dataclass(frozen=True)
class BazelDiffDecision:
  """Decision result from bazel-diff analysis.

  Attributes:
    decision: The decision type (FULL, SKIP, or IMPACTED).
    impacted_targets: Tuple of normalized impacted Bazel target labels.
    impacted_targets_file: Path to the file containing target patterns for
      Bazel's --target_pattern_file flag, if decision is IMPACTED.
    reason: Human-readable explanation for the decision.
    changed_files_count: Number of git changed files between base and head.
    elapsed_seconds: Wall-clock duration of the bazel-diff analysis in seconds.
  """

  decision: BazelDiffDecisionType
  impacted_targets: Tuple[str, ...] = ()
  impacted_targets_file: Optional[str] = None
  reason: str = ""
  changed_files_count: int = 0
  elapsed_seconds: float = 0.0


def normalize_target_label(label: str) -> str:
  """Normalizes Bazel target labels to standard repo forms."""
  label = label.strip()
  if not label:
    return ""

  # Root workspace bzlmod syntax: @@//... or @//... -> //...
  if label.startswith("@@//"):
    return label[2:]
  if label.startswith("@//"):
    return label[1:]

  # External repo canonical labels: @@+tsl+tsl//tsl/... or @@tsl//tsl/...
  # -> @tsl//tsl/...
  if label.startswith("@@") and "//" in label:
    repo_part, target_part = label[2:].split("//", 1)
    if "tsl" in repo_part:
      return f"@tsl//{target_part}"
    return f"@{repo_part}//{target_part}"

  return label


def target_matches_pattern(target: str, pattern: str) -> bool:
  """Checks if target label matches a Bazel target pattern."""
  if pattern.endswith("/..."):
    base = pattern[:-4]
    return (
        target == base
        or target.startswith(base + ":")
        or target.startswith(base + "/")
    )
  if pattern.endswith(":all") or pattern.endswith(":*"):
    package = pattern.rsplit(":", 1)[0]
    return target.startswith(package + ":")
  return target == pattern


def target_allowed_by_patterns(target: str, patterns: Sequence[str]) -> bool:
  """Checks if target is permitted by positive and negative patterns."""
  if not patterns:
    return True
  positive_patterns = [
      pattern for pattern in patterns if not pattern.startswith("-")
  ]
  negative_patterns = [
      pattern[1:] for pattern in patterns if pattern.startswith("-")
  ]
  if positive_patterns and not any(
      target_matches_pattern(target, pattern) for pattern in positive_patterns
  ):
    return False
  if any(
      target_matches_pattern(target, pattern) for pattern in negative_patterns
  ):
    return False
  return True


def filter_and_normalize_targets(
    targets: Iterable[str],
    keep_prefixes: Sequence[str] = KEEP_TARGET_PREFIXES,
    target_patterns: Sequence[str] = (),
) -> List[str]:
  """Normalizes and filters impacted targets against patterns."""
  filtered: List[str] = []
  seen = set()

  for raw_target in targets:
    line = raw_target.strip()
    if not line or line.startswith("#"):
      continue
    # Drop external synthetic query targets that are not directly buildable
    if line.startswith(("//external:", "@external:")):
      continue

    normalized = normalize_target_label(line)
    if not normalized:
      continue

    if any(normalized.startswith(prefix) for prefix in keep_prefixes):
      if target_patterns and not target_allowed_by_patterns(
          normalized, target_patterns
      ):
        continue
      if normalized not in seen:
        seen.add(normalized)
        filtered.append(normalized)

  return sorted(filtered)


def verify_sha256(file_path: str, expected_sha256: str) -> bool:
  """Verifies the SHA256 checksum of a file."""
  if not os.path.isfile(file_path):
    return False
  hasher = hashlib.sha256()
  with open(file_path, "rb") as f:
    while chunk := f.read(65536):
      hasher.update(chunk)
  return hasher.hexdigest().lower() == expected_sha256.lower()


def download_and_verify_jar(
    dest_path: Optional[str] = None,
    url: str = BAZEL_DIFF_JAR_URL,
    expected_sha256: str = BAZEL_DIFF_JAR_SHA256,
) -> str:
  """Downloads bazel-diff deploy jar with retries and verifies its SHA-256."""
  if dest_path is None:
    dest_path = os.path.join(tempfile.gettempdir(), "bazel-diff.jar")
  if verify_sha256(dest_path, expected_sha256):
    logging.info("Reusing existing verified bazel-diff jar at %s", dest_path)
    return dest_path

  os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
  temp_dest = f"{dest_path}.tmp.{os.getpid()}"

  command = [
      "curl",
      "-fLo",
      temp_dest,
      "--retry",
      "5",
      "--retry-connrefused",
      "--retry-all-errors",
      url,
  ]
  logging.info("Downloading bazel-diff jar via curl: %s", " ".join(command))
  subprocess.run(command, check=True)
  if not verify_sha256(temp_dest, expected_sha256):
    if os.path.exists(temp_dest):
      os.remove(temp_dest)
    raise ValueError(
        f"SHA-256 mismatch for downloaded bazel-diff jar from {url}"
    )

  os.replace(temp_dest, dest_path)
  return dest_path


def resolve_java_bin(
    workspace_dir: str = ".",
    startup_options: Optional[Sequence[str]] = None,
) -> str:
  """Resolves the java binary path, falling back to Bazel's embedded JDK."""
  if shutil.which("java"):
    return "java"
  java_home = os.environ.get("JAVA_HOME", "").strip()
  if java_home:
    candidate = os.path.join(java_home, "bin", "java")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
      return candidate
  try:
    cmd = ["bazel", *(startup_options or ()), "info", "java-home"]
    result = subprocess.run(
        cmd,
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    if isinstance(result.stdout, str):
      bazel_java_home = result.stdout.strip()
      if bazel_java_home:
        candidate = os.path.join(bazel_java_home, "bin", "java")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
          return candidate
  except (OSError, subprocess.SubprocessError) as e:
    logging.warning("Failed to query bazel info java-home: %s", e)
  return "java"


def run_generate_hashes(
    jar_path: str,
    workspace_dir: str,
    output_json: str,
    cquery_options: Optional[Sequence[str]] = None,
    use_cquery: bool = True,
    startup_options: Optional[Sequence[str]] = None,
    cwd: Optional[str] = None,
    java_bin: Optional[str] = None,
) -> None:
  """Executes bazel-diff generate-hashes for the current workspace state."""
  resolved_java = java_bin or resolve_java_bin(
      cwd or workspace_dir, startup_options=startup_options
  )
  fine_grained_repos = "@tsl,@@+tsl_extension+tsl" if use_cquery else "@tsl"
  args = [
      resolved_java,
      "-jar",
      jar_path,
      "generate-hashes",
      "-k",
      "-w",
      workspace_dir,
      output_json,
      f"--fineGrainedHashExternalRepos={fine_grained_repos}",
  ]
  if use_cquery:
    args.append("--useCquery")
    args.append(
        "--cqueryExpression=deps(//xla/...:all-targets +"
        " //build_tools/...:all-targets + @tsl//tsl/...:all-targets)"
    )
    if cquery_options:
      args.append(f"--cqueryCommandOptions={' '.join(cquery_options)}")
  if startup_options:
    args.append(f"--bazelStartupOptions={' '.join(startup_options)}")

  logging.info("Running bazel-diff generate-hashes: %s", " ".join(args))
  subprocess.run(args, cwd=cwd or workspace_dir, check=True)


def run_get_impacted_targets(
    jar_path: str,
    workspace_dir: str,
    starting_hashes: str,
    final_hashes: str,
    output_txt: str,
    cwd: Optional[str] = None,
    java_bin: Optional[str] = None,
) -> None:
  """Executes bazel-diff get-impacted-targets."""
  resolved_java = java_bin or resolve_java_bin(cwd or workspace_dir)
  args = [
      resolved_java,
      "-jar",
      jar_path,
      "get-impacted-targets",
      "-sh",
      starting_hashes,
      "-fh",
      final_hashes,
      "-w",
      workspace_dir,
      "-o",
      output_txt,
  ]
  logging.info("Running bazel-diff get-impacted-targets: %s", " ".join(args))
  subprocess.run(args, cwd=cwd or workspace_dir, check=True)


def report_decision(decision: BazelDiffDecision, build_name: str) -> None:
  """Emits decision summary to stdout and appends to $GITHUB_STEP_SUMMARY."""
  summary_line = (
      f"bazel-diff Build='{build_name}': "
      f"Decision={decision.decision.value}, "
      f"ChangedFiles={decision.changed_files_count}, "
      f"ImpactedTargets={len(decision.impacted_targets)}, "
      f"Reason='{decision.reason}', "
      f"Duration={decision.elapsed_seconds:.1f}s"
  )
  separator = "=" * 55
  sys.stdout.write(f"\n{separator}\n")
  sys.stdout.write(f"{summary_line}\n")
  if decision.decision == BazelDiffDecisionType.IMPACTED:
    sys.stdout.write(
        "Sample impacted targets (up to 10 of"
        f" {len(decision.impacted_targets)}):\n"
    )
    for target in decision.impacted_targets[:10]:
      sys.stdout.write(f"  {target}\n")
  sys.stdout.write(f"{separator}\n\n")
  sys.stdout.flush()

  step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
  if step_summary_file:
    try:
      with open(step_summary_file, "a", encoding="utf-8") as f:
        f.write("### bazel-diff Impact Analysis\n")
        f.write(
            "| Parameter | Value |\n"
            "|---|---|\n"
            f"| **Build** | `{build_name}` |\n"
            f"| **Decision** | **`{decision.decision.value}`** |\n"
            f"| **Changed files** | {decision.changed_files_count} |\n"
            f"| **Impacted targets** | {len(decision.impacted_targets)} |\n"
            f"| **Reason** | {decision.reason} |\n"
            f"| **Duration** | {decision.elapsed_seconds:.1f}s |\n\n"
        )
        if decision.decision == BazelDiffDecisionType.IMPACTED:
          displayed_targets = decision.impacted_targets[
              :_MAX_STEP_SUMMARY_TARGETS
          ]
          omitted_count = len(decision.impacted_targets) - len(
              displayed_targets
          )
          summary_label = "Impacted Targets List"
          if omitted_count > 0:
            summary_label += (
                f" (showing first {len(displayed_targets)} of"
                f" {len(decision.impacted_targets)})"
            )
          f.write(f"<details><summary>{summary_label}</summary>\n\n```\n")
          for target in displayed_targets:
            f.write(f"{target}\n")
          if omitted_count > 0:
            f.write(f"... ({omitted_count} more targets omitted)\n")
          f.write("```\n</details>\n\n")
    except OSError as e:
      logging.warning("Failed to write to GITHUB_STEP_SUMMARY: %s", e)


def dict_to_cli_options(options_dict: Any) -> List[str]:
  """Converts a dict of options to Bazel CLI flags."""
  if not isinstance(options_dict, dict):
    return []
  options = []
  for key, value in options_dict.items():
    if isinstance(value, bool) and value:
      options.append(f"--{key}")
    elif isinstance(value, (list, tuple)):
      options.extend(f"--{key}={item}" for item in value)
    else:
      options.append(f"--{key}={value}")
  return options


def get_cquery_command_options(build: BuildConfig) -> List[str]:
  """Extracts --config, --repo_env, and Starlark build flags from a Build."""
  options = [f"--config={config}" for config in build.configs]
  for key, value in build.repo_env.items():
    options.append(f"--repo_env={key}={value}")
  for key, value in build.options.items():
    if key.startswith(("//", "@")):
      if isinstance(value, bool) and value:
        options.append(f"--{key}")
      else:
        options.append(f"--{key}={value}")
  return options


# b/519689071: keywords in BUILD/.bzl changes that signal unconfigured hashing
# under-selection (label_flag, config_setting, alias).
FULL_RUN_BUILD_KEYWORDS: Tuple[str, ...] = (
    "label_flag(",
    "config_setting(",
    "alias(",
)


def _touches_build_keywords(
    changed_files: Sequence[str],
    base_sha: str,
    head_sha: str,
    cwd: str = ".",
) -> bool:
  """Returns True if any BUILD or .bzl diff contains sensitive keywords."""
  build_or_bzl_files = [
      file_path
      for file_path in changed_files
      if file_path.endswith((".bzl", "BUILD", "BUILD.bazel"))
  ]
  if not build_or_bzl_files:
    return False

  try:
    diff_base = change_detector.get_diff_base(base_sha, head_sha, cwd=cwd)
    safe_head = change_detector.validate_git_ref(head_sha)
    command = [
        "git",
        "diff",
        "-U0",
        "--end-of-options",
        diff_base,
        safe_head,
        "--",
        *build_or_bzl_files,
    ]
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=True
    )
    diff_text = result.stdout
    for keyword in FULL_RUN_BUILD_KEYWORDS:
      if keyword in diff_text:
        logging.info(
            "Found build keyword '%s' in diff; forcing full run", keyword
        )
        return True
  except (OSError, subprocess.SubprocessError, ValueError) as e:
    logging.warning("Failed to check build keywords: %s", e)
    return True
  return False


def compute_impacted_targets(
    build: BuildConfig,
    base_sha: str,
    head_sha: str = "HEAD",
    workspace_dir: str = ".",
    dest_jar_path: Optional[str] = None,
) -> BazelDiffDecision:
  """Computes impacted targets for a given build between base_sha and head_sha.

  Guaranteed fail-open: any unexpected error returns FULL decision so CI never
  fails due to a failure in bazel-diff.

  Args:
    build: Build object defining the target CI job configuration.
    base_sha: Base git SHA to compare against.
    head_sha: Head git SHA.
    workspace_dir: Absolute or relative path to the workspace root directory.
    dest_jar_path: Path where the bazel-diff jar will be cached.

  Returns:
    A BazelDiffDecision specifying whether to run FULL, SKIP, or IMPACTED tests.
  """
  start_time = time.time()
  workspace_dir = os.path.abspath(workspace_dir)
  if dest_jar_path is None:
    dest_jar_path = os.path.join(tempfile.gettempdir(), "bazel-diff.jar")

  try:
    base_sha = change_detector.validate_git_ref(base_sha)
    head_sha = change_detector.validate_git_ref(head_sha or "HEAD")
  except ValueError as e:
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.FULL,
        reason=f"Invalid git revision parameter: {e}",
        elapsed_seconds=time.time() - start_time,
    )

  # Record original ref currently checked out so we can faithfully restore it
  original_head = head_sha
  result = change_detector.run_git_best_effort(
      ["rev-parse", "HEAD"], cwd=workspace_dir
  )
  if result and result.returncode == 0 and result.stdout.strip():
    try:
      original_head = change_detector.validate_git_ref(result.stdout.strip())
    except ValueError:
      original_head = head_sha

  if head_sha == "HEAD":
    head_sha = original_head

  change_detector.ensure_base_fetched(base_sha, head_sha, cwd=workspace_dir)

  # 1. Changed files analysis
  try:
    changed_files = change_detector.get_changed_files(
        base_sha, head_sha, cwd=workspace_dir
    )
  except (OSError, subprocess.SubprocessError, ValueError) as e:
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.FULL,
        reason=f"Failed to get git changed files: {e}",
        elapsed_seconds=time.time() - start_time,
    )

  changed_count = len(changed_files)
  if change_detector.is_global_config_changed(changed_files):
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.FULL,
        reason="Global Bazel configuration modified",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  if _touches_build_keywords(
      changed_files, base_sha, head_sha, cwd=workspace_dir
  ):
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.FULL,
        reason=(
            "Diff touches label_flag, config_setting, or alias (b/519689071)"
        ),
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  # 2. Download bazel-diff
  try:
    jar_path = download_and_verify_jar(dest_jar_path)
  except (OSError, subprocess.SubprocessError, ValueError) as e:
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.FULL,
        reason=f"Failed to download/verify bazel-diff jar: {e}",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  # 3. Generate hashes with git checkout guarantees
  temp_dir = tempfile.mkdtemp(prefix="bazel_diff_")
  base_hashes_json = os.path.join(temp_dir, "base_hashes.json")
  head_hashes_json = os.path.join(temp_dir, "head_hashes.json")
  impacted_txt = os.path.join(temp_dir, "impacted_targets.txt")

  cquery_options = get_cquery_command_options(build)
  use_cquery = build.bazel_diff_use_cquery
  startup_options = None
  if build.startup_options:
    startup_options = dict_to_cli_options(build.startup_options)

  try:
    diff_base = change_detector.validate_git_ref(
        change_detector.get_diff_base(base_sha, head_sha, cwd=workspace_dir)
    )
    safe_head = change_detector.validate_git_ref(head_sha)

    # Generate base hashes
    subprocess.run(
        ["git", "checkout", "-f", diff_base],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
    )
    run_generate_hashes(
        jar_path,
        workspace_dir,
        base_hashes_json,
        cquery_options=cquery_options,
        use_cquery=use_cquery,
        startup_options=startup_options,
    )

    # Generate head hashes
    subprocess.run(
        ["git", "checkout", "-f", safe_head],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
    )
    run_generate_hashes(
        jar_path,
        workspace_dir,
        head_hashes_json,
        cquery_options=cquery_options,
        use_cquery=use_cquery,
        startup_options=startup_options,
    )

    # Determine impacted targets
    run_get_impacted_targets(
        jar_path,
        workspace_dir,
        base_hashes_json,
        head_hashes_json,
        impacted_txt,
    )

    with open(impacted_txt, "r", encoding="utf-8") as f:
      raw_targets = f.readlines()
  except (OSError, subprocess.SubprocessError, ValueError) as e:
    logging.warning("bazel-diff hashing/analysis failed: %s", e)
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.FULL,
        reason=f"Exception during bazel-diff run: {e}",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )
  finally:
    # Guaranteed restoration to original HEAD
    try:
      safe_original_head = change_detector.validate_git_ref(original_head)
      change_detector.run_git_best_effort(
          ["checkout", "-f", safe_original_head],
          cwd=workspace_dir,
      )
    except ValueError:
      logging.warning(
          "Skipping git checkout restore for invalid ref: %r", original_head
      )

  target_patterns = build.target_patterns
  filtered_targets = filter_and_normalize_targets(
      raw_targets, target_patterns=target_patterns
  )

  if not filtered_targets:
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.SKIP,
        impacted_targets=(),
        reason="No targets impacted after filtering",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  # Rewrite the impacted targets file with the filtered list
  filtered_impacted_file = os.path.join(
      temp_dir, "filtered_impacted_targets.txt"
  )
  try:
    with open(filtered_impacted_file, "w", encoding="utf-8") as f:
      for target in filtered_targets:
        f.write(f"{target}\n")
      if target_patterns:
        for pattern in target_patterns:
          if pattern.startswith("-"):
            f.write(f"{pattern}\n")
  except OSError as e:
    logging.warning("Failed to write filtered impacted targets file: %s", e)
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.FULL,
        reason=f"Exception writing impacted targets file: {e}",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  return BazelDiffDecision(
      decision=BazelDiffDecisionType.IMPACTED,
      impacted_targets=tuple(filtered_targets),
      impacted_targets_file=filtered_impacted_file,
      reason=f"Found {len(filtered_targets)} impacted targets",
      changed_files_count=changed_count,
      elapsed_seconds=time.time() - start_time,
  )


@dataclasses.dataclass
class CliBuildConfig:
  """Concrete BuildConfig constructed from CLI flags.

  Attributes:
    target_patterns: Sequence of Bazel target patterns to build or test.
    configs: Sequence of Bazel --config names.
    repo_env: Mapping of --repo_env variables.
    options: Mapping of additional Bazel flags.
    startup_options: Mapping of Bazel startup options.
    bazel_diff_use_cquery: Whether bazel-diff should use cquery (--useCquery).
  """

  target_patterns: Sequence[str] = ("//xla/...",)
  configs: Sequence[str] = ()
  repo_env: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  options: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  startup_options: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  bazel_diff_use_cquery: bool = False


def _write_github_output(
    github_output: str,
    *,
    decision: BazelDiffDecisionType,
    should_skip: bool,
    target_pattern_file: str = "",
) -> None:
  """Writes decision, should_skip, and target_pattern_file to $GITHUB_OUTPUT."""
  if not github_output:
    return
  try:
    with open(github_output, "a", encoding="utf-8") as f:
      f.write(f"decision={decision.value}\n")
      f.write(f"should_skip={'true' if should_skip else 'false'}\n")
      f.write(f"target_pattern_file={target_pattern_file}\n")
  except OSError as e:
    logging.warning(
        "Failed to write to GITHUB_OUTPUT (%s): %s", github_output, e
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses CLI arguments for standalone bazel-diff target selection."""
  parser = argparse.ArgumentParser(
      description="Compute impacted Bazel targets between two git revisions."
  )
  parser.add_argument(
      "--base_sha",
      default=os.environ.get("XLA_CI_BAZEL_DIFF_BASE_SHA", ""),
      help="Base git revision to compare against.",
  )
  parser.add_argument(
      "--head_sha",
      default=os.environ.get("XLA_CI_BAZEL_DIFF_HEAD_SHA", "HEAD") or "HEAD",
      help="Head git revision to compare.",
  )
  parser.add_argument(
      "--workspace_dir",
      default=".",
      help="Path to the Bazel workspace root.",
  )
  parser.add_argument(
      "--build_name",
      default="xla_ci",
      help="Build name used in step summary reports.",
  )
  parser.add_argument(
      "--target_pattern",
      action="append",
      dest="target_patterns",
      help="Bazel target pattern to filter impacted targets (repeatable).",
  )
  parser.add_argument(
      "--config",
      action="append",
      dest="configs",
      default=[],
      help="Bazel --config option for cquery (repeatable).",
  )
  parser.add_argument(
      "--repo_env",
      action="append",
      dest="repo_envs",
      default=[],
      help="Bazel --repo_env KEY=VALUE option for cquery (repeatable).",
  )
  parser.add_argument(
      "--bazelrc",
      default="",
      help="Optional path to a .bazelrc file passed as a Bazel startup option.",
  )
  parser.add_argument(
      "--use_cquery",
      action="store_true",
      help="Use bazel cquery instead of unconfigured query for target hashing.",
  )
  parser.add_argument(
      "--output_target_pattern_file",
      default="",
      help="Optional destination path for the filtered impacted targets file.",
  )
  parser.add_argument(
      "--github_output",
      default=os.environ.get("GITHUB_OUTPUT", ""),
      help="Path to GitHub Actions $GITHUB_OUTPUT file.",
  )
  return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> int:
  """Executes standalone bazel-diff target selection from parsed CLI flags.

  Args:
    args: Parsed command-line arguments from `_parse_args`.

  Returns:
    Process exit status (0 on completion or fail-open fallback).
  """
  disabled = os.environ.get("XLA_CI_DISABLE_BAZEL_DIFF", "false").lower() in (
      "true",
      "1",
      "yes",
  )
  base_sha = args.base_sha.strip()
  head_sha = args.head_sha.strip() or "HEAD"
  if disabled or not base_sha:
    _write_github_output(
        args.github_output,
        decision=BazelDiffDecisionType.FULL,
        should_skip=False,
    )
    return 0

  skip_decision = change_detector.evaluate_skip(
      base_sha, head_sha, cwd=args.workspace_dir, is_jax_build=False
  )
  if skip_decision.should_skip:
    change_detector.report_skip(skip_decision, args.build_name)
    _write_github_output(
        args.github_output,
        decision=BazelDiffDecisionType.SKIP,
        should_skip=True,
    )
    return 0

  target_patterns = (
      tuple(args.target_patterns) if args.target_patterns else ("//xla/...",)
  )
  repo_env = {}
  for entry in args.repo_envs:
    if "=" in entry:
      key, value = entry.split("=", 1)
      repo_env[key] = value
  startup_options = {"bazelrc": args.bazelrc} if args.bazelrc else {}
  build_config = CliBuildConfig(
      target_patterns=target_patterns,
      configs=tuple(args.configs),
      repo_env=repo_env,
      startup_options=startup_options,
      bazel_diff_use_cquery=args.use_cquery,
  )
  decision = compute_impacted_targets(
      build_config, base_sha, head_sha, workspace_dir=args.workspace_dir
  )
  report_decision(decision, args.build_name)

  if decision.decision == BazelDiffDecisionType.SKIP:
    _write_github_output(
        args.github_output,
        decision=BazelDiffDecisionType.SKIP,
        should_skip=True,
    )
    return 0

  if (
      decision.decision == BazelDiffDecisionType.IMPACTED
      and decision.impacted_targets_file
  ):
    target_file = decision.impacted_targets_file
    if args.output_target_pattern_file:
      try:
        shutil.copyfile(target_file, args.output_target_pattern_file)
        target_file = os.path.abspath(args.output_target_pattern_file)
      except OSError as e:
        logging.warning("Failed to copy target_pattern_file: %s", e)
        _write_github_output(
            args.github_output,
            decision=BazelDiffDecisionType.FULL,
            should_skip=False,
        )
        return 0
    _write_github_output(
        args.github_output,
        decision=BazelDiffDecisionType.IMPACTED,
        should_skip=False,
        target_pattern_file=target_file,
    )
    return 0

  _write_github_output(
      args.github_output,
      decision=BazelDiffDecisionType.FULL,
      should_skip=False,
  )
  return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
  """CLI entry point for standalone bazel-diff target selection in CI."""
  logging.basicConfig(level=logging.INFO)
  return run_cli(_parse_args(argv))


if __name__ == "__main__":
  sys.exit(main())
