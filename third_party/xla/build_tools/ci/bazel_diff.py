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

from collections.abc import Iterable, Sequence
import dataclasses
import enum
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, List, Optional, Tuple

BAZEL_DIFF_VERSION = "v46.1.0"
BAZEL_DIFF_JAR_URL = "https://github.com/Tinder/bazel-diff/releases/download/v46.1.0/bazel-diff_deploy.jar"
BAZEL_DIFF_JAR_SHA256 = (
    "1026c66304d262e26065fc0ccd0281766c4c8b63abd18ea1fc4366fe05453c1e"
)

# Global configuration files whose modification invalidates incremental diffs
# and necessitates a full test suite run.
GLOBAL_BAZEL_CONFIG_PATTERNS: Tuple[str, ...] = (
    r"(^|/)MODULE\.bazel$",
    r"(^|/)REPO\.bazel$",
    r"(^|/)WORKSPACE(\.bzlmod)?$",
    r"\.bazelrc$",
    r"\.bazelversion$",
    # b/519689071: flag-defining .bzl (e.g. flags.bzl label_flag redirects)
    r"\.bzl$",
)

# b/519689071: keywords in BUILD/.bzl changes that signal unconfigured hashing
# under-selection (label_flag, config_setting, alias).
FULL_RUN_BUILD_KEYWORDS: Tuple[str, ...] = (
    "label_flag",
    "config_setting",
    "alias",
)

# Regex matching files that only contain documentation or repository metadata.
DOCS_OR_METADATA_PATTERN = re.compile(
    r"(\.md$|^docs/|(^|/)OWNERS$|^LICENSE|^\.clang|^\.gitignore|^\.vscode/|\.png$|\.jpg$|\.jpeg$|\.svg$|\.webp$|\.gif$)"
)

# Target prefixes to keep in XLA's CI.
KEEP_TARGET_PREFIXES: Tuple[str, ...] = (
    "//xla/",
    "//xla:",
    "//build_tools/",
    "@tsl//tsl/",
)


class BazelDiffDecisionType(enum.Enum):
  FULL = "FULL"
  SKIP = "SKIP"
  IMPACTED = "IMPACTED"


@dataclasses.dataclass(frozen=True)
class BazelDiffDecision:
  """Decision result from bazel-diff analysis."""

  decision: BazelDiffDecisionType
  impacted_targets: Tuple[str, ...] = ()
  impacted_targets_file: Optional[str] = None
  reason: str = ""
  changed_files_count: int = 0
  elapsed_seconds: float = 0.0


def matches_any_pattern(path: str, patterns: Sequence[str]) -> bool:
  """Returns True if path matches any regex pattern in patterns."""
  return any(re.search(pattern, path) for pattern in patterns)


def is_global_config_changed(changed_files: Sequence[str]) -> bool:
  """Returns True if any changed file matches global bazel config patterns."""
  return any(
      matches_any_pattern(f, GLOBAL_BAZEL_CONFIG_PATTERNS)
      for f in changed_files
  )


def is_docs_or_metadata_only(changed_files: Sequence[str]) -> bool:
  """Returns True if all changed files are docs or non-build metadata."""
  if not changed_files:
    return False
  return all(DOCS_OR_METADATA_PATTERN.search(f) for f in changed_files)


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
    res = subprocess.run(
        ["git", "merge-base", base_sha, head_sha],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0 and res.stdout.strip():
      return res.stdout.strip()
  except Exception:  # pylint: disable=broad-exception-caught
    pass
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
    The merge-base SHA if available, otherwise base_sha.
  """
  merge_base = get_merge_base(base_sha, head_sha, cwd=cwd)
  return merge_base if merge_base else base_sha


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
  cmd = ["git", "diff", "--name-only", diff_base, head_sha]
  res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
  return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def touches_build_keywords(
    changed_files: Sequence[str],
    base_sha: str,
    head_sha: str,
    cwd: str = ".",
) -> bool:
  """Returns True if any BUILD or .bzl diff contains sensitive keywords (b/519689071)."""
  build_or_bzl_files = [
      f for f in changed_files if f.endswith((".bzl", "BUILD", "BUILD.bazel"))
  ]
  if not build_or_bzl_files:
    return False

  try:
    diff_base = get_diff_base(base_sha, head_sha, cwd=cwd)
    cmd = ["git", "diff", diff_base, head_sha, "--", *build_or_bzl_files]
    res = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=True
    )
    diff_text = res.stdout
    for kw in FULL_RUN_BUILD_KEYWORDS:
      if kw in diff_text:
        logging.info("Found build keyword '%s' in diff; forcing full run", kw)
        return True
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("Failed to check build keywords: %s", e)
    return True
  return False


def normalize_target_label(label: str) -> str:
  """Normalizes Bazel target labels, converting Bzlmod canonical labels to standard repo forms."""
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


def filter_and_normalize_targets(
    targets: Iterable[str],
    keep_prefixes: Sequence[str] = KEEP_TARGET_PREFIXES,
) -> List[str]:
  """Normalizes and filters impacted targets, keeping only those matching keep_prefixes."""
  filtered: List[str] = []
  seen = set()

  for raw in targets:
    line = raw.strip()
    if not line or line.startswith("#"):
      continue
    # Drop external synthetic query targets that are not directly buildable
    if line.startswith(("//external:", "@external:")):
      continue

    normalized = normalize_target_label(line)
    if not normalized:
      continue

    if any(normalized.startswith(prefix) for prefix in keep_prefixes):
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
  tmp_dest = f"{dest_path}.tmp.{os.getpid()}"

  # Try curl with retries first
  if shutil.which("curl"):
    cmd = [
        "curl",
        "-fLo",
        tmp_dest,
        "--retry",
        "5",
        "--retry-connrefused",
        "--retry-all-errors",
        url,
    ]
    logging.info("Downloading bazel-diff jar via curl: %s", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode == 0 and verify_sha256(tmp_dest, expected_sha256):
      os.replace(tmp_dest, dest_path)
      return dest_path

  # Fallback to python urllib if curl failed or is unavailable
  import urllib.request  # pylint: disable=g-import-not-at-top

  logging.info("Downloading bazel-diff jar via urllib: %s", url)
  urllib.request.urlretrieve(url, tmp_dest)
  if not verify_sha256(tmp_dest, expected_sha256):
    if os.path.exists(tmp_dest):
      os.remove(tmp_dest)
    raise ValueError(
        f"SHA-256 mismatch for downloaded bazel-diff jar from {url}"
    )

  os.replace(tmp_dest, dest_path)
  return dest_path


def run_generate_hashes(
    jar_path: str,
    workspace_dir: str,
    output_json: str,
    config: Optional[str] = None,
    use_cquery: bool = True,
    startup_options: Optional[Sequence[str]] = None,
    cwd: Optional[str] = None,
) -> None:
  """Executes bazel-diff generate-hashes for the current workspace state."""
  args = [
      "java",
      "-jar",
      jar_path,
      "generate-hashes",
      "-k",
      "-w",
      workspace_dir,
      output_json,
      "--fineGrainedHashExternalRepos=@tsl",
  ]
  if use_cquery:
    args.append("--useCquery")
    args.append(
        "--cqueryExpression=deps(//xla/...:all-targets +"
        " //build_tools/...:all-targets + @tsl//tsl/...:all-targets)"
    )
    if config:
      args.append(f"--cqueryCommandOptions=--config={config}")
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
) -> None:
  """Executes bazel-diff get-impacted-targets."""
  args = [
      "java",
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
    for t in decision.impacted_targets[:10]:
      sys.stdout.write(f"  {t}\n")
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
          f.write("<details><summary>Impacted Targets List</summary>\n\n```\n")
          for t in decision.impacted_targets:
            f.write(f"{t}\n")
          f.write("```\n</details>\n\n")
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to write to GITHUB_STEP_SUMMARY: %s", e)


def dict_to_cli_options(d: Any) -> List[str]:
  """Converts a dict of options to Bazel CLI flags."""
  if not isinstance(d, dict):
    return []
  opts = []
  for k, v in d.items():
    if isinstance(v, bool) and v:
      opts.append(f"--{k}")
    elif isinstance(v, (list, tuple)):
      opts.extend(f"--{k}={item}" for item in v)
    else:
      opts.append(f"--{k}={v}")
  return opts


def compute_impacted_targets(
    build: Any,
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

  # Record original ref currently checked out so we can faithfully restore it
  original_head = head_sha
  try:
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0 and res.stdout.strip():
      original_head = res.stdout.strip()
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  # Fetch base SHA if shallow
  try:
    subprocess.run(
        ["git", "fetch", "--depth=1", "origin", base_sha],
        cwd=workspace_dir,
        check=False,
        capture_output=True,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  try:
    if not get_merge_base(base_sha, original_head, cwd=workspace_dir):
      subprocess.run(
          ["git", "fetch", "--unshallow", "origin"],
          cwd=workspace_dir,
          check=False,
          capture_output=True,
      )
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  # 1. Changed files analysis
  try:
    changed_files = get_changed_files(
        base_sha, original_head, cwd=workspace_dir
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.FULL,
        reason=f"Failed to get git changed files: {e}",
        elapsed_seconds=time.time() - start_time,
    )

  changed_count = len(changed_files)
  if is_global_config_changed(changed_files):
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.FULL,
        reason="Global Bazel configuration or .bzl modified",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  if touches_build_keywords(
      changed_files, base_sha, original_head, cwd=workspace_dir
  ):
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.FULL,
        reason=(
            "Diff touches label_flag, config_setting, or alias (b/519689071)"
        ),
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  if is_docs_or_metadata_only(changed_files):
    return BazelDiffDecision(
        decision=BazelDiffDecisionType.SKIP,
        reason="Only documentation or repository metadata modified",
        changed_files_count=changed_count,
        elapsed_seconds=time.time() - start_time,
    )

  # 2. Download bazel-diff
  try:
    jar_path = download_and_verify_jar(dest_jar_path)
  except Exception as e:  # pylint: disable=broad-exception-caught
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

  config_name = build.configs[0] if getattr(build, "configs", ()) else None
  use_cquery = getattr(
      build, "bazel_diff_use_cquery", getattr(build, "use_cquery", True)
  )
  startup_opts = None
  if hasattr(build, "startup_options") and build.startup_options:
    startup_opts = dict_to_cli_options(build.startup_options)

  diff_base = get_diff_base(base_sha, original_head, cwd=workspace_dir)
  try:
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
        config=config_name,
        use_cquery=use_cquery,
        startup_options=startup_opts,
    )

    # Generate head hashes
    subprocess.run(
        ["git", "checkout", "-f", original_head],
        cwd=workspace_dir,
        check=True,
        capture_output=True,
    )
    run_generate_hashes(
        jar_path,
        workspace_dir,
        head_hashes_json,
        config=config_name,
        use_cquery=use_cquery,
        startup_options=startup_opts,
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

    filtered_targets = filter_and_normalize_targets(raw_targets)

    # Rewrite the impacted targets file with the filtered list
    filtered_impacted_file = os.path.join(
        temp_dir, "filtered_impacted_targets.txt"
    )
    with open(filtered_impacted_file, "w", encoding="utf-8") as f:
      for t in filtered_targets:
        f.write(f"{t}\n")

    if not filtered_targets:
      return BazelDiffDecision(
          decision=BazelDiffDecisionType.SKIP,
          impacted_targets=(),
          reason="No targets impacted after filtering",
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

  except Exception as e:  # pylint: disable=broad-exception-caught
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
      subprocess.run(
          ["git", "checkout", "-f", original_head],
          cwd=workspace_dir,
          check=False,
          capture_output=True,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      pass
