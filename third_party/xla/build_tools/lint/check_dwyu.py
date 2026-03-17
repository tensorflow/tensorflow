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
"""Find modified Bazel targets of allowed types for DWYU checking.

Finds Bazel packages affected by the current diff, extracts targets matching
an allowlist of rule types, and prints their labels. Intended to be used with
`bant dwyu` in CI to check that modified targets depend on what they use.

Usage:
  python3 build_tools/lint/check_dwyu.py --allowed_rules cc_library xla_test
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from typing import Sequence

from build_tools.lint import diff_parser

_RULE_START = re.compile(r"^(\w+)\s*\(", re.MULTILINE)
_NAME_ATTR = re.compile(r'name\s*=\s*"([^"]+)"')

DEFAULT_ALLOWED_RULES = ("cc_library", "xla_test", "xla_cc_test")


def get_diff(base_ref: str) -> str:
  """Run git diff against base_ref and return stdout."""
  proc = subprocess.run(
      ["git", "diff", base_ref, "HEAD"],
      capture_output=True,
      check=True,
      text=True,
  )
  return proc.stdout


def changed_files_from_diff(diff: str) -> list[str]:
  """Extract unique file paths from a parsed diff."""
  hunks = diff_parser.parse_hunks(diff)
  return sorted(set(h.file for h in hunks))


def find_packages(changed_files: list[str]) -> set[str]:
  """Find Bazel packages containing the changed files."""
  packages = set()
  for filepath in changed_files:
    dirpath = os.path.dirname(filepath)
    while dirpath:
      if os.path.isfile(os.path.join(dirpath, "BUILD")) or os.path.isfile(
          os.path.join(dirpath, "BUILD.bazel")
      ):
        packages.add(dirpath)
        break
      dirpath = os.path.dirname(dirpath)
  return packages


def extract_targets(build_content: str, allowed_rules: set[str]) -> list[str]:
  """Extract target names of allowed rule types from BUILD file content."""
  targets = []
  lines = build_content.split("\n")
  for i, line in enumerate(lines):
    m = _RULE_START.match(line.strip())
    if not m or m.group(1) not in allowed_rules:
      continue
    # name= is typically within the first few lines of a rule definition.
    for j in range(i, min(i + 5, len(lines))):
      nm = _NAME_ATTR.search(lines[j])
      if nm:
        targets.append(nm.group(1))
        break
  return targets


def find_affected_targets(
    packages: set[str], allowed_rules: set[str]
) -> list[str]:
  """Find targets of allowed types in the given packages."""
  targets = []
  for package in sorted(packages):
    for build_name in ("BUILD", "BUILD.bazel"):
      build_path = os.path.join(package, build_name)
      if not os.path.isfile(build_path):
        continue
      with open(build_path) as f:
        content = f.read()
      for target_name in extract_targets(content, allowed_rules):
        targets.append(f"//{package}:{target_name}")
      break
  return targets


def main(argv: Sequence[str]):
  parser = argparse.ArgumentParser(
      description="Find modified Bazel targets for DWYU checking."
  )
  parser.add_argument(
      "--allowed_rules",
      nargs="+",
      default=list(DEFAULT_ALLOWED_RULES),
      help="Rule types to include (default: %(default)s)",
  )
  parser.add_argument(
      "--base_ref",
      default="origin/main",
      help="Git ref to diff against (default: origin/main)",
  )
  args = parser.parse_args(argv[1:])
  allowed_rules = set(args.allowed_rules)

  diff = get_diff(args.base_ref)
  changed = changed_files_from_diff(diff)
  if not changed:
    logging.info("No files changed.")
    sys.exit(0)

  packages = find_packages(changed)
  if not packages:
    logging.info("No Bazel packages affected.")
    sys.exit(0)

  targets = find_affected_targets(packages, allowed_rules)
  if not targets:
    logging.info("No targets of allowed types found in affected packages.")
    sys.exit(0)

  logging.info("Found %d target(s) to check:", len(targets))
  for t in targets:
    logging.info("  %s", t)

  # Write targets to stdout, one per line, for consumption by bant.
  sys.stdout.write("\n".join(targets) + "\n")


if __name__ == "__main__":
  main(sys.argv)
