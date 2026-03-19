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

Finds Bazel targets whose source files (srcs/hdrs) were modified in the
current diff, and prints their labels. Intended to be used with `bant dwyu`
in CI to check that modified targets depend on what they use.

Only targets that directly include a modified file are checked, rather than
all targets in affected packages.

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
_STRING_LITERAL = re.compile(r'"([^"]+)"')

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


def _find_rule_end(lines: list[str], start: int) -> int:
  """Find the line index where the rule's parentheses are balanced."""
  depth = 0
  for i in range(start, len(lines)):
    depth += lines[i].count("(") - lines[i].count(")")
    if depth <= 0:
      return i
  return len(lines) - 1


def _extract_string_list(block: str, attr: str) -> list[str]:
  """Extract string literals from a list-valued attribute in a rule block."""
  pattern = re.compile(rf"{attr}\s*=\s*\[([^\]]*)\]", re.DOTALL)
  m = pattern.search(block)
  if not m:
    return []
  return _STRING_LITERAL.findall(m.group(1))


def extract_targets(
    build_content: str, allowed_rules: set[str]
) -> list[tuple[str, set[str]]]:
  """Extract target names and their source files from allowed rule types.

  Args:
    build_content: the contents of a BUILD file.
    allowed_rules: set of rule types to consider.

  Returns:
    A list of (target_name, source_files) tuples, where source_files
    is the set of filenames referenced in srcs and hdrs attributes.
  """
  targets = []
  lines = build_content.split("\n")
  for i, line in enumerate(lines):
    m = _RULE_START.match(line.strip())
    if not m or m.group(1) not in allowed_rules:
      continue
    # Find the target name.
    name = None
    for j in range(i, min(i + 5, len(lines))):
      nm = _NAME_ATTR.search(lines[j])
      if nm:
        name = nm.group(1)
        break
    if name is None:
      continue
    # Extract the full rule block to find srcs/hdrs.
    end = _find_rule_end(lines, i)
    block = "\n".join(lines[i : end + 1])
    source_files = set()
    source_files.update(_extract_string_list(block, "srcs"))
    source_files.update(_extract_string_list(block, "hdrs"))
    targets.append((name, source_files))
  return targets


def find_affected_targets(
    packages: set[str],
    allowed_rules: set[str],
    changed_basenames: dict[str, set[str]],
    build_files_changed: set[str],
) -> list[str]:
  """Find targets whose source files were modified.

  Args:
    packages: set of Bazel package paths to scan.
    allowed_rules: set of rule types to consider.
    changed_basenames: mapping from package path to set of changed file
      basenames within that package.
    build_files_changed: set of package paths whose BUILD files were modified.

  Returns:
    list of Bazel target labels that include modified files.
  """
  targets = []
  for package in sorted(packages):
    for build_name in ("BUILD", "BUILD.bazel"):
      build_path = os.path.join(package, build_name)
      if not os.path.isfile(build_path):
        continue
      with open(build_path) as f:
        content = f.read()
      pkg_changed = changed_basenames.get(package, set())
      build_changed = package in build_files_changed
      for target_name, source_files in extract_targets(content, allowed_rules):
        # Include target if: (1) any of its srcs/hdrs were modified, or
        # (2) the BUILD file itself was modified (deps may have changed).
        if build_changed or (source_files & pkg_changed):
          targets.append(f"//{package}:{target_name}")
      break
  return targets


def _group_changed_files_by_package(
    changed_files: list[str], packages: set[str]
) -> tuple[dict[str, set[str]], set[str]]:
  """Group changed file basenames by their Bazel package.

  Args:
    changed_files: list of changed file paths.
    packages: set of Bazel package paths to consider.

  Returns:
    A tuple of (changed_basenames, build_files_changed) where
    changed_basenames maps package path to set of changed file basenames,
    and build_files_changed is a set of package paths whose BUILD file
    was modified.
  """
  changed_basenames: dict[str, set[str]] = {}
  build_files_changed: set[str] = set()
  for filepath in changed_files:
    basename = os.path.basename(filepath)
    dirpath = os.path.dirname(filepath)
    # Walk up to find which package this file belongs to.
    while dirpath:
      if dirpath in packages:
        changed_basenames.setdefault(dirpath, set()).add(basename)
        if basename in ("BUILD", "BUILD.bazel"):
          build_files_changed.add(dirpath)
        break
      dirpath = os.path.dirname(dirpath)
  return changed_basenames, build_files_changed


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

  changed_basenames, build_files_changed = _group_changed_files_by_package(
      changed, packages
  )
  targets = find_affected_targets(
      packages, allowed_rules, changed_basenames, build_files_changed
  )
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
