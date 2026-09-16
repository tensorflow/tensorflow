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
current diff, and prints their labels. Intended to be used with `run_dwyu.py`
in CI to check that modified targets depend on what they use.

Only targets that directly include a modified file are checked, rather than
all targets in affected packages. BUILD-only changes do not select targets.

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

_RULE_START = re.compile(r"^(\w+)\s*\(", re.MULTILINE)
_NAME_ATTR = re.compile(r'name\s*=\s*"([^"]+)"')
_STRING_LITERAL = re.compile(r'"([^"]+)"')

DEFAULT_ALLOWED_RULES = ("cc_library", "xla_test", "xla_cc_test", "tsl_cc_test")


def package_label(package: str) -> str:
  """Return the Bazel identity of a package, including the vendored TSL repo."""
  if package == "third_party/tsl":
    return "@tsl//"
  if package.startswith("third_party/tsl/"):
    return "@tsl//" + package.removeprefix("third_party/tsl/")
  return "//" + package


def get_changed_files(base_ref: str) -> list[str]:
  """List changed paths since the merge base, excluding unrelated base changes."""
  proc = subprocess.run(
      [
          "git",
          "diff",
          "--name-only",
          "-z",
          "--no-renames",
          f"{base_ref}...HEAD",
          "--",
      ],
      capture_output=True,
      check=True,
      text=True,
  )
  return sorted(set(proc.stdout.split("\0")) - {""})


def find_packages(changed_files: list[str]) -> set[str]:
  """Find Bazel packages containing the changed files."""
  packages = set()
  for filepath in changed_files:
    dirpath = os.path.dirname(filepath)
    while True:
      if os.path.isfile(os.path.join(dirpath, "BUILD")) or os.path.isfile(
          os.path.join(dirpath, "BUILD.bazel")
      ):
        packages.add(dirpath)
        break
      if not dirpath:
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
    changed_files_by_package: dict[str, set[str]],
) -> list[str]:
  """Find targets whose source files were modified.

  Args:
    packages: set of Bazel package paths to scan.
    allowed_rules: set of rule types to consider.
    changed_files_by_package: mapping from package path to set of changed
      package-relative file paths.

  Returns:
    list of Bazel target labels that include modified files.
  """
  targets = []
  for package in sorted(packages):
    for build_name in ("BUILD.bazel", "BUILD"):
      build_path = os.path.join(package, build_name)
      if not os.path.isfile(build_path):
        continue
      with open(build_path) as f:
        content = f.read()
      pkg_changed = changed_files_by_package.get(package, set())
      label = package_label(package)
      local_label = "//" + label.split("//", 1)[1]
      for target_name, source_files in extract_targets(content, allowed_rules):
        # Sources can be package-relative paths or labels in this package.
        source_paths = set()
        for source in source_files:
          if source.startswith((f"{label}:", f"{local_label}:")):
            source_paths.add(source.split(":", 1)[1])
          elif source.startswith(":"):
            source_paths.add(source[1:])
          elif not source.startswith(("//", "@")):
            source_paths.add(source)
        if source_paths & pkg_changed:
          targets.append(f"{label}:{target_name}")
      break
  return targets


def _group_changed_files_by_package(
    changed_files: list[str], packages: set[str]
) -> dict[str, set[str]]:
  """Group changed file paths by their Bazel package.

  Args:
    changed_files: list of changed file paths.
    packages: set of Bazel package paths to consider.

  Returns:
    A mapping from package path to set of changed package-relative file paths.
    BUILD files are excluded: changing one does not modify its targets' sources.
  """
  changed_files_by_package: dict[str, set[str]] = {}
  for filepath in changed_files:
    if os.path.basename(filepath) in ("BUILD", "BUILD.bazel"):
      continue
    dirpath = os.path.dirname(filepath)
    # Walk up to find which package this file belongs to.
    while True:
      if dirpath in packages:
        changed_files_by_package.setdefault(dirpath, set()).add(
            os.path.relpath(filepath, dirpath or ".")
        )
        break
      if not dirpath:
        break
      dirpath = os.path.dirname(dirpath)
  return changed_files_by_package


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
      help=(
          "Git ref whose merge base with HEAD to diff against"
          " (default: origin/main)"
      ),
  )
  args = parser.parse_args(argv[1:])
  allowed_rules = set(args.allowed_rules)

  changed = get_changed_files(args.base_ref)
  if not changed:
    logging.info("No files changed.")
    sys.exit(0)

  packages = find_packages(changed)
  if not packages:
    logging.info("No Bazel packages affected.")
    sys.exit(0)

  changed_files_by_package = _group_changed_files_by_package(changed, packages)
  targets = find_affected_targets(
      packages, allowed_rules, changed_files_by_package
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
