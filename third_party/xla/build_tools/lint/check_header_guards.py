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
"""Checks that newly added .h files have standard header guards."""

import argparse
from collections.abc import Sequence
import logging
import re
import subprocess
import sys

# Compiled regular expressions for header guard checks.
_IFNDEF_RE = re.compile(r"^#ifndef\s+([A-Z0-9_]+_H_)\b", re.MULTILINE)
_DEFINE_RE = re.compile(r"^#define\s+([A-Z0-9_]+_H_)\b", re.MULTILINE)


def get_added_header_files() -> list[str]:
  """Gets the list of newly added or copied header files.

  This function compares the current HEAD with 'origin/main' and returns a list
  of '.h' files that have been added or copied in the current branch.

  Returns:
    A list of strings, where each string is the path to a new header file.
    Returns an empty list if there's an error running the git command.
  """
  cmd = [
      "git",
      "diff",
      "--name-only",
      "--diff-filter=ARC",
      "origin/main",
      "HEAD",
  ]
  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [f for f in result.stdout.splitlines() if f.endswith(".h")]
  except subprocess.CalledProcessError as e:
    logging.error("Error running git diff: %s", e)
    logging.error(e.stderr)
    # Returns empty list to avoid failing the build over git issues.
    return []


def check_file(path: str) -> tuple[bool, str]:
  """Checks if a given file has standard C++ header guards.

  Args:
    path: The path to the header file to check (str).

  Returns:
    A tuple (is_valid, error_message), where is_valid is True if the file has
    a valid header guard and False otherwise, and error_message is a string
    describing the error if is_valid is False.
  """
  with open(path, "r", encoding="utf-8") as f:
    content = f.read()
  # Check for #ifndef and #define at the top of the file.
  ifndef_match = _IFNDEF_RE.search(content)
  define_match = _DEFINE_RE.search(content)

  if not ifndef_match or not define_match:
    return (
        False,
        ("Missing or malformed #ifndef or #define guard (must follow"
         " PROJECT_PATH_FILE_H_ convention)"),
    )

  # Ensure they use the same guard name
  guard_name = ifndef_match.group(1)
  if define_match.group(1) != guard_name:
    return (
        False,
        f"Mismatched guard name: {guard_name} vs {define_match.group(1)}",
    )

  # Check for #endif with the guard comment
  # It usually looks like `#endif  // XLA_PATH_H_`
  endif_re = re.compile(
      r"^#endif\s+//\s*" + re.escape(guard_name) + r"\b", re.MULTILINE
  )
  if not endif_re.search(content):
    return False, f"Missing or malformed #endif comment for {guard_name}"

  return True, ""


def main(argv: Sequence[str]) -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Checks that newly added .h files have standard header guards."
      )
  )
  parser.parse_args(argv[1:])

  logging.basicConfig(level=logging.INFO)

  headers = get_added_header_files()

  failed = []
  for h in headers:
    is_valid, error_msg = check_file(h)
    if not is_valid:
      failed.append((h, error_msg))

  if failed:
    for f, msg in failed:
      logging.error("Error in %s: %s", f, msg)
    sys.exit(1)
  logging.info("Header guard check passed or no new headers found!")


if __name__ == "__main__":
  main(sys.argv)
