# Copyright 2023 The OpenXLA Authors.
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
"""Parses diffs into hunks."""
import dataclasses
import re
import subprocess
from typing import Generator, Iterable, TypeVar

_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class Hunk:
  """Represents a hunk of a diff."""

  file: str
  start: int
  length: int
  lines: list[str]

  def added_lines(self) -> Generator[tuple[int, str], None, None]:
    current_line_no = self.start
    for line in self.lines:
      if line.startswith("+"):
        yield current_line_no, line[1:]  # elide leading '+'
        current_line_no += 1
      elif line.startswith("-"):
        continue
      else:
        current_line_no += 1


def batch(
    iterable: Iterable[_T], n: int
) -> Generator[tuple[_T, ...], None, None]:
  """Splits an iterable into chunks of size n.

  TODO(ddunleavy): once python 3.12 is available, use itertools.batch.

  Arguments:
    iterable: the iterable to batch.
    n: the number of elements in each batch.

  Yields:
    A tuple of length n of the type that the iterable produces.
  """
  iterator = iter(iterable)
  while True:
    try:
      # Unnecessary list here, but a generator won't raise StopIteration,
      # instead it will raise RuntimeError: "generator raises StopIteration".
      # I'd rather have a list comprehension in place of a generator expression
      # than catch RuntimeError and have to inspect the payload to verify it's
      # the one I want to be catching.
      yield tuple([next(iterator) for _ in range(n)])
    except StopIteration:
      return


def parse_hunks(diff: str) -> list[Hunk]:
  """Parses a diff into hunks.

  Arguments:
    diff: The raw output of git diff.

  Returns:
    A list of Hunks.
  """
  diff_pattern = (
      r"diff --git a/.* b/(.*)\n"  # capture file name
      r"(?:\w+ file mode \d+\n)?"  # maybe 'new file mode 100644' or similar
      r"index .*\n"
      r"--- .*\n"
      r"\+\+\+ .*\n"
  )

  # capture line number and length from header
  hunk_header_pattern = r"@@ -\d+,\d+ \+(\d+),(\d+) @@.*\n"

  # ignore initial empty match
  raw_per_file_hunks = re.split(diff_pattern, diff)[1:]

  parsed_hunks = []

  for file, raw_hunks in batch(raw_per_file_hunks, 2):
    # ignore initial empty match
    hunks = re.split(hunk_header_pattern, raw_hunks, re.MULTILINE)[1:]
    for start, length, body in batch(hunks, 3):
      lines = body.split("\n")
      lines = lines if lines[-1] else lines[:-1]  # trim empty line
      parsed_hunks.append(Hunk(file, int(start), int(length), lines))

  return parsed_hunks


def get_git_diff_stdout() -> str:
  """Run git diff with appropriate arguments and capture stdout as a str."""
  proc = subprocess.run(
      ["git", "diff", "origin/main", "HEAD"],
      capture_output=True,
      check=True,
      text=True,
  )
  return proc.stdout
