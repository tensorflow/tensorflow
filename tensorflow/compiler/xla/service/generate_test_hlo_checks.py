#!/usr/bin/env python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Replace HLO instruction names with FileCheck variable captures.c.

Makes FileCheck-based tests on HLO more resilient to instruction name changes.
"""

import re
import shutil
import sys
import tempfile
from typing import Dict


ESCAPE_FILECHECK_VARNAME = re.compile(r"[^a-zA-Z0-9_]")


class FileCheckVarReplacer:
  """Replacer class for replacing HLO instructions by FileCheck captures."""

  _counter: int
  _replacement_cache: Dict[str, str]
  _check_instruction_matcher: re.Pattern = re.compile(r"^[^:]*CHECK[^:]*:.*=")
  _instr_name_matcher: re.Pattern = re.compile(r"%[\w-]+(\.\d+)?")

  def __init__(self):
    self._counter = -1
    self._replacement_cache = {}

  def replace_instruction_names_for_line(self, line: str) -> str:
    """Replaces all HLO instruction names by captured FileCheck variables.

    Works only for instruction definitions preceded by "CHECK-XXX: " directives.

    Args:
      line: One of test lines.

    Returns:
      A line with replacements applied.
    """
    if not self._check_instruction_matcher.match(line):
      # Reset internal storage non-matching lines
      self._counter = -1
      self._replacement_cache = {}
      return line

    return re.sub(self._instr_name_matcher, self._replacer, line)

  def _replacer(self, m: re.Match) -> str:
    instr_name = m.group(0)
    if instr_name in self._replacement_cache:
      return self._replacement_cache[instr_name]
    replacement_instr = self._generate_unique_varname(instr_name)
    self._replacement_cache[instr_name] = f"[[{replacement_instr}]]"
    return "".join([f"[[{replacement_instr}:", r"%[^ ]+", "]]"])

  def _generate_unique_varname(self, instr_name: str) -> str:
    self._counter += 1
    normalized_instr_name = ESCAPE_FILECHECK_VARNAME.sub(
        "_", instr_name.replace("%", ""))
    return f"{normalized_instr_name}_{self._counter}"


def replace_instruction_names(t: str) -> str:
  """Replaces all HLO instruction names by captured FileCheck variables.

  Args:
    t: Test text to replace

  Returns:
    Test with replacements applied.
  """
  f = FileCheckVarReplacer()
  out = []
  for line in t.split("\n"):
    out.append(f.replace_instruction_names_for_line(line))
  return "\n".join(out)


def main() -> None:
  argv = sys.argv
  if len(argv) != 2:
    raise Exception("Expecting exactly one filename argument (or -)")

  r = FileCheckVarReplacer()

  input_filename = argv[1]
  if input_filename == "-":
    # Read from input, write to stdout.
    for line in sys.stdin:
      sys.stdout.write(r.replace_instruction_names_for_line(line))
    return 0

  with open(input_filename) as f:
    # Replace contents of `input_filename`.
    fd, fname = tempfile.mkstemp()
    with open(fd, "w") as out_f:
      for line in f:
        out_f.write(r.replace_instruction_names_for_line(line))

  shutil.move(fname, input_filename)


if __name__ == "__main__":
  main()
