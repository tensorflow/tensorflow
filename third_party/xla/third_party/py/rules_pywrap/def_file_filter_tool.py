# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Tool to filter a windows .def file based on a filter .json."""
import argparse
import json
import re
from typing import Any, Dict, List, Pattern


def filter_def_file(
    def_file: str, filter_file: str, filtered_file: str
) -> None:
  """Filters a windows .def file based on a filter .json.

  Args:
    def_file: The path to the input windows .def file.
    filter_file: The path to the filter file (JSON format).
    filtered_file: The path to the output filtered windows .def file.
  """
  with open(filter_file, "r", encoding="utf-8") as filter_file_handle:
    filter_json: Dict[str, Any] = json.load(filter_file_handle)
    inclusion_patterns: List[str] = filter_json["global"] + ["EXPORTS", "*;*"]

    incl_patterns: List[Pattern[str]] = [
        re.compile(re.escape(p).replace("\\*", ".*"))
        for p in inclusion_patterns
    ]
    exclusion_patterns: List[str] = filter_json["local"]
    excl_patterns: List[Pattern[str]] = [
        re.compile(re.escape(p).replace("\\*", ".*"))
        for p in exclusion_patterns
    ]

  with open(def_file, "r") as orig_file, open(filtered_file, "w") as filt_file:
    for l in orig_file:
      if not matches_any(excl_patterns, l) or matches_any(incl_patterns, l):
        filt_file.write(l)


def matches_any(patterns: List[Pattern[str]], line: str) -> bool:
  """Checks if the line matches any of the given patterns.

  Args:
    patterns: A list of compiled regular expression patterns.
    line: The line to check for matches.

  Returns:
    True if the line matches any of the patterns, False otherwise.
  """
  stripped_line = line.strip()
  for pattern in patterns:
    if pattern.match(stripped_line):
      return True
  return False


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--def-file", required=True)
  parser.add_argument("--def-file-filter", required=True)
  parser.add_argument("--filtered-def-file", required=True)

  return parser.parse_args()


def main():
  args = parse_args()
  filter_def_file(args.def_file, args.def_file_filter, args.filtered_def_file)


if __name__ == "__main__":
  main()
