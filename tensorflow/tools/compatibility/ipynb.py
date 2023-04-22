# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""A module to support operations on ipynb files"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import re
import shutil
import tempfile

import six

CodeLine = collections.namedtuple("CodeLine", ["cell_number", "code"])

def is_python(cell):
  """Checks if the cell consists of Python code."""
  return (cell["cell_type"] == "code"  # code cells only
          and cell["source"]  # non-empty cells
          and not six.ensure_str(cell["source"][0]).startswith("%%")
         )  # multiline eg: %%bash


def process_file(in_filename, out_filename, upgrader):
  """The function where we inject the support for ipynb upgrade."""
  print("Extracting code lines from original notebook")
  raw_code, notebook = _get_code(in_filename)
  raw_lines = [cl.code for cl in raw_code]

  # The function follows the original flow from `upgrader.process_fil`
  with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:

    processed_file, new_file_content, log, process_errors = (
        upgrader.update_string_pasta("\n".join(raw_lines), in_filename))

    if temp_file and processed_file:
      new_notebook = _update_notebook(
          notebook, raw_code,
          six.ensure_str(new_file_content).split("\n"))
      json.dump(new_notebook, temp_file)
    else:
      raise SyntaxError(
          "Was not able to process the file: \n%s\n" % "".join(log))

    files_processed = processed_file
    report_text = upgrader._format_log(log, in_filename, out_filename)
    errors = process_errors

  shutil.move(temp_file.name, out_filename)

  return files_processed, report_text, errors


def skip_magic(code_line, magic_list):
  """Checks if the cell has magic, that is not Python-based.

  Args:
      code_line: A line of Python code
      magic_list: A list of jupyter "magic" exceptions

  Returns:
    If the line jupyter "magic" line, not Python line

   >>> skip_magic('!ls -laF', ['%', '!', '?'])
  True
  """

  for magic in magic_list:
    if six.ensure_str(code_line).startswith(magic):
      return True

  return False


def check_line_split(code_line):
  r"""Checks if a line was split with `\`.

  Args:
      code_line: A line of Python code

  Returns:
    If the line was split with `\`

  >>> skip_magic("!gcloud ml-engine models create ${MODEL} \\\n")
  True
  """

  return re.search(r"\\\s*\n$", code_line)


def _get_code(input_file):
  """Loads the ipynb file and returns a list of CodeLines."""

  raw_code = []

  with open(input_file) as in_file:
    notebook = json.load(in_file)

  cell_index = 0
  for cell in notebook["cells"]:
    if is_python(cell):
      cell_lines = cell["source"]

      is_line_split = False
      for line_idx, code_line in enumerate(cell_lines):

        # Sometimes, jupyter has more than python code
        # Idea is to comment these lines, for upgrade time
        if skip_magic(code_line, ["%", "!", "?"]) or is_line_split:
          # Found a special character, need to "encode"
          code_line = "###!!!" + six.ensure_str(code_line)

          # if this cell ends with `\` -> skip the next line
          is_line_split = check_line_split(code_line)

        if is_line_split:
          is_line_split = check_line_split(code_line)

        # Sometimes, people leave \n at the end of cell
        # in order to migrate only related things, and make the diff
        # the smallest -> here is another hack
        if (line_idx == len(cell_lines) -
            1) and six.ensure_str(code_line).endswith("\n"):
          code_line = six.ensure_str(code_line).replace("\n", "###===")

        # sometimes a line would start with `\n` and content after
        # that's the hack for this
        raw_code.append(
            CodeLine(cell_index,
                     six.ensure_str(code_line.rstrip()).replace("\n",
                                                                "###===")))

      cell_index += 1

  return raw_code, notebook


def _update_notebook(original_notebook, original_raw_lines, updated_code_lines):
  """Updates notebook, once migration is done."""

  new_notebook = copy.deepcopy(original_notebook)

  # validate that the number of lines is the same
  assert len(original_raw_lines) == len(updated_code_lines), \
    ("The lengths of input and converted files are not the same: "
     "{} vs {}".format(len(original_raw_lines), len(updated_code_lines)))

  code_cell_idx = 0
  for cell in new_notebook["cells"]:
    if not is_python(cell):
      continue

    applicable_lines = [
        idx for idx, code_line in enumerate(original_raw_lines)
        if code_line.cell_number == code_cell_idx
    ]

    new_code = [updated_code_lines[idx] for idx in applicable_lines]

    cell["source"] = "\n".join(new_code).replace("###!!!", "").replace(
        "###===", "\n")
    code_cell_idx += 1

  return new_notebook
