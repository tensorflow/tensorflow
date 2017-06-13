# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Library for operating on Python API Guide files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re


def md_files_in_dir(py_guide_src_dir):
  """Returns a list of filename (full_path, base) pairs for guide files."""
  all_in_dir = [(os.path.join(py_guide_src_dir, f), f)
                for f in os.listdir(py_guide_src_dir)]
  return [(full, f) for full, f in all_in_dir
          if os.path.isfile(full) and f.endswith('.md')]


class PyGuideParser(object):
  """Simple parsing of a guide .md file.

  Descendants can override the process_*() functions (called by process())
  to either record information from the guide, or call replace_line()
  to affect the return value of process().
  """

  def __init__(self):
    self._lines = None

  def process(self, full_path):
    """Read and process the file at `full_path`."""
    md_string = open(full_path).read()
    self._lines = md_string.split('\n')
    seen = set()

    in_blockquote = False
    for i, line in enumerate(self._lines):
      if '```' in line:
        in_blockquote = not in_blockquote

      if not in_blockquote and line.startswith('# '):
        self.process_title(i, line[2:])
      elif not in_blockquote and line.startswith('## '):
        section_title = line.strip()[3:]
        existing_tag = re.search(' {([^}]+)} *$', line)
        if existing_tag:
          tag = existing_tag.group(1)
        else:
          tag = re.sub('[^a-zA-Z0-9]+', '_', section_title)
          if tag in seen:
            suffix = 0
            while True:
              candidate = '%s_%d' % (tag, suffix)
              if candidate not in seen:
                tag = candidate
                break
        seen.add(tag)
        self.process_section(i, section_title, tag)

      elif in_blockquote:
        self.process_in_blockquote(i, line)
      else:
        self.process_line(i, line)

    ret = '\n'.join(self._lines)
    self._lines = None
    return ret

  def replace_line(self, line_number, line):
    """Replace the contents of line numbered `line_number` with `line`."""
    self._lines[line_number] = line

  def process_title(self, line_number, title):
    pass

  def process_section(self, line_number, section_title, tag):
    pass

  def process_in_blockquote(self, line_number, line):
    pass

  def process_line(self, line_number, line):
    pass
