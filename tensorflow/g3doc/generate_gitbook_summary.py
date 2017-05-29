#!/usr/bin/env python
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
""" # Script that use to generate `SUMMARY.md` for Gitbook.

## How to use

1. Install `gitbook` via following instructions in https://toolchain.gitbook.com/setup.html .
2. Run `python generate_gitbook_summary.py`.
3. Run `gitbook pdf` or `gitbook epub` to generate ebooks.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re


def format_output(docs_root, path):
  for line in open(path).readlines():
    match = re.search('^#\s*(.*)', line)
    if match:
      header = match.group(1)
      relative_path = os.path.relpath(path, docs_root)
      prefix = (len(relative_path.split(os.path.sep)) - 1) * "  "
      return "%s* [%s](%s)\n" % (prefix, header, relative_path)

  return ""


def _generate_summary(docs_root):
  yield "<!-- This file is machine generated: DO NOT EDIT! -->\n\n"

  for root, dir_names, file_names in os.walk(docs_root):
    relative_path = os.path.relpath(root, docs_root)
    if relative_path.startswith("_book"):
      continue

    index_file = os.path.join(root, "index.md")
    # Don't generate documents for those folders don't have `index.md`.
    if not os.path.exists(index_file):
      continue

    yield format_output(docs_root, index_file)

    for file_name in file_names:
      path = os.path.join(root, file_name)
      if path == index_file:
        continue

      if os.path.relpath(path, docs_root) in ["README.md", "SUMMARY.md"]:
        continue

      if not path.endswith(".md"):
        continue

      yield format_output(docs_root, path)


def generate_summary():
  docs_root = os.path.dirname(os.path.abspath(__file__))
  with open(os.path.join(docs_root, "SUMMARY.md"), "w") as output:
    for line in _generate_summary(docs_root):
      output.write(line)


if __name__ == "__main__":
  generate_summary()