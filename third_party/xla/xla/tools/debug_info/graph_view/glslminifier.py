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

"""A simple GLSL minifier tool.

Strips comments, redundant whitespaces, and empty lines from GLSL code.
"""

import argparse
import re


def minify_glsl(content: str) -> str:
  """Minifies GLSL content by removing comments and unnecessary whitespaces."""
  # Remove multi-line comments.
  content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

  lines = []
  for line in content.splitlines():
    # Remove single-line comments.
    line = re.sub(r'//.*', '', line)
    line = line.strip()
    if not line:
      continue

    # If it's a preprocessor directive, keep it as is (after strip).
    if line.startswith('#'):
      lines.append(line)
    else:
      # Remove extra spaces around common operators and comma/semicolon.
      line = re.sub(r'\s*([=+\-*/<>!&|{};,])\s*', r'\1', line)
      # Consolidate remaining multiple spaces into one.
      line = re.sub(r'\s+', ' ', line)
      lines.append(line)

  return '\n'.join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description='Minify GLSL files.')
  parser.add_argument('input', help='Path to input GLSL file')
  args = parser.parse_args()

  with open(args.input, 'r', encoding='utf-8') as f:
    content = f.read()

  minified = minify_glsl(content)
  print(minified, end='')


if __name__ == '__main__':
  main()
