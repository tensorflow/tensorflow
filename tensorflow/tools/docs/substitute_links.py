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

"""Convert @{symbol} to MarkDown links in the Python API guides."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from tensorflow.tools.docs import generate
from tensorflow.tools.docs import parser
from tensorflow.tools.docs import py_guide_parser


class UpdateTags(py_guide_parser.PyGuideParser):
  """Rewrites a Python guide so that each section has an explicit tag."""

  def process_section(self, line_number, section_title, tag):
    self.replace_line(line_number, '## %s {#%s}' % (section_title, tag))


def _main(input_dir, output_dir):
  """Convert all the files in `input_dir` and write results to `output_dir`."""
  visitor = generate.extract()

  header = '<!-- DO NOT EDIT! Automatically generated file. -->\n'

  # Iterate through all the source files and process them.
  tag_updater = UpdateTags()
  for dirpath, _, filenames in os.walk(input_dir):
    # How to get from `dirpath` to api_docs/python/
    relative_path_to_root = os.path.relpath(
        path=os.path.join(input_dir, 'api_docs/python'), start=dirpath)

    # Make the directory under output_dir.
    new_dir = os.path.join(output_dir,
                           os.path.relpath(path=dirpath, start=input_dir))
    try:
      if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    except OSError as e:
      print('Creating output dir "%s" failed: %s' % (new_dir, e))
      raise

    for base_name in filenames:
      full_in_path = os.path.join(dirpath, base_name)
      suffix = os.path.relpath(path=full_in_path, start=input_dir)
      full_out_path = os.path.join(output_dir, suffix)
      if not base_name.endswith('.md'):
        print('Copying non-md file %s...' % suffix)
        open(full_out_path, 'w').write(open(full_in_path).read())
        continue
      if dirpath.endswith('/api_guides/python'):
        print('Processing Python guide %s...' % base_name)
        md_string = tag_updater.process(full_in_path)
      else:
        print('Processing doc %s...' % suffix)
        md_string = open(full_in_path).read()

      output = parser.replace_references(
          md_string, relative_path_to_root, visitor.duplicate_of)
      open(full_out_path, 'w').write(header + output)

  print('Done.')


if __name__ == '__main__':
  argument_parser = argparse.ArgumentParser()
  argument_parser.add_argument(
      '--input_dir',
      type=str,
      default=None,
      required=True,
      help='Directory to copy docs from.'
  )
  argument_parser.add_argument(
      '--output_dir',
      type=str,
      default=None,
      required=True,
      help='Directory to write docs to.'
  )
  flags, _ = argument_parser.parse_known_args()
  _main(flags.input_dir, flags.output_dir)
