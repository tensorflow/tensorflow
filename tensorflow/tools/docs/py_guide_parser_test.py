# Lint as: python2, python3
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
"""Tests for py_guide_parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six

from tensorflow.python.platform import test
from tensorflow.tools.docs import py_guide_parser


class TestPyGuideParser(py_guide_parser.PyGuideParser):

  def __init__(self):
    self.calls = []
    py_guide_parser.PyGuideParser.__init__(self)

  def process_title(self, line_number, title):
    self.calls.append((line_number, 't', title))

  def process_section(self, line_number, section_title, tag):
    self.calls.append((line_number, 's', '%s : %s' % (section_title, tag)))

  def process_in_blockquote(self, line_number, line):
    self.calls.append((line_number, 'b', line))
    self.replace_line(line_number, six.ensure_str(line) + ' BQ')

  def process_line(self, line_number, line):
    self.calls.append((line_number, 'l', line))


class PyGuideParserTest(test.TestCase):

  def testBasics(self):
    tmp = os.path.join(test.get_temp_dir(), 'py_guide_parser_test.md')
    f = open(tmp, 'w')
    f.write("""# a title
a line
## a section
```shell
in a blockquote
```
out of blockquote
""")
    f.close()
    parser = TestPyGuideParser()
    result = parser.process(tmp)
    expected = """# a title
a line
## a section
```shell BQ
in a blockquote BQ
```
out of blockquote
"""
    self.assertEqual(expected, result)
    expected = [(0, 't', 'a title'),
                (1, 'l', 'a line'),
                (2, 's', 'a section : a_section'),
                (3, 'b', '```shell'),
                (4, 'b', 'in a blockquote'),
                (5, 'l', '```'),
                (6, 'l', 'out of blockquote'),
                (7, 'l', '')]
    self.assertEqual(expected, parser.calls)


if __name__ == '__main__':
  test.main()
