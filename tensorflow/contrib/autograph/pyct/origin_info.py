# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Container for origin source code information before AutoGraph compilation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.python.util import tf_inspect


class CodeLocation(
    collections.namedtuple('CodeLocation', ('file_path', 'line_number'))):
  """Location of a line of code.

  Attributes:
    file_path: text, the full path to the file containing the code.
    line_number: Int, the 1-based line number of the code in its file.
  """
  pass


class OriginInfo(
    collections.namedtuple('OriginInfo',
                           ('file_path', 'function_name', 'line_number',
                            'column_offset', 'source_code_line'))):
  """Container for information about the source code before conversion.

  Instances of this class contain information about the source code that
  transformed code originated from. Examples include:
    * line number
    * file name
    * original user code
  """

  def as_frame(self):
    """Makes a traceback frame tuple.

    Returns:
      A tuple of (file_path, line_number, function_name, source_code_line).
    """
    return (self.file_path, self.line_number, self.function_name,
            self.source_code_line)


# TODO(znado): Consider refactoring this into a Visitor.
def resolve(node, source, function=None):
  """Adds an origin information to all nodes inside the body of function.

  Args:
    node: The AST node for the function whose body nodes will be annotated.
    source: Text, the source code string for the function whose body nodes will
      be annotated.
    function: Callable, the function that will have all nodes inside of it
      annotation with an OriginInfo annotation with key anno.Basic.ORIGIN.  If
      it is None then only the line numbers and column offset will be set in the
      annotation, with the rest of the information being None.

  Returns:
    A tuple of the AST node for function and a String containing its source
    code.
  """
  if function:
    _, function_lineno = tf_inspect.getsourcelines(function)
    function_filepath = tf_inspect.getsourcefile(function)
  else:
    function_lineno = None
    function_filepath = None
  source_lines = source.split('\n')
  for n in gast.walk(node):
    if hasattr(n, 'lineno'):
      # n.lineno is relative to the start of the enclosing function, so need to
      # offset it by the line of the function.
      source_code_line = source_lines[n.lineno - 1]
      if function:
        source_lineno = n.lineno + function_lineno - 1
        function_name = function.__name__
      else:
        source_lineno = n.lineno
        function_name = None
      anno.setanno(
          n, anno.Basic.ORIGIN,
          OriginInfo(function_filepath, function_name, source_lineno,
                     n.col_offset, source_code_line))
