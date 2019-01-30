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
import os
import tokenize

import gast
import six

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.util import tf_inspect


class LineLocation(
    collections.namedtuple('LineLocation', ('filename', 'lineno'))):
  """Similar to Location, but without column information.

  Attributes:
    filename: Text
    lineno: int, 1-based
  """
  pass


class Location(
    collections.namedtuple('Location', ('filename', 'lineno', 'col_offset'))):
  """Encodes code location information.

  Attributes:
    filename: Text
    lineno: int, 1-based
    col_offset: int
  """

  @property
  def line_loc(self):
    return LineLocation(self.filename, self.lineno)


class OriginInfo(
    collections.namedtuple(
        'OriginInfo',
        ('loc', 'function_name', 'source_code_line', 'comment'))):
  """Container for information about the source code before conversion.

  Attributes:
    loc: Location
    function_name: Optional[Text]
    source_code_line: Text
    comment: Optional[Text]
  """

  def as_frame(self):
    """Returns a 4-tuple consistent with the return of traceback.extract_tb."""
    return (self.loc.filename, self.loc.lineno, self.function_name,
            self.source_code_line)

  def __repr__(self):
    return '{}:{}:{}'.format(
        os.path.split(self.loc.filename)[1], self.loc.lineno,
        self.loc.col_offset)


# TODO(mdan): This source map should be a class - easier to refer to.
def create_source_map(nodes, code, filename, indices_in_code):
  """Creates a source map between an annotated AST and the code it compiles to.

  Args:
    nodes: Iterable[ast.AST, ...]
    code: Text
    filename: Optional[Text]
    indices_in_code: Union[int, Iterable[int, ...]], the positions at which
        nodes appear in code. The parser always returns a module when parsing
        code. This argument indicates the position in that module's body at
        which the corresponding of node should appear.

  Returns:
    Dict[CodeLocation, OriginInfo], mapping locations in code to locations
    indicated by origin annotations in node.
  """
  reparsed_nodes = parser.parse_str(code)
  reparsed_nodes = [reparsed_nodes.body[i] for i in indices_in_code]

  resolve(reparsed_nodes, code)
  result = {}

  for before, after in ast_util.parallel_walk(nodes, reparsed_nodes):
    # Note: generated code might not be mapped back to its origin.
    # TODO(mdan): Generated code should always be mapped to something.
    origin_info = anno.getanno(before, anno.Basic.ORIGIN, default=None)
    final_info = anno.getanno(after, anno.Basic.ORIGIN, default=None)
    if origin_info is None or final_info is None:
      continue

    line_loc = LineLocation(filename, final_info.loc.lineno)

    existing_origin = result.get(line_loc)
    if existing_origin is not None:
      # Overlaps may exist because of child nodes, but almost never to
      # different line locations. Exception make decorated functions, where
      # both lines are mapped to the same line in the AST.

      # Line overlaps: keep bottom node.
      if existing_origin.loc.line_loc == origin_info.loc.line_loc:
        if existing_origin.loc.lineno >= origin_info.loc.lineno:
          continue

      # In case of overlaps, keep the leftmost node.
      if existing_origin.loc.col_offset <= origin_info.loc.col_offset:
        continue

    result[line_loc] = origin_info

  return result


# TODO(znado): Consider refactoring this into a Visitor.
# TODO(mdan): Does this work correctly with inner functions?
def resolve(nodes, source, function=None):
  """Adds an origin information to all nodes inside the body of function.

  Args:
    nodes: Union[ast.AST, Iterable[ast.AST, ...]]
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
  if not isinstance(nodes, (list, tuple)):
    nodes = (nodes,)

  if function:
    _, function_lineno = tf_inspect.getsourcelines(function)
    function_filepath = tf_inspect.getsourcefile(function)
  else:
    function_lineno = None
    function_filepath = None

  # TODO(mdan): Pull this to a separate utility.
  code_reader = six.StringIO(source)
  comment_map = {}
  for token in tokenize.generate_tokens(code_reader.readline):
    tok_type, tok_string, loc, _, _ = token
    srow, _ = loc
    if tok_type == tokenize.COMMENT:
      comment_map[srow] = tok_string.strip()[1:].strip()

  source_lines = source.split('\n')
  for node in nodes:
    for n in gast.walk(node):
      if not hasattr(n, 'lineno'):
        continue

      lineno_in_body = n.lineno

      source_code_line = source_lines[lineno_in_body - 1]
      if function:
        source_lineno = function_lineno + lineno_in_body
        function_name = function.__name__
      else:
        source_lineno = lineno_in_body
        function_name = None

      location = Location(function_filepath, source_lineno, n.col_offset)
      origin = OriginInfo(location, function_name,
                          source_code_line, comment_map.get(source_lineno))
      anno.setanno(n, anno.Basic.ORIGIN, origin)
