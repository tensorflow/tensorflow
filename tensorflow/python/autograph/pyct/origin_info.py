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
import difflib
import os
import tokenize

import gast
import six

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
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
    line_loc: LineLocation
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
    if self.loc.filename:
      return '{}:{}:{}'.format(
          os.path.split(self.loc.filename)[1], self.loc.lineno,
          self.loc.col_offset)
    return '<no file>:{}:{}'.format(self.loc.lineno, self.loc.col_offset)


# TODO(mdan): This source map should be a class - easier to refer to.
def create_source_map(nodes, code, filepath):
  """Creates a source map between an annotated AST and the code it compiles to.

  Note: this function assumes nodes nodes, code and filepath correspond to the
  same code.

  Args:
    nodes: Iterable[ast.AST, ...], one or more AST modes.
    code: Text, the source code in which nodes are found.
    filepath: Text

  Returns:
    Dict[LineLocation, OriginInfo], mapping locations in code to locations
    indicated by origin annotations in node.
  """
  reparsed_nodes = parser.parse(code, preamble_len=0, single_node=False)
  for node in reparsed_nodes:
    resolve(node, code, filepath, node.lineno, node.col_offset)

  source_map = {}

  try:
    for before, after in ast_util.parallel_walk(nodes, reparsed_nodes):
      # Note: generated code might not be mapped back to its origin.
      # TODO(mdan): Generated code should always be mapped to something.
      origin_info = anno.getanno(before, anno.Basic.ORIGIN, default=None)
      final_info = anno.getanno(after, anno.Basic.ORIGIN, default=None)
      if origin_info is None or final_info is None:
        continue

      # Note: the keys are by line only, excluding the column offset.
      line_loc = LineLocation(final_info.loc.filename, final_info.loc.lineno)

      existing_origin = source_map.get(line_loc)
      if existing_origin is not None:
        # Overlaps may exist because of child nodes, but almost never to
        # different line locations. Exception make decorated functions, where
        # both lines are mapped to the same line in the AST.

        # Line overlaps: keep bottom node.
        if existing_origin.loc.line_loc == origin_info.loc.line_loc:
          if existing_origin.loc.lineno >= origin_info.loc.lineno:
            continue

        # In case of column overlaps, keep the leftmost node.
        if existing_origin.loc.col_offset <= origin_info.loc.col_offset:
          continue

      source_map[line_loc] = origin_info

  except ValueError as err:
    new_msg = 'Inconsistent ASTs detected. This is a bug. Cause: \n'
    new_msg += str(err)
    new_msg += 'Diff:\n'

    for n, rn in zip(nodes, reparsed_nodes):
      nodes_str = pretty_printer.fmt(n, color=False, noanno=True)
      reparsed_nodes_str = pretty_printer.fmt(rn, color=False, noanno=True)
      diff = difflib.context_diff(
          nodes_str.split('\n'),
          reparsed_nodes_str.split('\n'),
          fromfile='Original nodes',
          tofile='Reparsed nodes',
          n=7)
      diff = '\n'.join(diff)
      new_msg += diff + '\n'
    raise ValueError(new_msg)

  return source_map


class _Function(object):

  def __init__(self, name):
    self.name = name


class OriginResolver(gast.NodeVisitor):
  """Annotates an AST with additional source information like file name."""

  def __init__(self, root_node, source_lines, comments_map,
               context_lineno, context_col_offset,
               filepath):
    self._source_lines = source_lines
    self._comments_map = comments_map

    if (hasattr(root_node, 'decorator_list') and root_node.decorator_list and
        hasattr(root_node.decorator_list[0], 'lineno')):
      # Typical case: functions. The line number of the first decorator
      # is more accurate than the line number of the function itself in
      # 3.8+. In earier versions they coincide.
      self._lineno_offset = context_lineno - root_node.decorator_list[0].lineno
    else:
      # Fall back to the line number of the root node.
      self._lineno_offset = context_lineno - root_node.lineno

    self._col_offset = context_col_offset - root_node.col_offset

    self._filepath = filepath

    self._function_stack = []

  def _absolute_lineno(self, node):
    return node.lineno + self._lineno_offset

  def _absolute_col_offset(self, node):
    return node.col_offset + self._col_offset

  def _attach_origin_info(self, node):
    if self._function_stack:
      function_name = self._function_stack[-1].name
    else:
      function_name = None

    source_code_line = self._source_lines[node.lineno - 1]
    comment = self._comments_map.get(node.lineno)

    loc = Location(self._filepath, self._absolute_lineno(node),
                   self._absolute_col_offset(node))
    origin = OriginInfo(loc, function_name, source_code_line, comment)
    anno.setanno(node, 'lineno', node.lineno)
    anno.setanno(node, anno.Basic.ORIGIN, origin)

  def visit(self, node):
    entered_function = False
    if isinstance(node, gast.FunctionDef):
      entered_function = True
      self._function_stack.append(_Function(node.name))

    if hasattr(node, 'lineno'):
      self._attach_origin_info(node)
    self.generic_visit(node)

    if entered_function:
      self._function_stack.pop()


def resolve(node, source, context_filepath, context_lineno, context_col_offset):
  """Adds origin information to an AST, based on the source it was loaded from.

  This allows us to map the original source code line numbers to generated
  source code.

  Note: the AST may be a part of a larger context (e.g. a function is part of
  a module that may contain other things). However, this function does not
  assume the source argument contains the entire context, nor that it contains
  only code corresponding to node itself. However, it assumes that node was
  parsed from the given source code.
  For this reason, two extra arguments are required, and they indicate the
  location of the node in the original context.

  Args:
    node: gast.AST, the AST to annotate.
    source: Text, the source code representing node.
    context_filepath: Text
    context_lineno: int
    context_col_offset: int
  """
  # TODO(mdan): Pull this to a separate utility.
  code_reader = six.StringIO(source)
  comments_map = {}
  try:
    for token in tokenize.generate_tokens(code_reader.readline):
      tok_type, tok_string, loc, _, _ = token
      srow, _ = loc
      if tok_type == tokenize.COMMENT:
        comments_map[srow] = tok_string.strip()[1:].strip()
  except tokenize.TokenError:
    if isinstance(node, gast.Lambda):
      # Source code resolution in older Python versions is brittle for
      # lambda functions, and may contain garbage.
      pass
    else:
      raise

  source_lines = source.split('\n')
  visitor = OriginResolver(node, source_lines, comments_map,
                           context_lineno, context_col_offset,
                           context_filepath)
  visitor.visit(node)


def resolve_entity(node, source, entity):
  """Like resolve, but extracts the context information from an entity."""
  lines, lineno = tf_inspect.getsourcelines(entity)
  filepath = tf_inspect.getsourcefile(entity)

  # Poor man's attempt at guessing the column offset: count the leading
  # whitespace. This might not work well with tabs.
  definition_line = lines[0]
  col_offset = len(definition_line) - len(definition_line.lstrip())

  resolve(node, source, filepath, lineno, col_offset)


def copy_origin(from_node, to_node):
  """Copies the origin info from a node to another, recursively."""
  origin = anno.Basic.ORIGIN.of(from_node, default=None)
  if origin is None:
    return
  if not isinstance(to_node, (list, tuple)):
    to_node = (to_node,)
  for node in to_node:
    for n in gast.walk(node):
      anno.setanno(n, anno.Basic.ORIGIN, origin)
