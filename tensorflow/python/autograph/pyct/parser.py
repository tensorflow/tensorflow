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
"""Converting code to AST.

Adapted from Tangent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import inspect
import linecache
import re
import sys
import textwrap
import tokenize

import astunparse
import gast
import six

from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.util import tf_inspect


PY2_PREAMBLE = textwrap.dedent("""
from __future__ import division
from __future__ import print_function
""")
PY3_PREAMBLE = ''
MAX_SIZE = 0

if sys.version_info >= (3, 9):
  astunparse = ast

if sys.version_info >= (3,):
  STANDARD_PREAMBLE = PY3_PREAMBLE
  MAX_SIZE = sys.maxsize
else:
  STANDARD_PREAMBLE = PY2_PREAMBLE
  MAX_SIZE = sys.maxint

STANDARD_PREAMBLE_LEN = STANDARD_PREAMBLE.count('__future__')


_LEADING_WHITESPACE = re.compile(r'\s*')


def _unfold_continuations(code_string):
  """Removes any backslash line continuations from the code."""
  return code_string.replace('\\\n', '')


def dedent_block(code_string):
  """Dedents a code so that its first line starts at row zero."""

  code_string = _unfold_continuations(code_string)

  token_gen = tokenize.generate_tokens(six.StringIO(code_string).readline)

  block_indentation = None
  tokens = []
  try:
    for tok in token_gen:
      tokens.append(tok)
  except tokenize.TokenError:
    # Resolution of lambda functions may yield incomplete code, which can
    # in turn generate this error. We silently ignore this error because the
    # parser may still be able to deal with it.
    pass

  for tok in tokens:
    tok_type, tok_string, _, _, _ = tok
    if tok_type == tokenize.INDENT:
      block_indentation = tok_string
      block_level = len(block_indentation)
      break
    elif tok_type not in (
        tokenize.NL, tokenize.NEWLINE, tokenize.STRING, tokenize.COMMENT):
      block_indentation = ''
      break

  if not block_indentation:
    return code_string

  block_level = len(block_indentation)
  first_indent_uses_tabs = '\t' in block_indentation
  for i, tok in enumerate(tokens):
    tok_type, tok_string, _, _, _ = tok
    if tok_type == tokenize.INDENT:
      if ((' ' in tok_string and first_indent_uses_tabs)
          or ('\t' in tok_string and not first_indent_uses_tabs)):
        # TODO(mdan): We could attempt to convert tabs to spaces by unix rule.
        # See:
        # https://docs.python.org/3/reference/lexical_analysis.html#indentation
        raise errors.UnsupportedLanguageElementError(
            'code mixing tabs and spaces for indentation is not allowed')
      if len(tok_string) >= block_level:
        tok_string = tok_string[block_level:]
      tokens[i] = (tok_type, tok_string)

  new_code = tokenize.untokenize(tokens)

  # Note: untokenize respects the line structure, but not the whitespace within
  # lines. For example, `def foo()` may be untokenized as `def foo ()`
  # So instead of using the output of dedent, we match the leading whitespace
  # on each line.
  dedented_code = []
  for line, new_line in zip(code_string.split('\n'), new_code.split('\n')):
    original_indent = re.match(_LEADING_WHITESPACE, line).group()
    new_indent = re.match(_LEADING_WHITESPACE, new_line).group()
    if len(original_indent) > len(new_indent):
      dedented_line = line[len(original_indent) - len(new_indent):]
    else:
      dedented_line = line
    dedented_code.append(dedented_line)
  new_code = '\n'.join(dedented_code)

  return new_code


def parse_entity(entity, future_features):
  """Returns the AST and source code of given entity.

  Args:
    entity: Any, Python function/method/class
    future_features: Iterable[Text], future features to use (e.g.
      'print_statement'). See
      https://docs.python.org/2/reference/simple_stmts.html#future

  Returns:
    gast.AST, Text: the parsed AST node; the source code that was parsed to
    generate the AST (including any prefixes that this function may have added).
  """
  if inspect_utils.islambda(entity):
    return _parse_lambda(entity)

  try:
    original_source = inspect_utils.getimmediatesource(entity)
  except OSError as e:
    raise errors.InaccessibleSourceCodeError(
        f'Unable to locate the source code of {entity}. Note that functions'
        ' defined in certain environments, like the interactive Python shell,'
        ' do not expose their source code. If that is the case, you should'
        ' define them in a .py source file. If you are certain the code is'
        ' graph-compatible, wrap the call using'
        f' @tf.autograph.experimental.do_not_convert. Original error: {e}')

  source = dedent_block(original_source)

  future_statements = tuple(
      'from __future__ import {}'.format(name) for name in future_features)
  source = '\n'.join(future_statements + (source,))

  return parse(source, preamble_len=len(future_features)), source


def _without_context(node, lines, minl, maxl):
  """Returns a clean node and source code without indenting and context."""
  for n in gast.walk(node):
    lineno = getattr(n, 'lineno', None)
    if lineno is not None:
      n.lineno = lineno - minl
    end_lineno = getattr(n, 'end_lineno', None)
    if end_lineno is not None:
      n.end_lineno = end_lineno - minl

  code_lines = lines[minl - 1:maxl]

  # Attempt to clean up surrounding context code.

  end_col_offset = getattr(node, 'end_col_offset', None)
  if end_col_offset is not None:
    # This is only available in 3.8.
    code_lines[-1] = code_lines[-1][:end_col_offset]

  col_offset = getattr(node, 'col_offset', None)
  if col_offset is None:
    # Older Python: try to find the "lambda" token. This is brittle.
    match = re.search(r'(?<!\w)lambda(?!\w)', code_lines[0])
    if match is not None:
      col_offset = match.start(0)

  if col_offset is not None:
    code_lines[0] = code_lines[0][col_offset:]

  code_block = '\n'.join([c.rstrip() for c in code_lines])

  return node, code_block


def _arg_name(node):
  if node is None:
    return None
  if isinstance(node, gast.Name):
    return node.id
  assert isinstance(node, str)
  return node


def _node_matches_argspec(node, func):
  """Returns True is node fits the argspec of func."""
  # TODO(mdan): Use just inspect once support for Python 2 is dropped.
  arg_spec = tf_inspect.getfullargspec(func)

  node_args = tuple(_arg_name(arg) for arg in node.args.args)
  if node_args != tuple(arg_spec.args):
    return False

  if arg_spec.varargs != _arg_name(node.args.vararg):
    return False

  if arg_spec.varkw != _arg_name(node.args.kwarg):
    return False

  node_kwonlyargs = tuple(_arg_name(arg) for arg in node.args.kwonlyargs)
  if node_kwonlyargs != tuple(arg_spec.kwonlyargs):
    return False

  return True


def _parse_lambda(lam):
  """Returns the AST and source code of given lambda function.

  Args:
    lam: types.LambdaType, Python function/method/class

  Returns:
    gast.AST, Text: the parsed AST node; the source code that was parsed to
    generate the AST (including any prefixes that this function may have added).
  """
  # TODO(mdan): Use a fast path if the definition is not multi-line.
  # We could detect that the lambda is in a multi-line expression by looking
  # at the surrounding code - an surrounding set of parentheses indicates a
  # potential multi-line definition.

  mod = inspect.getmodule(lam)
  f = inspect.getsourcefile(lam)
  def_line = lam.__code__.co_firstlineno

  # This method is more robust that just calling inspect.getsource(mod), as it
  # works in interactive shells, where getsource would fail. This is the
  # same procedure followed by inspect for non-modules:
  # https://github.com/python/cpython/blob/3.8/Lib/inspect.py#L772
  lines = linecache.getlines(f, mod.__dict__)
  source = ''.join(lines)

  # Narrow down to the last node starting before our definition node.
  all_nodes = parse(source, preamble_len=0, single_node=False)
  search_nodes = []
  for node in all_nodes:
    # Also include nodes without a line number, for safety. This is defensive -
    # we don't know whether such nodes might exist, and if they do, whether
    # they are not safe to skip.
    # TODO(mdan): Replace this check with an assertion or skip such nodes.
    if getattr(node, 'lineno', def_line) <= def_line:
      search_nodes.append(node)
    else:
      # Found a node starting past our lambda - can stop the search.
      break

  # Extract all lambda nodes from the shortlist.
  lambda_nodes = []
  for node in search_nodes:
    lambda_nodes.extend(
        n for n in gast.walk(node) if isinstance(n, gast.Lambda))

  # Filter down to lambda nodes which span our actual lambda.
  candidates = []
  for ln in lambda_nodes:
    minl, maxl = MAX_SIZE, 0
    for n in gast.walk(ln):
      minl = min(minl, getattr(n, 'lineno', minl))
      lineno = getattr(n, 'lineno', maxl)
      end_lineno = getattr(n, 'end_lineno', None)
      if end_lineno is not None:
        # end_lineno is more precise, but lineno should almost always work too.
        lineno = end_lineno
      maxl = max(maxl, lineno)
    if minl <= def_line <= maxl:
      candidates.append((ln, minl, maxl))

  # Happy path: exactly one node found.
  if len(candidates) == 1:
    (node, minl, maxl), = candidates  # pylint:disable=unbalanced-tuple-unpacking
    return _without_context(node, lines, minl, maxl)

  elif not candidates:
    lambda_codes = '\n'.join([unparse(l) for l in lambda_nodes])
    raise errors.UnsupportedLanguageElementError(
        f'could not parse the source code of {lam}:'
        f' no matching AST found among candidates:\n{lambda_codes}')

  # Attempt to narrow down selection by signature is multiple nodes are found.
  matches = [v for v in candidates if _node_matches_argspec(v[0], lam)]
  if len(matches) == 1:
    (node, minl, maxl), = matches
    return _without_context(node, lines, minl, maxl)

  # Give up if could not narrow down to a single node.
  matches = '\n'.join(
      'Match {}:\n{}\n'.format(i, unparse(node, include_encoding_marker=False))
      for i, (node, _, _) in enumerate(matches))
  raise errors.UnsupportedLanguageElementError(
      f'could not parse the source code of {lam}: found multiple definitions'
      ' with identical signatures at the location. This error'
      ' may be avoided by defining each lambda on a single line and with'
      f' unique argument names. The matching definitions were:\n{matches}')


# TODO(mdan): This should take futures as input instead.
def parse(src, preamble_len=0, single_node=True):
  """Returns the AST of given piece of code.

  Args:
    src: Text
    preamble_len: Int, indicates leading nodes in the parsed AST which should be
      dropped.
    single_node: Bool, whether `src` is assumed to be represented by exactly one
      AST node.

  Returns:
    ast.AST
  """
  module_node = gast.parse(src)
  nodes = module_node.body
  if preamble_len:
    nodes = nodes[preamble_len:]
  if single_node:
    if len(nodes) != 1:
      raise ValueError('expected exactly one node, got {}'.format(nodes))
    return nodes[0]
  return nodes


def parse_expression(src):
  """Returns the AST of given identifier.

  Args:
    src: A piece of code that represents a single Python expression
  Returns:
    A gast.AST object.
  Raises:
    ValueError: if src does not consist of a single Expression.
  """
  src = STANDARD_PREAMBLE + src.strip()
  node = parse(src, preamble_len=STANDARD_PREAMBLE_LEN, single_node=True)
  if __debug__:
    if not isinstance(node, gast.Expr):
      raise ValueError(
          'expected exactly one node of type Expr, got {}'.format(node))
  return node.value


def unparse(node, indentation=None, include_encoding_marker=True):
  """Returns the source code of given AST.

  Args:
    node: The code to compile, as an AST object.
    indentation: Unused, deprecated. The returning code will always be indented
      at 4 spaces.
    include_encoding_marker: Bool, whether to include a comment on the first
      line to explicitly specify UTF-8 encoding.

  Returns:
    code: The source code generated from the AST object
    source_mapping: A mapping between the user and AutoGraph generated code.
  """
  del indentation  # astunparse doesn't allow configuring it.
  if not isinstance(node, (list, tuple)):
    node = (node,)

  codes = []
  if include_encoding_marker:
    codes.append('# coding=utf-8')
  for n in node:
    if isinstance(n, gast.AST):
      ast_n = gast.gast_to_ast(n)
    else:
      ast_n = n

    if astunparse is ast:
      ast.fix_missing_locations(ast_n)  # Only ast needs to call this.
    codes.append(astunparse.unparse(ast_n).strip())

  return '\n'.join(codes)
