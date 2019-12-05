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

import re
import textwrap
import tokenize

import astor
import gast
import six

from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils


STANDARD_PREAMBLE = textwrap.dedent("""
    from __future__ import division
    from __future__ import print_function
""")
STANDARD_PREAMBLE_LEN = 2


_LEADING_WHITESPACE = re.compile(r'\s*')


def dedent_block(code_string):
  """Dedents a code so that its first line starts at row zero."""

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
            'code mixing tabs and spaces for intentation is not allowed')
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


def _attempt_to_parse_normal_source(source, future_features):
  return parse(source, preamble_len=len(future_features)), source


def _attempt_to_parse_lambda_source(source, original_source,
                                    future_features, try_fallback=True):
  """Parsing function specialized on dealing with lambdas.

  Lambda functions, only hold the raw code lines which defined
  them, which may include surrounding tokens and may be syntactically
  invalid out of context. For example:

      l = (
          lambda x: x,)[0]

  will have the dedented source "lambda x: x,)[0]"
  This function makes an attempt to stip away the garbage by looking at the
  information in the syntax error.

  Args:
    source: the processed source code of `entity`.
    original_source: the source code of `entity`, as it was reported
        by `inspect.getsource`.
    future_features: see `parse`.
    try_fallback: whether to attempt to remove extra code from `source` before
        one more attempt to parse it.
  Returns:
    Same as `parse`.
  """

  try:
    return parse(source, preamble_len=len(future_features)), source

  # Note: the ValueError may be raised by parse.
  except (SyntaxError, ValueError) as e:
    def fail():
      raise errors.UnsupportedLanguageElementError(
          'could not parse the source code:'
          '\n\n{}\n'
          'This error may be avoided by creating the lambda in a standalone'
          ' statement.\n'.format(original_source))

    if not try_fallback:
      fail()

    lines = source.split('\n')
    lineno, offset = e.lineno, e.offset  # 1-based

    # Give up if there's nothing we can chip away.
    if len(lines) == lineno and len(lines[-1]) == offset:
      fail()

    # Drop all lines following the error location
    # TODO(mdan): What's with the pylint errors?
    lines = lines[:lineno]  # pylint:disable=invalid-slice-index
    # Drop all characters following the error location
    lines[-1] = lines[-1][:offset - 1]  # pylint:disable=invalid-slice-index
    source = '\n'.join(lines)

    return _attempt_to_parse_lambda_source(
        source, original_source, future_features, try_fallback=False)


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
  try:
    original_source = inspect_utils.getimmediatesource(entity)
  except (IOError, OSError) as e:
    raise ValueError(
        'Unable to locate the source code of {}. Note that functions defined'
        ' in certain environments, like the interactive Python shell do not'
        ' expose their source code. If that is the case, you should to define'
        ' them in a .py source file. If you are certain the code is'
        ' graph-compatible, wrap the call using'
        ' @tf.autograph.do_not_convert. Original error: {}'.format(entity, e))

  source = dedent_block(original_source)

  future_statements = tuple(
      'from __future__ import {}'.format(name) for name in future_features)
  source = '\n'.join(future_statements + (source,))

  if inspect_utils.islambda(entity):
    return _attempt_to_parse_lambda_source(
        source, original_source, future_features)
  else:
    return _attempt_to_parse_normal_source(source, future_features)


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
      raise ValueError('expected exactly one node node, found {}'.format(nodes))
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
          'expected a single expression, found instead {}'.format(node))
  return node.value


def unparse(node, indentation='  ', include_encoding_marker=True):
  """Returns the source code of given AST.

  Args:
    node: The code to compile, as an AST object.
    indentation: The string to use for indentation.
    include_encoding_marker: Bool, thether to include a comment on the first
      line to explicitly specify UTF-8 encoding.

  Returns:
    code: The source code generated from the AST object
    source_mapping: A mapping between the user and AutoGraph generated code.
  """
  if not isinstance(node, (list, tuple)):
    node = (node,)
  generator = astor.code_gen.SourceGenerator(indentation, False,
                                             astor.string_repr.pretty_string)

  for n in node:
    if isinstance(n, gast.AST):
      n = gast.gast_to_ast(n)
    generator.visit(n)
    generator.result.append('\n')

  # In some versions of Python, literals may appear as actual values. This
  # ensures everything is string.
  code = ''.join(map(str, generator.result))

  # Strip leading blank lines.
  code_lines = code.split('\n')
  trimmed_code_lines = []
  for l in code_lines:
    if l.rstrip() or trimmed_code_lines:
      trimmed_code_lines.append(l)
  code = '\n'.join(trimmed_code_lines)

  # Work around the reference cycle generated by astor.
  # See https://github.com/berkerpeksag/astor/blob/55dd323f7d8d696610c703c0296763c567685c31/astor/code_gen.py#L162  # pylint:disable=line-too-long
  # Reference cycles are quite disliked by TensorFlow's tests.
  if hasattr(generator, 'write'):
    generator.write = None
  del generator

  if include_encoding_marker:
    code = '# coding=utf-8\n' + code

  return code
