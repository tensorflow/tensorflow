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
"""A node transformer that includes utilities for SCT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import gast
import six

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import compiler
from tensorflow.contrib.autograph.pyct import pretty_printer


class AutographParseError(SyntaxError):
  pass


def try_ast_to_source(node):
  try:
    return compiler.ast_to_source(node)
  except AssertionError:
    return '<could not convert AST to source>'


class Base(gast.NodeTransformer):
  """Base class for specialized transformers.

  Scope-local state tracking: to keep state across nodes, at the level of
  (possibly nested) scopes, use enter/exit_local_scope and set/get_local.
  You must call enter/exit_local_scope manually, but the transformer detects
  when they are not properly paired.
  """

  def __init__(self, context):
    """Initialize the transformer. Subclasses should call this.

    Args:
      context: An EntityContext.
    """
    self._lineno = 0
    self._col_offset = 0
    self.context = context
    self._enclosing_entities = []

    # A stack that allows keeping mutable, scope-local state where scopes may be
    # nested. For example, it can be used to track the usage of break
    # statements in each loop, where loops may be nested.
    self._local_scope_state = []
    self.enter_local_scope()

  @property
  def enclosing_entities(self):
    return tuple(self._enclosing_entities)

  @property
  def locel_scope_level(self):
    return len(self._local_scope_state)

  def enter_local_scope(self):
    self._local_scope_state.append({})

  def exit_local_scope(self):
    return self._local_scope_state.pop()

  def set_local(self, name, value):
    self._local_scope_state[-1][name] = value

  def get_local(self, name, default=None):
    return self._local_scope_state[-1].get(name, default)

  def debug_print(self, node):
    """Helper method useful for debugging."""
    if __debug__:
      print(pretty_printer.fmt(node))
    return node

  def visit_block(self, nodes):
    """Helper equivalent to generic_visit, but for node lists."""
    results = []
    for node in nodes:
      replacement = self.visit(node)
      if replacement:
        if isinstance(replacement, (list, tuple)):
          results.extend(replacement)
        else:
          results.append(replacement)
    return results

  # TODO(mdan): Once we have error tracing, we may be able to just go to SSA.
  def apply_to_single_assignments(self, targets, values, apply_fn):
    """Applies a fuction to each individual assignment.

    This function can process a possibly-unpacked (e.g. a, b = c, d) assignment.
    It tries to break down the unpacking if possible. In effect, it has the same
    effect as passing the assigned values in SSA form to apply_fn.

    Examples:

    The following will result in apply_fn(a, c), apply_fn(b, d):

        a, b = c, d

    The following will result in apply_fn(a, c[0]), apply_fn(b, c[1]):

        a, b = c

    The following will result in apply_fn(a, (b, c)):

        a = b, c

    It uses the visitor pattern to allow subclasses to process single
    assignments individually.

    Args:
      targets: list, tuple of or individual AST node. Should be used with the
          targets field of an ast.Assign node.
      values: an AST node.
      apply_fn: a function of a single argument, which will be called with the
          respective nodes of each single assignment. The signaure is
          apply_fn(target, value), no return value.
    """
    if not isinstance(targets, (list, tuple)):
      targets = (targets,)
    for target in targets:
      if isinstance(target, (gast.Tuple, gast.List)):
        for i in range(len(target.elts)):
          target_el = target.elts[i]
          if isinstance(values, (gast.Tuple, gast.List)):
            value_el = values.elts[i]
          else:
            value_el = gast.Subscript(values, gast.Index(i), ctx=gast.Store())
          self.apply_to_single_assignments(target_el, value_el, apply_fn)
      else:
        # TODO(mdan): Look into allowing to rewrite the AST here.
        apply_fn(target, values)

  def visit(self, node):
    source_code = self.context.source_code
    source_file = self.context.source_file
    did_enter_function = False
    local_scope_state_size = len(self._local_scope_state)

    try:
      if isinstance(node, (gast.FunctionDef, gast.ClassDef, gast.Lambda)):
        self._enclosing_entities.append(node)
        did_enter_function = True

      if source_code and hasattr(node, 'lineno'):
        self._lineno = node.lineno
        self._col_offset = node.col_offset
      if anno.hasanno(node, anno.Basic.SKIP_PROCESSING):
        return node
      return super(Base, self).visit(node)

    except (ValueError, AttributeError, KeyError, NotImplementedError,
            AssertionError) as e:
      msg = '%s: %s\nOffending source:\n%s\n\nOccurred at node:\n%s' % (
          e.__class__.__name__, str(e), try_ast_to_source(node),
          pretty_printer.fmt(node, color=False))
      if source_code:
        line = source_code.splitlines()[self._lineno - 1]
      else:
        line = '<no source available>'
      six.reraise(AutographParseError,
                  AutographParseError(
                      msg,
                      (source_file, self._lineno, self._col_offset + 1, line)),
                  sys.exc_info()[2])
    finally:
      if did_enter_function:
        self._enclosing_entities.pop()

      if local_scope_state_size != len(self._local_scope_state):
        raise AssertionError(
            'Inconsistent local scope stack. Before entering node %s, the'
            ' stack had length %d, after exit it has length %d. This'
            ' indicates enter_local_scope and exit_local_scope are not'
            ' well paired.')
