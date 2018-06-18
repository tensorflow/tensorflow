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


# TODO(mdan): Use namedtuple.
class EntityInfo(object):
  """Contains information about a Python entity. Immutable.

  Examples of entities include functions and classes.

  Attributes:
    source_code: The entity's source code.
    source_file: The entity's source file.
    namespace: Dict[str, ], containing symbols visible to the entity
        (excluding parameters).
    arg_values: dict[str->*], containing parameter values, if known.
    arg_types: dict[str->*], containing parameter types, if known.
    owner_type: The surrounding class type of the function, if present.
  """

  # TODO(mdan): Remove the default and update tests.
  def __init__(self, source_code, source_file, namespace, arg_values, arg_types,
               owner_type):
    self.source_code = source_code
    self.source_file = source_file
    self.namespace = namespace
    self.arg_values = {} if arg_values is None else arg_values
    self.arg_types = {} if arg_types is None else arg_types
    self.owner_type = owner_type


class Base(gast.NodeTransformer):
  """Base class for general-purpose code transformers transformers.

  This is an extension of ast.NodeTransformer that provides a few additional
  functions, like state tracking within the scope of arbitrary node, helpers
  for processing code blocks, debugging, mapping of transformed code to
  original code, and others.

  Scope-local state tracking: to keep state across nodes, at the level of
  (possibly nested) scopes, use enter/exit_local_scope and set/get_local.
  You must call enter/exit_local_scope manually, but the transformer detects
  when they are not properly paired.
  """

  # TODO(mdan): Document all extra features.

  def __init__(self, entity_info):
    """Initialize the transformer. Subclasses should call this.

    Args:
      entity_info: An EntityInfo object.
    """
    self._lineno = 0
    self._col_offset = 0
    self.entity_info = entity_info
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
  def local_scope_level(self):
    return len(self._local_scope_state)

  def enter_local_scope(self, inherit=None):
    """Marks entry into a new local scope.

    Args:
      inherit: Optional enumerable of variable names to copy from the
          parent scope.
    """
    scope_entered = {}
    if inherit:
      this_scope = self._local_scope_state[-1]
      for name in inherit:
        if name in this_scope:
          scope_entered[name] = this_scope[name]
    self._local_scope_state.append(scope_entered)

  def exit_local_scope(self, keep=None):
    """Marks exit from the current local scope.

    Args:
      keep: Optional enumerable of variable names to copy into the
          parent scope.
    Returns:
      A dict containing the scope that has just been exited.
    """
    scope_left = self._local_scope_state.pop()
    if keep:
      this_scope = self._local_scope_state[-1]
      for name in keep:
        if name in scope_left:
          this_scope[name] = scope_left[name]
    return scope_left

  def set_local(self, name, value):
    self._local_scope_state[-1][name] = value

  def get_local(self, name, default=None):
    return self._local_scope_state[-1].get(name, default)

  def debug_print(self, node):
    """Helper method useful for debugging."""
    if __debug__:
      print(pretty_printer.fmt(node))
    return node

  def visit_block(self, nodes, before_visit=None, after_visit=None):
    """A more powerful version of generic_visit for statement blocks.

    An example of a block is the body of an if statement.

    This function allows specifying a postprocessing callback (the
    after_visit argument) argument which can be used to move nodes to a new
    destination. This is done by after_visit by returning a non-null
    second return value, e.g. return new_node, new_destination.

    For example, a transformer could perform the following move:

        foo()
        bar()
        baz()

        foo()
        if cond:
          bar()
          baz()

    The above could be done with a postprocessor of this kind:

        def after_visit(node):
          if node_is_function_call(bar):
            new_container_node = build_cond()
            new_container_node.body.append(node)
            return new_container_node, new_container_node.body
          else:
            # Once we set a new destination, all subsequent items will be
            # moved to it, so we don't need to explicitly handle baz.
            return node, None

    Args:
      nodes: enumerable of AST node objects
      before_visit: optional callable that is called before visiting each item
          in nodes
      after_visit: optional callable that takes in an AST node and
          returns a tuple (new_node, new_destination). It is called after
          visiting each item in nodes. Is used in the same was as the
          visit_* methods: new_node will replace the node; if not None,
          new_destination must be a list, and subsequent nodes will be placed
          in this list instead of the list returned by visit_block.
    Returns:
      A list of AST node objects containing the transformed items fron nodes,
      except those nodes that have been relocated using after_visit.
    """
    results = []
    node_destination = results
    for node in nodes:
      if before_visit:
        # TODO(mdan): We can modify node here too, if ever needed.
        before_visit()

      replacement = self.visit(node)

      if after_visit and replacement:
        replacement, new_destination = after_visit(replacement)
      else:
        new_destination = None

      if replacement:
        if isinstance(replacement, (list, tuple)):
          node_destination.extend(replacement)
        else:
          node_destination.append(replacement)

      # Allow the postprocessor to reroute the remaining nodes to a new list.
      if new_destination is not None:
        node_destination = new_destination
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

  def _get_source(self, node):
    try:
      return compiler.ast_to_source(node)
    except AssertionError:
      return '<could not convert AST to source>'

  def visit(self, node):
    source_code = self.entity_info.source_code
    source_file = self.entity_info.source_file
    did_enter_function = False
    local_scope_size_at_entry = len(self._local_scope_state)

    try:
      if isinstance(node, (gast.FunctionDef, gast.ClassDef, gast.Lambda)):
        did_enter_function = True

      if did_enter_function:
        self._enclosing_entities.append(node)

      if source_code and hasattr(node, 'lineno'):
        self._lineno = node.lineno
        self._col_offset = node.col_offset

      if not anno.hasanno(node, anno.Basic.SKIP_PROCESSING):
        result = super(Base, self).visit(node)

      # On exception, the local scope integrity is not guaranteed.
      if did_enter_function:
        self._enclosing_entities.pop()

      if local_scope_size_at_entry != len(self._local_scope_state):
        raise AssertionError(
            'Inconsistent local scope stack. Before entering node %s, the'
            ' stack had length %d, after exit it has length %d. This'
            ' indicates enter_local_scope and exit_local_scope are not'
            ' well paired.' % (
                node,
                local_scope_size_at_entry,
                len(self._local_scope_state)
            ))
      return result

    except (ValueError, AttributeError, KeyError, NotImplementedError) as e:
      msg = '%s: %s\nOffending source:\n%s\n\nOccurred at node:\n%s' % (
          e.__class__.__name__, str(e), self._get_source(node),
          pretty_printer.fmt(node, color=False))
      if source_code:
        line = source_code.splitlines()[self._lineno - 1]
      else:
        line = '<no source available>'
      # TODO(mdan): Avoid the printing of the original exception.
      # In other words, we need to find how to suppress the "During handling
      # of the above exception, another exception occurred" message.
      six.reraise(AutographParseError,
                  AutographParseError(
                      msg,
                      (source_file, self._lineno, self._col_offset + 1, line)),
                  sys.exc_info()[2])
