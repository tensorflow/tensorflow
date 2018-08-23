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
"""Activity analysis.

Requires qualified name annotations (see qual_names.py).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import qual_names
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.contrib.autograph.pyct.static_analysis.annos import NodeAnno

# TODO(mdan): Add support for PY3 (e.g. Param vs arg).
# TODO(alexbw): Ignore named literals (e.g. None)


class Scope(object):
  """Encloses local symbol definition and usage information.

  This can track for instance whether a symbol is modified in the current scope.
  Note that scopes do not necessarily align with Python's scopes. For example,
  the body of an if statement may be considered a separate scope.

  Attributes:
    modified: identifiers modified in this scope
    created: identifiers created in this scope
    used: identifiers referenced in this scope
  """

  def __init__(self, parent, isolated=True, add_unknown_symbols=False):
    """Create a new scope.

    Args:
      parent: A Scope or None.
      isolated: Whether the scope is isolated, that is, whether variables
          created in this scope should be visible to the parent scope.
      add_unknown_symbols: Whether to handle attributed and subscripts
          without having first seen the base name.
          E.g., analyzing the statement 'x.y = z' without first having seen 'x'.
    """
    self.isolated = isolated
    self.parent = parent
    self.add_unknown_symbols = add_unknown_symbols
    self.modified = set()
    # TODO(mdan): Completely remove this.
    self.created = set()
    self.used = set()
    self.params = {}
    self.returned = set()

  # TODO(mdan): Rename to `locals`
  @property
  def referenced(self):
    if not self.isolated and self.parent is not None:
      return self.used | self.parent.referenced
    return self.used

  def __repr__(self):
    return 'Scope{r=%s, c=%s, w=%s}' % (tuple(self.used), tuple(self.created),
                                        tuple(self.modified))

  def copy_from(self, other):
    """Recursively copies the contents of this scope from another scope."""
    if (self.parent is None) != (other.parent is None):
      raise ValueError('cannot copy scopes of different structures')
    if other.parent is not None:
      self.parent.copy_from(other.parent)
    self.isolated = other.isolated
    self.modified = copy.copy(other.modified)
    self.created = copy.copy(other.created)
    self.used = copy.copy(other.used)
    self.params = copy.copy(other.params)
    self.returned = copy.copy(other.returned)

  @classmethod
  def copy_of(cls, other):
    if other.parent is not None:
      parent = cls.copy_of(other.parent)
    else:
      parent = None
    new_copy = cls(parent)
    new_copy.copy_from(other)
    return new_copy

  def merge_from(self, other):
    if (self.parent is None) != (other.parent is None):
      raise ValueError('cannot merge scopes of different structures')
    if other.parent is not None:
      self.parent.merge_from(other.parent)
    self.modified |= other.modified
    self.created |= other.created
    self.used |= other.used
    self.params.update(other.params)
    self.returned |= other.returned

  def has(self, name):
    if name in self.modified:
      return True
    elif self.parent is not None:
      return self.parent.has(name)
    return False

  def mark_read(self, name):
    self.used.add(name)
    if self.parent is not None and name not in self.created:
      self.parent.mark_read(name)

  def mark_param(self, name, owner):
    self.params[name] = owner

  def mark_creation(self, name, writes_create_symbol=False):
    """Mark a qualified name as created."""
    if name.is_composite():
      parent = name.parent
      if not writes_create_symbol:
        return
      else:
        if not self.has(parent):
          if self.add_unknown_symbols:
            self.mark_read(parent)
          else:
            raise ValueError('Unknown symbol "%s".' % parent)
    self.created.add(name)

  def mark_write(self, name):
    """Marks the given symbol as modified in the current scope."""
    self.modified.add(name)
    if self.isolated:
      self.mark_creation(name)
    else:
      if self.parent is None:
        self.mark_creation(name)
      else:
        if not self.parent.has(name):
          self.mark_creation(name)
        self.parent.mark_write(name)

  def mark_returned(self, name):
    self.returned.add(name)
    if not self.isolated and self.parent is not None:
      self.parent.mark_returned(name)


class ActivityAnalyzer(transformer.Base):
  """Annotates nodes with local scope information.

  See Scope.

  The use of this class requires that qual_names.resolve() has been called on
  the node. This class will ignore nodes have not been
  annotated with their qualified names.
  """

  def __init__(self, context, parent_scope=None, add_unknown_symbols=False):
    super(ActivityAnalyzer, self).__init__(context)
    self.scope = Scope(parent_scope, None, add_unknown_symbols)
    self._in_return_statement = False
    self._in_aug_assign = False

  @property
  def _in_constructor(self):
    if len(self.enclosing_entities) > 1:
      innermost = self.enclosing_entities[-1]
      parent = self.enclosing_entities[-2]
      return isinstance(parent, gast.ClassDef) and innermost.name == '__init__'
    return False

  def _node_sets_self_attribute(self, node):
    if anno.hasanno(node, anno.Basic.QN):
      qn = anno.getanno(node, anno.Basic.QN)
      # TODO(mdan): The 'self' argument is not guaranteed to be called 'self'.
      if qn.has_attr and qn.parent.qn == ('self',):
        return True
    return False

  def _track_symbol(self,
                    node,
                    composite_writes_alter_parent=False,
                    writes_create_symbol=False):
    # A QN may be missing when we have an attribute (or subscript) on a function
    # call. Example: a().b
    if not anno.hasanno(node, anno.Basic.QN):
      return
    qn = anno.getanno(node, anno.Basic.QN)

    if isinstance(node.ctx, gast.Store):
      self.scope.mark_write(qn)
      if qn.is_composite and composite_writes_alter_parent:
        self.scope.mark_write(qn.parent)
      if writes_create_symbol:
        self.scope.mark_creation(qn, writes_create_symbol=True)
      if self._in_aug_assign:
        self.scope.mark_read(qn)
    elif isinstance(node.ctx, gast.Load):
      self.scope.mark_read(qn)
    elif isinstance(node.ctx, gast.Param):
      # Param contexts appear in function defs, so they have the meaning of
      # defining a variable.
      self.scope.mark_write(qn)
      self.scope.mark_param(qn, self.enclosing_entities[-1])
    else:
      raise ValueError('Unknown context %s for node %s.' % (type(node.ctx), qn))

    anno.setanno(node, NodeAnno.IS_LOCAL, self.scope.has(qn))

    if self._in_return_statement:
      self.scope.mark_returned(qn)

  def _enter_scope(self, isolated):
    self.scope = Scope(self.scope, isolated=isolated)

  def _exit_scope(self):
    self.scope = self.scope.parent

  def _process_statement(self, node):
    self._enter_scope(False)
    node = self.generic_visit(node)
    anno.setanno(node, anno.Static.SCOPE, self.scope)
    self._exit_scope()
    return node

  def visit_Expr(self, node):
    return self._process_statement(node)

  def visit_Return(self, node):
    self._in_return_statement = True
    node = self._process_statement(node)
    self._in_return_statement = False
    return node

  def visit_Assign(self, node):
    return self._process_statement(node)

  def visit_AugAssign(self, node):
    # Special rules for AugAssign. In Assign, the target is only written,
    # but in AugAssig (e.g. a += b), the target is both read and written.
    self._in_aug_assign = True
    node = self._process_statement(node)
    self._in_aug_assign = False
    return node

  def visit_Name(self, node):
    node = self.generic_visit(node)
    self._track_symbol(node)
    return node

  def visit_Attribute(self, node):
    node = self.generic_visit(node)
    if self._in_constructor and self._node_sets_self_attribute(node):
      self._track_symbol(
          node, composite_writes_alter_parent=True, writes_create_symbol=True)
    else:
      self._track_symbol(node)
    return node

  def visit_Subscript(self, node):
    node = self.generic_visit(node)
    # Subscript writes (e.g. a[b] = "value") are considered to modify
    # both the element itself (a[b]) and its parent (a).
    self._track_symbol(node)
    return node

  def visit_Print(self, node):
    self._enter_scope(False)
    node.values = self.visit_block(node.values)
    anno.setanno(node, anno.Static.SCOPE, self.scope)
    anno.setanno(node, NodeAnno.ARGS_SCOPE, self.scope)
    self._exit_scope()
    return node

  def visit_Assert(self, node):
    return self._process_statement(node)

  def visit_Call(self, node):
    self._enter_scope(False)
    node.args = self.visit_block(node.args)
    node.keywords = self.visit_block(node.keywords)
    # TODO(mdan): Account starargs, kwargs
    anno.setanno(node, NodeAnno.ARGS_SCOPE, self.scope)
    self._exit_scope()
    node.func = self.visit(node.func)
    return node

  def _process_block_node(self, node, block, scope_name):
    self._enter_scope(False)
    block = self.visit_block(block)
    anno.setanno(node, scope_name, self.scope)
    self._exit_scope()
    return node

  def _process_parallel_blocks(self, parent, children):
    # Because the scopes are not isolated, processing any child block
    # modifies the parent state causing the other child blocks to be
    # processed incorrectly. So we need to checkpoint the parent scope so that
    # each child sees the same context.
    before_parent = Scope.copy_of(self.scope)
    after_children = []
    for child, scope_name in children:
      self.scope.copy_from(before_parent)
      parent = self._process_block_node(parent, child, scope_name)
      after_child = Scope.copy_of(self.scope)
      after_children.append(after_child)
    for after_child in after_children:
      self.scope.merge_from(after_child)
    return parent

  def visit_arguments(self, node):
    return self._process_statement(node)

  def visit_FunctionDef(self, node):
    # The FunctionDef node itself has a Scope object that tracks the creation
    # of its name, along with the usage of any decorator accompany it.
    self._enter_scope(False)
    node.decorator_list = self.visit_block(node.decorator_list)
    self.scope.mark_write(qual_names.QN(node.name))
    anno.setanno(node, anno.Static.SCOPE, self.scope)
    self._exit_scope()

    # A separate Scope tracks the actual function definition.
    self._enter_scope(True)
    node.args = self.visit(node.args)

    # Track the body separately. This is for compatibility reasons, it may not
    # be strictly needed.
    self._enter_scope(False)
    node.body = self.visit_block(node.body)
    anno.setanno(node, NodeAnno.BODY_SCOPE, self.scope)
    self._exit_scope()

    self._exit_scope()
    return node

  def visit_With(self, node):
    self._enter_scope(False)
    node = self.generic_visit(node)
    anno.setanno(node, NodeAnno.BODY_SCOPE, self.scope)
    self._exit_scope()
    return node

  def visit_withitem(self, node):
    return self._process_statement(node)

  def visit_If(self, node):
    self._enter_scope(False)
    node.test = self.visit(node.test)
    anno.setanno(node, NodeAnno.COND_SCOPE, self.scope)
    anno.setanno(node.test, anno.Static.SCOPE, self.scope)
    self._exit_scope()
    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_For(self, node):
    self._enter_scope(False)
    node.target = self.visit(node.target)
    node.iter = self.visit(node.iter)
    anno.setanno(node.iter, anno.Static.SCOPE, self.scope)
    self._exit_scope()
    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_While(self, node):
    self._enter_scope(False)
    node.test = self.visit(node.test)
    anno.setanno(node, NodeAnno.COND_SCOPE, self.scope)
    anno.setanno(node.test, anno.Static.SCOPE, self.scope)
    self._exit_scope()
    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node


def resolve(node, context, parent_scope=None):
  return ActivityAnalyzer(context, parent_scope).visit(node)
