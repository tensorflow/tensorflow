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
"""Activity analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.contrib.autograph.pyct.qual_names import QN
from tensorflow.contrib.autograph.pyct.static_analysis.annos import NodeAnno

# TODO(mdan): Add support for PY3 (e.g. Param vs arg).


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

  def __init__(self, parent, isolated=True):
    """Create a new scope.

    Args:
      parent: A Scope or None.
      isolated: Whether the scope is isolated, that is, whether variables
          created in this scope should be visible to the parent scope.
    """
    self.isolated = isolated
    self.parent = parent
    self.modified = set()
    self.created = set()
    self.used = set()
    self.params = set()
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
    self.params |= other.params
    self.returned |= other.returned

  def has(self, name):
    if name in self.modified or name in self.params:
      return True
    elif self.parent is not None:
      return self.parent.has(name)
    return False

  def is_modified_since_entry(self, name):
    if name in self.modified:
      return True
    elif self.parent is not None and not self.isolated:
      return self.parent.is_modified_since_entry(name)
    return False

  def is_param(self, name):
    if name in self.params:
      return True
    elif self.parent is not None and not self.isolated:
      return self.parent.is_param(name)
    return False

  def mark_read(self, name):
    self.used.add(name)
    if self.parent is not None and name not in self.created:
      self.parent.mark_read(name)

  def mark_param(self, name):
    self.params.add(name)

  def mark_creation(self, name):
    if name.is_composite():
      parent = name.parent
      if self.has(parent):
        # This is considered mutation of the parent, not creation.
        # TODO(mdan): Is that really so?
        return
      else:
        raise ValueError('Unknown symbol "%s".' % parent)
    self.created.add(name)

  def mark_write(self, name):
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
  """Annotates nodes with local scope information. See Scope."""

  def __init__(self, context, parent_scope):
    super(ActivityAnalyzer, self).__init__(context)
    self.scope = Scope(parent_scope)
    self._in_return_statement = False

  def _track_symbol(self, node):
    # This can happen when we have an attribute (or subscript) on a function
    # call.  Example: a().b
    if not anno.hasanno(node, anno.Basic.QN):
      return
    qn = anno.getanno(node, anno.Basic.QN)

    if isinstance(node.ctx, gast.Store):
      self.scope.mark_write(qn)
    elif isinstance(node.ctx, gast.Load):
      self.scope.mark_read(qn)
    elif isinstance(node.ctx, gast.Param):
      # Param contexts appear in function defs, so they have the meaning of
      # defining a variable.
      # TODO(mdan): This bay be incorrect with nested functions.
      # For nested functions, we'll have to add the notion of hiding args from
      # the parent scope, not writing to them.
      self.scope.mark_creation(qn)
      self.scope.mark_param(qn)
    else:
      raise ValueError('Unknown context %s for node %s.' % (type(node.ctx), qn))

    anno.setanno(node, NodeAnno.IS_LOCAL, self.scope.has(qn))
    anno.setanno(node, NodeAnno.IS_MODIFIED_SINCE_ENTRY,
                 self.scope.is_modified_since_entry(qn))
    anno.setanno(node, NodeAnno.IS_PARAM, self.scope.is_param(qn))

    if self._in_return_statement:
      self.scope.mark_returned(qn)

  def visit_Name(self, node):
    self.generic_visit(node)
    self._track_symbol(node)
    return node

  def visit_Attribute(self, node):
    self.generic_visit(node)
    self._track_symbol(node)
    return node

  def visit_Print(self, node):
    current_scope = self.scope
    args_scope = Scope(current_scope)
    self.scope = args_scope
    for n in node.values:
      self.visit(n)
    anno.setanno(node, NodeAnno.ARGS_SCOPE, args_scope)
    self.scope = current_scope
    return node

  def visit_Call(self, node):
    current_scope = self.scope
    args_scope = Scope(current_scope, isolated=False)
    self.scope = args_scope
    for n in node.args:
      self.visit(n)
    # TODO(mdan): Account starargs, kwargs
    for n in node.keywords:
      self.visit(n)
    anno.setanno(node, NodeAnno.ARGS_SCOPE, args_scope)
    self.scope = current_scope
    self.visit(node.func)
    return node

  def _process_block_node(self, node, block, scope_name):
    current_scope = self.scope
    block_scope = Scope(current_scope, isolated=False)
    self.scope = block_scope
    for n in block:
      self.visit(n)
    anno.setanno(node, scope_name, block_scope)
    self.scope = current_scope
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

  def visit_FunctionDef(self, node):
    if self.scope:
      qn = QN(node.name)
      self.scope.mark_write(qn)
    current_scope = self.scope
    body_scope = Scope(current_scope, isolated=True)
    self.scope = body_scope
    self.generic_visit(node)
    anno.setanno(node, NodeAnno.BODY_SCOPE, body_scope)
    self.scope = current_scope
    return node

  def visit_With(self, node):
    current_scope = self.scope
    with_scope = Scope(current_scope, isolated=False)
    self.scope = with_scope
    self.generic_visit(node)
    anno.setanno(node, NodeAnno.BODY_SCOPE, with_scope)
    self.scope = current_scope
    return node

  def visit_If(self, node):
    current_scope = self.scope
    cond_scope = Scope(current_scope, isolated=False)
    self.scope = cond_scope
    self.visit(node.test)
    anno.setanno(node, NodeAnno.COND_SCOPE, cond_scope)
    self.scope = current_scope

    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_For(self, node):
    self.visit(node.target)
    self.visit(node.iter)
    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_While(self, node):
    current_scope = self.scope
    cond_scope = Scope(current_scope, isolated=False)
    self.scope = cond_scope
    self.visit(node.test)
    anno.setanno(node, NodeAnno.COND_SCOPE, cond_scope)
    self.scope = current_scope

    node = self._process_parallel_blocks(node,
                                         ((node.body, NodeAnno.BODY_SCOPE),
                                          (node.orelse, NodeAnno.ORELSE_SCOPE)))
    return node

  def visit_Return(self, node):
    self._in_return_statement = True
    node = self.generic_visit(node)
    self._in_return_statement = False
    return node


def resolve(node, context, parent_scope=None):
  return ActivityAnalyzer(context, parent_scope).visit(node)
