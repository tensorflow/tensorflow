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
"""Access information (reads, writes) resolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import transformer

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
    self.modified = copy.copy(other.modified)
    self.created = copy.copy(other.created)
    self.used = copy.copy(other.used)
    self.params = copy.copy(other.params)
    self.returned = copy.copy(other.returned)

  def merge_from(self, other):
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


class AccessResolver(transformer.Base):
  """Annotates nodes with local scope information. See Scope."""

  def __init__(self, context):
    super(AccessResolver, self).__init__(context)
    self.scope = Scope(None)
    self._in_return_statement = False

  def visit_Name(self, node):
    # TODO(mdan): This is insufficient for object fields, e.g. hp.learning_rate.
    self.generic_visit(node)
    if isinstance(node.ctx, gast.Store):
      self.scope.mark_write(node.id)
    elif isinstance(node.ctx, gast.Load):
      anno.setanno(node, 'is_local', self.scope.has(node.id))
      self.scope.mark_read(node.id)
    elif isinstance(node.ctx, gast.Param):
      # Param contexts appear in function defs, so they have the meaning of
      # defining a variable.
      # TODO(mdan): This bay be incorrect with nested functions.
      # For nested functions, we'll have to add the notion of hiding args from
      # the parent scope, not writing to them.
      self.scope.mark_creation(node.id)
      self.scope.mark_param(node.id)
    else:
      raise ValueError('Unknown context %s for node %s.' % (type(node.ctx),
                                                            node.id))
    anno.setanno(node, 'is_modified_since_entry',
                 self.scope.is_modified_since_entry(node.id))
    anno.setanno(node, 'is_param', self.scope.is_param(node.id))

    if self._in_return_statement:
      self.scope.mark_returned(node.id)
    return node

  def visit_Print(self, node):
    current_scope = self.scope
    args_scope = Scope(current_scope)
    self.scope = args_scope
    for n in node.values:
      self.visit(n)
    anno.setanno(node, 'args_scope', args_scope)
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
    anno.setanno(node, 'args_scope', args_scope)
    self.scope = current_scope
    self.visit(node.func)
    return node

  def _process_block_node(self, node, block, scope_name):
    current_scope = self.scope
    block_scope = Scope(current_scope, isolated=False)
    self.scope = block_scope
    for n in block:
      self.visit(n)
    anno.setanno(node, '%s_scope' % scope_name, block_scope)
    self.scope = current_scope
    return node

  def _process_parallel_blocks(self, parent, children):
    # Because the scopes are not isolated, processing any child block
    # modifies the parent state causing the other child blocks to be
    # processed incorrectly. So we need to checkpoint the parent scope so that
    # each child sees the same context.
    before_parent = Scope(None)
    before_parent.copy_from(self.scope)
    after_children = []
    for child, name in children:
      self.scope.copy_from(before_parent)
      parent = self._process_block_node(parent, child, name)
      after_child = Scope(None)
      after_child.copy_from(self.scope)
      after_children.append(after_child)
    for after_child in after_children:
      self.scope.merge_from(after_child)
    for child, name in children:
      # TODO(mdan): We don't need this - we have the parent link from scope.
      anno.setanno(parent, '%s_parent_scope' % name, self.scope)
    return parent

  def visit_If(self, node):
    self.visit(node.test)
    node = self._process_parallel_blocks(
        node, ((node.body, 'body'), (node.orelse, 'orelse')))
    return node

  def visit_For(self, node):
    self.visit(node.target)
    self.visit(node.iter)
    node = self._process_parallel_blocks(
        node, ((node.body, 'body'), (node.orelse, 'orelse')))
    return node

  def visit_While(self, node):
    self.visit(node.test)
    node = self._process_parallel_blocks(
        node, ((node.body, 'body'), (node.orelse, 'orelse')))
    return node

  def visit_Return(self, node):
    self._in_return_statement = True
    node = self.generic_visit(node)
    self._in_return_statement = False
    return node


def resolve(node, context):
  return AccessResolver(context).visit(node)
