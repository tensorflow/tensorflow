# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Converter for list operations.

This includes converting Python lists to TensorArray/TensorList.
"""

# TODO(mdan): Elaborate the logic here.
# TODO(mdan): Does it even make sense to attempt to try to use TAs?
# The current rule (always convert to TensorArray) is naive and insufficient.
# In general, a better mechanism could look like:
#   * convert to TensorList by default
#   * leave as Python list if the user explicitly forbids it
#   * convert to TensorArray only when complete write once behavior can be
#     guaranteed (e.g. list comprehensions)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.autograph.core import converter
from tensorflow.contrib.autograph.lang import directives
from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import templates
from tensorflow.contrib.autograph.pyct.static_analysis.annos import NodeAnno


# Tags for local state.
POP_USES = 'pop_uses'


class ListTransformer(converter.Base):
  """Converts lists and related operations to their TF counterpart."""

  def visit_List(self, node):
    node = self.generic_visit(node)
    template = """
      ag__.new_list(elements)
    """
    return templates.replace_as_expression(template, elements=node)

  def _replace_append_call(self, node):
    assert len(node.args) == 1
    assert isinstance(node.func, gast.Attribute)
    template = """
      target = ag__.list_append(target, element)
    """
    return templates.replace(
        template,
        target=node.func.value,
        element=node.args[0])

  def _replace_pop_call(self, node):
    # Expressions that use pop() are converted to a statement + expression.
    #
    # For example:
    #
    #   print(target.pop())
    #
    # ... is converted to:
    #
    #   target, target_pop = ag__.list_pop(target)
    #   print(target_pop)
    #
    # Here, we just generate the variable name and swap it in,
    # and _generate_pop_operation will handle the rest.
    #
    # Multiple uses of pop() are allowed:
    #
    #   print(tartget.pop(), target.pop())
    #   print(tartget.pop().pop())
    #
    assert isinstance(node.func, gast.Attribute)
    scope = anno.getanno(node, NodeAnno.ARGS_SCOPE)
    target_node = node.func.value

    # Attempt to use a related name if one exists. Otherwise use something
    # generic.
    if anno.hasanno(target_node, anno.Basic.QN):
      target_name = anno.getanno(target_node, anno.Basic.QN).ssf()
    else:
      target_name = 'list_'
    pop_var_name = self.ctx.namer.new_symbol(target_name, scope.referenced)

    pop_uses = self.get_local(POP_USES, [])
    pop_uses.append((node, pop_var_name))
    self.set_local(POP_USES, pop_uses)

    return templates.replace_as_expression('var_name', var_name=pop_var_name)

  def _replace_stack_call(self, node):
    assert len(node.args) == 1
    dtype = self.get_definition_directive(
        node.args[0],
        directives.set_element_type,
        'dtype',
        default=templates.replace_as_expression('None'))
    template = """
      ag__.list_stack(
          target,
          opts=ag__.ListStackOpts(
              element_dtype=dtype,
              original_call=orig_call))
    """
    return templates.replace_as_expression(
        template,
        dtype=dtype,
        target=node.args[0],
        orig_call=node.func)

  def visit_Call(self, node):
    node = self.generic_visit(node)

    # TODO(mdan): This is insufficient if target is a function argument.
    # In the case of function arguments, we need to add the list to the
    # function's return value, because it is being modified.
    # TODO(mdan): Checking just the name is brittle, can it be improved?
    if isinstance(node.func, gast.Attribute):
      func_name = node.func.attr
      if func_name == 'append' and (len(node.args) == 1):
        node = self._replace_append_call(node)
      elif func_name == 'pop' and (len(node.args) <= 1):
        node = self._replace_pop_call(node)
      elif (func_name == 'stack' and (len(node.args) == 1) and
            (not node.keywords or node.keywords[0].arg == 'strict')):
        # This avoids false positives with keyword args.
        # TODO(mdan): handle kwargs properly.
        node = self._replace_stack_call(node)

    return node

  def _generate_pop_operation(self, original_call_node, pop_var_name):
    assert isinstance(original_call_node.func, gast.Attribute)

    if original_call_node.args:
      pop_element = original_call_node.args[0]
    else:
      pop_element = parser.parse_expression('None')

    # The call will be something like "target.pop()", and the dtype is hooked to
    # target, hence the func.value.
    # TODO(mdan): For lists of lists, this won't work.
    # The reason why it won't work is because it's unclear how to annotate
    # the list as a "list of lists with a certain element type" when using
    # operations like `l.pop().pop()`.
    dtype = self.get_definition_directive(
        original_call_node.func.value,
        directives.set_element_type,
        'dtype',
        default=templates.replace_as_expression('None'))
    shape = self.get_definition_directive(
        original_call_node.func.value,
        directives.set_element_type,
        'shape',
        default=templates.replace_as_expression('None'))

    template = """
      target, pop_var_name = ag__.list_pop(
          target, element,
          opts=ag__.ListPopOpts(element_dtype=dtype, element_shape=shape))
    """
    return templates.replace(
        template,
        target=original_call_node.func.value,
        pop_var_name=pop_var_name,
        element=pop_element,
        dtype=dtype,
        shape=shape)

  def _postprocess_statement(self, node):
    """Inserts any separate pop() calls that node may use."""
    pop_uses = self.get_local(POP_USES, None)
    if pop_uses:
      replacements = []
      for original_call_node, pop_var_name in pop_uses:
        replacements.extend(
            self._generate_pop_operation(original_call_node, pop_var_name))
      replacements.append(node)
      node = replacements
    self.exit_local_scope()
    return node, None

  # TODO(mdan): Should we have a generic visit_block instead?
  # Right now it feels that a visit_block would add too much magic that's
  # hard to follow.

  def _visit_and_process_block(self, block):
    return self.visit_block(
        block,
        before_visit=self.enter_local_scope,
        after_visit=self._postprocess_statement)

  def visit_FunctionDef(self, node):
    node.args = self.generic_visit(node.args)
    node.decorator_list = self.visit_block(node.decorator_list)
    node.body = self._visit_and_process_block(node.body)
    return node

  def visit_For(self, node):
    node.target = self.visit(node.target)
    node.body = self._visit_and_process_block(node.body)
    node.orelse = self._visit_and_process_block(node.orelse)
    return node

  def visit_While(self, node):
    node.test = self.visit(node.test)
    node.body = self._visit_and_process_block(node.body)
    node.orelse = self._visit_and_process_block(node.orelse)
    return node

  def visit_If(self, node):
    node.test = self.visit(node.test)
    node.body = self._visit_and_process_block(node.body)
    node.orelse = self._visit_and_process_block(node.orelse)
    return node

  def visit_With(self, node):
    node.items = self.visit_block(node.items)
    node.body = self._visit_and_process_block(node.body)
    return node


def transform(node, ctx):
  return ListTransformer(ctx).visit(node)
