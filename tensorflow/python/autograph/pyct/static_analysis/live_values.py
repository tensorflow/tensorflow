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
"""Live value resolution.

Live values are extracted from the known execution context.

Requires activity and reaching definitions analyses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import transformer


class LiveValueResolver(transformer.Base):
  """Annotates nodes with live values."""

  def __init__(self, context, literals):
    super(LiveValueResolver, self).__init__(context)
    self.literals = literals

  def visit_ClassDef(self, node):
    self.generic_visit(node)
    anno.setanno(node, 'live_val', self.entity_info.namespace[node.name])
    return node

  def visit_Name(self, node):
    self.generic_visit(node)
    if isinstance(node.ctx, gast.Load):
      defs = anno.getanno(node, anno.Static.DEFINITIONS, ())

      is_defined = bool(defs)
      has_single_def = len(defs) == 1

      if not is_defined:
        if node.id in self.literals:
          anno.setanno(node, 'live_val', self.literals[node.id])
        elif node.id in self.entity_info.namespace:
          obj = self.entity_info.namespace[node.id]
          anno.setanno(node, 'live_val', obj)
          if hasattr(obj, '__name__'):
            anno.setanno(node, 'fqn', (obj.__name__,))
          elif hasattr(obj, '__class__'):
            obj_class = obj.__class__
            anno.setanno(node, 'fqn',
                         (obj_class.__module__, obj_class.__name__))
          else:
            # If the symbol value is for example a primitive, then it will not
            # have a name.
            pass
        elif node.id in inspect_utils.SPECIAL_BUILTINS:
          # Note: if the user redefined any of these symbols, then they would
          # be visible in the namespace and we would never reach this branch.
          anno.setanno(
              node, 'live_val', inspect_utils.SPECIAL_BUILTINS[node.id])
        else:
          pass
          # TODO(mdan): Should we raise an error here?
          # Can encounter this when:
          #  * a symbol truly lacks reference
          #  * a symbol is new, like the new name of a function we just renamed.
      else:
        pass
        # TODO(mdan): Attempt to trace its value through the local chain.
        # TODO(mdan): Use type annotations as fallback.

      if has_single_def:
        def_, = defs
        # Note: param_of is a weakref.
        if def_.param_of and def_.param_of() is self.enclosing_entities[0]:
          if node.id in self.entity_info.arg_values:
            obj = self.entity_info.arg_values[node.id]
            anno.setanno(node, 'live_val', obj)
            anno.setanno(node, 'fqn', (obj.__class__.__name__,))
    return node

  def visit_Attribute(self, node):
    self.generic_visit(node)
    if anno.hasanno(node.value, 'live_val'):
      assert anno.hasanno(node.value, 'fqn')
      parent_object = anno.getanno(node.value, 'live_val')

      anno.setanno(node, 'parent_type', type(parent_object))
      anno.setanno(node, 'fqn', anno.getanno(node.value, 'fqn') + (node.attr,))
      if hasattr(parent_object, node.attr):
        # This can happen when the attribute's creation and use depend on the
        # same static condition, for example:
        #
        #  if cond:
        #    foo.bar = baz
        #  if cond:
        #    x = foo.bar
        #
        anno.setanno(node, 'live_val', getattr(parent_object, node.attr))

    # TODO(mdan): Investigate the role built-in annotations can play here.
    elif anno.hasanno(node.value, 'type'):
      parent_type = anno.getanno(node.value, 'type')
      if hasattr(parent_type, node.attr):
        # This should hold for static members like methods.
        # This would not hold for dynamic members like function attributes.
        # For the dynamic case, we simply leave the node without an annotation,
        # and let downstream consumers figure out what to do.
        anno.setanno(node, 'parent_type', parent_type)
        anno.setanno(node, 'live_val', getattr(parent_type, node.attr))
        anno.setanno(node, 'fqn',
                     anno.getanno(node.value, 'type_fqn') + (node.attr,))
    elif isinstance(node.value, gast.Name):
      # TODO(mdan): Figure out what to do when calling attribute on local object
      # Maybe just leave as-is?
      pass
    return node


def resolve(node, context, literals):
  return LiveValueResolver(context, literals).visit(node)
