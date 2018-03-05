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

Requires activity analysis annotations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import transformer
from tensorflow.contrib.py2tf.pyct.static_analysis.annos import NodeAnno


class LiveValueResolver(transformer.Base):
  """Annotates nodes with live values."""

  def __init__(self, context, literals):
    super(LiveValueResolver, self).__init__(context)
    self.literals = literals

  def visit_ClassDef(self, node):
    self.generic_visit(node)
    anno.setanno(node, 'live_val', self.context.namespace[node.name])
    return node

  def visit_Name(self, node):
    self.generic_visit(node)
    if isinstance(node.ctx, gast.Load):
      assert anno.hasanno(node, NodeAnno.IS_LOCAL), node
      symbol_is_local = anno.getanno(node, NodeAnno.IS_LOCAL)
      assert anno.hasanno(node, NodeAnno.IS_MODIFIED_SINCE_ENTRY), node
      symbol_is_modified = anno.getanno(node, NodeAnno.IS_MODIFIED_SINCE_ENTRY)
      assert anno.hasanno(node, NodeAnno.IS_PARAM), node
      symbol_is_param = anno.getanno(node, NodeAnno.IS_PARAM)

      if not symbol_is_local and not symbol_is_param:
        if node.id in self.literals:
          anno.setanno(node, 'live_val', self.literals[node.id])
          # TODO(mdan): Could live values have FQNs? i.e. 'a'.join()
        elif node.id in self.context.namespace:
          obj = self.context.namespace[node.id]
          anno.setanno(node, 'live_val', obj)
          anno.setanno(node, 'fqn', (obj.__name__,))
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

      if not symbol_is_modified:
        if node.id in self.context.arg_values:
          obj = self.context.arg_values[node.id]
          anno.setanno(node, 'live_val', obj)
          anno.setanno(node, 'fqn', (obj.__class__.__name__,))
    return node

  def visit_Attribute(self, node):
    self.generic_visit(node)
    if anno.hasanno(node.value, 'live_val'):
      assert anno.hasanno(node.value, 'fqn')
      parent_object = anno.getanno(node.value, 'live_val')
      if not hasattr(parent_object, node.attr):
        raise AttributeError('%s has no attribute %s' % (parent_object,
                                                         node.attr))
      anno.setanno(node, 'parent_type', type(parent_object))
      anno.setanno(node, 'live_val', getattr(parent_object, node.attr))
      anno.setanno(node, 'fqn', anno.getanno(node.value, 'fqn') + (node.attr,))
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
      stem_name = node.value
      # All nonlocal symbols should be fully resolved.
      assert anno.hasanno(stem_name, NodeAnno.IS_LOCAL), stem_name
      # TODO(mdan): Figure out what to do when calling attribute on local object
      # Maybe just leave as-is?
    return node


def resolve(node, context, literals):
  return LiveValueResolver(context, literals).visit(node)
