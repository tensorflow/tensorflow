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
"""Handles decorators.

Note: this module only deals with functions whose decorators are still recorded
in the AST. This does not always happen. See the unit test for an example.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.util import tf_inspect


class DecoratorsTransformer(converter.Base):
  """Converts or removes decorators."""

  def visit_FunctionDef(self, node):
    self.generic_visit(node)
    kept_decorators = []
    for dec in node.decorator_list:
      if isinstance(dec, gast.Call):
        dec_func = dec.func
      else:
        dec_func = dec

      # Special cases.
      # TODO(mdan): Is there any way we can treat these more generically?
      # We may want to forego using decorators altogether if we can't
      # properly support them.
      if isinstance(dec_func, gast.Name) and dec_func.id in ('classmethod',):
        # Assumption: decorators are only visible in the AST when converting
        # a function inline (via another decorator).
        # In that case, the converted function is no longer part of the
        # original object that it was declared into.
        # This is currently verified by tests.
        continue

      original_dec = anno.getanno(dec_func, anno.Basic.QN)
      dec_value = anno.getanno(dec_func, 'live_val')

      if dec_value in self.ctx.program.autograph_decorators:
        # AutoGraph decorators do not need to be preserved.
        continue

      # When using foo.bar.baz, we only really need to grab foo and import
      # that.
      dec_support_node = dec_func
      while isinstance(dec_support_node, gast.Attribute):
        dec_support_node = dec_support_node.value

      if not anno.hasanno(dec_support_node, 'live_val'):
        raise ValueError(
            'could not resolve symbol "%s" when looking up decorator "%s"' %
            (anno.getanno(dec_support_node, anno.Basic.QN), original_dec))

      dec_support = anno.getanno(dec_support_node, 'live_val')
      # The tuple contains:
      #  * the AST that represents the decorator
      #  * the entity supporting the decorator (i.e., what we need to import)
      #  * the name of the module that needs to be imported for this decorator
      #    to properly resolve.
      # Examples:
      #  for foo.bar, the tuple is (<ast>, <module foo>, 'foo')
      #  for baz, the tuple is (<ast>, <module baz.__module__>, 'baz')
      kept_decorators.append((dec, dec_support,
                              anno.getanno(dec_support_node, anno.Basic.QN)))

    for _, dec_support, name in kept_decorators:
      if tf_inspect.ismodule(dec_support):
        self.ctx.program.additional_imports.add(
            'import %s as %s' % (dec_support.__name__, name))
      else:
        if dec_support.__module__ == '__main__':
          raise ValueError(
              'decorator "%s" was not allowed because it is declared '
              'in the module "%s". To fix this, declare it in a separate '
              'module that we can import it from.' % (dec_support,
                                                      dec_support.__module__))
        self.ctx.program.additional_imports.add(
            'from %s import %s' % (dec_support.__module__, name))

    node.decorator_list = [dec for dec, _, _ in kept_decorators]
    return node


def transform(node, ctx):
  return DecoratorsTransformer(ctx).visit(node)
