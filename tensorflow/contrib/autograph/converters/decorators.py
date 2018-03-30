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

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import pretty_printer


class DecoratorsTransformer(gast.NodeTransformer):
  """Converts or removes decorators."""

  def __init__(self, remove_decorators):
    self.remove_decorators = remove_decorators
    self.additional_dependencies = set()

  # pylint:disable=invalid-name

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

      if not anno.hasanno(dec_func, 'live_val'):
        raise ValueError(
            'Could not resolve decorator: %s' % pretty_printer.fmt(dec_func))

      dec_value = anno.getanno(dec_func, 'live_val')
      if dec_value not in self.remove_decorators:
        kept_decorators.append((dec, dec_value))

    for _, dec_value in kept_decorators:
      if dec_value.__module__ == '__main__':
        raise ValueError(
            'decorator "%s" was not allowed because it is declared '
            'in the module "%s". To fix this, declare it in a separate '
            'module that we can import it from.' % (dec_value,
                                                    dec_value.__module__))
      else:
        self.additional_dependencies.add(dec_value)

    node.decorator_list = [dec for dec, _ in kept_decorators]
    return node

  # pylint:enable=invalid-name


def transform(node, remove_decorators):
  transformer = DecoratorsTransformer(remove_decorators)
  node = transformer.visit(node)
  return node, transformer.additional_dependencies
