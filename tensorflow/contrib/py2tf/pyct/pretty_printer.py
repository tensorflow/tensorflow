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
"""Print an AST tree in a form more readable than ast.dump."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast
import termcolor


class PrettyPrinter(gast.NodeVisitor):
  """Print AST nodes."""

  def __init__(self):
    self.indent_lvl = 0
    self.result = ''

  def _type(self, node):
    return termcolor.colored(node.__class__.__name__, None, attrs=['bold'])

  def _field(self, name):
    return termcolor.colored(name, 'blue')

  def _value(self, name):
    return termcolor.colored(name, 'magenta')

  def _warning(self, name):
    return termcolor.colored(name, 'red')

  def _indent(self):
    return termcolor.colored('| ' * self.indent_lvl, None, attrs=['dark'])

  def _print(self, s):
    self.result += s
    self.result += '\n'

  def generic_visit(self, node, name=None):
    if node._fields:
      cont = ':'
    else:
      cont = '()'

    if name:
      self._print('%s%s=%s%s' % (self._indent(), self._field(name),
                                 self._type(node), cont))
    else:
      self._print('%s%s%s' % (self._indent(), self._type(node), cont))

    self.indent_lvl += 1
    for f in node._fields:
      if not hasattr(node, f):
        self._print('%s%s' % (self._indent(), self._warning('%s=<unset>' % f)))
        continue
      v = getattr(node, f)
      if isinstance(v, list):
        if v:
          self._print('%s%s=[' % (self._indent(), self._field(f)))
          self.indent_lvl += 1
          for n in v:
            self.generic_visit(n)
          self.indent_lvl -= 1
          self._print('%s]' % (self._indent()))
        else:
          self._print('%s%s=[]' % (self._indent(), self._field(f)))
      elif isinstance(v, gast.AST):
        self.generic_visit(v, f)
      elif isinstance(v, str):
        self._print('%s%s=%s' % (self._indent(), self._field(f),
                                 self._value('"%s"' % v)))
      else:
        self._print('%s%s=%s' % (self._indent(), self._field(f),
                                 self._value(v)))
    self.indent_lvl -= 1


def fmt(node):
  printer = PrettyPrinter()
  if isinstance(node, (list, tuple)):
    for n in node:
      printer.visit(n)
  else:
    printer.visit(node)
  return printer.result
