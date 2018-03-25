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

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import pretty_printer


class PyFlowParseError(SyntaxError):
  pass


def try_ast_to_source(node):
  try:
    return compiler.ast_to_source(node)
  except AssertionError:
    return '<could not convert AST to source>'


class Base(gast.NodeTransformer):
  """Base class for specialized transformers."""

  def __init__(self, context):
    """Initialize the transformer. Subclasses should call this.

    Args:
      context: An EntityContext.
    """
    self._lineno = 0
    self._col_offset = 0
    self.context = context

  def debug_print(self, node):
    """Helper method useful for debugging."""
    if __debug__:
      print(pretty_printer.fmt(node))
    return node

  def visit(self, node):
    source_code = self.context.source_code
    source_file = self.context.source_file
    try:
      if source_code and hasattr(node, 'lineno'):
        self._lineno = node.lineno
        self._col_offset = node.col_offset
      if anno.hasanno(node, anno.Basic.SKIP_PROCESSING):
        return node
      return super(Base, self).visit(node)
    except (ValueError, AttributeError, KeyError, NotImplementedError,
            AssertionError) as e:
      msg = '%s: %s\nOffending source:\n%s\n\nOccurred at node:\n%s' % (
          e.__class__.__name__, str(e), try_ast_to_source(node),
          pretty_printer.fmt(node, color=False))
      if source_code:
        line = source_code.splitlines()[self._lineno - 1]
      else:
        line = '<no source available>'
      six.reraise(PyFlowParseError,
                  PyFlowParseError(
                      msg,
                      (source_file, self._lineno, self._col_offset + 1, line)),
                  sys.exc_info()[2])
