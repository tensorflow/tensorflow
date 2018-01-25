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

import gast

from tensorflow.contrib.py2tf.pyct import pretty_printer


class PyFlowParseError(SyntaxError):
  pass


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

  def visit(self, node):
    try:
      source_code = self.context.source_code
      source_file = self.context.source_file
      if source_code and hasattr(node, 'lineno'):
        self._lineno = node.lineno
        self._col_offset = node.col_offset
      return super(Base, self).visit(node)
    except ValueError as e:
      msg = '%s\nOccurred at node:\n%s' % (str(e), pretty_printer.fmt(node))
      if source_code:
        line = self._source.splitlines()[self._lineno - 1]
      else:
        line = '<no source available>'
      raise PyFlowParseError(
          msg, (source_file, self._lineno, self._col_offset + 1, line))
