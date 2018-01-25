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
"""Handles decorators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import pretty_printer


class DecoratorsTransformer(gast.NodeTransformer):
  """Converts or removes decorators."""

  def __init__(self, remove_decorators):
    self.remove_decorators = remove_decorators

  # pylint:disable=invalid-name

  def visit_FunctionDef(self, node):
    self.generic_visit(node)
    for dec in node.decorator_list:
      if isinstance(dec, gast.Call):
        dec = dec.func
      if not anno.hasanno(dec, 'live_val'):
        raise ValueError(
            'Could not resolve decorator: %s' % pretty_printer.fmt(dec))
      dec_value = anno.getanno(dec, 'live_val')
      if dec_value in self.remove_decorators:
        continue
      raise ValueError('Dont know how to convert decorators for now.')
    node.decorator_list = []
    return node

  # pylint:enable=invalid-name


def transform(node, remove_decorators):
  transformer = DecoratorsTransformer(remove_decorators)
  node = transformer.visit(node)
  return node
