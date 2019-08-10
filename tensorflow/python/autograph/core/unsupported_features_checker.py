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
"""Checkers for detecting unsupported Python features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast


class UnsupportedFeaturesChecker(gast.NodeVisitor):
  """Quick check for Python features we know we don't support.

  Any features detected will cause AutoGraph to not compile a function.
  """

  def visit_Attribute(self, node):
    if (node.attr is not None
        and node.attr.startswith('__') and not node.attr.endswith('__')):
      raise NotImplementedError(
          'Mangled names are not yet supported by AutoGraph')

  # These checks could potentially be replaced with inspect.isgeneratorfunction
  # to avoid a getsource/parse/ast-walk round trip.
  def visit_Yield(self, node):
    raise NotImplementedError('Generators are not supported by AutoGraph')

  def visit_YieldFrom(self, node):
    raise NotImplementedError('Generators are not supported by AutoGraph')


def verify(node):
  UnsupportedFeaturesChecker().visit(node)
