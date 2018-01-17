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
"""Handling annotations on AST nodes.

Adapted from Tangent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def getanno(node, key, field_name='___pyct_anno'):
  return getattr(node, field_name)[key]


def hasanno(node, key, field_name='___pyct_anno'):
  return hasattr(node, field_name) and key in getattr(node, field_name)


def setanno(node, key, value, field_name='___pyct_anno'):
  annotations = getattr(node, field_name, {})
  setattr(node, field_name, annotations)
  annotations[key] = value

  # So that the annotations survive gast_to_ast() and ast_to_gast()
  if field_name not in node._fields:
    node._fields += (field_name,)
