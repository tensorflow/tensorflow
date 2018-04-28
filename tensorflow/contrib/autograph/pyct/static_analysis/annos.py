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
"""Annotations used by the static analyzer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum


class NoValue(Enum):

  def __repr__(self):
    return self.name


class NodeAnno(NoValue):
  """Additional annotations used by the static analyzer.

  These are in addition to the basic annotations declared in anno.py.
  """

  # Symbols
  # These flags are boolean.
  IS_LOCAL = 'Symbol is local to the function scope being analyzed.'
  IS_PARAM = 'Symbol is a parameter to the function being analyzed.'
  IS_MODIFIED_SINCE_ENTRY = (
      'Symbol has been explicitly replaced in the current function scope.')

  # Scopes
  # Scopes are represented by objects of type activity.Scope.
  ARGS_SCOPE = 'The scope for the argument list of a function call.'
  COND_SCOPE = 'The scope for the test node of a conditional statement.'
  BODY_SCOPE = (
      'The scope for the main body of a statement (True branch for if '
      'statements, main body for loops).')
  ORELSE_SCOPE = (
      'The scope for the orelse body of a statement (False branch for if '
      'statements, orelse body for loops).')

  # Type and Value annotations
  # Type annotations are represented by objects of type type_info.Type.
  STATIC_INFO = (
      'The type or value information that should be asserted about the entity '
      'referenced by the symbol holding this annotation, irrespective of the '
      'execution context.')
