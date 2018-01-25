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
"""Conversion context containers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class EntityContext(object):
  """Contains information about an entity, like source code.

  Attributes:
    namer: Namer that matches the contract of all converters.
    source_code: The entity's source code.
    source_file: The entity's source file.
    namespace: Dict[str->*], containing symbols visible to the entity
        (excluding parameters).
    arg_values: Dict[str->*], containing parameter values, if known.
    arg_types: Dict[str->*], containing parameter types, if known.
  """

  def __init__(self, namer, source_code, source_file, namespace, arg_values,
               arg_types):
    self.namer = namer
    self.source_code = source_code
    self.source_file = source_file
    self.namespace = namespace
    self.arg_values = {} if arg_values is None else arg_values
    self.arg_types = {} if arg_types is None else arg_types
