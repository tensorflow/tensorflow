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
"""Symbol naming utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import tf_inspect


class Namer(object):
  """Implementation of the namer interfaces required by various converters.

  This implementation performs additional tasks like keeping track of the
  function calls that have been encountered and replaced with calls to their
  corresponding compiled counterparts.

  Interfaces currently implemented:
    * call_trees.FunctionNamer
    * control_flow.SymbolNamer
    * side_effect_guards.SymbolNamer
  """

  def __init__(self, global_namespace, name_map=None):
    self.global_namespace = global_namespace

    self.renamed_calls = {}
    if name_map is not None:
      self.renamed_calls.update(name_map)

    self.generated_names = set()

  def compiled_class_name(self, original_name, live_object=None):
    """See call_trees.FunctionNamer.compiled_class_name."""
    if live_object is not None and live_object in self.renamed_calls:
      return self.renamed_calls[live_object]

    new_name_root = 'Tf%s' % original_name
    new_name = new_name_root
    n = 0
    while new_name in self.global_namespace:
      n += 1
      new_name = '%s_%d' % (new_name_root, n)
    if live_object is not None:
      self.renamed_calls[live_object] = new_name
    self.generated_names.add(new_name)
    return new_name

  def compiled_function_name(self,
                             original_name,
                             live_object=None,
                             owner_type=None):
    """See call_trees.FunctionNamer.compiled_function_name."""
    if live_object is not None and live_object in self.renamed_calls:
      return self.renamed_calls[live_object]

    if owner_type is None:
      # Top level functions: rename
      new_name_root = 'tf__%s' % original_name
      new_name = new_name_root
      n = 0
      while new_name in self.global_namespace:
        n += 1
        new_name = '%s_%d' % (new_name_root, n)
    else:
      if tf_inspect.isclass(owner_type):
        # Class members: do not rename (the entire class will be renamed)
        new_name = original_name
      else:
        raise NotImplementedError('Member function "%s" of non-class type: %s' %
                                  (original_name, owner_type))

    if live_object is not None:
      self.renamed_calls[live_object] = new_name
    self.generated_names.add(new_name)
    return new_name

  def new_symbol(self, name_root, reserved_locals):
    """See control_flow.SymbolNamer.new_symbol."""
    new_name = name_root
    n = 0
    while (new_name in self.global_namespace
           or new_name in reserved_locals
           or new_name in self.generated_names):
      n += 1
      new_name = '%s_%d' % (name_root, n)

    self.generated_names.add(new_name)
    return new_name
