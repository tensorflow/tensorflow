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

import enum

from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.utils import misc


class _NamingStyle(enum.Enum):
  SNAKE = 1
  CAMEL = 2


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

  def __init__(self, global_namespace, recursive, name_map, partial_types):
    self.global_namespace = global_namespace
    self.recursive = recursive
    self.partial_types = partial_types

    self.renamed_calls = {}
    if name_map is not None:
      self.renamed_calls.update(name_map)

    self.generated_names = set()

  def _as_symbol_name(self, fqn, style=_NamingStyle.SNAKE):
    """Returns a symbol name that matches a fully-qualified name.

    The returned name is safe to use for Python symbols. Any special characters
    present in fqn are replaced according to the style argument.

    Examples:

      self._as_symbol_name('foo.bar', style=_NamingStyle.CAMEL) == 'FooBar'
      self._as_symbol_name('foo.bar', style=_NamingStyle.SNAKE) == 'foo_bar'

    See the unit tests for more examples.

    Args:
      fqn: Union[Text, Tuple[Text]] a fully-qualified symbol name. The qualifier
        may include module, class names, attributes, etc.
      style: _NamingStyle
    Returns:
      Text
    """
    assert style in _NamingStyle

    if isinstance(fqn, tuple):
      cn = '.'.join(fqn)
    else:
      cn = fqn

    # Until we clean up the whole FQN mechanism, `fqn` may not be
    # canonical, that is, in can appear as ('foo.bar', 'baz')
    # This replaces any characters that might remain because of that.
    pieces = cn.split('.')

    if style == _NamingStyle.CAMEL:
      pieces = tuple(misc.capitalize_initial(p) for p in pieces)
      return ''.join(pieces)
    elif style == _NamingStyle.SNAKE:
      return '_'.join(pieces)

  def compiled_class_name(self, original_fqn, live_entity=None):
    """See call_trees.FunctionNamer.compiled_class_name."""
    if live_entity is not None and live_entity in self.renamed_calls:
      return self.renamed_calls[live_entity]

    canonical_name = self._as_symbol_name(
        original_fqn, style=_NamingStyle.CAMEL)
    new_name_root = 'Tf%s' % canonical_name
    new_name = new_name_root
    n = 0
    while new_name in self.global_namespace:
      n += 1
      new_name = '%s_%d' % (new_name_root, n)

    self.generated_names.add(new_name)
    if live_entity is not None:
      self.renamed_calls[live_entity] = new_name
    return new_name

  def compiled_function_name(self,
                             original_fqn,
                             live_entity=None,
                             owner_type=None):
    """See call_trees.FunctionNamer.compiled_function_name."""
    if not self.recursive:
      return None, False

    if (live_entity is not None and inspect_utils.islambda(live_entity)):
      return None, False

    if owner_type is not None and owner_type not in self.partial_types:
      # Members are not renamed when part of an entire converted class.
      return None, False

    if live_entity is not None and live_entity in self.renamed_calls:
      return self.renamed_calls[live_entity], True

    canonical_name = self._as_symbol_name(
        original_fqn, style=_NamingStyle.SNAKE)
    new_name_root = 'tf__%s' % canonical_name
    new_name = new_name_root
    n = 0
    while new_name in self.global_namespace:
      n += 1
      new_name = '%s_%d' % (new_name_root, n)

    if live_entity is not None:
      self.renamed_calls[live_entity] = new_name
    self.generated_names.add(new_name)

    return new_name, True

  def new_symbol(self, name_root, reserved_locals):
    """See control_flow.SymbolNamer.new_symbol."""
    # reserved_locals may contain QNs.
    all_reserved_locals = set()
    for s in reserved_locals:
      if isinstance(s, qual_names.QN):
        all_reserved_locals.update(s.qn)
      elif isinstance(s, str):
        all_reserved_locals.add(s)
      else:
        raise ValueError('Unexpected symbol type "%s"' % type(s))

    pieces = name_root.split('_')
    if pieces[-1].isdigit():
      name_root = '_'.join(pieces[:-1])
      n = int(pieces[-1])
    else:
      n = 0
    new_name = name_root

    while (new_name in self.global_namespace or
           new_name in all_reserved_locals or new_name in self.generated_names):
      n += 1
      new_name = '%s_%d' % (name_root, n)

    self.generated_names.add(new_name)
    return new_name
