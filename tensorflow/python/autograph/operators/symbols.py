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
"""Abstract representation of composite symbols that can used in staging code.

This provides a way to checkpoint the values of symbols that may be undefined
entering staged control flow. This checkpointing is necessary to prevent some
unintended side-effects. For example checkpointing prevents side-effects in one
branch of a conditional from leaking into another.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.operators import special_values


is_undefined = special_values.is_undefined
Undefined = special_values.Undefined


class Symbol(object):
  """Representation of a simple or composite Python symbol.

  Subclasses should implement `maybe_compute_value(self)` that returns the value
  corresponding to the symbol or Undefined if no such value exists.
  """

  def __init__(self, name):
    self.name = name


class ValueSymbol(Symbol):
  """Representation of a simple Python symbol with a concrete value.

  This includes variables and literals. Since we are reifying undefined symbols
  `Undefined` is also a valid value.
  """

  def __init__(self, name, value):
    super(ValueSymbol, self).__init__(name)
    self.value = value

  def maybe_compute_value(self):
    return self.value


class AttributeAccessSymbol(Symbol):
  """Representation of Python attribute access e.g. `a.b`."""

  def __init__(self, parent_symbol, attr_name):
    super(AttributeAccessSymbol, self).__init__(
        parent_symbol.name + '.' + attr_name)
    self.attr_name = attr_name
    self.parent_symbol = parent_symbol

  def maybe_compute_value(self):
    """Compute the value corresponding to the attribute access or `Undefined`.

    This will be `Undefined` if no such value exists either because there is no
    such attribute or if the base is itself undefined.

    Returns:
      value corresponding to the attribute access or `Undefined`
    """
    parent_value = self.parent_symbol.maybe_compute_value()
    if (is_undefined(parent_value) or
        getattr(parent_value, self.attr_name, None) is None):
      return Undefined(self.name)
    else:
      return parent_value.__getattribute__(self.attr_name)


class SubscriptSymbol(Symbol):
  """Representation of Python subscript access e.g. `a[b]`."""

  def __init__(self, parent_symbol, index_symbol):
    super(SubscriptSymbol, self).__init__(
        parent_symbol.name + '[' + index_symbol.name + ']')
    self.index_symbol = index_symbol
    self.parent_symbol = parent_symbol

  def maybe_compute_value(self):
    """Compute the value corresponding to the subscript access or `Undefined`.

    This will be `Undefined` if no such value exists either because there is no
    element corresponding to the given subscript or if the base itself is
    not defined.

    Returns:
      value corresponding to the subscript access or `Undefined`
    """
    parent_value = self.parent_symbol.maybe_compute_value()
    index_value = self.index_symbol.maybe_compute_value()
    if is_undefined(parent_value) or is_undefined(index_value):
      return Undefined(self.name)
    else:
      try:
        return parent_value[index_value]
      except (IndexError, KeyError, TypeError):
        # Reify the lack of an object for the given index/key
        # This allows us to define them later without regret
        return Undefined(self.name)
