# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities used to capture Python idioms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Undefined(object):
  """Represents an undefined symbol in Python.

  This is used to reify undefined symbols, which is required to use the
  functional form of loops.
  Example:

    while n > 0:
      n = n - 1
      s = n
    return s  # Runtime error if n == 0

  This is valid Python code and will not result in an error as long as n
  is positive. The use of this class is to stay as close to Python semantics
  as possible for staged code of this nature.

  Converted version of the above showing the possible usage of this class:

    s = Undefined('s')
    init_state = (s,)
    s = while_loop(cond, body, init_state)
    return s  # s is an instance of Undefined if the loop never runs

  Attributes:
    symbol_name: Text, identifier for the undefined symbol
  """

  def __init__(self, symbol_name):
    # TODO(aqj) Possibly remove this after Symbols are fully integrated.
    self.symbol_name = symbol_name


def is_undefined(value):
  """Checks whether Autograph has determined that a given value is undefined.

  This only works in places where Autograph reifies undefined symbols. Note that
  if this function is passed a truly undefined symbol the call-site will raise
  NameError.

  Args:
    value: value to test for undefinedness
  Returns:
    Boolean, whether the input value is undefined.
  """
  return isinstance(value, Undefined)


# TODO(mdan): Refactor as a RetVal object, aggregating the value and do_return.
class UndefinedReturnValue(object):
  """Represents a default return value from a function (None in Python)."""
  pass


def retval(value):
  """Returns the actual value that a return statement should produce."""
  if isinstance(value, UndefinedReturnValue):
    return None
  return value


def is_undefined_return(value):
  """Checks whether `value` is the default return value."""
  return isinstance(value, UndefinedReturnValue)
