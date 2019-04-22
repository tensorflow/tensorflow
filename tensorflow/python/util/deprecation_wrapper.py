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
"""Provides wrapper for TensorFlow modules to support deprecation messages.

TODO(annarev): potentially merge with LazyLoader.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import types

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import tf_stack
from tensorflow.tools.compatibility import all_renames_v2


_PER_MODULE_WARNING_LIMIT = 5


def get_rename_v2(name):
  if name not in all_renames_v2.symbol_renames:
    return None
  return all_renames_v2.symbol_renames[name]


def _call_location():
  # We want to get stack frame 2 frames up from current frame,
  # i.e. above _getattr__ and _call_location calls.
  stack = tf_stack.extract_stack_file_and_line(max_length=3)
  if not stack:  # should never happen as we're in a function
    return 'UNKNOWN'
  frame = stack[0]
  return '{}:{}'.format(frame.file, frame.line)


def contains_deprecation_decorator(decorators):
  return any(
      d.decorator_name == 'deprecated' for d in decorators)


def has_deprecation_decorator(symbol):
  """Checks if given object has a deprecation decorator.

  We check if deprecation decorator is in decorators as well as
  whether symbol is a class whose __init__ method has a deprecation
  decorator.
  Args:
    symbol: Python object.

  Returns:
    True if symbol has deprecation decorator.
  """
  decorators, symbol = tf_decorator.unwrap(symbol)
  if contains_deprecation_decorator(decorators):
    return True
  if tf_inspect.isfunction(symbol):
    return False
  if not tf_inspect.isclass(symbol):
    return False
  if not hasattr(symbol, '__init__'):
    return False
  init_decorators, _ = tf_decorator.unwrap(symbol.__init__)
  return contains_deprecation_decorator(init_decorators)


class DeprecationWrapper(types.ModuleType):
  """Wrapper for TensorFlow modules to support deprecation messages."""

  # TODO(annarev): remove unused_depr_to_canonical once estimator stops
  # passing it in the next nightly build.
  def __init__(self, wrapped, module_name, unused_depr_to_canonical=None):  # pylint: disable=super-on-old-class
    # Prefix all local attributes with _dw_ so that we can
    # handle them differently in attribute access methods.
    self._dw_wrapped_module = wrapped
    self._dw_module_name = module_name
    self._dw_deprecated_printed = set()  # names we already printed warning for
    self.__file__ = wrapped.__file__
    self.__name__ = wrapped.__name__
    if hasattr(self._dw_wrapped_module, '__all__'):
      self.__all__ = self._dw_wrapped_module.__all__
    else:
      self.__all__ = dir(self._dw_wrapped_module)
    self._dw_warning_count = 0
    super(DeprecationWrapper, self).__init__(wrapped.__name__)

  def __getattr__(self, name):
    if name.startswith('_dw_'):
      raise AttributeError('Accessing local variables before they are created.')
    attr = getattr(self._dw_wrapped_module, name)
    if (self._dw_warning_count < _PER_MODULE_WARNING_LIMIT and
        name not in self._dw_deprecated_printed):
      if self._dw_module_name:
        full_name = 'tf.%s.%s' % (self._dw_module_name, name)
      else:
        full_name = 'tf.%s' % name
      rename = get_rename_v2(full_name)
      if rename and not has_deprecation_decorator(attr):
        call_location = _call_location()
        if not call_location.startswith('<'):  # skip locations in Python source
          logging.warning(
              'From %s: The name %s is deprecated. Please use %s instead.\n',
              _call_location(), full_name, rename)
          self._dw_deprecated_printed.add(name)
          self._dw_warning_count += 1
    return attr

  def __setattr__(self, arg, val):  # pylint: disable=super-on-old-class
    if arg.startswith('_dw_'):
      super(DeprecationWrapper, self).__setattr__(arg, val)
    else:
      setattr(self._dw_wrapped_module, arg, val)

  def __dir__(self):
    return dir(self._dw_wrapped_module)

  def __delattr__(self, name):  # pylint: disable=super-on-old-class
    if name.startswith('_dw_'):
      super(DeprecationWrapper, self).__delattr__(name)
    else:
      delattr(self._dw_wrapped_module, name)

  def __repr__(self):
    return self._dw_wrapped_module.__repr__()

  def __getstate__(self):
    return self.__name__

  def __setstate__(self, d):
    # pylint: disable=protected-access
    self.__init__(
        sys.modules[d]._dw_wrapped_module,
        sys.modules[d]._dw_module_name)
    # pylint: enable=protected-access
