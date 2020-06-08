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
"""Provides wrapper for TensorFlow modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import types

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import tf_stack
from tensorflow.tools.compatibility import all_renames_v2


_PER_MODULE_WARNING_LIMIT = 1


def get_rename_v2(name):
  if name not in all_renames_v2.symbol_renames:
    return None
  return all_renames_v2.symbol_renames[name]


def _call_location():
  # We want to get stack frame 3 frames up from current frame,
  # i.e. above __getattr__, _tfmw_add_deprecation_warning,
  # and _call_location calls.
  stack = tf_stack.extract_stack(limit=4)
  if not stack:  # should never happen as we're in a function
    return 'UNKNOWN'
  frame = stack[0]
  return '{}:{}'.format(frame.filename, frame.lineno)


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


class TFModuleWrapper(types.ModuleType):
  """Wrapper for TF modules to support deprecation messages and lazyloading."""

  def __init__(  # pylint: disable=super-on-old-class
      self,
      wrapped,
      module_name,
      public_apis=None,
      deprecation=True,
      has_lite=False):  # pylint: enable=super-on-old-class
    super(TFModuleWrapper, self).__init__(wrapped.__name__)
    # A cache for all members which do not print deprecations (any more).
    self._tfmw_attr_map = {}
    self.__dict__.update(wrapped.__dict__)
    # Prefix all local attributes with _tfmw_ so that we can
    # handle them differently in attribute access methods.
    self._tfmw_wrapped_module = wrapped
    self._tfmw_module_name = module_name
    self._tfmw_public_apis = public_apis
    self._tfmw_print_deprecation_warnings = deprecation
    self._tfmw_has_lite = has_lite
    # Set __all__ so that import * work for lazy loaded modules
    if self._tfmw_public_apis:
      self._tfmw_wrapped_module.__all__ = list(self._tfmw_public_apis.keys())
      self.__all__ = list(self._tfmw_public_apis.keys())
    else:
      if hasattr(self._tfmw_wrapped_module, '__all__'):
        self.__all__ = self._tfmw_wrapped_module.__all__
      else:
        self._tfmw_wrapped_module.__all__ = [
            attr for attr in dir(self._tfmw_wrapped_module)
            if not attr.startswith('_')
        ]
        self.__all__ = self._tfmw_wrapped_module.__all__

    # names we already checked for deprecation
    self._tfmw_deprecated_checked = set()
    self._tfmw_warning_count = 0

  def _tfmw_add_deprecation_warning(self, name, attr):
    """Print deprecation warning for attr with given name if necessary."""
    if (self._tfmw_warning_count < _PER_MODULE_WARNING_LIMIT and
        name not in self._tfmw_deprecated_checked):

      self._tfmw_deprecated_checked.add(name)

      if self._tfmw_module_name:
        full_name = 'tf.%s.%s' % (self._tfmw_module_name, name)
      else:
        full_name = 'tf.%s' % name
      rename = get_rename_v2(full_name)
      if rename and not has_deprecation_decorator(attr):
        call_location = _call_location()
        # skip locations in Python source
        if not call_location.startswith('<'):
          logging.warning(
              'From %s: The name %s is deprecated. Please use %s instead.\n',
              _call_location(), full_name, rename)
          self._tfmw_warning_count += 1
          return True
    return False

  def _tfmw_import_module(self, name):
    symbol_loc_info = self._tfmw_public_apis[name]
    if symbol_loc_info[0]:
      module = importlib.import_module(symbol_loc_info[0])
      attr = getattr(module, symbol_loc_info[1])
    else:
      attr = importlib.import_module(symbol_loc_info[1])
    setattr(self._tfmw_wrapped_module, name, attr)
    self.__dict__[name] = attr
    return attr

  def __getattribute__(self, name):  # pylint: disable=super-on-old-class
    # Handle edge case where we unpickle and the object is not initialized yet
    # and does not have _tfmw_attr_map attribute. Otherwise, calling
    # __getattribute__ on __setstate__ will result in infinite recursion where
    # we keep trying to get _tfmw_wrapped_module in __getattr__.
    try:
      attr_map = object.__getattribute__(self, '_tfmw_attr_map')
    except AttributeError:
      self._tfmw_attr_map = attr_map = {}

    try:
      # Use cached attrs if available
      return attr_map[name]
    except KeyError:
      # Make sure we do not import from tensorflow/lite/__init__.py
      if name == 'lite':
        if self._tfmw_has_lite:
          attr = self._tfmw_import_module(name)
          setattr(self._tfmw_wrapped_module, 'lite', attr)
          attr_map[name] = attr
          return attr

      # Placeholder for Google-internal contrib error

      attr = super(TFModuleWrapper, self).__getattribute__(name)

      # Return and cache dunders and our own members.
      if name.startswith('__') or name.startswith('_tfmw_'):
        attr_map[name] = attr
        return attr

      # Print deprecations, only cache functions after deprecation warnings have
      # stopped.
      if not (self._tfmw_print_deprecation_warnings and
              self._tfmw_add_deprecation_warning(name, attr)):
        attr_map[name] = attr
      return attr

  def __getattr__(self, name):
    try:
      attr = getattr(self._tfmw_wrapped_module, name)
    except AttributeError:
      # Placeholder for Google-internal contrib error

      if not self._tfmw_public_apis:
        raise
      if name not in self._tfmw_public_apis:
        raise
      attr = self._tfmw_import_module(name)

    if self._tfmw_print_deprecation_warnings:
      self._tfmw_add_deprecation_warning(name, attr)
    return attr

  def __setattr__(self, arg, val):  # pylint: disable=super-on-old-class
    if not arg.startswith('_tfmw_'):
      setattr(self._tfmw_wrapped_module, arg, val)
      self.__dict__[arg] = val
      if arg not in self.__all__ and arg != '__all__':
        self.__all__.append(arg)
      if arg in self._tfmw_attr_map:
        self._tfmw_attr_map[arg] = val
    super(TFModuleWrapper, self).__setattr__(arg, val)

  def __dir__(self):
    if self._tfmw_public_apis:
      return list(
          set(self._tfmw_public_apis.keys()).union(
              set([
                  attr for attr in dir(self._tfmw_wrapped_module)
                  if not attr.startswith('_')
              ])))
    else:
      return dir(self._tfmw_wrapped_module)

  def __delattr__(self, name):  # pylint: disable=super-on-old-class
    if name.startswith('_tfmw_'):
      super(TFModuleWrapper, self).__delattr__(name)
    else:
      delattr(self._tfmw_wrapped_module, name)

  def __repr__(self):
    return self._tfmw_wrapped_module.__repr__()

  def __reduce__(self):
    return importlib.import_module, (self.__name__,)
