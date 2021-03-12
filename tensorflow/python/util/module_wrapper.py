# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
import inspect

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import fast_module_type
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2

FastModuleType = fast_module_type.get_fast_module_type_class()
_PER_MODULE_WARNING_LIMIT = 1


def get_rename_v2(name):
  if name not in all_renames_v2.symbol_renames:
    return None
  return all_renames_v2.symbol_renames[name]


def _call_location():
  """Extracts the caller filename and line number as a string.

  Returns:
    A string describing the caller source location.
  """
  frame = inspect.currentframe()
  assert frame.f_back.f_code.co_name == '_tfmw_add_deprecation_warning', (
      'This function should be called directly from '
      '_tfmw_add_deprecation_warning, as the caller is identified '
      'heuristically by chopping off the top stack frames.')

  # We want to get stack frame 3 frames up from current frame,
  # i.e. above __getattr__, _tfmw_add_deprecation_warning,
  # and _call_location calls.
  for _ in range(3):
    parent = frame.f_back
    if parent is None:
      break
    frame = parent
  return '{}:{}'.format(frame.f_code.co_filename, frame.f_lineno)


def contains_deprecation_decorator(decorators):
  return any(d.decorator_name == 'deprecated' for d in decorators)


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


class TFModuleWrapper(FastModuleType):
  """Wrapper for TF modules to support deprecation messages and lazyloading."""

  def __init__(
      self,
      wrapped,
      module_name,
      public_apis=None,
      deprecation=True,
      has_lite=False):
    super(TFModuleWrapper, self).__init__(wrapped.__name__)
    FastModuleType.set_getattr_callback(self, TFModuleWrapper._getattr)
    FastModuleType.set_getattribute_callback(self,
                                             TFModuleWrapper._getattribute)
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
    """Lazily loading the modules."""
    symbol_loc_info = self._tfmw_public_apis[name]
    if symbol_loc_info[0]:
      module = importlib.import_module(symbol_loc_info[0])
      attr = getattr(module, symbol_loc_info[1])
    else:
      attr = importlib.import_module(symbol_loc_info[1])
    setattr(self._tfmw_wrapped_module, name, attr)
    self.__dict__[name] = attr
    # Cache the pair
    self._fastdict_insert(name, attr)
    return attr

  def _getattribute(self, name):
    # pylint: disable=g-doc-return-or-yield,g-doc-args
    """Imports and caches pre-defined API.

    Warns if necessary.

    This method is a replacement for __getattribute__(). It will be added into
    the extended python module as a callback to reduce API overhead.
    """
    # Avoid infinite recursions
    func__fastdict_insert = object.__getattribute__(self, '_fastdict_insert')

    # Make sure we do not import from tensorflow/lite/__init__.py
    if name == 'lite':
      if self._tfmw_has_lite:
        attr = self._tfmw_import_module(name)
        setattr(self._tfmw_wrapped_module, 'lite', attr)
        func__fastdict_insert(name, attr)
        return attr
  # Placeholder for Google-internal contrib error

    attr = object.__getattribute__(self, name)

    # Return and cache dunders and our own members.
    # This is necessary to guarantee successful construction.
    # In addition, all the accessed attributes used during the construction must
    # begin with "__" or "_tfmw" or "_fastdict_".
    if name.startswith('__') or name.startswith('_tfmw_') or name.startswith(
        '_fastdict_'):
      func__fastdict_insert(name, attr)
      return attr

    # Print deprecations, only cache functions after deprecation warnings have
    # stopped.
    if not (self._tfmw_print_deprecation_warnings and
            self._tfmw_add_deprecation_warning(name, attr)):
      func__fastdict_insert(name, attr)

    return attr

  def _getattr(self, name):
    # pylint: disable=g-doc-return-or-yield,g-doc-args
    """Imports and caches pre-defined API.

    Warns if necessary.

    This method is a replacement for __getattr__(). It will be added into the
    extended python module as a callback to reduce API overhead. Instead of
    relying on implicit AttributeError handling, this added callback function
    will
    be called explicitly from the extended C API if the default attribute lookup
    fails.
    """
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

  def __setattr__(self, arg, val):
    if not arg.startswith('_tfmw_'):
      setattr(self._tfmw_wrapped_module, arg, val)
      self.__dict__[arg] = val
      if arg not in self.__all__ and arg != '__all__':
        self.__all__.append(arg)
      # Update the cache
      if self._fastdict_key_in(arg):
        self._fastdict_insert(arg, val)
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

  def __delattr__(self, name):
    if name.startswith('_tfmw_'):
      super(TFModuleWrapper, self).__delattr__(name)
    else:
      delattr(self._tfmw_wrapped_module, name)

  def __repr__(self):
    return self._tfmw_wrapped_module.__repr__()

  def __reduce__(self):
    return importlib.import_module, (self.__name__,)
