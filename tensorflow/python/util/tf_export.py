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
"""Utilities for exporting TensorFlow symbols to the API.

Exporting a function or a class:

To export a function or a class use tf_export decorator. For e.g.:
```python
@tf_export('foo', 'bar.foo')
def foo(...):
  ...
```

If a function is assigned to a variable, you can export it by calling
tf_export explicitly. For e.g.:
```python
foo = get_foo(...)
tf_export('foo', 'bar.foo')(foo)
```


Exporting a constant
```python
foo = 1
tf_export('consts.foo').export_constant(__name__, 'foo')
```
"""
import collections
import functools
import sys

from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

ESTIMATOR_API_NAME = 'estimator'
KERAS_API_NAME = 'keras'
TENSORFLOW_API_NAME = 'tensorflow'

# List of subpackage names used by TensorFlow components. Have to check that
# TensorFlow core repo does not export any symbols under these names.
SUBPACKAGE_NAMESPACES = [ESTIMATOR_API_NAME]

_Attributes = collections.namedtuple(
    'ExportedApiAttributes', ['names', 'constants'])

# Attribute values must be unique to each API.
API_ATTRS = {
    TENSORFLOW_API_NAME: _Attributes(
        '_tf_api_names',
        '_tf_api_constants'),
    ESTIMATOR_API_NAME: _Attributes(
        '_estimator_api_names',
        '_estimator_api_constants'),
    KERAS_API_NAME: _Attributes(
        '_keras_api_names',
        '_keras_api_constants')
}

API_ATTRS_V1 = {
    TENSORFLOW_API_NAME: _Attributes(
        '_tf_api_names_v1',
        '_tf_api_constants_v1'),
    ESTIMATOR_API_NAME: _Attributes(
        '_estimator_api_names_v1',
        '_estimator_api_constants_v1'),
    KERAS_API_NAME: _Attributes(
        '_keras_api_names_v1',
        '_keras_api_constants_v1')
}


class SymbolAlreadyExposedError(Exception):
  """Raised when adding API names to symbol that already has API names."""
  pass


class InvalidSymbolNameError(Exception):
  """Raised when trying to export symbol as an invalid or unallowed name."""
  pass

_NAME_TO_SYMBOL_MAPPING = dict()


def get_symbol_from_name(name):
  return _NAME_TO_SYMBOL_MAPPING.get(name)


def get_canonical_name_for_symbol(
    symbol, api_name=TENSORFLOW_API_NAME,
    add_prefix_to_v1_names=False):
  """Get canonical name for the API symbol.

  Example:
  ```python
  from tensorflow.python.util import tf_export
  cls = tf_export.get_symbol_from_name('keras.optimizers.Adam')

  # Gives `<class 'keras.optimizer_v2.adam.Adam'>`
  print(cls)

  # Gives `keras.optimizers.Adam`
  print(tf_export.get_canonical_name_for_symbol(cls, api_name='keras'))
  ```

  Args:
    symbol: API function or class.
    api_name: API name (tensorflow or estimator).
    add_prefix_to_v1_names: Specifies whether a name available only in V1
      should be prefixed with compat.v1.

  Returns:
    Canonical name for the API symbol (for e.g. initializers.zeros) if
    canonical name could be determined. Otherwise, returns None.
  """
  if not hasattr(symbol, '__dict__'):
    return None
  api_names_attr = API_ATTRS[api_name].names
  _, undecorated_symbol = tf_decorator.unwrap(symbol)
  if api_names_attr not in undecorated_symbol.__dict__:
    return None
  api_names = getattr(undecorated_symbol, api_names_attr)
  deprecated_api_names = undecorated_symbol.__dict__.get(
      '_tf_deprecated_api_names', [])

  canonical_name = get_canonical_name(api_names, deprecated_api_names)
  if canonical_name:
    return canonical_name

  # If there is no V2 canonical name, get V1 canonical name.
  api_names_attr = API_ATTRS_V1[api_name].names
  api_names = getattr(undecorated_symbol, api_names_attr)
  v1_canonical_name = get_canonical_name(api_names, deprecated_api_names)
  if add_prefix_to_v1_names:
    return 'compat.v1.%s' % v1_canonical_name
  return v1_canonical_name


def get_canonical_name(api_names, deprecated_api_names):
  """Get preferred endpoint name.

  Args:
    api_names: API names iterable.
    deprecated_api_names: Deprecated API names iterable.
  Returns:
    Returns one of the following in decreasing preference:
    - first non-deprecated endpoint
    - first endpoint
    - None
  """
  non_deprecated_name = next(
      (name for name in api_names if name not in deprecated_api_names),
      None)
  if non_deprecated_name:
    return non_deprecated_name
  if api_names:
    return api_names[0]
  return None


def get_v1_names(symbol):
  """Get a list of TF 1.* names for this symbol.

  Args:
    symbol: symbol to get API names for.

  Returns:
    List of all API names for this symbol including TensorFlow and
    Estimator names.
  """
  names_v1 = []
  tensorflow_api_attr_v1 = API_ATTRS_V1[TENSORFLOW_API_NAME].names
  estimator_api_attr_v1 = API_ATTRS_V1[ESTIMATOR_API_NAME].names
  keras_api_attr_v1 = API_ATTRS_V1[KERAS_API_NAME].names

  if not hasattr(symbol, '__dict__'):
    return names_v1
  if tensorflow_api_attr_v1 in symbol.__dict__:
    names_v1.extend(getattr(symbol, tensorflow_api_attr_v1))
  if estimator_api_attr_v1 in symbol.__dict__:
    names_v1.extend(getattr(symbol, estimator_api_attr_v1))
  if keras_api_attr_v1 in symbol.__dict__:
    names_v1.extend(getattr(symbol, keras_api_attr_v1))
  return names_v1


def get_v2_names(symbol):
  """Get a list of TF 2.0 names for this symbol.

  Args:
    symbol: symbol to get API names for.

  Returns:
    List of all API names for this symbol including TensorFlow and
    Estimator names.
  """
  names_v2 = []
  tensorflow_api_attr = API_ATTRS[TENSORFLOW_API_NAME].names
  estimator_api_attr = API_ATTRS[ESTIMATOR_API_NAME].names
  keras_api_attr = API_ATTRS[KERAS_API_NAME].names

  if not hasattr(symbol, '__dict__'):
    return names_v2
  if tensorflow_api_attr in symbol.__dict__:
    names_v2.extend(getattr(symbol, tensorflow_api_attr))
  if estimator_api_attr in symbol.__dict__:
    names_v2.extend(getattr(symbol, estimator_api_attr))
  if keras_api_attr in symbol.__dict__:
    names_v2.extend(getattr(symbol, keras_api_attr))
  return names_v2


def get_v1_constants(module):
  """Get a list of TF 1.* constants in this module.

  Args:
    module: TensorFlow module.

  Returns:
    List of all API constants under the given module including TensorFlow and
    Estimator constants.
  """
  constants_v1 = []
  tensorflow_constants_attr_v1 = API_ATTRS_V1[TENSORFLOW_API_NAME].constants
  estimator_constants_attr_v1 = API_ATTRS_V1[ESTIMATOR_API_NAME].constants

  if hasattr(module, tensorflow_constants_attr_v1):
    constants_v1.extend(getattr(module, tensorflow_constants_attr_v1))
  if hasattr(module, estimator_constants_attr_v1):
    constants_v1.extend(getattr(module, estimator_constants_attr_v1))
  return constants_v1


def get_v2_constants(module):
  """Get a list of TF 2.0 constants in this module.

  Args:
    module: TensorFlow module.

  Returns:
    List of all API constants under the given module including TensorFlow and
    Estimator constants.
  """
  constants_v2 = []
  tensorflow_constants_attr = API_ATTRS[TENSORFLOW_API_NAME].constants
  estimator_constants_attr = API_ATTRS[ESTIMATOR_API_NAME].constants

  if hasattr(module, tensorflow_constants_attr):
    constants_v2.extend(getattr(module, tensorflow_constants_attr))
  if hasattr(module, estimator_constants_attr):
    constants_v2.extend(getattr(module, estimator_constants_attr))
  return constants_v2


class api_export(object):  # pylint: disable=invalid-name
  """Provides ways to export symbols to the TensorFlow API."""

  def __init__(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Export under the names *args (first one is considered canonical).

    Args:
      *args: API names in dot delimited format.
      **kwargs: Optional keyed arguments.
        v1: Names for the TensorFlow V1 API. If not set, we will use V2 API
          names both for TensorFlow V1 and V2 APIs.
        overrides: List of symbols that this is overriding
          (those overrided api exports will be removed). Note: passing overrides
          has no effect on exporting a constant.
        api_name: Name of the API you want to generate (e.g. `tensorflow` or
          `estimator`). Default is `tensorflow`.
        allow_multiple_exports: Allow symbol to be exported multiple time under
          different names.
    """
    self._names = args
    self._names_v1 = kwargs.get('v1', args)
    if 'v2' in kwargs:
      raise ValueError('You passed a "v2" argument to tf_export. This is not '
                       'what you want. Pass v2 names directly as positional '
                       'arguments instead.')
    self._api_name = kwargs.get('api_name', TENSORFLOW_API_NAME)
    self._overrides = kwargs.get('overrides', [])
    self._allow_multiple_exports = kwargs.get('allow_multiple_exports', False)
    self._is_deprecated = kwargs.get('is_deprecated', False)
    self._deprecation_inst = kwargs.get('deprecation_inst', None)

    self._validate_symbol_names()

  def _validate_symbol_names(self):
    """Validate you are exporting symbols under an allowed package.

    We need to ensure things exported by tf_export, estimator_export, etc.
    export symbols under disjoint top-level package names.

    For TensorFlow, we check that it does not export anything under subpackage
    names used by components (estimator, keras, etc.).

    For each component, we check that it exports everything under its own
    subpackage.

    Raises:
      InvalidSymbolNameError: If you try to export symbol under disallowed name.
    """
    all_symbol_names = set(self._names) | set(self._names_v1)
    if self._api_name == TENSORFLOW_API_NAME:
      for subpackage in SUBPACKAGE_NAMESPACES:
        if any(n.startswith(subpackage) for n in all_symbol_names):
          raise InvalidSymbolNameError(
              '@tf_export is not allowed to export symbols under %s.*' % (
                  subpackage))
    else:
      if not all(n.startswith(self._api_name) for n in all_symbol_names):
        raise InvalidSymbolNameError(
            'Can only export symbols under package name of component. '
            'e.g. tensorflow_estimator must export all symbols under '
            'tf.estimator')

  def __call__(self, func):
    """Calls this decorator.

    Args:
      func: decorated symbol (function or class).

    Returns:
      The input function with _tf_api_names attribute set.

    Raises:
      SymbolAlreadyExposedError: Raised when a symbol already has API names
        and kwarg `allow_multiple_exports` not set.
    """
    api_names_attr = API_ATTRS[self._api_name].names
    api_names_attr_v1 = API_ATTRS_V1[self._api_name].names
    # Undecorate overridden names
    for f in self._overrides:
      _, undecorated_f = tf_decorator.unwrap(f)
      delattr(undecorated_f, api_names_attr)
      delattr(undecorated_f, api_names_attr_v1)

    _, undecorated_func = tf_decorator.unwrap(func)
    self.set_attr(undecorated_func, api_names_attr, self._names)
    self.set_attr(undecorated_func, api_names_attr_v1, self._names_v1)

    for name in self._names:
      _NAME_TO_SYMBOL_MAPPING[name] = func
    for name_v1 in self._names_v1:
      _NAME_TO_SYMBOL_MAPPING['compat.v1.%s' % name_v1] = func

    if self._is_deprecated:
      from tensorflow.python.util import deprecation  # pylint: disable=g-import-not-at-top
      deprecation_wrapper = deprecation.deprecated(
          None, self._deprecation_inst, warn_once=True)
      return deprecation_wrapper(func)
    return func

  def set_attr(self, func, api_names_attr, names):
    # Check for an existing api. We check if attribute name is in
    # __dict__ instead of using hasattr to verify that subclasses have
    # their own _tf_api_names as opposed to just inheriting it.
    if api_names_attr in func.__dict__:
      if not self._allow_multiple_exports:
        raise SymbolAlreadyExposedError(
            'Symbol %s is already exposed as %s.' %
            (func.__name__, getattr(func, api_names_attr)))  # pylint: disable=protected-access
    setattr(func, api_names_attr, names)

  def export_constant(self, module_name, name):
    """Store export information for constants/string literals.

    Export information is stored in the module where constants/string literals
    are defined.

    e.g.
    ```python
    foo = 1
    bar = 2
    tf_export("consts.foo").export_constant(__name__, 'foo')
    tf_export("consts.bar").export_constant(__name__, 'bar')
    ```

    Args:
      module_name: (string) Name of the module to store constant at.
      name: (string) Current constant name.
    """
    module = sys.modules[module_name]
    api_constants_attr = API_ATTRS[self._api_name].constants
    api_constants_attr_v1 = API_ATTRS_V1[self._api_name].constants

    if not hasattr(module, api_constants_attr):
      setattr(module, api_constants_attr, [])
    # pylint: disable=protected-access
    getattr(module, api_constants_attr).append(
        (self._names, name))

    if not hasattr(module, api_constants_attr_v1):
      setattr(module, api_constants_attr_v1, [])
    getattr(module, api_constants_attr_v1).append(
        (self._names_v1, name))


def kwarg_only(f):
  """A wrapper that throws away all non-kwarg arguments."""
  f_argspec = tf_inspect.getargspec(f)

  def wrapper(*args, **kwargs):
    if args:
      raise TypeError(
          '{f} only takes keyword args (possible keys: {kwargs}). '
          'Please pass these args as kwargs instead.'
          .format(f=f.__name__, kwargs=f_argspec.args))
    return f(**kwargs)

  return tf_decorator.make_decorator(f, wrapper, decorator_argspec=f_argspec)


tf_export = functools.partial(api_export, api_name=TENSORFLOW_API_NAME)
estimator_export = functools.partial(
    api_export,
    api_name=ESTIMATOR_API_NAME,
    is_deprecated=True,
    deprecation_inst='Use tf.keras instead.')
keras_export = functools.partial(api_export, api_name=KERAS_API_NAME)
