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
tf_export("consts.foo").export_constant(__name__, 'foo')
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow.python.util import tf_decorator


class SymbolAlreadyExposedError(Exception):
  """Raised when adding API names to symbol that already has API names."""
  pass


class tf_export(object):  # pylint: disable=invalid-name
  """Provides ways to export symbols to the TensorFlow API."""

  def __init__(self, *args, **kwargs):
    """Export under the names *args (first one is considered canonical).

    Args:
      *args: API names in dot delimited format.
      **kwargs: Optional keyed arguments.
          overrides: List of symbols that this is overriding
          (those overrided api exports will be removed). Note: passing overrides
          has no effect on exporting a constant.
          allow_multiple_exports: Allows exporting the same symbol multiple
          times with multiple `tf_export` usages. Prefer however, to list all
          of the exported names in a single `tf_export` usage when possible.

    """
    self._names = args
    self._overrides = kwargs.get('overrides', [])
    self._allow_multiple_exports = kwargs.get(
        'allow_multiple_exports', False)

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
    # Undecorate overridden names
    for f in self._overrides:
      _, undecorated_f = tf_decorator.unwrap(f)
      del undecorated_f._tf_api_names  # pylint: disable=protected-access

    _, undecorated_func = tf_decorator.unwrap(func)

    # Check for an existing api. We check if attribute name is in
    # __dict__ instead of using hasattr to verify that subclasses have
    # their own _tf_api_names as opposed to just inheriting it.
    if '_tf_api_names' in undecorated_func.__dict__:
      if self._allow_multiple_exports:
        undecorated_func._tf_api_names += self._names  # pylint: disable=protected-access
      else:
        raise SymbolAlreadyExposedError(
            'Symbol %s is already exposed as %s.' %
            (undecorated_func.__name__, undecorated_func._tf_api_names))  # pylint: disable=protected-access
    else:
      undecorated_func._tf_api_names = self._names  # pylint: disable=protected-access
    return func

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
    if not hasattr(module, '_tf_api_constants'):
      module._tf_api_constants = []  # pylint: disable=protected-access
    # pylint: disable=protected-access
    module._tf_api_constants.append((self._names, name))

