# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Python utilities required by Keras."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import marshal
import sys
import time
import types as python_types

import numpy as np
import six

from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

_GLOBAL_CUSTOM_OBJECTS = {}


class CustomObjectScope(object):
  """Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

  Code within a `with` statement will be able to access custom objects
  by name. Changes to global custom objects persist
  within the enclosing `with` statement. At end of the `with` statement,
  global custom objects are reverted to state
  at beginning of the `with` statement.

  Example:

  Consider a custom object `MyObject`

  ```python
      with CustomObjectScope({'MyObject':MyObject}):
          layer = Dense(..., kernel_regularizer='MyObject')
          # save, load, etc. will recognize custom object by name
  ```
  """

  def __init__(self, *args):
    self.custom_objects = args
    self.backup = None

  def __enter__(self):
    self.backup = _GLOBAL_CUSTOM_OBJECTS.copy()
    for objects in self.custom_objects:
      _GLOBAL_CUSTOM_OBJECTS.update(objects)
    return self

  def __exit__(self, *args, **kwargs):
    _GLOBAL_CUSTOM_OBJECTS.clear()
    _GLOBAL_CUSTOM_OBJECTS.update(self.backup)


def custom_object_scope(*args):
  """Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

  Convenience wrapper for `CustomObjectScope`.
  Code within a `with` statement will be able to access custom objects
  by name. Changes to global custom objects persist
  within the enclosing `with` statement. At end of the `with` statement,
  global custom objects are reverted to state
  at beginning of the `with` statement.

  Example:

  Consider a custom object `MyObject`

  ```python
      with custom_object_scope({'MyObject':MyObject}):
          layer = Dense(..., kernel_regularizer='MyObject')
          # save, load, etc. will recognize custom object by name
  ```

  Arguments:
      *args: Variable length list of dictionaries of name,
          class pairs to add to custom objects.

  Returns:
      Object of type `CustomObjectScope`.
  """
  return CustomObjectScope(*args)


def get_custom_objects():
  """Retrieves a live reference to the global dictionary of custom objects.

  Updating and clearing custom objects using `custom_object_scope`
  is preferred, but `get_custom_objects` can
  be used to directly access `_GLOBAL_CUSTOM_OBJECTS`.

  Example:

  ```python
      get_custom_objects().clear()
      get_custom_objects()['MyObject'] = MyObject
  ```

  Returns:
      Global dictionary of names to classes (`_GLOBAL_CUSTOM_OBJECTS`).
  """
  return _GLOBAL_CUSTOM_OBJECTS


def serialize_keras_object(instance):
  _, instance = tf_decorator.unwrap(instance)
  if instance is None:
    return None
  if hasattr(instance, 'get_config'):
    return {
        'class_name': instance.__class__.__name__,
        'config': instance.get_config()
    }
  if hasattr(instance, '__name__'):
    return instance.__name__
  else:
    raise ValueError('Cannot serialize', instance)


def deserialize_keras_object(identifier,
                             module_objects=None,
                             custom_objects=None,
                             printable_module_name='object'):
  if isinstance(identifier, dict):
    # In this case we are dealing with a Keras config dictionary.
    config = identifier
    if 'class_name' not in config or 'config' not in config:
      raise ValueError('Improper config format: ' + str(config))
    class_name = config['class_name']
    if custom_objects and class_name in custom_objects:
      cls = custom_objects[class_name]
    elif class_name in _GLOBAL_CUSTOM_OBJECTS:
      cls = _GLOBAL_CUSTOM_OBJECTS[class_name]
    else:
      module_objects = module_objects or {}
      cls = module_objects.get(class_name)
      if cls is None:
        raise ValueError('Unknown ' + printable_module_name + ': ' + class_name)
    if hasattr(cls, 'from_config'):
      arg_spec = tf_inspect.getargspec(cls.from_config)
      custom_objects = custom_objects or {}

      if 'custom_objects' in arg_spec.args:
        return cls.from_config(
            config['config'],
            custom_objects=dict(
                list(_GLOBAL_CUSTOM_OBJECTS.items()) +
                list(custom_objects.items())))
      with CustomObjectScope(custom_objects):
        return cls.from_config(config['config'])
    else:
      # Then `cls` may be a function returning a class.
      # in this case by convention `config` holds
      # the kwargs of the function.
      custom_objects = custom_objects or {}
      with CustomObjectScope(custom_objects):
        return cls(**config['config'])
  elif isinstance(identifier, six.string_types):
    function_name = identifier
    if custom_objects and function_name in custom_objects:
      fn = custom_objects.get(function_name)
    elif function_name in _GLOBAL_CUSTOM_OBJECTS:
      fn = _GLOBAL_CUSTOM_OBJECTS[function_name]
    else:
      fn = module_objects.get(function_name)
      if fn is None:
        raise ValueError('Unknown ' + printable_module_name + ':' +
                         function_name)
    return fn
  else:
    raise ValueError('Could not interpret serialized ' + printable_module_name +
                     ': ' + identifier)


def func_dump(func):
  """Serializes a user defined function.

  Arguments:
      func: the function to serialize.

  Returns:
      A tuple `(code, defaults, closure)`.
  """
  code = marshal.dumps(func.__code__).decode('raw_unicode_escape')
  defaults = func.__defaults__
  if func.__closure__:
    closure = tuple(c.cell_contents for c in func.__closure__)
  else:
    closure = None
  return code, defaults, closure


def func_load(code, defaults=None, closure=None, globs=None):
  """Deserializes a user defined function.

  Arguments:
      code: bytecode of the function.
      defaults: defaults of the function.
      closure: closure of the function.
      globs: dictionary of global objects.

  Returns:
      A function object.
  """
  if isinstance(code, (tuple, list)):  # unpack previous dump
    code, defaults, closure = code
    if isinstance(defaults, list):
      defaults = tuple(defaults)
  code = marshal.loads(code.encode('raw_unicode_escape'))
  if globs is None:
    globs = globals()
  return python_types.FunctionType(
      code, globs, name=code.co_name, argdefs=defaults, closure=closure)


class Progbar(object):
  """Displays a progress bar.

  Arguments:
      target: Total number of steps expected, None if unknown.
      interval: Minimum visual progress update interval (in seconds).
  """

  def __init__(self, target, width=30, verbose=1, interval=0.05):
    self.width = width
    if target is None:
      target = -1
    self.target = target
    self.sum_values = {}
    self.unique_values = []
    self.start = time.time()
    self.last_update = 0
    self.interval = interval
    self.total_width = 0
    self.seen_so_far = 0
    self.verbose = verbose

  def update(self, current, values=None, force=False):
    """Updates the progress bar.

    Arguments:
        current: Index of current step.
        values: List of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        force: Whether to force visual progress update.
    """
    values = values or []
    for k, v in values:
      if k not in self.sum_values:
        self.sum_values[k] = [
            v * (current - self.seen_so_far), current - self.seen_so_far
        ]
        self.unique_values.append(k)
      else:
        self.sum_values[k][0] += v * (current - self.seen_so_far)
        self.sum_values[k][1] += (current - self.seen_so_far)
    self.seen_so_far = current

    now = time.time()
    if self.verbose == 1:
      if not force and (now - self.last_update) < self.interval:
        return

      prev_total_width = self.total_width
      sys.stdout.write('\b' * prev_total_width)
      sys.stdout.write('\r')

      if self.target is not -1:
        numdigits = int(np.floor(np.log10(self.target))) + 1
        barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
        bar = barstr % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
          bar += ('=' * (prog_width - 1))
          if current < self.target:
            bar += '>'
          else:
            bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
        sys.stdout.write(bar)
        self.total_width = len(bar)

      if current:
        time_per_unit = (now - self.start) / current
      else:
        time_per_unit = 0
      eta = time_per_unit * (self.target - current)
      info = ''
      if current < self.target and self.target is not -1:
        info += ' - ETA: %ds' % eta
      else:
        info += ' - %ds' % (now - self.start)
      for k in self.unique_values:
        info += ' - %s:' % k
        if isinstance(self.sum_values[k], list):
          avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
          if abs(avg) > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        else:
          info += ' %s' % self.sum_values[k]

      self.total_width += len(info)
      if prev_total_width > self.total_width:
        info += ((prev_total_width - self.total_width) * ' ')

      sys.stdout.write(info)
      sys.stdout.flush()

      if current >= self.target:
        sys.stdout.write('\n')

    if self.verbose == 2:
      if current >= self.target:
        info = '%ds' % (now - self.start)
        for k in self.unique_values:
          info += ' - %s:' % k
          avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
          if avg > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        sys.stdout.write(info + '\n')

    self.last_update = now

  def add(self, n, values=None):
    self.update(self.seen_so_far + n, values)
