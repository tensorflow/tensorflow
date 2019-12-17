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

import binascii
import codecs
import marshal
import os
import re
import sys
import time
import types as python_types

import numpy as np
import six

from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export

_GLOBAL_CUSTOM_OBJECTS = {}
_GLOBAL_CUSTOM_NAMES = {}


@keras_export('keras.utils.CustomObjectScope')
class CustomObjectScope(object):
  """Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

  Code within a `with` statement will be able to access custom objects
  by name. Changes to global custom objects persist
  within the enclosing `with` statement. At end of the `with` statement,
  global custom objects are reverted to state
  at beginning of the `with` statement.

  Example:

  Consider a custom object `MyObject` (e.g. a class):

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


@keras_export('keras.utils.custom_object_scope')
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


@keras_export('keras.utils.get_custom_objects')
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


def serialize_keras_class_and_config(cls_name, cls_config):
  """Returns the serialization of the class with the given config."""
  return {'class_name': cls_name, 'config': cls_config}


@keras_export('keras.utils.register_keras_serializable')
def register_keras_serializable(package='Custom', name=None):
  """Registers an object with the Keras serialization framework.

  This decorator injects the decorated class or function into the Keras custom
  object dictionary, so that it can be serialized and deserialized without
  needing an entry in the user-provided custom object dict. It also injects a
  function that Keras will call to get the object's serializable string key.

  Note that to be serialized and deserialized, classes must implement the
  `get_config()` method. Functions do not have this requirement.

  The object will be registered under the key 'package>name' where `name`,
  defaults to the object name if not passed.

  Arguments:
    package: The package that this class belongs to.
    name: The name to serialize this class under in this package. If None, the
      class's name will be used.

  Returns:
    A decorator that registers the decorated class with the passed names.
  """

  def decorator(arg):
    """Registers a class with the Keras serialization framework."""
    class_name = name if name is not None else arg.__name__
    registered_name = package + '>' + class_name

    if tf_inspect.isclass(arg) and not hasattr(arg, 'get_config'):
      raise ValueError(
          'Cannot register a class that does not have a get_config() method.')

    if registered_name in _GLOBAL_CUSTOM_OBJECTS:
      raise ValueError(
          '%s has already been registered to %s' %
          (registered_name, _GLOBAL_CUSTOM_OBJECTS[registered_name]))

    if arg in _GLOBAL_CUSTOM_NAMES:
      raise ValueError('%s has already been registered to %s' %
                       (arg, _GLOBAL_CUSTOM_NAMES[arg]))
    _GLOBAL_CUSTOM_OBJECTS[registered_name] = arg
    _GLOBAL_CUSTOM_NAMES[arg] = registered_name

    return arg

  return decorator


def _get_name_or_custom_name(obj):
  if obj in _GLOBAL_CUSTOM_NAMES:
    return _GLOBAL_CUSTOM_NAMES[obj]
  else:
    return obj.__name__


@keras_export('keras.utils.serialize_keras_object')
def serialize_keras_object(instance):
  """Serialize Keras object into JSON."""
  _, instance = tf_decorator.unwrap(instance)
  if instance is None:
    return None

  if hasattr(instance, 'get_config'):
    config = instance.get_config()
    serialization_config = {}
    for key, item in config.items():
      if isinstance(item, six.string_types):
        serialization_config[key] = item
        continue

      # Any object of a different type needs to be converted to string or dict
      # for serialization (e.g. custom functions, custom classes)
      try:
        serialized_item = serialize_keras_object(item)
        if isinstance(serialized_item, dict) and not isinstance(item, dict):
          serialized_item['__passive_serialization__'] = True
        serialization_config[key] = serialized_item
      except ValueError:
        serialization_config[key] = item

    name = _get_name_or_custom_name(instance.__class__)
    return serialize_keras_class_and_config(name, serialization_config)
  if hasattr(instance, '__name__'):
    return _get_name_or_custom_name(instance)
  raise ValueError('Cannot serialize', instance)


def _get_custom_objects_by_name(item, custom_objects=None):
  """Returns the item if it is in either local or global custom objects."""
  if item in _GLOBAL_CUSTOM_OBJECTS:
    return _GLOBAL_CUSTOM_OBJECTS[item]
  elif custom_objects and item in custom_objects:
    return custom_objects[item]
  return None


def class_and_config_for_serialized_keras_object(
    config,
    module_objects=None,
    custom_objects=None,
    printable_module_name='object'):
  """Returns the class name and config for a serialized keras object."""
  if (not isinstance(config, dict) or 'class_name' not in config or
      'config' not in config):
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

  cls_config = config['config']
  deserialized_objects = {}
  for key, item in cls_config.items():
    if isinstance(item, dict) and '__passive_serialization__' in item:
      deserialized_objects[key] = deserialize_keras_object(
          item,
          module_objects=module_objects,
          custom_objects=custom_objects,
          printable_module_name='config_item')
    elif (isinstance(item, six.string_types) and
          tf_inspect.isfunction(
              _get_custom_objects_by_name(item, custom_objects))):
      # Handle custom functions here. When saving functions, we only save the
      # function's name as a string. If we find a matching string in the custom
      # objects during deserialization, we convert the string back to the
      # original function.
      # Note that a potential issue is that a string field could have a naming
      # conflict with a custom function name, but this should be a rare case.
      # This issue does not occur if a string field has a naming conflict with
      # a custom object, since the config of an object will always be a dict.
      deserialized_objects[key] = _get_custom_objects_by_name(
          item, custom_objects)
  for key, item in deserialized_objects.items():
    cls_config[key] = deserialized_objects[key]

  return (cls, cls_config)


@keras_export('keras.utils.deserialize_keras_object')
def deserialize_keras_object(identifier,
                             module_objects=None,
                             custom_objects=None,
                             printable_module_name='object'):
  if identifier is None:
    return None

  if isinstance(identifier, dict):
    # In this case we are dealing with a Keras config dictionary.
    config = identifier
    (cls, cls_config) = class_and_config_for_serialized_keras_object(
        config, module_objects, custom_objects, printable_module_name)

    if hasattr(cls, 'from_config'):
      arg_spec = tf_inspect.getfullargspec(cls.from_config)
      custom_objects = custom_objects or {}

      if 'custom_objects' in arg_spec.args:
        return cls.from_config(
            cls_config,
            custom_objects=dict(
                list(_GLOBAL_CUSTOM_OBJECTS.items()) +
                list(custom_objects.items())))
      with CustomObjectScope(custom_objects):
        return cls.from_config(cls_config)
    else:
      # Then `cls` may be a function returning a class.
      # in this case by convention `config` holds
      # the kwargs of the function.
      custom_objects = custom_objects or {}
      with CustomObjectScope(custom_objects):
        return cls(**cls_config)
  elif isinstance(identifier, six.string_types):
    object_name = identifier
    if custom_objects and object_name in custom_objects:
      obj = custom_objects.get(object_name)
    elif object_name in _GLOBAL_CUSTOM_OBJECTS:
      obj = _GLOBAL_CUSTOM_OBJECTS[object_name]
    else:
      obj = module_objects.get(object_name)
      if obj is None:
        raise ValueError('Unknown ' + printable_module_name + ':' + object_name)
    # Classes passed by name are instantiated with no args, functions are
    # returned as-is.
    if tf_inspect.isclass(obj):
      return obj()
    return obj
  elif tf_inspect.isfunction(identifier):
    # If a function has already been deserialized, return as is.
    return identifier
  else:
    raise ValueError('Could not interpret serialized %s: %s' %
                     (printable_module_name, identifier))


def func_dump(func):
  """Serializes a user defined function.

  Arguments:
      func: the function to serialize.

  Returns:
      A tuple `(code, defaults, closure)`.
  """
  if os.name == 'nt':
    raw_code = marshal.dumps(func.__code__).replace(b'\\', b'/')
    code = codecs.encode(raw_code, 'base64').decode('ascii')
  else:
    raw_code = marshal.dumps(func.__code__)
    code = codecs.encode(raw_code, 'base64').decode('ascii')
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

  def ensure_value_to_cell(value):
    """Ensures that a value is converted to a python cell object.

    Arguments:
        value: Any value that needs to be casted to the cell type

    Returns:
        A value wrapped as a cell object (see function "func_load")
    """
    def dummy_fn():
      # pylint: disable=pointless-statement
      value  # just access it so it gets captured in .__closure__

    cell_value = dummy_fn.__closure__[0]
    if not isinstance(value, type(cell_value)):
      return cell_value
    return value

  if closure is not None:
    closure = tuple(ensure_value_to_cell(_) for _ in closure)
  try:
    raw_code = codecs.decode(code.encode('ascii'), 'base64')
  except (UnicodeEncodeError, binascii.Error):
    raw_code = code.encode('raw_unicode_escape')
  code = marshal.loads(raw_code)
  if globs is None:
    globs = globals()
  return python_types.FunctionType(
      code, globs, name=code.co_name, argdefs=defaults, closure=closure)


def has_arg(fn, name, accept_all=False):
  """Checks if a callable accepts a given keyword argument.

  Arguments:
      fn: Callable to inspect.
      name: Check if `fn` can be called with `name` as a keyword argument.
      accept_all: What to return if there is no parameter called `name`
                  but the function accepts a `**kwargs` argument.

  Returns:
      bool, whether `fn` accepts a `name` keyword argument.
  """
  arg_spec = tf_inspect.getfullargspec(fn)
  if accept_all and arg_spec.varkw is not None:
    return True
  return name in arg_spec.args


@keras_export('keras.utils.Progbar')
class Progbar(object):
  """Displays a progress bar.

  Arguments:
      target: Total number of steps expected, None if unknown.
      width: Progress bar width on screen.
      verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over time. Metrics in this list
          will be displayed as-is. All others will be averaged
          by the progbar before display.
      interval: Minimum visual progress update interval (in seconds).
      unit_name: Display name for step counts (usually "step" or "sample").
  """

  def __init__(self, target, width=30, verbose=1, interval=0.05,
               stateful_metrics=None, unit_name='step'):
    self.target = target
    self.width = width
    self.verbose = verbose
    self.interval = interval
    self.unit_name = unit_name
    if stateful_metrics:
      self.stateful_metrics = set(stateful_metrics)
    else:
      self.stateful_metrics = set()

    self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                              sys.stdout.isatty()) or
                             'ipykernel' in sys.modules or
                             'posix' in sys.modules or
                             'PYCHARM_HOSTED' in os.environ)
    self._total_width = 0
    self._seen_so_far = 0
    # We use a dict + list to avoid garbage collection
    # issues found in OrderedDict
    self._values = {}
    self._values_order = []
    self._start = time.time()
    self._last_update = 0

  def update(self, current, values=None):
    """Updates the progress bar.

    Arguments:
        current: Index of current step.
        values: List of tuples:
            `(name, value_for_last_step)`.
            If `name` is in `stateful_metrics`,
            `value_for_last_step` will be displayed as-is.
            Else, an average of the metric over time will be displayed.
    """
    values = values or []
    for k, v in values:
      if k not in self._values_order:
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        # In the case that progress bar doesn't have a target value in the first
        # epoch, both on_batch_end and on_epoch_end will be called, which will
        # cause 'current' and 'self._seen_so_far' to have the same value. Force
        # the minimal value to 1 here, otherwise stateful_metric will be 0s.
        value_base = max(current - self._seen_so_far, 1)
        if k not in self._values:
          self._values[k] = [v * value_base, value_base]
        else:
          self._values[k][0] += v * value_base
          self._values[k][1] += value_base
      else:
        # Stateful metrics output a numeric value. This representation
        # means "take an average from a single value" but keeps the
        # numeric formatting.
        self._values[k] = [v, 1]
    self._seen_so_far = current

    now = time.time()
    info = ' - %.0fs' % (now - self._start)
    if self.verbose == 1:
      if (now - self._last_update < self.interval and
          self.target is not None and current < self.target):
        return

      prev_total_width = self._total_width
      if self._dynamic_display:
        sys.stdout.write('\b' * prev_total_width)
        sys.stdout.write('\r')
      else:
        sys.stdout.write('\n')

      if self.target is not None:
        numdigits = int(np.log10(self.target)) + 1
        bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
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
      else:
        bar = '%7d/Unknown' % current

      self._total_width = len(bar)
      sys.stdout.write(bar)

      if current:
        time_per_unit = (now - self._start) / current
      else:
        time_per_unit = 0
      if self.target is not None and current < self.target:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600,
                                         (eta % 3600) // 60,
                                         eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta

        info = ' - ETA: %s' % eta_format
      else:
        if time_per_unit >= 1 or time_per_unit == 0:
          info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
        else:
          info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)

      for k in self._values_order:
        info += ' - %s:' % k
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        else:
          info += ' %s' % self._values[k]

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += (' ' * (prev_total_width - self._total_width))

      if self.target is not None and current >= self.target:
        info += '\n'

      sys.stdout.write(info)
      sys.stdout.flush()

    elif self.verbose == 2:
      if self.target is not None and current >= self.target:
        numdigits = int(np.log10(self.target)) + 1
        count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
        info = count + info
        for k in self._values_order:
          info += ' - %s:' % k
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        info += '\n'

        sys.stdout.write(info)
        sys.stdout.flush()

    self._last_update = now

  def add(self, n, values=None):
    self.update(self._seen_so_far + n, values)


def make_batches(size, batch_size):
  """Returns a list of batch indices (tuples of indices).

  Arguments:
      size: Integer, total size of the data to slice into batches.
      batch_size: Integer, batch size.

  Returns:
      A list of tuples of array indices.
  """
  num_batches = int(np.ceil(size / float(batch_size)))
  return [(i * batch_size, min(size, (i + 1) * batch_size))
          for i in range(0, num_batches)]


def slice_arrays(arrays, start=None, stop=None):
  """Slice an array or list of arrays.

  This takes an array-like, or a list of
  array-likes, and outputs:
      - arrays[start:stop] if `arrays` is an array-like
      - [x[start:stop] for x in arrays] if `arrays` is a list

  Can also work on list/array of indices: `slice_arrays(x, indices)`

  Arguments:
      arrays: Single array or list of arrays.
      start: can be an integer index (start index)
          or a list/array of indices
      stop: integer (stop index); should be None if
          `start` was a list.

  Returns:
      A slice of the array(s).

  Raises:
      ValueError: If the value of start is a list and stop is not None.
  """
  if arrays is None:
    return [None]
  if isinstance(start, list) and stop is not None:
    raise ValueError('The stop argument has to be None if the value of start '
                     'is a list.')
  elif isinstance(arrays, list):
    if hasattr(start, '__len__'):
      # hdf5 datasets only support list objects as indices
      if hasattr(start, 'shape'):
        start = start.tolist()
      return [None if x is None else x[start] for x in arrays]
    return [
        None if x is None else
        None if not hasattr(x, '__getitem__') else x[start:stop] for x in arrays
    ]
  else:
    if hasattr(start, '__len__'):
      if hasattr(start, 'shape'):
        start = start.tolist()
      return arrays[start]
    if hasattr(start, '__getitem__'):
      return arrays[start:stop]
    return [None]


def to_list(x):
  """Normalizes a list/tensor into a list.

  If a tensor is passed, we return
  a list of size 1 containing the tensor.

  Arguments:
      x: target object to be normalized.

  Returns:
      A list.
  """
  if isinstance(x, list):
    return x
  return [x]


def object_list_uid(object_list):
  """Creates a single string from object ids."""
  object_list = nest.flatten(object_list)
  return ', '.join(str(abs(id(x))) for x in object_list)


def to_snake_case(name):
  intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
  insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
  # If the class is private the name starts with "_" which is not secure
  # for creating scopes. We prefix the name with "private" in this case.
  if insecure[0] != '_':
    return insecure
  return 'private' + insecure


def is_all_none(structure):
  iterable = nest.flatten(structure)
  # We cannot use Python's `any` because the iterable may return Tensors.
  for element in iterable:
    if element is not None:
      return False
  return True


def check_for_unexpected_keys(name, input_dict, expected_values):
  unknown = set(input_dict.keys()).difference(expected_values)
  if unknown:
    raise ValueError('Unknown entries in {} dictionary: {}. Only expected '
                     'following keys: {}'.format(name, list(unknown),
                                                 expected_values))


def validate_kwargs(kwargs, allowed_kwargs,
                    error_message='Keyword argument not understood:'):
  """Checks that all keyword arguments are in the set of allowed keys."""
  for kwarg in kwargs:
    if kwarg not in allowed_kwargs:
      raise TypeError(error_message, kwarg)
