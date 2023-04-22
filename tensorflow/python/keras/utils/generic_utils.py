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

import binascii
import codecs
import importlib
import marshal
import os
import re
import sys
import threading
import time
import types as python_types
import warnings
import weakref

import numpy as np

from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import keras_export

_GLOBAL_CUSTOM_OBJECTS = {}
_GLOBAL_CUSTOM_NAMES = {}

# Flag that determines whether to skip the NotImplementedError when calling
# get_config in custom models and layers. This is only enabled when saving to
# SavedModel, when the config isn't required.
_SKIP_FAILED_SERIALIZATION = False
# If a layer does not have a defined config, then the returned config will be a
# dictionary with the below key.
_LAYER_UNDEFINED_CONFIG_KEY = 'layer was saved without config'


@keras_export('keras.utils.custom_object_scope',  # pylint: disable=g-classes-have-attributes
              'keras.utils.CustomObjectScope')
class CustomObjectScope(object):
  """Exposes custom classes/functions to Keras deserialization internals.

  Under a scope `with custom_object_scope(objects_dict)`, Keras methods such
  as `tf.keras.models.load_model` or `tf.keras.models.model_from_config`
  will be able to deserialize any custom object referenced by a
  saved config (e.g. a custom layer or metric).

  Example:

  Consider a custom regularizer `my_regularizer`:

  ```python
  layer = Dense(3, kernel_regularizer=my_regularizer)
  config = layer.get_config()  # Config contains a reference to `my_regularizer`
  ...
  # Later:
  with custom_object_scope({'my_regularizer': my_regularizer}):
    layer = Dense.from_config(config)
  ```

  Args:
      *args: Dictionary or dictionaries of `{name: object}` pairs.
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


@keras_export('keras.utils.get_custom_objects')
def get_custom_objects():
  """Retrieves a live reference to the global dictionary of custom objects.

  Updating and clearing custom objects using `custom_object_scope`
  is preferred, but `get_custom_objects` can
  be used to directly access the current collection of custom objects.

  Example:

  ```python
  get_custom_objects().clear()
  get_custom_objects()['MyObject'] = MyObject
  ```

  Returns:
      Global dictionary of names to classes (`_GLOBAL_CUSTOM_OBJECTS`).
  """
  return _GLOBAL_CUSTOM_OBJECTS


# Store a unique, per-object ID for shared objects.
#
# We store a unique ID for each object so that we may, at loading time,
# re-create the network properly.  Without this ID, we would have no way of
# determining whether a config is a description of a new object that
# should be created or is merely a reference to an already-created object.
SHARED_OBJECT_KEY = 'shared_object_id'


SHARED_OBJECT_DISABLED = threading.local()
SHARED_OBJECT_LOADING = threading.local()
SHARED_OBJECT_SAVING = threading.local()


# Attributes on the threadlocal variable must be set per-thread, thus we
# cannot initialize these globally. Instead, we have accessor functions with
# default values.
def _shared_object_disabled():
  """Get whether shared object handling is disabled in a threadsafe manner."""
  return getattr(SHARED_OBJECT_DISABLED, 'disabled', False)


def _shared_object_loading_scope():
  """Get the current shared object saving scope in a threadsafe manner."""
  return getattr(SHARED_OBJECT_LOADING, 'scope', NoopLoadingScope())


def _shared_object_saving_scope():
  """Get the current shared object saving scope in a threadsafe manner."""
  return getattr(SHARED_OBJECT_SAVING, 'scope', None)


class DisableSharedObjectScope(object):
  """A context manager for disabling handling of shared objects.

  Disables shared object handling for both saving and loading.

  Created primarily for use with `clone_model`, which does extra surgery that
  is incompatible with shared objects.
  """

  def __enter__(self):
    SHARED_OBJECT_DISABLED.disabled = True
    self._orig_loading_scope = _shared_object_loading_scope()
    self._orig_saving_scope = _shared_object_saving_scope()

  def __exit__(self, *args, **kwargs):
    SHARED_OBJECT_DISABLED.disabled = False
    SHARED_OBJECT_LOADING.scope = self._orig_loading_scope
    SHARED_OBJECT_SAVING.scope = self._orig_saving_scope


class NoopLoadingScope(object):
  """The default shared object loading scope. It does nothing.

  Created to simplify serialization code that doesn't care about shared objects
  (e.g. when serializing a single object).
  """

  def get(self, unused_object_id):
    return None

  def set(self, object_id, obj):
    pass


class SharedObjectLoadingScope(object):
  """A context manager for keeping track of loaded objects.

  During the deserialization process, we may come across objects that are
  shared across multiple layers. In order to accurately restore the network
  structure to its original state, `SharedObjectLoadingScope` allows us to
  re-use shared objects rather than cloning them.
  """

  def __enter__(self):
    if _shared_object_disabled():
      return NoopLoadingScope()

    global SHARED_OBJECT_LOADING
    SHARED_OBJECT_LOADING.scope = self
    self._obj_ids_to_obj = {}
    return self

  def get(self, object_id):
    """Given a shared object ID, returns a previously instantiated object.

    Args:
      object_id: shared object ID to use when attempting to find already-loaded
        object.

    Returns:
      The object, if we've seen this ID before. Else, `None`.
    """
    # Explicitly check for `None` internally to make external calling code a
    # bit cleaner.
    if object_id is None:
      return
    return self._obj_ids_to_obj.get(object_id)

  def set(self, object_id, obj):
    """Stores an instantiated object for future lookup and sharing."""
    if object_id is None:
      return
    self._obj_ids_to_obj[object_id] = obj

  def __exit__(self, *args, **kwargs):
    global SHARED_OBJECT_LOADING
    SHARED_OBJECT_LOADING.scope = NoopLoadingScope()


class SharedObjectConfig(dict):
  """A configuration container that keeps track of references.

  `SharedObjectConfig` will automatically attach a shared object ID to any
  configs which are referenced more than once, allowing for proper shared
  object reconstruction at load time.

  In most cases, it would be more proper to subclass something like
  `collections.UserDict` or `collections.Mapping` rather than `dict` directly.
  Unfortunately, python's json encoder does not support `Mapping`s. This is
  important functionality to retain, since we are dealing with serialization.

  We should be safe to subclass `dict` here, since we aren't actually
  overriding any core methods, only augmenting with a new one for reference
  counting.
  """

  def __init__(self, base_config, object_id, **kwargs):
    self.ref_count = 1
    self.object_id = object_id
    super(SharedObjectConfig, self).__init__(base_config, **kwargs)

  def increment_ref_count(self):
    # As soon as we've seen the object more than once, we want to attach the
    # shared object ID. This allows us to only attach the shared object ID when
    # it's strictly necessary, making backwards compatibility breakage less
    # likely.
    if self.ref_count == 1:
      self[SHARED_OBJECT_KEY] = self.object_id
    self.ref_count += 1


class SharedObjectSavingScope(object):
  """Keeps track of shared object configs when serializing."""

  def __enter__(self):
    if _shared_object_disabled():
      return None

    global SHARED_OBJECT_SAVING

    # Serialization can happen at a number of layers for a number of reasons.
    # We may end up with a case where we're opening a saving scope within
    # another saving scope. In that case, we'd like to use the outermost scope
    # available and ignore inner scopes, since there is not (yet) a reasonable
    # use case for having these nested and distinct.
    if _shared_object_saving_scope() is not None:
      self._passthrough = True
      return _shared_object_saving_scope()
    else:
      self._passthrough = False

    SHARED_OBJECT_SAVING.scope = self
    self._shared_objects_config = weakref.WeakKeyDictionary()
    self._next_id = 0
    return self

  def get_config(self, obj):
    """Gets a `SharedObjectConfig` if one has already been seen for `obj`.

    Args:
      obj: The object for which to retrieve the `SharedObjectConfig`.

    Returns:
      The SharedObjectConfig for a given object, if already seen. Else,
        `None`.
    """
    try:
      shared_object_config = self._shared_objects_config[obj]
    except (TypeError, KeyError):
      # If the object is unhashable (e.g. a subclass of `AbstractBaseClass`
      # that has not overridden `__hash__`), a `TypeError` will be thrown.
      # We'll just continue on without shared object support.
      return None
    shared_object_config.increment_ref_count()
    return shared_object_config

  def create_config(self, base_config, obj):
    """Create a new SharedObjectConfig for a given object."""
    shared_object_config = SharedObjectConfig(base_config, self._next_id)
    self._next_id += 1
    try:
      self._shared_objects_config[obj] = shared_object_config
    except TypeError:
      # If the object is unhashable (e.g. a subclass of `AbstractBaseClass`
      # that has not overridden `__hash__`), a `TypeError` will be thrown.
      # We'll just continue on without shared object support.
      pass
    return shared_object_config

  def __exit__(self, *args, **kwargs):
    if not getattr(self, '_passthrough', False):
      global SHARED_OBJECT_SAVING
      SHARED_OBJECT_SAVING.scope = None


def serialize_keras_class_and_config(
    cls_name, cls_config, obj=None, shared_object_id=None):
  """Returns the serialization of the class with the given config."""
  base_config = {'class_name': cls_name, 'config': cls_config}

  # We call `serialize_keras_class_and_config` for some branches of the load
  # path. In that case, we may already have a shared object ID we'd like to
  # retain.
  if shared_object_id is not None:
    base_config[SHARED_OBJECT_KEY] = shared_object_id

  # If we have an active `SharedObjectSavingScope`, check whether we've already
  # serialized this config. If so, just use that config. This will store an
  # extra ID field in the config, allowing us to re-create the shared object
  # relationship at load time.
  if _shared_object_saving_scope() is not None and obj is not None:
    shared_object_config = _shared_object_saving_scope().get_config(obj)
    if shared_object_config is None:
      return _shared_object_saving_scope().create_config(base_config, obj)
    return shared_object_config

  return base_config


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

  Args:
    package: The package that this class belongs to.
    name: The name to serialize this class under in this package. If None, the
      class' name will be used.

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


@keras_export('keras.utils.get_registered_name')
def get_registered_name(obj):
  """Returns the name registered to an object within the Keras framework.

  This function is part of the Keras serialization and deserialization
  framework. It maps objects to the string names associated with those objects
  for serialization/deserialization.

  Args:
    obj: The object to look up.

  Returns:
    The name associated with the object, or the default Python name if the
      object is not registered.
  """
  if obj in _GLOBAL_CUSTOM_NAMES:
    return _GLOBAL_CUSTOM_NAMES[obj]
  else:
    return obj.__name__


@tf_contextlib.contextmanager
def skip_failed_serialization():
  global _SKIP_FAILED_SERIALIZATION
  prev = _SKIP_FAILED_SERIALIZATION
  try:
    _SKIP_FAILED_SERIALIZATION = True
    yield
  finally:
    _SKIP_FAILED_SERIALIZATION = prev


@keras_export('keras.utils.get_registered_object')
def get_registered_object(name, custom_objects=None, module_objects=None):
  """Returns the class associated with `name` if it is registered with Keras.

  This function is part of the Keras serialization and deserialization
  framework. It maps strings to the objects associated with them for
  serialization/deserialization.

  Example:
  ```
  def from_config(cls, config, custom_objects=None):
    if 'my_custom_object_name' in config:
      config['hidden_cls'] = tf.keras.utils.get_registered_object(
          config['my_custom_object_name'], custom_objects=custom_objects)
  ```

  Args:
    name: The name to look up.
    custom_objects: A dictionary of custom objects to look the name up in.
      Generally, custom_objects is provided by the user.
    module_objects: A dictionary of custom objects to look the name up in.
      Generally, module_objects is provided by midlevel library implementers.

  Returns:
    An instantiable class associated with 'name', or None if no such class
      exists.
  """
  if name in _GLOBAL_CUSTOM_OBJECTS:
    return _GLOBAL_CUSTOM_OBJECTS[name]
  elif custom_objects and name in custom_objects:
    return custom_objects[name]
  elif module_objects and name in module_objects:
    return module_objects[name]
  return None


# pylint: disable=g-bad-exception-name
class CustomMaskWarning(Warning):
  pass
# pylint: enable=g-bad-exception-name


@keras_export('keras.utils.serialize_keras_object')
def serialize_keras_object(instance):
  """Serialize a Keras object into a JSON-compatible representation.

  Calls to `serialize_keras_object` while underneath the
  `SharedObjectSavingScope` context manager will cause any objects re-used
  across multiple layers to be saved with a special shared object ID. This
  allows the network to be re-created properly during deserialization.

  Args:
    instance: The object to serialize.

  Returns:
    A dict-like, JSON-compatible representation of the object's config.
  """
  _, instance = tf_decorator.unwrap(instance)
  if instance is None:
    return None

  # pylint: disable=protected-access
  #
  # For v1 layers, checking supports_masking is not enough. We have to also
  # check whether compute_mask has been overridden.
  supports_masking = (getattr(instance, 'supports_masking', False)
                      or (hasattr(instance, 'compute_mask')
                          and not is_default(instance.compute_mask)))
  if supports_masking and is_default(instance.get_config):
    warnings.warn('Custom mask layers require a config and must override '
                  'get_config. When loading, the custom mask layer must be '
                  'passed to the custom_objects argument.',
                  category=CustomMaskWarning)
  # pylint: enable=protected-access

  if hasattr(instance, 'get_config'):
    name = get_registered_name(instance.__class__)
    try:
      config = instance.get_config()
    except NotImplementedError as e:
      if _SKIP_FAILED_SERIALIZATION:
        return serialize_keras_class_and_config(
            name, {_LAYER_UNDEFINED_CONFIG_KEY: True})
      raise e
    serialization_config = {}
    for key, item in config.items():
      if isinstance(item, str):
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

    name = get_registered_name(instance.__class__)
    return serialize_keras_class_and_config(
        name, serialization_config, instance)
  if hasattr(instance, '__name__'):
    return get_registered_name(instance)
  raise ValueError('Cannot serialize', instance)


def get_custom_objects_by_name(item, custom_objects=None):
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
  if (not isinstance(config, dict)
      or 'class_name' not in config
      or 'config' not in config):
    raise ValueError('Improper config format: ' + str(config))

  class_name = config['class_name']
  cls = get_registered_object(class_name, custom_objects, module_objects)
  if cls is None:
    raise ValueError(
        'Unknown {}: {}. Please ensure this object is '
        'passed to the `custom_objects` argument. See '
        'https://www.tensorflow.org/guide/keras/save_and_serialize'
        '#registering_the_custom_object for details.'
        .format(printable_module_name, class_name))

  cls_config = config['config']
  # Check if `cls_config` is a list. If it is a list, return the class and the
  # associated class configs for recursively deserialization. This case will
  # happen on the old version of sequential model (e.g. `keras_version` ==
  # "2.0.6"), which is serialized in a different structure, for example
  # "{'class_name': 'Sequential',
  #   'config': [{'class_name': 'Embedding', 'config': ...}, {}, ...]}".
  if isinstance(cls_config, list):
    return (cls, cls_config)

  deserialized_objects = {}
  for key, item in cls_config.items():
    if key == 'name':
      # Assume that the value of 'name' is a string that should not be
      # deserialized as a function. This avoids the corner case where
      # cls_config['name'] has an identical name to a custom function and
      # gets converted into that function.
      deserialized_objects[key] = item
    elif isinstance(item, dict) and '__passive_serialization__' in item:
      deserialized_objects[key] = deserialize_keras_object(
          item,
          module_objects=module_objects,
          custom_objects=custom_objects,
          printable_module_name='config_item')
    # TODO(momernick): Should this also have 'module_objects'?
    elif (isinstance(item, str) and
          tf_inspect.isfunction(get_registered_object(item, custom_objects))):
      # Handle custom functions here. When saving functions, we only save the
      # function's name as a string. If we find a matching string in the custom
      # objects during deserialization, we convert the string back to the
      # original function.
      # Note that a potential issue is that a string field could have a naming
      # conflict with a custom function name, but this should be a rare case.
      # This issue does not occur if a string field has a naming conflict with
      # a custom object, since the config of an object will always be a dict.
      deserialized_objects[key] = get_registered_object(item, custom_objects)
  for key, item in deserialized_objects.items():
    cls_config[key] = deserialized_objects[key]

  return (cls, cls_config)


@keras_export('keras.utils.deserialize_keras_object')
def deserialize_keras_object(identifier,
                             module_objects=None,
                             custom_objects=None,
                             printable_module_name='object'):
  """Turns the serialized form of a Keras object back into an actual object.

  This function is for mid-level library implementers rather than end users.

  Importantly, this utility requires you to provide the dict of `module_objects`
  to use for looking up the object config; this is not populated by default.
  If you need a deserialization utility that has preexisting knowledge of
  built-in Keras objects, use e.g. `keras.layers.deserialize(config)`,
  `keras.metrics.deserialize(config)`, etc.

  Calling `deserialize_keras_object` while underneath the
  `SharedObjectLoadingScope` context manager will cause any already-seen shared
  objects to be returned as-is rather than creating a new object.

  Args:
    identifier: the serialized form of the object.
    module_objects: A dictionary of built-in objects to look the name up in.
      Generally, `module_objects` is provided by midlevel library implementers.
    custom_objects: A dictionary of custom objects to look the name up in.
      Generally, `custom_objects` is provided by the end user.
    printable_module_name: A human-readable string representing the type of the
      object. Printed in case of exception.

  Returns:
    The deserialized object.

  Example:

  A mid-level library implementer might want to implement a utility for
  retrieving an object from its config, as such:

  ```python
  def deserialize(config, custom_objects=None):
     return deserialize_keras_object(
       identifier,
       module_objects=globals(),
       custom_objects=custom_objects,
       name="MyObjectType",
     )
  ```

  This is how e.g. `keras.layers.deserialize()` is implemented.
  """
  if identifier is None:
    return None

  if isinstance(identifier, dict):
    # In this case we are dealing with a Keras config dictionary.
    config = identifier
    (cls, cls_config) = class_and_config_for_serialized_keras_object(
        config, module_objects, custom_objects, printable_module_name)

    # If this object has already been loaded (i.e. it's shared between multiple
    # objects), return the already-loaded object.
    shared_object_id = config.get(SHARED_OBJECT_KEY)
    shared_object = _shared_object_loading_scope().get(shared_object_id)  # pylint: disable=assignment-from-none
    if shared_object is not None:
      return shared_object

    if hasattr(cls, 'from_config'):
      arg_spec = tf_inspect.getfullargspec(cls.from_config)
      custom_objects = custom_objects or {}

      if 'custom_objects' in arg_spec.args:
        deserialized_obj = cls.from_config(
            cls_config,
            custom_objects=dict(
                list(_GLOBAL_CUSTOM_OBJECTS.items()) +
                list(custom_objects.items())))
      else:
        with CustomObjectScope(custom_objects):
          deserialized_obj = cls.from_config(cls_config)
    else:
      # Then `cls` may be a function returning a class.
      # in this case by convention `config` holds
      # the kwargs of the function.
      custom_objects = custom_objects or {}
      with CustomObjectScope(custom_objects):
        deserialized_obj = cls(**cls_config)

    # Add object to shared objects, in case we find it referenced again.
    _shared_object_loading_scope().set(shared_object_id, deserialized_obj)

    return deserialized_obj

  elif isinstance(identifier, str):
    object_name = identifier
    if custom_objects and object_name in custom_objects:
      obj = custom_objects.get(object_name)
    elif object_name in _GLOBAL_CUSTOM_OBJECTS:
      obj = _GLOBAL_CUSTOM_OBJECTS[object_name]
    else:
      obj = module_objects.get(object_name)
      if obj is None:
        raise ValueError(
            'Unknown {}: {}. Please ensure this object is '
            'passed to the `custom_objects` argument. See '
            'https://www.tensorflow.org/guide/keras/save_and_serialize'
            '#registering_the_custom_object for details.'
            .format(printable_module_name, object_name))

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

  Args:
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

  Args:
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

    Args:
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

  Args:
      fn: Callable to inspect.
      name: Check if `fn` can be called with `name` as a keyword argument.
      accept_all: What to return if there is no parameter called `name` but the
        function accepts a `**kwargs` argument.

  Returns:
      bool, whether `fn` accepts a `name` keyword argument.
  """
  arg_spec = tf_inspect.getfullargspec(fn)
  if accept_all and arg_spec.varkw is not None:
    return True
  return name in arg_spec.args or name in arg_spec.kwonlyargs


@keras_export('keras.utils.Progbar')
class Progbar(object):
  """Displays a progress bar.

  Args:
      target: Total number of steps expected, None if unknown.
      width: Progress bar width on screen.
      verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
      stateful_metrics: Iterable of string names of metrics that should *not* be
        averaged over time. Metrics in this list will be displayed as-is. All
        others will be averaged by the progbar before display.
      interval: Minimum visual progress update interval (in seconds).
      unit_name: Display name for step counts (usually "step" or "sample").
  """

  def __init__(self,
               target,
               width=30,
               verbose=1,
               interval=0.05,
               stateful_metrics=None,
               unit_name='step'):
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

    self._time_after_first_step = None

  def update(self, current, values=None, finalize=None):
    """Updates the progress bar.

    Args:
        current: Index of current step.
        values: List of tuples: `(name, value_for_last_step)`. If `name` is in
          `stateful_metrics`, `value_for_last_step` will be displayed as-is.
          Else, an average of the metric over time will be displayed.
        finalize: Whether this is the last update for the progress bar. If
          `None`, defaults to `current >= self.target`.
    """
    if finalize is None:
      if self.target is None:
        finalize = False
      else:
        finalize = current >= self.target

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
      if now - self._last_update < self.interval and not finalize:
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

      time_per_unit = self._estimate_step_duration(current, now)

      if self.target is None or finalize:
        if time_per_unit >= 1 or time_per_unit == 0:
          info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
        else:
          info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)
      else:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600,
                                         (eta % 3600) // 60, eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta

        info = ' - ETA: %s' % eta_format

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

      if finalize:
        info += '\n'

      sys.stdout.write(info)
      sys.stdout.flush()

    elif self.verbose == 2:
      if finalize:
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

  def _estimate_step_duration(self, current, now):
    """Estimate the duration of a single step.

    Given the step number `current` and the corresponding time `now`
    this function returns an estimate for how long a single step
    takes. If this is called before one step has been completed
    (i.e. `current == 0`) then zero is given as an estimate. The duration
    estimate ignores the duration of the (assumed to be non-representative)
    first step for estimates when more steps are available (i.e. `current>1`).
    Args:
      current: Index of current step.
      now: The current time.
    Returns: Estimate of the duration of a single step.
    """
    if current:
      # there are a few special scenarios here:
      # 1) somebody is calling the progress bar without ever supplying step 1
      # 2) somebody is calling the progress bar and supplies step one mulitple
      #    times, e.g. as part of a finalizing call
      # in these cases, we just fall back to the simple calculation
      if self._time_after_first_step is not None and current > 1:
        time_per_unit = (now - self._time_after_first_step) / (current - 1)
      else:
        time_per_unit = (now - self._start) / current

      if current == 1:
        self._time_after_first_step = now
      return time_per_unit
    else:
      return 0

  def _update_stateful_metrics(self, stateful_metrics):
    self.stateful_metrics = self.stateful_metrics.union(stateful_metrics)


def make_batches(size, batch_size):
  """Returns a list of batch indices (tuples of indices).

  Args:
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

  Args:
      arrays: Single array or list of arrays.
      start: can be an integer index (start index) or a list/array of indices
      stop: integer (stop index); should be None if `start` was a list.

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

  Args:
      x: target object to be normalized.

  Returns:
      A list.
  """
  if isinstance(x, list):
    return x
  return [x]


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


def validate_kwargs(kwargs,
                    allowed_kwargs,
                    error_message='Keyword argument not understood:'):
  """Checks that all keyword arguments are in the set of allowed keys."""
  for kwarg in kwargs:
    if kwarg not in allowed_kwargs:
      raise TypeError(error_message, kwarg)


def validate_config(config):
  """Determines whether config appears to be a valid layer config."""
  return isinstance(config, dict) and _LAYER_UNDEFINED_CONFIG_KEY not in config


def default(method):
  """Decorates a method to detect overrides in subclasses."""
  method._is_default = True  # pylint: disable=protected-access
  return method


def is_default(method):
  """Check if a method is decorated with the `default` wrapper."""
  return getattr(method, '_is_default', False)


def populate_dict_with_module_objects(target_dict, modules, obj_filter):
  for module in modules:
    for name in dir(module):
      obj = getattr(module, name)
      if obj_filter(obj):
        target_dict[name] = obj


class LazyLoader(python_types.ModuleType):
  """Lazily import a module, mainly to avoid pulling in large dependencies."""

  def __init__(self, local_name, parent_module_globals, name):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    super(LazyLoader, self).__init__(name)

  def _load(self):
    """Load the module and insert it into the parent's globals."""
    # Import the target module and insert it into the parent's namespace
    module = importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module
    # Update this object's dict so that if someone keeps a reference to the
    #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
    #   that fail).
    self.__dict__.update(module.__dict__)
    return module

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)


# Aliases

custom_object_scope = CustomObjectScope  # pylint: disable=invalid-name
