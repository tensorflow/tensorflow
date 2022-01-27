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
"""Serialization Registration for SavedModel.

revived_types registration will be migrated to this infrastructure.

See the Advanced saving section in go/savedmodel-configurability.
This API is approved for TF internal use only.
"""
import collections
import re

from tensorflow.python.util import tf_inspect


# Only allow valid file/directory characters
_VALID_REGISTERED_NAME = re.compile(r"^[a-zA-Z0-9._-]+$")


class _PredicateRegistry(object):
  """Registry with predicate-based lookup.

  See the documentation for `register_checkpoint_saver` and
  `register_serializable` for reasons why predicates are required over a
  class-based registry.

  Since this class is used for global registries, each object must be registered
  to unique names (an error is raised if there are naming conflicts). The lookup
  searches the predicates in reverse order, so that later-registered predicates
  are executed first.
  """
  __slots__ = ("_registry_name", "_registered_map", "_registered_predicates",
               "_registered_names")

  def __init__(self, name):
    self._registry_name = name
    # Maps registered name -> object
    self._registered_map = {}
    # Maps registered name -> predicate
    self._registered_predicates = {}
    # Stores names in the order of registration
    self._registered_names = []

  @property
  def name(self):
    return self._registry_name

  def register(self, package, name, predicate, candidate):
    """Registers a candidate object under the package, name and predicate."""
    if not isinstance(package, str) or not isinstance(name, str):
      raise TypeError(
          f"The package and name registered to a {self.name} must be strings, "
          f"got: package={type(package)}, name={type(name)}")
    if not callable(predicate):
      raise TypeError(
          f"The predicate registered to a {self.name} must be callable, "
          f"got: {type(predicate)}")
    registered_name = package + "." + name
    if not _VALID_REGISTERED_NAME.match(registered_name):
      raise ValueError(
          f"Invalid registered {self.name}. Please check that the package and "
          f"name follow the regex '{_VALID_REGISTERED_NAME.pattern}': "
          f"(package='{package}', name='{name}')")
    if registered_name in self._registered_map:
      raise ValueError(
          f"The name '{registered_name}' has already been registered to a "
          f"{self.name}. Found: {self._registered_map[registered_name]}")

    self._registered_map[registered_name] = candidate
    self._registered_predicates[registered_name] = predicate
    self._registered_names.append(registered_name)

  def lookup(self, obj):
    """Looks up the registered object using the predicate.

    Args:
      obj: Object to pass to each of the registered predicates to look up the
        registered object.
    Returns:
      The object registered with the first passing predicate.
    Raises:
      LookupError if the object does not match any of the predicate functions.
    """
    return self._registered_map[self.get_registered_name(obj)]

  def name_lookup(self, registered_name):
    """Looks up the registered object using the registered name."""
    try:
      return self._registered_map[registered_name]
    except KeyError:
      raise LookupError(f"The {self.name} registry does not have name "
                        f"'{registered_name}' registered.")

  def get_registered_name(self, obj):
    for registered_name in reversed(self._registered_names):
      predicate = self._registered_predicates[registered_name]
      if predicate(obj):
        return registered_name
    raise LookupError(f"Could not find matching {self.name} for {type(obj)}.")

  def get_predicate(self, registered_name):
    try:
      return self._registered_predicates[registered_name]
    except KeyError:
      raise LookupError(f"The {self.name} registry does not have name "
                        f"'{registered_name}' registered.")

  def get_registrations(self):
    return self._registered_predicates

_class_registry = _PredicateRegistry("serializable class")
_saver_registry = _PredicateRegistry("checkpoint saver")


def get_registered_class_name(obj):
  try:
    return _class_registry.get_registered_name(obj)
  except LookupError:
    return None


def get_registered_class(registered_name):
  try:
    return _class_registry.name_lookup(registered_name)
  except LookupError:
    return None


def register_serializable(package="Custom", name=None, predicate=None):  # pylint: disable=unused-argument
  """Decorator for registering a serializable class.

  THIS METHOD IS STILL EXPERIMENTAL AND MAY CHANGE AT ANY TIME.

  Registered classes will be saved with a name generated by combining the
  `package` and `name` arguments. When loading a SavedModel, modules saved with
  this registered name will be created using the `_deserialize_from_proto`
  method.

  By default, only direct instances of the registered class will be saved/
  restored with the `serialize_from_proto`/`deserialize_from_proto` methods. To
  extend the registration to subclasses, use the `predicate argument`:

  ```python
  class A(tf.Module):
    pass

  register_serializable(
      package="Example", predicate=lambda obj: isinstance(obj, A))(A)
  ```

  Args:
    package: The package that this class belongs to.
    name: The name to serialize this class under in this package. If None, the
      class's name will be used.
    predicate: An optional function that takes a single Trackable argument, and
      determines whether that object should be serialized with this `package`
      and `name`. The default predicate checks whether the object's type exactly
      matches the registered class. Predicates are executed in the reverse order
      that they are added (later registrations are checked first).

  Returns:
    A decorator that registers the decorated class with the passed names and
    predicate.
  """
  def decorator(arg):
    """Registers a class with the serialization framework."""
    nonlocal predicate
    if not tf_inspect.isclass(arg):
      raise TypeError("Registered serializable must be a class: {}".format(arg))

    class_name = name if name is not None else arg.__name__
    if predicate is None:
      predicate = lambda x: isinstance(x, arg)
    _class_registry.register(package, class_name, predicate, arg)
    return arg

  return decorator


RegisteredSaver = collections.namedtuple(
    "RegisteredSaver", ["name", "predicate", "save_fn", "restore_fn"])
_REGISTERED_SAVERS = {}
_REGISTERED_SAVER_NAMES = []  # Stores names in the order of registration


def register_checkpoint_saver(package="Custom",
                              name=None,
                              predicate=None,
                              save_fn=None,
                              restore_fn=None):
  """Registers functions which checkpoints & restores objects with custom steps.

  If you have a class that requires complicated coordination between multiple
  objects when checkpointing, then you will need to register a custom saver
  and restore function. An example of this is a custom Variable class that
  splits the variable across different objects and devices, and needs to write
  checkpoints that are compatible with different configurations of devices.

  The registered save and restore functions are used in checkpoints and
  SavedModel.

  Please make sure you are familiar with the concepts in the [Checkpointing
  guide](https://www.tensorflow.org/guide/checkpoint), and ops used to save the
  V2 checkpoint format:

  * io_ops.SaveV2
  * io_ops.MergeV2Checkpoints
  * io_ops.RestoreV2

  **Predicate**

  The predicate is a filter that will run on every `Trackable` object connected
  to the root object. This function determines whether a `Trackable` should use
  the registered functions.

  Example: `lambda x: isinstance(x, CustomClass)`

  **Custom save function**

  This is how checkpoint saving works normally:
  1. Gather all of the Trackables with saveable values.
  2. For each Trackable, gather all of the saveable tensors.
  3. Save checkpoint shards (grouping tensors by device) with SaveV2
  4. Merge the shards with MergeCheckpointV2. This combines all of the shard's
     metadata, and renames them to follow the standard shard pattern.

  When a saver is registered, Trackables that pass the registered `predicate`
  are automatically marked as having saveable values. Next, the custom save
  function replaces steps 2 and 3 of the saving process. Finally, the shards
  returned by the custom save function are merged with the other shards.

  The save function takes in a dictionary of `Trackables` and a `file_prefix`
  string. The function should save checkpoint shards using the SaveV2 op, and
  list of the shard prefixes. SaveV2 is currently required to work a correctly,
  because the code merges all of the returned shards, and the `restore_fn` will
  only be given the prefix of the merged checkpoint. If you need to be able to
  save and restore from unmerged shards, please file a feature request.

  Specification and example of the save function:

  ```
  def save_fn(trackables, file_prefix):
    # trackables: A dictionary mapping unique string identifiers to trackables
    # file_prefix: A unique file prefix generated using the registered name.
    ...
    # Gather the tensors to save.
    ...
    io_ops.SaveV2(file_prefix, tensor_names, shapes_and_slices, tensors)
    return file_prefix  # Returns a tensor or a list of string tensors
  ```

  **Custom restore function**

  Normal checkpoint restore behavior:
  1. Gather all of the Trackables that have saveable values.
  2. For each Trackable, get the names of the desired tensors to extract from
     the checkpoint.
  3. Use RestoreV2 to read the saved values, and pass the restored tensors to
     the corresponding Trackables.

  The custom restore function replaces steps 2 and 3.

  The restore function also takes a dictionary of `Trackables` and a
  `merged_prefix` string. The `merged_prefix` is different from the
  `file_prefix`, since it contains the renamed shard paths. To read from the
  merged checkpoint, you must use `RestoreV2(merged_prefix, ...)`.

  Specification:

  ```
  def restore_fn(trackables, merged_prefix):
    # trackables: A dictionary mapping unique string identifiers to Trackables
    # merged_prefix: File prefix of the merged shard names.

    restored_tensors = io_ops.restore_v2(
        merged_prefix, tensor_names, shapes_and_slices, dtypes)
    ...
    # Restore the checkpoint values for the given Trackables.
  ```

  Args:
    package: Optional, the package that this class belongs to.
    name: (Required) The name of this saver, which is saved to the checkpoint.
      When a checkpoint is restored, the name and package are used to find the
      the matching restore function. The name and package are also used to
      generate a unique file prefix that is passed to the save_fn.
    predicate: (Required) A function that returns a boolean indicating whether a
      `Trackable` object should be checkpointed with this function. Predicates
      are executed in the reverse order that they are added (later registrations
      are checked first).
    save_fn: (Required) A function that takes a dictionary of trackables and a
      file prefix as the arguments, writes the checkpoint shards for the given
      Trackables, and returns the list of shard prefixes.
    restore_fn: (Required) A function that takes a dictionary of trackables and
      a file prefix as the arguments and restores the trackable values.

  Raises:
    ValueError: if the package and name are already registered.
  """
  if not callable(save_fn):
    raise TypeError(f"The save_fn must be callable, got: {type(save_fn)}")
  if not callable(restore_fn):
    raise TypeError(f"The restore_fn must be callable, got: {type(restore_fn)}")

  _saver_registry.register(package, name, predicate, (save_fn, restore_fn))


def get_registered_saver_name(trackable):
  """Returns the name of the registered saver to use with Trackable."""
  try:
    return _saver_registry.get_registered_name(trackable)
  except LookupError:
    return None


def get_save_function(registered_name):
  """Returns save function registered to name."""
  return _saver_registry.name_lookup(registered_name)[0]


def get_restore_function(registered_name):
  """Returns restore function registered to name."""
  return _saver_registry.name_lookup(registered_name)[1]


def validate_restore_function(trackable, registered_name):
  """Validates whether the trackable can be restored with the saver.

  When using a checkpoint saved with a registered saver, that same saver must
  also be also registered when loading. The name of that saver is saved to the
  checkpoint and set in the `registered_name` arg.

  Args:
    trackable: A `Trackable` object.
    registered_name: String name of the expected registered saver. This argument
      should be set using the name saved in a checkpoint.

  Raises:
    ValueError if the saver could not be found, or if the predicate associated
      with the saver does not pass.
  """
  try:
    _saver_registry.name_lookup(registered_name)
  except LookupError:
    raise ValueError(
        f"Error when restoring object {trackable} from checkpoint. This "
        "object was saved using a registered saver named "
        f"'{registered_name}', but this saver cannot be found in the "
        "current context.")
  if not _saver_registry.get_predicate(registered_name)(trackable):
    raise ValueError(
        f"Object {trackable} was saved with the registered saver named "
        f"'{registered_name}'. However, this saver cannot be used to restore the "
        "object because the predicate does not pass.")
