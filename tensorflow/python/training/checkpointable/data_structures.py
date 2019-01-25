"""Checkpointable data structures."""
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import operator
import sys

import six

from tensorflow.python.ops import variables
from tensorflow.python.saved_model import revived_types
from tensorflow.python.training.checkpointable import base
from tensorflow.python.training.checkpointable import layer_utils


class NoDependency(object):
  """Allows attribute assignment to `Checkpointable` objects with no dependency.

  Example usage:
  ```python
  obj = Checkpointable()
  obj.has_dependency = tf.Variable(0., name="dep")
  obj.no_dependency = NoDependency(tf.Variable(1., name="nodep"))
  assert obj.no_dependency.name == "nodep:0"
  ```

  `obj` in this example has a dependency on the variable "dep", and both
  attributes contain un-wrapped `Variable` objects.

  `NoDependency` also works with `tf.keras.Model`, but only for checkpoint
  dependencies: wrapping a `Layer` in `NoDependency` will assign the (unwrapped)
  `Layer` to the attribute without a checkpoint dependency, but the `Model` will
  still track the `Layer` (so it will appear in `Model.layers`, and its
  variables will appear in `Model.variables`).
  """

  def __init__(self, value):
    self.value = value


def _wrap_or_unwrap(value):
  """Wraps basic data structures, unwraps NoDependency objects."""
  if isinstance(value, NoDependency):
    return value.value
  if isinstance(value, base.Checkpointable):
    return value  # Skip conversion for already checkpointable objects.
  elif isinstance(value, dict):
    return _DictWrapper(value)
  elif isinstance(value, list):
    return _ListWrapper(value)
  else:
    return value
  # TODO(allenl): Handle other common data structures. Tuples will require
  # special casing (tuple subclasses are not weak referenceable, so replacement
  # with a wrapper that subclasses tuple on attribute assignment works poorly,
  # and replacement with a wrapper that isn't a tuple is also problematic),
  # probably a tree traversal where the leaves are non-tuples(/namedtuples) to
  # come up with names. Dictionaries should look like lists.


def sticky_attribute_assignment(checkpointable, name, value):
  """Adds dependencies, generally called from __setattr__.

  This behavior is shared between Checkpointable and Model.

  Respects NoDependency indicators, but otherwise makes checkpointable objects
  out of common data structures and tracks objects by their attribute names.

  Args:
    checkpointable: The object to add dependencies to (generally the one having
      an attribute assigned).
    name: The attribute name being assigned.
    value: The value being assigned. Not necessarily a checkpointable object.

  Returns:
    The value which should be stored in the attribute (unwrapped from a
    NoDependency object if necessary).
  """
  if isinstance(value, NoDependency):
    add_dependency = False
  else:
    add_dependency = True
  value = _wrap_or_unwrap(value)
  if not add_dependency:
    return value
  if isinstance(value, base.Checkpointable):
    checkpointable._track_checkpointable(  # pylint: disable=protected-access
        value, name=name,
        # Allow the user to switch the Checkpointable which is tracked by this
        # name, since assigning a new variable to an attribute has
        # historically been fine (e.g. Adam did this).
        overwrite=True)
  return value


class CheckpointableDataStructure(base.Checkpointable):
  """Base class for data structures which contain checkpointable objects."""

  def __init__(self):
    self.trainable = True
    self._extra_variables = []

  def _track_value(self, value, name):
    """Add a dependency on `value`."""
    value = sticky_attribute_assignment(
        checkpointable=self, value=value, name=name)
    if isinstance(value, variables.Variable):
      self._extra_variables.append(value)
    if not isinstance(value, base.Checkpointable):
      raise ValueError(
          ("Only checkpointable objects (such as Layers or Optimizers) may be "
           "stored in a List object. Got %s, which does not inherit from "
           "Checkpointable.") % (value,))
    if hasattr(value, "_use_resource_variables"):
      # In subclassed models, legacy layers (tf.layers) must always use
      # resource variables.
      value._use_resource_variables = True  # pylint: disable=protected-access
    return value

  @property
  def _values(self):
    """An iterable/sequence which may contain checkpointable objects."""
    raise NotImplementedError("Abstract method")

  @property
  def _layers(self):
    """All Layers and Layer containers, including empty containers."""
    # Filter objects on demand so that wrapper objects use values from the thing
    # they're wrapping if out of sync.
    collected = []
    for obj in self._values:
      if (isinstance(obj, CheckpointableDataStructure)
          or layer_utils.is_layer(obj)
          or layer_utils.has_weights(obj)):
        collected.append(obj)
    return collected

  @property
  def layers(self):
    return layer_utils.filter_empty_layer_containers(self._layers)

  @property
  def trainable_weights(self):
    return layer_utils.gather_trainable_weights(
        trainable=self.trainable,
        sub_layers=self._layers,
        extra_variables=self._extra_variables)

  @property
  def non_trainable_weights(self):
    return layer_utils.gather_non_trainable_weights(
        trainable=self.trainable,
        sub_layers=self._layers,
        extra_variables=self._extra_variables)

  @property
  def weights(self):
    return self.trainable_weights + self.non_trainable_weights

  @property
  def trainable_variables(self):
    return self.trainable_weights

  @property
  def non_trainable_variables(self):
    return self.non_trainable_weights

  @property
  def variables(self):
    return self.weights

  @property
  def updates(self):
    """Aggregate updates from any `Layer` instances."""
    # Updates and conditional losses are forwarded as-is rather than being
    # filtered based on inputs, since this is just a container and won't ever
    # have any inputs.
    aggregated = []
    for layer in self.layers:
      if hasattr(layer, "updates"):
        aggregated += layer.updates
    return aggregated

  @property
  def losses(self):
    """Aggregate losses from any `Layer` instances."""
    aggregated = []
    for layer in self.layers:
      if hasattr(layer, "losses"):
        aggregated += layer.losses
    return aggregated

  def __hash__(self):
    # Support object-identity hashing, so these structures can be used as keys
    # in sets/dicts.
    return id(self)

  def __eq__(self, other):
    # Similar to Tensors, checkpointable data structures use object-identity
    # equality to support set/dict membership.
    return self is other


class List(CheckpointableDataStructure, collections.Sequence):
  """An append-only sequence type which is checkpointable.

  Maintains checkpoint dependencies on its contents (which must also be
  checkpointable), and forwards any `Layer` metadata such as updates and losses.

  Note that `List` is purely a container. It lets a `tf.keras.Model` or
  other checkpointable object know about its contents, but does not call any
  `Layer` instances which are added to it. To indicate a sequence of `Layer`
  instances which should be called sequentially, use `tf.keras.Sequential`.

  Example usage:
  ```python
  class HasList(tf.keras.Model):

    def __init__(self):
      super(HasList, self).__init__()
      self.layer_list = tf.contrib.checkpoint.List([layers.Dense(3)])
      self.layer_list.append(layers.Dense(4))

    def call(self, x):
      aggregation = 0.
      for l in self.layer_list:
        x = l(x)
        aggregation += tf.reduce_sum(x)
      return aggregation
  ```

  This kind of wrapping is necessary because `Checkpointable` objects do not
  (yet) deeply inspect regular Python data structures, so for example assigning
  a regular list (`self.layer_list = [layers.Dense(3)]`) does not create a
  checkpoint dependency and does not add the `Layer` instance's weights to its
  parent `Model`.
  """

  def __init__(self, *args, **kwargs):
    """Construct a new sequence. Arguments are passed to `list()`."""
    super(List, self).__init__()
    self._storage = self._make_storage(*args, **kwargs)
    for index, element in enumerate(self._storage):
      self._storage[index] = self._track_value(
          element, name=self._name_element(index))

  def copy(self):
    return type(self)(copy.copy(self._storage))

  def __copy__(self):
    return self.copy()

  def __deepcopy__(self, memo):
    return type(self)(copy.deepcopy(self._storage, memo))

  def _make_storage(self, *args, **kwargs):
    """Determines the backing storage (overridden in subclasses)."""
    return list(*args, **kwargs)

  def _name_element(self, index):
    return "%d" % (index,)

  @property
  def _values(self):
    return self

  def append(self, value):
    """Add a new checkpointable value."""
    value = self._track_value(value, self._name_element(len(self._storage)))
    self._storage.append(value)

  def extend(self, values):
    """Add a sequence of checkpointable values."""
    for value in values:
      self.append(value)

  def __iadd__(self, values):
    self.extend(values)
    return self

  def __add__(self, other):
    return self.__class__(self._storage + getattr(other, "_storage", other))

  def __imul__(self, y):
    if y <= 0:
      raise ValueError(
          "List only supports append, multiplying in place by %d removes "
          "elements." % y)

    n = len(self._storage)
    for _ in range(y - 1):
      for i in range(n):
        self.append(self._storage[i])

    return self

  def __mul__(self, n):
    return self.__class__(self._storage * n)

  def __rmul__(self, n):
    return self * n

  def __radd__(self, other):
    return self + other

  def __getitem__(self, key):
    return self._storage[key]

  def __getslice__(self, i, j):
    return self._storage[slice(i, j)]

  def __len__(self):
    return len(self._storage)

  def __repr__(self):
    return "List(%s)" % (repr(self._storage),)

  def __sizeof__(self):
    return super(List, self).__sizeof__() + sys.getsizeof(self._storage)


# TODO(tomhennigan) Update to collections.UserList?
class _ListWrapper(List, collections.MutableSequence,
                   # Shadowed, but there for isinstance checks.
                   list):
  """Wraps the built-in `list` to support restore-on-create for variables.

  Unlike `List`, this sequence type is mutable in the same ways built-in lists
  are. Instead of throwing an error immediately like `List`, it records
  problematic mutations (e.g. assigning a new element to a position already
  occupied, meaning both elements get the same names at different times) and
  refuses to save.

  On assignment to an attribute of a Model or Checkpointable object, Python
  lists are replaced with _ListWrapper. Wrapping a list in a
  `tf.contrib.checkpoint.NoDependency` object prevents this.
  """

  def __init__(self, wrapped_list):
    """Construct a new list wrapper.

    Args:
      wrapped_list: The initial value of the data structure. A shallow copy may
        be maintained for error checking. `wrapped_list` itself should not be
        modified directly after constructing the `_ListWrapper`, and if changes
        are detected the `_ListWrapper` will throw an exception on save.
    """
    # Monotonic flags which indicate this object would not be restored properly,
    # and therefore should throw an error on save to avoid giving the impression
    # that restoring it will work.
    self._non_append_mutation = False
    self._external_modification = False
    super(_ListWrapper, self).__init__(wrapped_list)
    self._last_wrapped_list_snapshot = list(self._storage)

  # pylint: disable=protected-access
  def __copy__(self):
    copied = super(_ListWrapper, self).__copy__()
    copied._non_append_mutation = self._non_append_mutation
    copied._external_modification = self._external_modification
    return copied

  def __deepcopy__(self, memo):
    copied = super(_ListWrapper, self).__deepcopy__(memo)
    copied._non_append_mutation = self._non_append_mutation
    copied._external_modification = self._external_modification
    return copied
  # pylint: enable=protected-access

  def _make_storage(self, wrapped_list):
    """Use the user's original list for storage."""
    return wrapped_list

  def _check_external_modification(self):
    """Checks for any changes to the wrapped list not through the wrapper."""
    if self._external_modification or self._non_append_mutation:
      return
    if self._storage != self._last_wrapped_list_snapshot:
      self._external_modification = True
      self._last_wrapped_list_snapshot = None

  def _update_snapshot(self):
    """Acknowledges tracked changes to the wrapped list."""
    if self._external_modification or self._non_append_mutation:
      return
    self._last_wrapped_list_snapshot = list(self._storage)

  @property
  def _checkpoint_dependencies(self):
    self._check_external_modification()
    if self._non_append_mutation:
      raise ValueError(
          ("Unable to save the object %s (a list wrapper constructed to track "
           "checkpointable TensorFlow objects). A list element was replaced "
           "(__setitem__, __setslice__), deleted (__delitem__, __delslice__), "
           "or moved (sort). In order to support restoration on object "
           "creation, tracking is exclusively for append-only data structures."
           "\n\nIf you don't need this list checkpointed, wrap it in a "
           "tf.contrib.checkpoint.NoDependency object; it will be "
           "automatically un-wrapped and subsequently ignored." % (self,)))
    if self._external_modification:
      raise ValueError(
          ("Unable to save the object %s (a list wrapper constructed to track "
           "checkpointable TensorFlow objects). The wrapped list was modified "
           "outside the wrapper (its final value was %s, its value when a "
           "checkpoint dependency was added was %s), which breaks restoration "
           "on object creation.\n\nIf you don't need this list checkpointed, "
           "wrap it in a tf.contrib.checkpoint.NoDependency object; it will be "
           "automatically un-wrapped and subsequently ignored." % (
               self, self._storage, self._last_wrapped_list_snapshot)))
    return super(_ListWrapper, self)._checkpoint_dependencies

  def __delitem__(self, key):
    self._non_append_mutation = True
    del self._storage[key]

  def __setitem__(self, key, value):
    self._check_external_modification()

    if isinstance(key, slice):
      # Note: this is quite inefficient, but the list API supports a broad range
      # of slice setters (e.g. truncate, extend, replace) and immitating this
      # for a range of Python versions is non-trivial.
      storage_copy = list(self._storage)
      self._storage[key] = value

      len_before = len(storage_copy)
      len_now = len(self._storage)
      for i in range(max(len_before, len_now)):
        value_now = self._storage[i] if i < len_now else None
        value_before = storage_copy[i] if i < len_before else None

        if isinstance(value_before, base.Checkpointable):
          self._non_append_mutation = True

        if value_now is not None and value_now != value_before:
          self._storage[i] = self._track_value(self._storage[i],
                                               self._name_element(i))

    else:
      if isinstance(self._storage[key], base.Checkpointable):
        self._non_append_mutation = True
      self._storage[key] = self._track_value(value, self._name_element(key))

    self._update_snapshot()

  def append(self, value):
    """Add a new checkpointable value."""
    self._check_external_modification()
    super(_ListWrapper, self).append(value)
    self._update_snapshot()

  def extend(self, values):
    """Add a sequence of checkpointable values."""
    self._check_external_modification()
    super(_ListWrapper, self).extend(values)
    self._update_snapshot()

  def __eq__(self, other):
    return self._storage == getattr(other, "_storage", other)

  def __ne__(self, other):
    return self._storage != getattr(other, "_storage", other)

  def __lt__(self, other):
    return self._storage < getattr(other, "_storage", other)

  def __le__(self, other):
    return self._storage <= getattr(other, "_storage", other)

  def __gt__(self, other):
    return self._storage > getattr(other, "_storage", other)

  def __ge__(self, other):
    return self._storage >= getattr(other, "_storage", other)

  def __hash__(self):
    # List wrappers need to compare like regular lists, and so like regular
    # lists they don't belong in hash tables.
    raise TypeError("unhashable type: 'ListWrapper'")

  def insert(self, index, obj):
    self._non_append_mutation = True
    self._storage.insert(index, obj)

  def sort(self):
    self._non_append_mutation = True
    self._storage.sort()

  def __setslice__(self, i, j, y):
    self.__setitem__(slice(i, j), y)

  def __delslice__(self, i, j):
    self._non_append_mutation = True
    del self._storage[slice(i, j)]

  def _track_value(self, value, name):
    """Allows storage of non-checkpointable objects."""
    try:
      value = super(_ListWrapper, self)._track_value(value=value, name=name)
    except ValueError:
      # Even if this value isn't checkpointable, we need to make sure
      # NoDependency objects get unwrapped.
      value = sticky_attribute_assignment(
          checkpointable=self, value=value, name=name)
    return value

  def __repr__(self):
    return "ListWrapper(%s)" % (repr(self._storage),)


class Mapping(CheckpointableDataStructure, collections.Mapping):
  """An append-only checkpointable mapping data structure with string keys.

  Maintains checkpoint dependencies on its contents (which must also be
  checkpointable), named based on its keys.

  Note that once a key has been added, it may not be deleted or replaced. If
  names may not be unique, see `tf.contrib.checkpoint.UniqueNameTracker`.
  """

  def __init__(self, *args, **kwargs):
    """Construct a new sequence. Arguments are passed to `dict()`."""
    super(Mapping, self).__init__()
    self._storage = self._make_storage(*args, **kwargs)
    self._storage.update(
        {key: self._track_value(
            value, name=self._name_element(key))
         for key, value in self._storage.items()})

  def __copy__(self):
    return type(self)(copy.copy(self._storage))

  def __deepcopy__(self, memo):
    return type(self)(copy.deepcopy(self._storage, memo))

  def _make_storage(self, *args, **kwargs):
    return dict(*args, **kwargs)

  @property
  def _values(self):
    # Sort items deterministically by key
    ordered = list(zip(*sorted(self.items(), key=lambda it: it[0])))
    if ordered:
      return ordered[1]
    return []

  def _name_element(self, key):
    if not isinstance(key, six.string_types):
      raise TypeError(
          "Mapping accepts only string keys, but got a key %s."
          % repr(key))
    return str(key)

  def __setitem__(self, key, value):
    name = self._name_element(key)
    value = self._track_value(value, name=name)
    current_value = self._storage.setdefault(key, value)
    if current_value is not value:
      raise ValueError(
          ("Mappings are an append-only data structure. Tried to overwrite the "
           "key '%s' with value %s, but it already contains %s")
          % (key, value, current_value))

  def update(self, *args, **kwargs):
    for key, value in dict(*args, **kwargs).items():
      self[key] = value

  def __getitem__(self, key):
    return self._storage[key]

  def __len__(self):
    return len(self._storage)

  def __repr__(self):
    return "Mapping(%s)" % (repr(self._storage),)

  def __iter__(self):
    return iter(self._storage)


# Unlike _ListWrapper, having _DictWrapper inherit from dict and pass isinstance
# checks seems infeasible. CPython will not call Python methods/properties on
# dictionary subclasses when running e.g. {}.update(dict_subclass), and instead
# collects elements directly from dict_subclass's C structs. So subclassing dict
# implies that the storage has to be "self" (i.e. the C structs for the object
# must be updated correctly), but we also need that storage to be the wrapped
# dictionary to avoid synchronization bugs (un-tracked external modifications
# should still show up when the dict is accessed through the wrapper). Monkey
# patching all of the "wrapped" dict's methods instead of creating a wrapper
# object is an option, but not a very attractive one (replacing methods without
# creating reference cycles is difficult, and then dicts would need to be
# special cased everywhere as being checkpointable).
class _DictWrapper(Mapping, collections.MutableMapping):
  """Wraps built-in dicts to support restore-on-create for variables.

  _DictWrapper is to Mapping as _ListWrapper is to List. Unlike Mapping,
  _DictWrapper allows non-string keys and values and arbitrary mutations (delete
  keys, reassign values). Like _ListWrapper, these mutations mean that
  _DictWrapper will raise an exception on save.
  """

  def __new__(cls, *args):
    if len(args) == 1 and isinstance(args[0], dict):
      return super(_DictWrapper, cls).__new__(cls)
    else:
      # Allow construction from a sequence, e.g. for nest.pack_sequence_as. In
      # this case there's nothing to wrap, so we make a normal dictionary. Also
      # allows constructing empty instances of the _DictWrapper type, as Session
      # is wont to do (and again there's nothing to wrap, so a normal dictionary
      # makes more sense).
      return dict(*args)

  def __init__(self, wrapped_dict):
    self._non_string_key = False
    self._non_append_mutation = False
    self._external_modification = False
    super(_DictWrapper, self).__init__(wrapped_dict)
    self._update_snapshot()

  # pylint: disable=protected-access
  def __copy__(self):
    copied = super(_DictWrapper, self).__copy__()
    copied._non_append_mutation = self._non_append_mutation
    copied._external_modification = self._external_modification
    copied._non_string_key = self._non_string_key
    return copied

  def __deepcopy__(self, memo):
    copied = super(_DictWrapper, self).__deepcopy__(memo)
    copied._non_append_mutation = self._non_append_mutation
    copied._external_modification = self._external_modification
    copied._non_string_key = self._non_string_key
    return copied
  # pylint: enable=protected-access

  def _make_storage(self, wrapped_dict):
    """Re-use the wrapped dict for storage (to force them to be in sync)."""
    return wrapped_dict

  @property
  def _checkpoint_dependencies(self):
    """Check that the object is saveable before listing its dependencies."""
    self._check_external_modification()
    if self._non_string_key:
      raise ValueError(
          "Unable to save the object %s (a dictionary wrapper constructed "
          "automatically on attribute assignment). The wrapped dictionary "
          "contains a non-string key which maps to a checkpointable object or "
          "mutable data structure.\n\nIf you don't need this dictionary "
          "checkpointed, wrap it in a tf.contrib.checkpoint.NoDependency "
          "object; it will be automatically un-wrapped and subsequently "
          "ignored." % (self,))
    if self._non_append_mutation:
      raise ValueError(
          "Unable to save the object %s (a dictionary wrapper constructed "
          "automatically on attribute assignment). A key mapping to a "
          "checkpointable object was overwritten or deleted, which would "
          "cause problems for restoration.\n\nIf you don't need this "
          "dictionary checkpointed, wrap it in a "
          "tf.contrib.checkpoint.NoDependency object; it will be automatically "
          "un-wrapped and subsequently ignored." % (self,))
    if self._external_modification:
      raise ValueError(
          "Unable to save the object %s (a dictionary wrapper constructed "
          "automatically on attribute assignment). The wrapped dictionary was "
          "modified outside the wrapper (its final value was %s, its value "
          "when a checkpoint dependency was added was %s), which breaks "
          "restoration on object creation.\n\nIf you don't need this "
          "dictionary checkpointed, wrap it in a "
          "tf.contrib.checkpoint.NoDependency object; it will be automatically "
          "un-wrapped and subsequently ignored." % (
              self, self, self._last_wrapped_dict_snapshot))
    assert not self._dirty  # Any reason for dirtiness should have an exception.
    return super(_DictWrapper, self)._checkpoint_dependencies

  @property
  def _dirty(self):
    """Check if there has already been a mutation which prevents saving."""
    return (self._external_modification
            or self._non_append_mutation
            or self._non_string_key)

  def _check_external_modification(self):
    """Checks for any changes to the wrapped dict not through the wrapper."""
    if self._dirty:
      return
    if self != self._last_wrapped_dict_snapshot:
      self._external_modification = True
      self._last_wrapped_dict_snapshot = None

  def _update_snapshot(self):
    """Acknowledges tracked changes to the wrapped dict."""
    if self._dirty:
      return
    self._last_wrapped_dict_snapshot = dict(self)

  def _track_value(self, value, name):
    """Allows storage of non-checkpointable objects."""
    if isinstance(name, six.string_types):
      string_key = True
    else:
      name = "-non_string_key"
      string_key = False
    try:
      no_dependency = isinstance(value, NoDependency)
      value = super(_DictWrapper, self)._track_value(value=value, name=name)
      if not (string_key or no_dependency):
        # A non-string key maps to a checkpointable value. This data structure
        # is not saveable.
        self._non_string_key = True
      return value
    except ValueError:
      # Even if this value isn't checkpointable, we need to make sure
      # NoDependency objects get unwrapped.
      return sticky_attribute_assignment(
          checkpointable=self, value=value, name=name)

  def _name_element(self, key):
    """Don't throw errors for non-string keys."""
    if isinstance(key, six.string_types):
      return super(_DictWrapper, self)._name_element(key)
    else:
      return key

  def __setitem__(self, key, value):
    """Allow any modifications, but possibly mark the wrapper as unsaveable."""
    self._check_external_modification()
    no_dep = isinstance(value, NoDependency)
    if isinstance(key, six.string_types):
      existing_dependency = self._lookup_dependency(key)
      value = self._track_value(value, name=key)
    else:
      value = _wrap_or_unwrap(value)
      existing_dependency = None
      if not no_dep and isinstance(value, base.Checkpointable):
        # Non-string keys are OK as long as we have no reason to add a
        # dependency on the value (either because the value is not
        # checkpointable, or because it was wrapped in a NoDependency object).
        self._non_string_key = True
    current_value = self._storage.setdefault(key, value)
    if current_value is not value:
      if ((not no_dep and isinstance(value, base.Checkpointable))
          # We don't want to just check that the existing object is
          # checkpointable, since it may have been wrapped in a NoDependency
          # object.
          or existing_dependency is not None):
        # A checkpointable object was replaced under the same key; this means
        # that restoring would be error-prone, so we'll throw an exception on
        # save.
        self._non_append_mutation = True
      self._storage[key] = value

    self._update_snapshot()

  def __delitem__(self, key):
    self._check_external_modification()
    existing_value = self[key]
    if isinstance(existing_value, base.Checkpointable):
      # Deleting tracked checkpointable values means restoring is problematic,
      # so we'll throw an exception on save.
      self._non_append_mutation = True
    del self._storage[key]
    self._update_snapshot()

  def __repr__(self):
    return "DictWrapper(%s)" % (repr(self._storage),)

  def __hash__(self):
    raise TypeError("unhashable type: 'DictWrapper'")

  def __eq__(self, other):
    return self._storage == getattr(other, "_storage", other)

  def update(self, *args, **kwargs):
    for key, value in dict(*args, **kwargs).items():
      self[key] = value

revived_types.register_revived_type(
    "checkpointable_dict_wrapper",
    lambda obj: isinstance(obj, _DictWrapper),
    versions=[revived_types.VersionedTypeRegistration(
        # Standard dependencies are enough to reconstruct the checkpointable
        # items in dictionaries, so we don't need to save any extra information.
        object_factory=lambda proto: _DictWrapper({}),
        version=1,
        min_producer_version=1,
        min_consumer_version=1,
        setter=operator.setitem)])


def _set_list_item(list_object, index_string, value):
  item_index = int(index_string)
  if len(list_object) <= item_index:
    list_object.extend([None] * (1 + item_index - len(list_object)))
  list_object[item_index] = value


revived_types.register_revived_type(
    "checkpointable_list_wrapper",
    lambda obj: isinstance(obj, _ListWrapper),
    versions=[revived_types.VersionedTypeRegistration(
        object_factory=lambda proto: _ListWrapper([]),
        version=1,
        min_producer_version=1,
        min_consumer_version=1,
        setter=_set_list_item)])
