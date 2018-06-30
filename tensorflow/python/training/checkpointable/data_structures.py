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

import six

from tensorflow.python.ops import variables
from tensorflow.python.training.checkpointable import base as checkpointable_lib
from tensorflow.python.training.checkpointable import layer_utils


# TODO(allenl): We could track regular Python data structures which get assigned
# to Checkpointable objects. Making this work with restore-on-create would be
# tricky; we'd need to re-create nested structures with our own wrapped objects
# on assignment to an attribute, and track the user's original structure to make
# sure they don't modify it except through the wrappers (since we could save the
# user's updated structure, but would have no way to support restore-on-create
# for those modifications).
# TODO(allenl): A dictionary data structure would be good too.
class CheckpointableDataStructure(checkpointable_lib.CheckpointableBase):
  """Base class for data structures which contain checkpointable objects."""

  def __init__(self):
    self._layers = []
    self.trainable = True
    self._extra_variables = []

  def _track_value(self, value, name):
    """Add a dependency on `value`."""
    if isinstance(value, checkpointable_lib.CheckpointableBase):
      self._track_checkpointable(value, name=name)
      if isinstance(value, variables.Variable):
        self._extra_variables.append(value)
    else:
      raise ValueError(
          ("Only checkpointable objects (such as Layers or Optimizers) may be "
           "stored in a List object. Got %s, which does not inherit from "
           "CheckpointableBase.") % (value,))
    if (isinstance(value, CheckpointableDataStructure)
        or layer_utils.is_layer(value)):
      if value not in self._layers:
        self._layers.append(value)
        if hasattr(value, "_use_resource_variables"):
          # In subclassed models, legacy layers (tf.layers) must always use
          # resource variables.
          value._use_resource_variables = True  # pylint: disable=protected-access

  @property
  def layers(self):
    return self._layers

  @property
  def trainable_weights(self):
    return layer_utils.gather_trainable_weights(
        trainable=self.trainable,
        sub_layers=self.layers,
        extra_variables=self._extra_variables)

  @property
  def non_trainable_weights(self):
    return layer_utils.gather_non_trainable_weights(
        trainable=self.trainable,
        sub_layers=self.layers,
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
      aggregated += layer.updates
    return aggregated

  @property
  def losses(self):
    """Aggregate losses from any `Layer` instances."""
    aggregated = []
    for layer in self.layers:
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
    self._storage = list(*args, **kwargs)
    for index, element in enumerate(self._storage):
      self._track_value(element, name=self._name_element(index))

  def _name_element(self, index):
    return "%d" % (index,)

  def append(self, value):
    """Add a new checkpointable value."""
    self._track_value(value, self._name_element(len(self._storage)))
    self._storage.append(value)

  def extend(self, values):
    """Add a sequence of checkpointable values."""
    for index_offset, value in enumerate(values):
      self._track_value(
          value, name=self._name_element(len(self._storage) + index_offset))
    self._storage.extend(values)

  def __iadd__(self, values):
    self.extend(values)
    return self

  def __add__(self, other):
    if isinstance(other, List):
      return List(self._storage + other._storage)  # pylint: disable=protected-access
    else:
      return List(self._storage + other)

  def __getitem__(self, key):
    return self._storage[key]

  def __len__(self):
    return len(self._storage)

  def __repr__(self):
    return "List(%s)" % (repr(self._storage),)


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
    self._storage = dict(*args, **kwargs)
    for key, value in self._storage.items():
      self._track_value(value, name=self._name_element(key))

  def _name_element(self, key):
    if not isinstance(key, six.string_types):
      raise TypeError(
          "Mapping accepts only string keys, but got a key %s."
          % repr(key))
    return str(key)

  def __setitem__(self, key, value):
    current_value = self._storage.setdefault(key, value)
    if current_value is not value:
      raise ValueError(
          ("Mappings are an append-only data structure. Tried to overwrite the "
           "key '%s' with value %s, but it already contains %s")
          % (key, value, current_value))
    self._track_value(value, name=self._name_element(key))

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
