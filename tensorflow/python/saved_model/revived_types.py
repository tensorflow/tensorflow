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
"""Handles types registrations for tf.saved_model.load."""

import operator

from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.trackable import data_structures
from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.saved_model.load.VersionedTypeRegistration", v1=[])
class VersionedTypeRegistration(object):
  """Holds information about one version of a revived type."""

  def __init__(self, object_factory, version, min_producer_version,
               min_consumer_version, bad_consumers=None, setter=setattr):
    """Identify a revived type version.

    Args:
      object_factory: A callable which takes a SavedUserObject proto and returns
        a trackable object. Dependencies are added later via `setter`.
      version: An integer, the producer version of this wrapper type. When
        making incompatible changes to a wrapper, add a new
        `VersionedTypeRegistration` with an incremented `version`. The most
        recent version will be saved, and all registrations with a matching
        identifier will be searched for the highest compatible version to use
        when loading.
      min_producer_version: The minimum producer version number required to use
        this `VersionedTypeRegistration` when loading a proto.
      min_consumer_version: `VersionedTypeRegistration`s with a version number
        less than `min_consumer_version` will not be used to load a proto saved
        with this object. `min_consumer_version` should be set to the lowest
        version number which can successfully load protos saved by this
        object. If no matching registration is available on load, the object
        will be revived with a generic trackable type.

        `min_consumer_version` and `bad_consumers` are a blunt tool, and using
        them will generally break forward compatibility: previous versions of
        TensorFlow will revive newly saved objects as opaque trackable
        objects rather than wrapped objects. When updating wrappers, prefer
        saving new information but preserving compatibility with previous
        wrapper versions. They are, however, useful for ensuring that
        previously-released buggy wrapper versions degrade gracefully rather
        than throwing exceptions when presented with newly-saved SavedModels.
      bad_consumers: A list of consumer versions which are incompatible (in
        addition to any version less than `min_consumer_version`).
      setter: A callable with the same signature as `setattr` to use when adding
        dependencies to generated objects.
    """
    self.setter = setter
    self.identifier = None  # Set after registration
    self._object_factory = object_factory
    self.version = version
    self._min_consumer_version = min_consumer_version
    self._min_producer_version = min_producer_version
    if bad_consumers is None:
      bad_consumers = []
    self._bad_consumers = bad_consumers

  def to_proto(self):
    """Create a SavedUserObject proto."""
    # For now wrappers just use dependencies to save their state, so the
    # SavedUserObject doesn't depend on the object being saved.
    # TODO(allenl): Add a wrapper which uses its own proto.
    return saved_object_graph_pb2.SavedUserObject(
        identifier=self.identifier,
        version=versions_pb2.VersionDef(
            producer=self.version,
            min_consumer=self._min_consumer_version,
            bad_consumers=self._bad_consumers))

  def from_proto(self, proto):
    """Recreate a trackable object from a SavedUserObject proto."""
    return self._object_factory(proto)

  def should_load(self, proto):
    """Checks if this object should load the SavedUserObject `proto`."""
    if proto.identifier != self.identifier:
      return False
    if self.version < proto.version.min_consumer:
      return False
    if proto.version.producer < self._min_producer_version:
      return False
    for bad_version in proto.version.bad_consumers:
      if self.version == bad_version:
        return False
    return True


# string identifier -> (predicate, [VersionedTypeRegistration])
_REVIVED_TYPE_REGISTRY = {}
_TYPE_IDENTIFIERS = []


@tf_export("__internal__.saved_model.load.register_revived_type", v1=[])
def register_revived_type(identifier, predicate, versions):
  """Register a type for revived objects.

  Args:
    identifier: A unique string identifying this class of objects.
    predicate: A Boolean predicate for this registration. Takes a
      trackable object as an argument. If True, `type_registration` may be
      used to save and restore the object.
    versions: A list of `VersionedTypeRegistration` objects.
  """
  # Keep registrations in order of version. We always use the highest matching
  # version (respecting the min consumer version and bad consumers).
  versions.sort(key=lambda reg: reg.version, reverse=True)
  if not versions:
    raise AssertionError("Need at least one version of a registered type.")
  version_numbers = set()
  for registration in versions:
    # Copy over the identifier for use in generating protos
    registration.identifier = identifier
    if registration.version in version_numbers:
      raise AssertionError(
          f"Got multiple registrations with version {registration.version} for "
          f"type {identifier}.")
    version_numbers.add(registration.version)

  if identifier in _REVIVED_TYPE_REGISTRY:
    raise AssertionError(f"Duplicate registrations for type '{identifier}'")

  _REVIVED_TYPE_REGISTRY[identifier] = (predicate, versions)
  _TYPE_IDENTIFIERS.append(identifier)


def serialize(obj):
  """Create a SavedUserObject from a trackable object."""
  for identifier in _TYPE_IDENTIFIERS:
    predicate, versions = _REVIVED_TYPE_REGISTRY[identifier]
    if predicate(obj):
      # Always uses the most recent version to serialize.
      return versions[0].to_proto()
  return None


def deserialize(proto):
  """Create a trackable object from a SavedUserObject proto.

  Args:
    proto: A SavedUserObject to deserialize.

  Returns:
    A tuple of (trackable, assignment_fn) where assignment_fn has the same
    signature as setattr and should be used to add dependencies to
    `trackable` when they are available.
  """
  _, type_registrations = _REVIVED_TYPE_REGISTRY.get(
      proto.identifier, (None, None))
  if type_registrations is not None:
    for type_registration in type_registrations:
      if type_registration.should_load(proto):
        return (type_registration.from_proto(proto), type_registration.setter)
  return None


@tf_export("__internal__.saved_model.load.registered_identifiers", v1=[])
def registered_identifiers():
  """Return all the current registered revived object identifiers.

  Returns:
    A set of strings.
  """
  return _REVIVED_TYPE_REGISTRY.keys()


@tf_export("__internal__.saved_model.load.get_setter", v1=[])
def get_setter(proto):
  """Gets the registered setter function for the SavedUserObject proto.

  See VersionedTypeRegistration for info about the setter function.

  Args:
    proto: SavedUserObject proto

  Returns:
    setter function
  """
  _, type_registrations = _REVIVED_TYPE_REGISTRY.get(
      proto.identifier, (None, None))
  if type_registrations is not None:
    for type_registration in type_registrations:
      if type_registration.should_load(proto):
        return type_registration.setter
  return None


register_revived_type(
    "trackable_dict_wrapper",
    # pylint: disable=protected-access
    lambda obj: isinstance(obj, data_structures._DictWrapper),
    versions=[
        VersionedTypeRegistration(
            # Standard dependencies are enough to reconstruct the trackable
            # items in dictionaries, so we don't need to save any extra
            # information.
            # pylint: disable=protected-access
            object_factory=lambda proto: data_structures._DictWrapper({}),
            version=1,
            min_producer_version=1,
            min_consumer_version=1,
            setter=operator.setitem,
        )
    ],
)


register_revived_type(
    "trackable_list_wrapper",
    lambda obj: isinstance(obj, data_structures.ListWrapper),
    versions=[
        VersionedTypeRegistration(
            object_factory=lambda proto: data_structures.ListWrapper([]),
            version=1,
            min_producer_version=1,
            min_consumer_version=1,
            setter=data_structures.set_list_item,
        )
    ],
)


# Revive tuples as lists so we can append any dependencies during loading.
register_revived_type(
    "trackable_tuple_wrapper",
    # pylint: disable=protected-access
    lambda obj: isinstance(obj, data_structures._TupleWrapper),
    versions=[
        VersionedTypeRegistration(
            object_factory=lambda proto: data_structures.ListWrapper([]),
            version=1,
            min_producer_version=1,
            min_consumer_version=1,
            setter=data_structures.set_tuple_item,
        )
    ],
)
