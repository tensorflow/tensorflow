"""Utilities for collecting objects based on "is" comparison."""
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import weakref

from tensorflow.python.util.compat import collections_abc


# LINT.IfChange
class _ObjectIdentityWrapper(object):
  """Wraps an object, mapping __eq__ on wrapper to "is" on wrapped.

  Since __eq__ is based on object identity, it's safe to also define __hash__
  based on object ids. This lets us add unhashable types like trackable
  _ListWrapper objects to object-identity collections.
  """

  __slots__ = ["_wrapped", "__weakref__"]

  def __init__(self, wrapped):
    self._wrapped = wrapped

  @property
  def unwrapped(self):
    return self._wrapped

  def _assert_type(self, other):
    if not isinstance(other, _ObjectIdentityWrapper):
      raise TypeError("Cannot compare wrapped object with unwrapped object")

  def __lt__(self, other):
    self._assert_type(other)
    return id(self._wrapped) < id(other._wrapped)  # pylint: disable=protected-access

  def __gt__(self, other):
    self._assert_type(other)
    return id(self._wrapped) > id(other._wrapped)  # pylint: disable=protected-access

  def __eq__(self, other):
    if other is None:
      return False
    self._assert_type(other)
    return self._wrapped is other._wrapped  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    # Wrapper id() is also fine for weakrefs. In fact, we rely on
    # id(weakref.ref(a)) == id(weakref.ref(a)) and weakref.ref(a) is
    # weakref.ref(a) in _WeakObjectIdentityWrapper.
    return id(self._wrapped)

  def __repr__(self):
    return "<{} wrapping {!r}>".format(type(self).__name__, self._wrapped)


class _WeakObjectIdentityWrapper(_ObjectIdentityWrapper):

  __slots__ = ()

  def __init__(self, wrapped):
    super(_WeakObjectIdentityWrapper, self).__init__(weakref.ref(wrapped))

  @property
  def unwrapped(self):
    return self._wrapped()


class Reference(_ObjectIdentityWrapper):
  """Reference that refers an object.

  ```python
  x = [1]
  y = [1]

  x_ref1 = Reference(x)
  x_ref2 = Reference(x)
  y_ref2 = Reference(y)

  print(x_ref1 == x_ref2)
  ==> True

  print(x_ref1 == y)
  ==> False
  ```
  """

  __slots__ = ()

  # Disabling super class' unwrapped field.
  unwrapped = property()

  def deref(self):
    """Returns the referenced object.

    ```python
    x_ref = Reference(x)
    print(x is x_ref.deref())
    ==> True
    ```
    """
    return self._wrapped


class ObjectIdentityDictionary(collections_abc.MutableMapping):
  """A mutable mapping data structure which compares using "is".

  This is necessary because we have trackable objects (_ListWrapper) which
  have behavior identical to built-in Python lists (including being unhashable
  and comparing based on the equality of their contents by default).
  """

  __slots__ = ["_storage"]

  def __init__(self):
    self._storage = {}

  def _wrap_key(self, key):
    return _ObjectIdentityWrapper(key)

  def __getitem__(self, key):
    return self._storage[self._wrap_key(key)]

  def __setitem__(self, key, value):
    self._storage[self._wrap_key(key)] = value

  def __delitem__(self, key):
    del self._storage[self._wrap_key(key)]

  def __len__(self):
    return len(self._storage)

  def __iter__(self):
    for key in self._storage:
      yield key.unwrapped

  def __repr__(self):
    return "ObjectIdentityDictionary(%s)" % repr(self._storage)


class ObjectIdentityWeakKeyDictionary(ObjectIdentityDictionary):
  """Like weakref.WeakKeyDictionary, but compares objects with "is"."""

  __slots__ = ["__weakref__"]

  def _wrap_key(self, key):
    return _WeakObjectIdentityWrapper(key)

  def __len__(self):
    # Iterate, discarding old weak refs
    return len(list(self._storage))

  def __iter__(self):
    keys = self._storage.keys()
    for key in keys:
      unwrapped = key.unwrapped
      if unwrapped is None:
        del self[key]
      else:
        yield unwrapped


class ObjectIdentitySet(collections_abc.MutableSet):
  """Like the built-in set, but compares objects with "is"."""

  __slots__ = ["_storage", "__weakref__"]

  def __init__(self, *args):
    self._storage = set(self._wrap_key(obj) for obj in list(*args))

  @staticmethod
  def _from_storage(storage):
    result = ObjectIdentitySet()
    result._storage = storage  # pylint: disable=protected-access
    return result

  def _wrap_key(self, key):
    return _ObjectIdentityWrapper(key)

  def __contains__(self, key):
    return self._wrap_key(key) in self._storage

  def discard(self, key):
    self._storage.discard(self._wrap_key(key))

  def add(self, key):
    self._storage.add(self._wrap_key(key))

  def update(self, items):
    self._storage.update([self._wrap_key(item) for item in items])

  def clear(self):
    self._storage.clear()

  def intersection(self, items):
    return self._storage.intersection([self._wrap_key(item) for item in items])

  def difference(self, items):
    return ObjectIdentitySet._from_storage(
        self._storage.difference([self._wrap_key(item) for item in items]))

  def __len__(self):
    return len(self._storage)

  def __iter__(self):
    keys = list(self._storage)
    for key in keys:
      yield key.unwrapped


class ObjectIdentityWeakSet(ObjectIdentitySet):
  """Like weakref.WeakSet, but compares objects with "is"."""

  __slots__ = ()

  def _wrap_key(self, key):
    return _WeakObjectIdentityWrapper(key)

  def __len__(self):
    # Iterate, discarding old weak refs
    return len([_ for _ in self])

  def __iter__(self):
    keys = list(self._storage)
    for key in keys:
      unwrapped = key.unwrapped
      if unwrapped is None:
        self.discard(key)
      else:
        yield unwrapped
# LINT.ThenChange(//tensorflow/python/keras/utils/object_identity.py)
