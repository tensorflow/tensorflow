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
"""Utilities related to layer/model functionality."""

# TODO(b/110718070): Move these functions back to tensorflow/python/keras/utils
# once __init__ files no longer require all of tf.keras to be imported together.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import weakref

from tensorflow.python.util import object_identity

try:
  # typing module is only used for comment type annotations.
  import typing  # pylint: disable=g-import-not-at-top, unused-import
except ImportError:
  pass


def is_layer(obj):
  """Implicit check for Layer-like objects."""
  # TODO(b/110718070): Replace with isinstance(obj, base_layer.Layer).
  return hasattr(obj, "_is_layer") and not isinstance(obj, type)


def has_weights(obj):
  """Implicit check for Layer-like objects."""
  # TODO(b/110718070): Replace with isinstance(obj, base_layer.Layer).
  has_weight = (hasattr(type(obj), "trainable_weights")
                and hasattr(type(obj), "non_trainable_weights"))

  return has_weight and not isinstance(obj, type)


def invalidate_recursive_cache(key):
  """Convenience decorator to invalidate the cache when setting attributes."""
  def outer(f):
    @functools.wraps(f)
    def wrapped(self, value):
      sentinel = getattr(self, "_attribute_sentinel")  # type: AttributeSentinel
      sentinel.invalidate(key)
      return f(self, value)
    return wrapped
  return outer


class MutationSentinel(object):
  """Container for tracking whether a property is in a cached state."""
  _in_cached_state = False

  def mark_as(self, value):  # type: (MutationSentinel, bool) -> bool
    may_affect_upstream = (value != self._in_cached_state)
    self._in_cached_state = value
    return may_affect_upstream

  @property
  def in_cached_state(self):
    return self._in_cached_state


class AttributeSentinel(object):
  """Container for managing attribute cache state within a Layer.

  The cache can be invalidated either on an individual basis (for instance when
  an attribute is mutated) or a layer-wide basis (such as when a new dependency
  is added).
  """

  def __init__(self, always_propagate=False):
    self._parents = weakref.WeakSet()
    self.attributes = collections.defaultdict(MutationSentinel)

    # The trackable data structure containers are simple pass throughs. They
    # don't know or care about particular attributes. As a result, they will
    # consider themselves to be in a cached state, so it's up to the Layer
    # which contains them to terminate propagation.
    self.always_propagate = always_propagate

  def __repr__(self):
    return "{}\n  {}".format(
        super(AttributeSentinel, self).__repr__(),
        {k: v.in_cached_state for k, v in self.attributes.items()})

  def add_parent(self, node):
    # type: (AttributeSentinel, AttributeSentinel) -> None

    # Properly tracking removal is quite challenging; however since this is only
    # used to invalidate a cache it's alright to be overly conservative. We need
    # to invalidate the cache of `node` (since it has implicitly gained a child)
    # but we don't need to invalidate self since attributes should not depend on
    # parent Layers.
    self._parents.add(node)
    node.invalidate_all()

  def get(self, key):
    # type: (AttributeSentinel, str) -> bool
    return self.attributes[key].in_cached_state

  def _set(self, key, value):
    # type: (AttributeSentinel, str, bool) -> None
    may_affect_upstream = self.attributes[key].mark_as(value)
    if may_affect_upstream or self.always_propagate:
      for node in self._parents:  # type: AttributeSentinel
        node.invalidate(key)

  def mark_cached(self, key):
    # type: (AttributeSentinel, str) -> None
    self._set(key, True)

  def invalidate(self, key):
    # type: (AttributeSentinel, str) -> None
    self._set(key, False)

  def invalidate_all(self):
    # Parents may have different keys than their children, so we locally
    # invalidate but use the `invalidate_all` method of parents.
    for key in self.attributes.keys():
      self.attributes[key].mark_as(False)

    for node in self._parents:
      node.invalidate_all()


def filter_empty_layer_containers(layer_list):
  """Filter out empty Layer-like containers and uniquify."""
  # TODO(b/130381733): Make this an attribute in base_layer.Layer.
  existing = object_identity.ObjectIdentitySet()
  to_visit = layer_list[::-1]
  while to_visit:
    obj = to_visit.pop()
    if obj in existing:
      continue
    existing.add(obj)
    if is_layer(obj):
      yield obj
    else:
      sub_layers = getattr(obj, "layers", None) or []

      # Trackable data structures will not show up in ".layers" lists, but
      # the layers they contain will.
      to_visit.extend(sub_layers[::-1])
