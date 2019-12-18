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


def cache_recursive_attribute(key):
  """Decorator to cache Layer properties which recursively depend on sub-layers.

  A number of attributes in Keras Layers take the form:

  ```
  @property
  def thing(self):
    return self._thing or any(layer.thing for layer in self.layers)
  ```

  This means that checking these properties (e.g. dynamic, stateful, etc) must
  traverse the entire graph of layers to determine whether any descent has
  changed its state. This decorator adds a mechanism for Layers and trackable
  data structures to broadcast mutations (including the addition or deletion
  of layers) and allows the top level layer to safely cache results. In general,
  if computing an attribute triggers a depth first search it is a good candidate
  for this caching mechanism.

  The architecture is optimized for safety and correctness rather than absolute
  optimality. This manifests in two ways:
    1) Parents are never removed. It is possible for layer A to depend on layer
       B but subsequently remove that dependency. In that case, layer B will
       continue to broadcast its mutations to layer A until either A or B is
       deleted. However because the only effect is to invalidate a cache this
       does not affect correctness. (And robustly removing dependencies is
       difficult and error prone.)

    2) Layers aggressively invalidate their caches when there is any ambiguity
       of whether or not it is necessary. For instance, consider the following:
       ```
       class MyLayer(tf.keras.layers.Layer):
         def __init__(self):
           super(MyLayer, self).__init__()

           sub_layer = tf.keras.layers.Dense(1)
           self.sub_layers = [
               sub_layer  # This will be picked up, converted to a ListWrapper,
                          # and added to self._layers
           ]

           # Include the layer twice.
           self.sub_layers.append(sub_layer)

           # Remove one copy, but one copy remains.
           self.sub_layers.pop()
       ```
       In the example layer above, the set of tracked layers actually doesn't
       change; however to know that in the general case the Layer needs
       significant machinery to reason about what, if anything, has changed.
       By invalidating on every mutation we don't need to concern ourselves
       with the many types of mutations (append, pop, in-place replacement)
       and their specific semantics.

  Because mutations to layers are expected to be infrequent, this very
  conservative approach captures the vast majority of the performance gains from
  caching recursive properties while still remaining quite lightweight and easy
  to reason about.

  `tracking.cached_per_instance` provides a more detailed performance analysis
  of the WeakKeyDictionary cache pattern.

  Args:
    key: A string indicating which field is being cached. While not strictly
         necessary (since it could be obtained from f.__name__), it forces
         deliberate behavior when caching an attribute.

  Returns:
    A caching decorater specialized to `key`.
  """
  cache = weakref.WeakKeyDictionary()
  def outer(f):
    """Attribute cache which has been specialized."""

    @functools.wraps(f)
    def wrapped(self):
      """Cache aware version of `f`."""

      # Sentinels are unique per Layer/Trackable, but can be hashed. (Unlike
      # some trackable data structures.) Consequently it makes sense to use the
      # sentinel as a cache key rather than `self`.
      sentinel = getattr(self, "_attribute_sentinel")  # type: AttributeSentinel

      if not sentinel.get(key) or sentinel not in cache:
        cache[sentinel] = f(self)
        sentinel.mark_cached(key)
      output = cache[sentinel]
      return output

    return wrapped
  return outer


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


def gather_trainable_weights(trainable, sub_layers, extra_variables):
  """Lists the trainable weights for an object with sub-layers.

  Args:
    trainable: Whether the object collecting the variables is trainable.
    sub_layers: A flat list of Layer objects owned by this object, to collect
      variables from.
    extra_variables: Any extra variables to include. Their `.trainable` property
      is used to categorize them.

  Returns:
    A list of collected trainable weights/variables.
  """
  if not trainable:
    return []
  weights = []
  for layer in sub_layers:
    weights += layer.trainable_weights
  trainable_extra_variables = [
      v for v in extra_variables if v.trainable]
  return weights + trainable_extra_variables


def gather_non_trainable_weights(trainable, sub_layers, extra_variables):
  """Lists the non-trainable weights for an object with sub-layers.

  Args:
    trainable: Whether the object collecting the variables is trainable.
    sub_layers: A flat list of Layer objects owned by this object, to collect
      variables from.
    extra_variables: Any extra variables to include. Their `.trainable` property
      is used to categorize them.

  Returns:
    A list of collected non-trainable weights/variables.
  """
  trainable_extra_variables = []
  non_trainable_extra_variables = []
  for v in extra_variables:
    if v.trainable:
      trainable_extra_variables.append(v)
    else:
      non_trainable_extra_variables.append(v)
  weights = []
  for layer in sub_layers:
    weights += layer.non_trainable_weights
  if not trainable:
    trainable_weights = []
    for layer in sub_layers:
      trainable_weights += layer.trainable_weights
    return (trainable_weights + trainable_extra_variables
            + weights + non_trainable_extra_variables)
  return weights + non_trainable_extra_variables
