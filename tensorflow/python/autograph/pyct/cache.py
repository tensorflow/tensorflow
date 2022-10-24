# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Caching utilities."""

import inspect
import weakref


# TODO(mdan): Add a garbage collection hook for cleaning up modules.
class _TransformedFnCache(object):
  """Generic hierarchical cache for transformed functions.

  The keys are soft references (i.e. they are discarded when the key is
  destroyed) created from the source function by `_get_key`. The subkeys are
  strong references and can be any value. Typically they identify different
  kinds of transformation.
  """

  __slots__ = ('_cache',)

  def __init__(self):
    self._cache = weakref.WeakKeyDictionary()

  def _get_key(self, entity):
    raise NotImplementedError('subclasses must override')

  def has(self, entity, subkey):
    key = self._get_key(entity)
    parent = self._cache.get(key, None)
    if parent is None:
      return False
    return subkey in parent

  def __getitem__(self, entity):
    key = self._get_key(entity)
    parent = self._cache.get(key, None)
    if parent is None:
      # The bucket is initialized to support this usage:
      #   cache[key][subkey] = value
      self._cache[key] = parent = {}
    return parent

  def __len__(self):
    return len(self._cache)


class CodeObjectCache(_TransformedFnCache):
  """A function cache based on code objects.

  Code objects are good proxies for the source code of a function.

  This cache efficiently handles functions that share code objects, such as
  functions defined in a loop, bound methods, etc.

  The cache falls back to the function object, if it doesn't have a code object.
  """

  def _get_key(self, entity):
    if hasattr(entity, '__code__'):
      return entity.__code__
    else:
      return entity


class UnboundInstanceCache(_TransformedFnCache):
  """A function cache based on unbound function objects.

  Using the function for the cache key allows efficient handling of object
  methods.

  Unlike the _CodeObjectCache, this discriminates between different functions
  even if they have the same code. This is needed for decorators that may
  masquerade as another function.
  """

  def _get_key(self, entity):
    if inspect.ismethod(entity):
      return entity.__func__
    return entity


