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

"""A key-value[] store that implements reservoir sampling on the values."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import threading


class Reservoir(object):
  """A map-to-arrays container, with deterministic Reservoir Sampling.

  Items are added with an associated key. Items may be retrieved by key, and
  a list of keys can also be retrieved. If size is not zero, then it dictates
  the maximum number of items that will be stored with each key. Once there are
  more items for a given key, they are replaced via reservoir sampling, such
  that each item has an equal probability of being included in the sample.

  Deterministic means that for any given seed and bucket size, the sequence of
  values that are kept for any given tag will always be the same, and that this
  is independent of any insertions on other tags. That is:

  >>> separate_reservoir = reservoir.Reservoir(10)
  >>> interleaved_reservoir = reservoir.Reservoir(10)
  >>> for i in xrange(100):
  >>>   separate_reservoir.AddItem('key1', i)
  >>> for i in xrange(100):
  >>>   separate_reservoir.AddItem('key2', i)
  >>> for i in xrange(100):
  >>>   interleaved_reservoir.AddItem('key1', i)
  >>>   interleaved_reservoir.AddItem('key2', i)

  separate_reservoir and interleaved_reservoir will be in identical states.

  See: https://en.wikipedia.org/wiki/Reservoir_sampling

  Adding items has amortized O(1) runtime.

  """

  def __init__(self, size, seed=0, always_keep_last=True):
    """Creates a new reservoir.

    Args:
      size: The number of values to keep in the reservoir for each tag. If 0,
        all values will be kept.
      seed: The seed of the random number generator to use when sampling.
        Different values for |seed| will produce different samples from the same
        input items.
      always_keep_last: Whether to always keep the latest seen item in the
        end of the reservoir. Defaults to True.

    Raises:
      ValueError: If size is negative or not an integer.
    """
    if size < 0 or size != round(size):
      raise ValueError('size must be nonegative integer, was %s' % size)
    self._buckets = collections.defaultdict(
        lambda: _ReservoirBucket(size, random.Random(seed), always_keep_last))
    # _mutex guards the keys - creating new keys, retrieving by key, etc
    # the internal items are guarded by the ReservoirBuckets' internal mutexes
    self._mutex = threading.Lock()

  def Keys(self):
    """Return all the keys in the reservoir.

    Returns:
      ['list', 'of', 'keys'] in the Reservoir.
    """
    with self._mutex:
      return list(self._buckets.keys())

  def Items(self, key):
    """Return items associated with given key.

    Args:
      key: The key for which we are finding associated items.

    Raises:
      KeyError: If the key is not found in the reservoir.

    Returns:
      [list, of, items] associated with that key.
    """
    with self._mutex:
      if key not in self._buckets:
        raise KeyError('Key %s was not found in Reservoir' % key)
      bucket = self._buckets[key]
    return bucket.Items()

  def AddItem(self, key, item, f=lambda x: x):
    """Add a new item to the Reservoir with the given tag.

    If the reservoir has not yet reached full size, the new item is guaranteed
    to be added. If the reservoir is full, then behavior depends on the
    always_keep_last boolean.

    If always_keep_last was set to true, the new item is guaranteed to be added
    to the reservoir, and either the previous last item will be replaced, or
    (with low probability) an older item will be replaced.

    If always_keep_last was set to false, then the new item will replace an
    old item with low probability.

    If f is provided, it will be applied to transform item (lazily, iff item is
      going to be included in the reservoir).

    Args:
      key: The key to store the item under.
      item: The item to add to the reservoir.
      f: An optional function to transform the item prior to addition.
    """
    with self._mutex:
      bucket = self._buckets[key]
    bucket.AddItem(item, f)

  def FilterItems(self, filterFn, key=None):
    """Filter items within a Reservoir, using a filtering function.

    Args:
      filterFn: A function that returns True for the items to be kept.
      key: An optional bucket key to filter. If not specified, will filter all
        all buckets.

    Returns:
      The number of items removed.
    """
    with self._mutex:
      if key:
        if key in self._buckets:
          return self._buckets[key].FilterItems(filterFn)
        else:
          return 0
      else:
        return sum(bucket.FilterItems(filterFn)
                   for bucket in self._buckets.values())


class _ReservoirBucket(object):
  """A container for items from a stream, that implements reservoir sampling.

  It always stores the most recent item as its final item.
  """

  def __init__(self, _max_size, _random=None, always_keep_last=True):
    """Create the _ReservoirBucket.

    Args:
      _max_size: The maximum size the reservoir bucket may grow to. If size is
        zero, the bucket has unbounded size.
      _random: The random number generator to use. If not specified, defaults to
        random.Random(0).
      always_keep_last: Whether the latest seen item should always be included
        in the end of the bucket.

    Raises:
      ValueError: if the size is not a nonnegative integer.
    """
    if _max_size < 0 or _max_size != round(_max_size):
      raise ValueError('_max_size must be nonegative int, was %s' % _max_size)
    self.items = []
    # This mutex protects the internal items, ensuring that calls to Items and
    # AddItem are thread-safe
    self._mutex = threading.Lock()
    self._max_size = _max_size
    self._num_items_seen = 0
    if _random is not None:
      self._random = _random
    else:
      self._random = random.Random(0)
    self.always_keep_last = always_keep_last

  def AddItem(self, item, f=lambda x: x):
    """Add an item to the ReservoirBucket, replacing an old item if necessary.

    The new item is guaranteed to be added to the bucket, and to be the last
    element in the bucket. If the bucket has reached capacity, then an old item
    will be replaced. With probability (_max_size/_num_items_seen) a random item
    in the bucket will be popped out and the new item will be appended
    to the end. With probability (1 - _max_size/_num_items_seen)
    the last item in the bucket will be replaced.

    Since the O(n) replacements occur with O(1/_num_items_seen) likelihood,
    the amortized runtime is O(1).

    Args:
      item: The item to add to the bucket.
      f: A function to transform item before addition, if it will be kept in
        the reservoir.
    """
    with self._mutex:
      if len(self.items) < self._max_size or self._max_size == 0:
        self.items.append(f(item))
      else:
        r = self._random.randint(0, self._num_items_seen)
        if r < self._max_size:
          self.items.pop(r)
          self.items.append(f(item))
        elif self.always_keep_last:
          self.items[-1] = f(item)
      self._num_items_seen += 1

  def FilterItems(self, filterFn):
    """Filter items in a ReservoirBucket, using a filtering function.

    Filtering items from the reservoir bucket must update the
    internal state variable self._num_items_seen, which is used for determining
    the rate of replacement in reservoir sampling. Ideally, self._num_items_seen
    would contain the exact number of items that have ever seen by the
    ReservoirBucket and satisfy filterFn. However, the ReservoirBucket does not
    have access to all items seen -- it only has access to the subset of items
    that have survived sampling (self.items). Therefore, we estimate
    self._num_items_seen by scaling it by the same ratio as the ratio of items
    not removed from self.items.

    Args:
      filterFn: A function that returns True for items to be kept.

    Returns:
      The number of items removed from the bucket.
    """
    with self._mutex:
      size_before = len(self.items)
      self.items = list(filter(filterFn, self.items))
      size_diff = size_before - len(self.items)

      # Estimate a correction the number of items seen
      prop_remaining = len(self.items) / float(
          size_before) if size_before > 0 else 0
      self._num_items_seen = int(round(self._num_items_seen * prop_remaining))
      return size_diff

  def Items(self):
    """Get all the items in the bucket."""
    with self._mutex:
      return list(self.items)
