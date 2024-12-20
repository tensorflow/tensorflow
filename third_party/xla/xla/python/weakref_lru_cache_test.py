# Copyright 2023 The OpenXLA Authors.
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

import gc
import threading
import time
import weakref

from absl.testing import absltest

from xla.python import xla_client


class WeakrefLRUCacheTest(absltest.TestCase):

  def testMultiThreaded(self):
    insert_evs = [threading.Event() for _ in range(2)]
    insert_evs_i = 0

    class WRKey:
      pass

    class ClashingKey:

      def __eq__(self, other):
        return False

      def __hash__(self):
        return 333  # induce maximal caching problems.

    class GilReleasingCacheKey:

      def __eq__(self, other):
        nonlocal insert_evs_i
        if isinstance(other, GilReleasingCacheKey) and insert_evs_i < len(
            insert_evs
        ):
          insert_evs[insert_evs_i].set()
          insert_evs_i += 1
          time.sleep(0.01)
        return False

      def __hash__(self):
        return 333  # induce maximal caching problems.

    def CacheFn(obj, gil_releasing_cache_key):
      del obj
      del gil_releasing_cache_key
      return None

    cache = xla_client.weakref_lru_cache(lambda: None, CacheFn, 2048)

    wrkey = WRKey()

    def Body():
      for insert_ev in insert_evs:
        insert_ev.wait()
        for _ in range(20):
          cache(wrkey, ClashingKey())

    t = threading.Thread(target=Body)
    t.start()
    for _ in range(3):
      cache(wrkey, GilReleasingCacheKey())
    t.join()

  def testAnotherMultiThreaded(self):
    num_workers = 5
    barrier = threading.Barrier(num_workers)
    cache = xla_client.weakref_lru_cache(lambda: None, lambda x, y: y, 2048)

    class WRKey:
      pass

    def WorkerAddToCache():
      barrier.wait()
      wrkey = WRKey()
      for i in range(10):
        cache(wrkey, i)

    def WorkerCleanCache():
      barrier.wait()
      for _ in range(10):
        cache.cache_clear()

    workers = [
        threading.Thread(target=WorkerAddToCache)
        for _ in range(num_workers - 1)
    ] + [threading.Thread(target=WorkerCleanCache)]

    for t in workers:
      t.start()

    for t in workers:
      t.join()

  def testKwargsDictOrder(self):
    miss_id = 0

    class WRKey:
      pass

    def CacheFn(obj, kwkey1, kwkey2):
      del obj, kwkey1, kwkey2
      nonlocal miss_id
      miss_id += 1
      return miss_id

    cache = xla_client.weakref_lru_cache(lambda: None, CacheFn, 4)

    wrkey = WRKey()

    self.assertEqual(cache(wrkey, kwkey1="a", kwkey2="b"), 1)
    self.assertEqual(cache(wrkey, kwkey1="b", kwkey2="a"), 2)
    self.assertEqual(cache(wrkey, kwkey2="b", kwkey1="a"), 1)

  def testGetKeys(self):
    def CacheFn(obj, arg):
      del obj
      return arg + "extra"

    cache = xla_client.weakref_lru_cache(lambda: None, CacheFn, 4)

    class WRKey:
      pass

    wrkey = WRKey()

    self.assertEmpty(cache.cache_keys())
    cache(wrkey, "arg1")
    cache(wrkey, "arg2")
    self.assertLen(cache.cache_keys(), 2)

  def testNonWeakreferenceableKey(self):
    class NonWRKey:
      __slots__ = ()

    non_wr_key = NonWRKey()
    with self.assertRaises(TypeError):
      weakref.ref(non_wr_key)

    cache = xla_client.weakref_lru_cache(lambda: None, lambda x: 2048)
    for _ in range(100):
      with self.assertRaises(TypeError):
        cache(non_wr_key)

  def testCrashingKey(self):
    class WRKey:
      pass

    class CrashingKey:
      # A key that raises exceptions if eq or hash is called.

      def __eq__(self, other):
        raise ValueError("eq")

      def __hash__(self):
        raise ValueError("hash")

    cache = xla_client.weakref_lru_cache(lambda: None, lambda x, y: y, 2048)
    wrkey = WRKey()
    with self.assertRaises(ValueError):
      for _ in range(100):
        cache(wrkey, CrashingKey())

  def testPrintingStats(self):
    class WRKey:
      pass

    cache = xla_client.weakref_lru_cache(lambda: None, lambda x, y: y, 2048)
    wrkey = WRKey()
    for i in range(10):
      cache(wrkey, i)
    for i in range(5):
      cache(wrkey, i)

    self.assertEqual(
        repr(cache.cache_info()),
        "WeakrefLRUCache(hits=5, misses=10, maxsize=2048, currsize=10)",
    )

  def testGCKeys(self):
    class WRKey:

      def __init__(self, x):
        self.x = x

      def __eq__(self, other):
        return self.x == other.x

      def __hash__(self):
        return hash(self.x)

    cache = xla_client.weakref_lru_cache(lambda: None, lambda x, y: y, 2048)
    keys = [WRKey(i) for i in range(10)]
    for i in range(10):
      cache(keys[i], i)

    # Delete some keys, to exercise the weakref callback behavior.
    del keys[::2]

    for key in keys:
      cache(key, 7)

  def testTpTraverse(self):
    class WRKey:
      pass

    def CacheContextFn():
      return None

    def CallFn(x, y, *args, **kwargs):
      del x, args, kwargs
      return y

    cache = xla_client.weakref_lru_cache(CacheContextFn, CallFn, 2048)

    keys = [WRKey() for _ in range(10)]
    values = [str(i) for i in range(10)]
    args = [str(i) for i in range(10)]
    kwargs = {"a": "b"}

    for key, value in zip(keys, values):
      cache(key, value, *args, **kwargs)

    expected_refs = (
        [
            CacheContextFn,
            CallFn,
            xla_client._xla.WeakrefLRUCache,
            kwargs,
        ]
        + [weakref.getweakrefs(key)[0] for key in keys]
        + values
        + args
    )

    # Can't use assertContainsSubset because it doesn't support kwargs since
    # dicts aren't hashable.
    for ref in expected_refs:
      self.assertIn(ref, gc.get_referents(cache))


if __name__ == "__main__":
  absltest.main()
