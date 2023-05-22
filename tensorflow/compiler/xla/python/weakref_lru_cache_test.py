# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

import threading
import time

from absl.testing import absltest

from tensorflow.compiler.xla.python import xla_client


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


if __name__ == "__main__":
  absltest.main()
