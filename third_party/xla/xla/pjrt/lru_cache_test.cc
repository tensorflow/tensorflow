/* Copyright 2021 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/pjrt/lru_cache.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

using ::testing::UnorderedElementsAreArray;

// Returns the key-value entries in the provided cache. The returned entries are
// invalidated if the provided cache is modified or destructed.
template <typename K, typename V>
std::vector<std::tuple<const K&, const V&>> entries(
    const LRUCache<K, V>& cache) {
  std::vector<std::tuple<const K&, const V&>> entries;
  cache.ForEach(
      [&entries](const auto& k, const auto& v) { entries.push_back({k, v}); });
  return entries;
}

TEST(LRUCache, Basics) {
  LRUCache<int, int>::LRUList list(3);
  LRUCache<int, int> cache(&list);
  EXPECT_EQ(3, cache.Capacity());
  EXPECT_EQ(0, cache.Size());

  EXPECT_EQ(0, cache.GetOrCreateIfAbsent(0, [](int) { return 0; }));
  EXPECT_EQ(1, cache.Size());
  std::vector<std::tuple<int, int>> want = {{0, 0}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(1, cache.GetOrCreateIfAbsent(1, [](int) { return 1; }));
  EXPECT_EQ(2, cache.Size());
  want = {{0, 0}, {1, 1}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(2, cache.GetOrCreateIfAbsent(2, [](int) { return 2; }));
  EXPECT_EQ(3, cache.Size());
  want = {{0, 0}, {1, 1}, {2, 2}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(0, cache.GetOrCreateIfAbsent(0, [](int) { return 3; }));
  EXPECT_EQ(3, cache.Size());
  want = {{1, 1}, {2, 2}, {0, 0}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(4, cache.GetOrCreateIfAbsent(3, [](int) { return 4; }));
  EXPECT_EQ(3, cache.Size());
  want = {{2, 2}, {0, 0}, {3, 4}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(2, cache.GetOrCreateIfAbsent(2, [](int) { return 5; }));
  EXPECT_EQ(3, cache.Size());
  want = {{0, 0}, {3, 4}, {2, 2}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(6, cache.GetOrCreateIfAbsent(1, [](int) { return 6; }));
  EXPECT_EQ(3, cache.Size());
  want = {{3, 4}, {2, 2}, {1, 6}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  cache.Clear();
  EXPECT_EQ(0, cache.Size());
  want = {};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(6, cache.GetOrCreateIfAbsent(1, [](int) { return 6; }));
  EXPECT_EQ(1, cache.Size());
  want = {{1, 6}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));
}

TEST(LRUCache, Remove) {
  LRUCache<int, int>::LRUList list(3);
  LRUCache<int, int> cache(&list);

  EXPECT_EQ(0, cache.GetOrCreateIfAbsent(0, [](int) { return 0; }));
  EXPECT_EQ(1, cache.Size());
  std::vector<std::tuple<int, int>> want = {{0, 0}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(1, cache.GetOrCreateIfAbsent(1, [](int) { return 1; }));
  EXPECT_EQ(2, cache.Size());
  want = {{0, 0}, {1, 1}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(2, cache.GetOrCreateIfAbsent(2, [](int) { return 2; }));
  EXPECT_EQ(3, cache.Size());
  want = {{0, 0}, {1, 1}, {2, 2}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(0, cache.GetOrCreateIfAbsent(0, [](int) { return 3; }));
  EXPECT_EQ(3, cache.Size());
  want = {{1, 1}, {2, 2}, {0, 0}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  cache.Remove(2);
  EXPECT_EQ(2, cache.Size());
  want = {{1, 1}, {0, 0}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  cache.Remove(0);
  EXPECT_EQ(1, cache.Size());
  want = {{1, 1}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  EXPECT_EQ(4, cache.GetOrCreateIfAbsent(0, [](int) { return 4; }));
  EXPECT_EQ(2, cache.Size());
  want = {{1, 1}, {0, 4}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));

  cache.Remove(1);
  EXPECT_EQ(1, cache.Size());
  want = {{0, 4}};
  EXPECT_THAT(entries(cache), UnorderedElementsAreArray(want));
}

TEST(LRUCache, SharedLRUList) {
  LRUCache<int, int>::LRUList list(2);
  LRUCache<int, int> cache1(&list);
  LRUCache<int, int> cache2(&list);
  EXPECT_EQ(2, list.Capacity());

  EXPECT_EQ(0, cache1.Size());
  EXPECT_EQ(0, cache2.Size());
  EXPECT_EQ(0, cache1.GetOrCreateIfAbsent(0, [](int) { return 0; }));
  EXPECT_EQ(1, list.Size());
  EXPECT_EQ(1, cache1.Size());
  EXPECT_EQ(0, cache2.Size());

  EXPECT_EQ(1, cache2.GetOrCreateIfAbsent(1, [](int) { return 1; }));
  EXPECT_EQ(2, list.Size());
  EXPECT_EQ(1, cache1.Size());
  EXPECT_EQ(1, cache2.Size());
  EXPECT_EQ(2, cache1.GetOrCreateIfAbsent(2, [](int) { return 2; }));
  EXPECT_EQ(2, list.Size());
  EXPECT_EQ(1, cache1.Size());
  EXPECT_EQ(1, cache2.Size());

  EXPECT_EQ(1, cache2.GetOrCreateIfAbsent(1, [](int) { return -1; }));
  EXPECT_EQ(2, list.Size());
  EXPECT_EQ(1, cache1.Size());
  EXPECT_EQ(1, cache2.Size());

  cache1.Clear();
  EXPECT_EQ(1, list.Size());
  EXPECT_EQ(0, cache1.Size());
  EXPECT_EQ(1, cache2.Size());

  EXPECT_EQ(1, cache2.GetOrCreateIfAbsent(1, [](int) { return 4; }));
  EXPECT_EQ(1, list.Size());
  EXPECT_EQ(0, cache1.Size());
  EXPECT_EQ(1, cache2.Size());
  EXPECT_EQ(7, cache1.GetOrCreateIfAbsent(7, [](int) { return 7; }));
  EXPECT_EQ(2, list.Size());
  EXPECT_EQ(1, cache1.Size());
  EXPECT_EQ(1, cache2.Size());

  list.Clear();
  EXPECT_EQ(0, list.Size());
  EXPECT_EQ(0, cache1.Size());
  EXPECT_EQ(0, cache2.Size());
  EXPECT_EQ(2, cache1.GetOrCreateIfAbsent(2, [](int) { return 2; }));
}

TEST(LRUCache, RandomInsertions) {
  LRUCache<int, int>::LRUList list(7);
  LRUCache<int, int> cache(&list);
  std::random_device rng;
  std::uniform_int_distribution<int> dist(0, 100);

  for (int i = 0; i < 1000; ++i) {
    EXPECT_LE(cache.Size(), std::min(cache.Capacity(), i));
    int key = dist(rng);
    int k = -1;
    int v = cache.GetOrCreateIfAbsent(key, [&](int k_arg) {
      CHECK_EQ(k_arg, key);
      k = k_arg;
      return k_arg * 37;
    });
    EXPECT_TRUE(k == -1 || k == key);
    EXPECT_EQ(v, key * 37);
  }
}

TEST(LRUCache, RandomOperations) {
  constexpr int num_lists = 3;
  constexpr int num_iterations = 10000;

  using Cache = LRUCache<int, int>;
  using List = Cache::LRUList;
  std::vector<std::unique_ptr<List>> lists;
  std::vector<std::unique_ptr<Cache>> caches;
  for (int i = 0; i < num_lists; ++i) {
    lists.push_back(std::make_unique<List>(8));
    caches.push_back(std::make_unique<Cache>(lists.back().get()));
    caches.push_back(std::make_unique<Cache>(lists.back().get()));
  }

  absl::BitGen bitgen;
  for (int i = 0; i < num_iterations; ++i) {
    Cache& cache = *caches[absl::Uniform(bitgen, 0u, caches.size())];
    const int key = absl::Uniform(bitgen, 0, 1000);
    double fraction = absl::Uniform(bitgen, 0, 1.0);
    if (fraction < 0.6) {
      auto f = [](int k) { return k * 37; };
      EXPECT_EQ(f(key), cache.GetOrCreateIfAbsent(key, f));
    } else if (fraction < 0.9) {
      cache.Remove(key);
    } else {
      cache.Clear();
    }
  }
}

TEST(LRUCache, ReentrantClear) {
  struct Value {
    explicit Value(LRUCache<int, std::shared_ptr<Value>>* cache)
        : cache(cache) {}
    ~Value() { cache->Clear(); }

    LRUCache<int, std::shared_ptr<Value>>* cache;
  };

  LRUCache<int, std::shared_ptr<Value>>::LRUList list(3);
  LRUCache<int, std::shared_ptr<Value>> cache(&list);

  cache.GetOrCreateIfAbsent(
      0, [&](int) { return std::make_shared<Value>(&cache); });
  cache.Clear();
}

struct Int {
  int val;
};

struct IntHash {
  std::size_t operator()(Int*) const { return 0; }
};

struct IntEq {
  bool operator()(Int* x, Int* y) const { return x->val == y->val; }
};

TEST(LRUCache, ChangingKeysDoesntCrash) {
  // This test checks the behavior of LRUCache in the face of changing keys.
  // When keys change, the cache may return the wrong result but should never
  // crash.
  {
    LRUCache<Int*, int, IntHash, IntEq>::LRUList list(2);
    LRUCache<Int*, int, IntHash, IntEq> cache(&list);
    Int a{1};
    Int b{2};
    Int c{3};
    cache.GetOrCreateIfAbsent(&a, [](Int*) { return 1; });
    cache.GetOrCreateIfAbsent(&b, [](Int*) { return 2; });
    a.val = b.val;
    cache.GetOrCreateIfAbsent(&c, [](Int*) { return 3; });
  }

  {
    LRUCache<Int*, int, IntHash, IntEq>::LRUList list(2);
    LRUCache<Int*, int, IntHash, IntEq> cache(&list);
    Int a{1};
    Int b{2};
    Int c{3};
    cache.GetOrCreateIfAbsent(&a, [](Int*) { return 1; });
    cache.GetOrCreateIfAbsent(&b, [](Int*) { return 2; });
    b.val = a.val;
    cache.GetOrCreateIfAbsent(&c, [](Int*) { return 3; });
  }
}

TEST(LRUCache, ChangingKeysReturnsWrongValue) {
  // In this test, we insert key x and value 1 into the cache. We then change
  // the value of x to be equal to y and look up y in the cache. Because x is
  // equal to y and because of the hash collision between the original value of
  // x and y, the cache returns 1 even though 1 was not computed with a value
  // equal to y.
  LRUCache<Int*, int, IntHash, IntEq>::LRUList list(1);
  LRUCache<Int*, int, IntHash, IntEq> cache(&list);
  Int x{1};
  Int y{2};
  cache.GetOrCreateIfAbsent(&x, [](Int*) { return 1; });
  x.val = y.val;
  EXPECT_EQ(cache.GetOrCreateIfAbsent(&y, [](Int*) { return 2; }), 1);
}

TEST(LRUCache, NonDefaultConstructibleKeys) {
  struct Pair {
    Pair(int x, int y) : x(x), y(y) {}
    int x = 0;
    int y = 0;
  };

  struct PairHash {
    std::size_t operator()(Pair) { return 0; };
  };

  struct PairEq {
    bool operator()(Pair, Pair) { return true; };
  };

  using Cache = LRUCache<Pair, Pair, PairHash, PairEq>;
  Cache::LRUList list(10);
  Cache cache(&list);
  cache.GetOrCreateIfAbsent(Pair(1, 2), [](const Pair& p) { return p; });
}

}  // namespace
}  // namespace xla
