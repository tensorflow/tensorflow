/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <array>

#include "absl/time/clock.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/presized_cuckoo_map.h"

namespace tensorflow {
namespace {

TEST(PresizedCuckooMapTest, MultiplyHigh) {
  struct Testcase {
    uint64 x;
    uint64 y;
    uint64 result;
  };
  std::array<Testcase, 7> testcases{
      {{0, 0, 0},
       {0xffffffff, 0xffffffff, 0},
       {0x2, 0xf000000000000000, 1},
       {0x3, 0xf000000000000000, 2},
       {0x3, 0xf000000000000001, 2},
       {0x3, 0xffffffffffffffff, 2},
       {0xffffffffffffffff, 0xffffffffffffffff, 0xfffffffffffffffe}}};
  for (auto &tc : testcases) {
    EXPECT_EQ(tc.result, presized_cuckoo_map::multiply_high_u64(tc.x, tc.y));
  }
}

TEST(PresizedCuckooMapTest, Basic) {
  PresizedCuckooMap<int> pscm(1000);
  EXPECT_TRUE(pscm.InsertUnique(1, 2));
  int out;
  EXPECT_TRUE(pscm.Find(1, &out));
  EXPECT_EQ(out, 2);
}

TEST(PresizedCuckooMapTest, Prefetch) {
  {
    PresizedCuckooMap<int64> pscm(2);
    EXPECT_TRUE(pscm.InsertUnique(1, 2));
    // Works for both present and absent keys.
    pscm.PrefetchKey(1);
    pscm.PrefetchKey(2);
  }

  // Do not run in debug mode, when prefetch is not implemented, or when
  // sanitizers are enabled.
#if defined(NDEBUG) && defined(__GNUC__) && !defined(ADDRESS_SANITIZER) && \
    !defined(MEMORY_SANITIZER) && !defined(THREAD_SANITIZER) &&            \
    !defined(UNDEFINED_BEHAVIOR_SANITIZER)
  const auto now = [] { return absl::Now(); };

  // Make size enough to not fit in L2 cache (16.7 Mb)
  static constexpr int size = 1 << 22;
  PresizedCuckooMap<int64> pscm(size);
  for (int i = 0; i < size; ++i) {
    pscm.InsertUnique(i, i);
  }

  absl::Duration no_prefetch, prefetch;
  int64 out;
  for (int iter = 0; iter < 10; ++iter) {
    auto time = now();
    for (int i = 0; i < size; ++i) {
      testing::DoNotOptimize(pscm.Find(i, &out));
    }
    no_prefetch += now() - time;

    time = now();
    for (int i = 0; i < size; ++i) {
      pscm.PrefetchKey(i + 20);
      testing::DoNotOptimize(pscm.Find(i, &out));
    }
    prefetch += now() - time;
  }

  // no_prefetch is at least 30% slower.
  EXPECT_GE(1.0 * no_prefetch / prefetch, 1.3);
#endif
}

TEST(PresizedCuckooMapTest, TooManyItems) {
  static constexpr int kTableSize = 1000;
  PresizedCuckooMap<int> pscm(kTableSize);
  for (uint64 i = 0; i < kTableSize; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64)));
    ASSERT_TRUE(pscm.InsertUnique(key, i));
  }
  // Try to over-fill the table.  A few of these
  // inserts will succeed, but should start failing.
  uint64 failed_at = 0;
  for (uint64 i = kTableSize; i < (2 * kTableSize); i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64)));
    if (!pscm.InsertUnique(key, i)) {
      failed_at = i;
      break;
    }
  }
  // Requirement 1:  Table must return failure when it's full.
  EXPECT_NE(failed_at, 0);

  // Requirement 2:  Table must preserve all items inserted prior
  // to the failure.
  for (uint64 i = 0; i < failed_at; i++) {
    int out;
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64)));
    EXPECT_TRUE(pscm.Find(key, &out));
    EXPECT_EQ(out, i);
  }
}

TEST(PresizedCuckooMapTest, ZeroSizeMap) {
  PresizedCuckooMap<int> pscm(0);
  int out;
  for (uint64 i = 0; i < 100; i++) {
    EXPECT_FALSE(pscm.Find(i, &out));
  }
}

TEST(PresizedCuckooMapTest, RepeatedClear) {
  PresizedCuckooMap<int> pscm(2);
  int out;
  for (int i = 0; i < 100; ++i) {
    pscm.InsertUnique(0, 0);
    pscm.InsertUnique(1, 1);
    EXPECT_TRUE(pscm.Find(0, &out));
    EXPECT_EQ(0, out);
    EXPECT_TRUE(pscm.Find(1, &out));
    EXPECT_EQ(1, out);
    pscm.Clear(2);
    EXPECT_FALSE(pscm.Find(0, &out));
    EXPECT_FALSE(pscm.Find(1, &out));
  }
}

void RunFill(int64 table_size) {
  PresizedCuckooMap<int> pscm(table_size);
  for (int64 i = 0; i < table_size; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64)));
    EXPECT_TRUE(pscm.InsertUnique(key, i));
  }
  for (int64 i = 0; i < table_size; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64)));
    int out;
    EXPECT_TRUE(pscm.Find(key, &out));
    EXPECT_EQ(out, i);
  }
}

TEST(PresizedCuckooMapTest, Fill) {
  for (int64 table_size = 10; table_size <= 5000000; table_size *= 71) {
    RunFill(table_size);
  }
}

TEST(PresizedCuckooMapTest, Duplicates) {
  static constexpr int kSmallTableSize = 1000;
  PresizedCuckooMap<int> pscm(kSmallTableSize);

  for (uint64 i = 0; i < kSmallTableSize; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(uint64)));
    EXPECT_TRUE(pscm.InsertUnique(key, i));
  }

  for (uint64 i = 0; i < kSmallTableSize; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(uint64)));
    EXPECT_FALSE(pscm.InsertUnique(key, i));
  }
}

static void CalculateKeys(uint64 num, std::vector<uint64> *dst) {
  dst->resize(num);
  for (uint64 i = 0; i < num; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(uint64)));
    dst->at(i) = key;
  }
}

static void BM_CuckooFill(int iters, int arg) {
  uint64 table_size = arg;
  testing::StopTiming();
  std::vector<uint64> calculated_keys;
  CalculateKeys(table_size, &calculated_keys);
  testing::StartTiming();
  for (int iter = 0; iter < iters; iter++) {
    PresizedCuckooMap<int> pscm(table_size);
    for (uint64 i = 0; i < table_size; i++) {
      pscm.InsertUnique(calculated_keys[i], i);
    }
  }
}

BENCHMARK(BM_CuckooFill)->Arg(1000)->Arg(10000000);

static void BM_CuckooRead(int iters, int arg) {
  uint64 table_size = arg;
  testing::StopTiming();
  std::vector<uint64> calculated_keys;
  CalculateKeys(table_size, &calculated_keys);
  PresizedCuckooMap<int> pscm(table_size);
  for (uint64 i = 0; i < table_size; i++) {
    pscm.InsertUnique(calculated_keys[i], i);
  }
  testing::StartTiming();
  uint64_t defeat_optimization = 0;
  for (int i = 0; i < iters; i++) {
    uint64 key_index = i % table_size;  // May slow down bench!
    int out = 0;
    pscm.Find(calculated_keys[key_index], &out);
    defeat_optimization += out;
  }
  if (defeat_optimization == 0) {
    printf("Preventing the compiler from eliding the inner loop\n");
  }
}

BENCHMARK(BM_CuckooRead)->Arg(1000)->Arg(10000000);

}  // namespace
}  // namespace tensorflow
