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
  PresizedCuckooMap<int64> pscm(2);
  EXPECT_TRUE(pscm.InsertUnique(1, 2));
  // Works for both present and absent keys.
  pscm.PrefetchKey(1);
  pscm.PrefetchKey(2);
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

void RunFill(int64_t table_size) {
  PresizedCuckooMap<int> pscm(table_size);
  for (int64_t i = 0; i < table_size; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64)));
    EXPECT_TRUE(pscm.InsertUnique(key, i));
  }
  for (int64_t i = 0; i < table_size; i++) {
    uint64 key =
        Fingerprint64(string(reinterpret_cast<char *>(&i), sizeof(int64)));
    int out;
    EXPECT_TRUE(pscm.Find(key, &out));
    EXPECT_EQ(out, i);
  }
}

TEST(PresizedCuckooMapTest, Fill) {
  for (int64_t table_size = 10; table_size <= 5000000; table_size *= 71) {
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

void BM_CuckooFill(::testing::benchmark::State &state) {
  const int arg = state.range(0);

  uint64 table_size = arg;
  std::vector<uint64> calculated_keys;
  CalculateKeys(table_size, &calculated_keys);
  for (auto s : state) {
    PresizedCuckooMap<int> pscm(table_size);
    for (uint64 i = 0; i < table_size; i++) {
      pscm.InsertUnique(calculated_keys[i], i);
    }
  }
}

BENCHMARK(BM_CuckooFill)->Arg(1000)->Arg(10000000);

void BM_CuckooRead(::testing::benchmark::State &state) {
  const int arg = state.range(0);

  uint64 table_size = arg;
  std::vector<uint64> calculated_keys;
  CalculateKeys(table_size, &calculated_keys);
  PresizedCuckooMap<int> pscm(table_size);
  for (uint64 i = 0; i < table_size; i++) {
    pscm.InsertUnique(calculated_keys[i], i);
  }

  int i = 0;
  for (auto s : state) {
    // Avoid using '%', which is expensive.
    uint64 key_index = i;
    ++i;
    if (i == table_size) i = 0;

    int out = 0;
    pscm.Find(calculated_keys[key_index], &out);
    tensorflow::testing::DoNotOptimize(out);
  }
}

BENCHMARK(BM_CuckooRead)->Arg(1000)->Arg(10000000);

}  // namespace
}  // namespace tensorflow
