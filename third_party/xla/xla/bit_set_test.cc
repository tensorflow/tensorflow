/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/bit_set.h"

#include <random>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

using ::benchmark::DoNotOptimize;
using ::testing::benchmark::State;

class BruteForceBitSet {
 public:
  void Set(int index) { bits_.insert(index); }

  void SetAll(int num_bits) {
    for (int i = 0; i < num_bits; ++i) {
      bits_.insert(i);
    }
  }

  void Clear(int index) { bits_.erase(index); }

  bool Test(int index) const { return bits_.contains(index); }

  BruteForceBitSet& operator|=(const BruteForceBitSet& other) {
    bits_.insert(other.bits_.begin(), other.bits_.end());
    return *this;
  }

 private:
  absl::flat_hash_set<int> bits_;
};

TEST(InlinedBitSetTest, DefaultInitializedAllFalse) {
  InlinedBitSet<> bits(130);

  EXPECT_FALSE(bits.Test(0));
  EXPECT_FALSE(bits.Test(63));
  EXPECT_FALSE(bits.Test(64));
  EXPECT_FALSE(bits.Test(127));
  EXPECT_FALSE(bits.Test(128));
}

TEST(InlinedBitSetTest, Set) {
  InlinedBitSet<> bits(130);

  bits.Set(0);
  bits.Set(63);
  bits.Set(64);
  bits.Set(128);

  EXPECT_TRUE(bits.Test(0));
  EXPECT_TRUE(bits.Test(63));
  EXPECT_TRUE(bits.Test(64));
  EXPECT_FALSE(bits.Test(65));
  EXPECT_FALSE(bits.Test(127));
  EXPECT_TRUE(bits.Test(128));
}

TEST(InlinedBitSetTest, Clear) {
  InlinedBitSet<> bits(130);
  bits.Set(63);
  bits.Set(64);

  bits.Clear(63);

  EXPECT_FALSE(bits.Test(63));
  EXPECT_TRUE(bits.Test(64));
}

TEST(InlinedBitSetTest, SetAllZeroBits) {
  InlinedBitSet<> bits(0);

  bits.SetAll(0);
}

TEST(InlinedBitSetTest, SetAllExactWordBoundary) {
  InlinedBitSet<> bits(64);

  bits.SetAll(64);

  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(bits.Test(i));
  }
}

TEST(InlinedBitSetTest, SetAllNonExactBoundary) {
  InlinedBitSet<> bits(70);

  bits.SetAll(70);

  for (int i = 0; i < 70; ++i) {
    EXPECT_TRUE(bits.Test(i));
  }
}

TEST(InlinedBitSetTest, SetAllLessThanCapacity) {
  InlinedBitSet<> bits(128);

  bits.SetAll(50);

  for (int i = 0; i < 50; ++i) {
    EXPECT_TRUE(bits.Test(i));
  }
  for (int i = 50; i < 128; ++i) {
    EXPECT_FALSE(bits.Test(i));
  }
}

TEST(InlinedBitSetTest, SetAllZeroOnNonEmpty) {
  InlinedBitSet<> bits(64);

  bits.SetAll(0);

  for (int i = 0; i < 64; ++i) {
    EXPECT_FALSE(bits.Test(i));
  }
}

TEST(InlinedBitSetTest, BitwiseOr) {
  InlinedBitSet<> a(130);
  InlinedBitSet<> b(130);
  a.Set(10);
  a.Set(100);
  b.Set(70);
  b.Set(100);

  a |= b;

  EXPECT_TRUE(a.Test(10));
  EXPECT_TRUE(a.Test(70));
  EXPECT_TRUE(a.Test(100));
  EXPECT_FALSE(a.Test(0));
  EXPECT_FALSE(a.Test(129));
}

TEST(InlinedBitSetTest, MatchesBruteForceOnRandomOperations) {
  constexpr int kNumBits = 200;
  constexpr int kNumOps = 500;
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> op_dist(0, 3);
  std::uniform_int_distribution<int> bit_dist(0, kNumBits - 1);
  std::uniform_int_distribution<int> count_dist(0, kNumBits);
  InlinedBitSet<> bits(kNumBits);
  InlinedBitSet<> other_bits(kNumBits);
  BruteForceBitSet bf_bits;
  BruteForceBitSet bf_other_bits;

  for (int step = 0; step < kNumOps; ++step) {
    const int idx = bit_dist(rng);
    other_bits.Set(idx);
    bf_other_bits.Set(idx);
    switch (op_dist(rng)) {
      case 0:
        bits.Set(idx);
        bf_bits.Set(idx);
        break;
      case 1:
        bits.Clear(idx);
        bf_bits.Clear(idx);
        break;
      case 2: {
        const int count = count_dist(rng);
        bits.SetAll(count);
        bf_bits.SetAll(count);
        break;
      }
      case 3:
        bits |= other_bits;
        bf_bits |= bf_other_bits;
        break;
    }
  }

  for (int i = 0; i < kNumBits; ++i) {
    EXPECT_EQ(bits.Test(i), bf_bits.Test(i)) << "Mismatch at index " << i;
  }
}

// Reference implementation of DenseBitSets: rows of bits as a vector of
// vectors.
class BruteForceBitSetRows {
 public:
  BruteForceBitSetRows(int num_rows, int num_bits)
      : rows_(num_rows, std::vector<bool>(num_bits, false)) {}

  void Set(int row, int bit) { rows_[row][bit] = true; }
  bool Test(int row, int bit) const { return rows_[row][bit]; }

  void Or(int row, int other) {
    for (int bit = 0; bit < rows_[row].size(); ++bit) {
      rows_[row][bit] = rows_[row][bit] || rows_[other][bit];
    }
  }

  void OrIgnoringBit(int row, int other, int ignored_bit) {
    for (int bit = 0; bit < rows_[row].size(); ++bit) {
      if (bit != ignored_bit) {
        rows_[row][bit] = rows_[row][bit] || rows_[other][bit];
      }
    }
  }

 private:
  std::vector<std::vector<bool>> rows_;
};

TEST(DenseBitSetsTest, EmptyShapesAreAllowed) {
  DenseBitSets no_rows(0, 130);
  EXPECT_EQ(no_rows.num_rows(), 0);
  EXPECT_EQ(no_rows.num_bits(), 130);

  DenseBitSets no_bits(3, 0);
  EXPECT_EQ(no_bits.num_rows(), 3);
  EXPECT_EQ(no_bits.num_bits(), 0);
  no_bits.Or(0, 1);  // Rows of zero words.

  DenseBitSets empty(0, 0);
  EXPECT_EQ(empty.num_rows(), 0);
  EXPECT_EQ(empty.num_bits(), 0);
}

TEST(DenseBitSetsTest, DefaultInitializedAllFalse) {
  DenseBitSets bits(3, 130);

  EXPECT_EQ(bits.num_rows(), 3);
  EXPECT_EQ(bits.num_bits(), 130);
  for (int row = 0; row < 3; ++row) {
    EXPECT_FALSE(bits.Test(row, 0));
    EXPECT_FALSE(bits.Test(row, 63));
    EXPECT_FALSE(bits.Test(row, 64));
    EXPECT_FALSE(bits.Test(row, 129));
  }
}

TEST(DenseBitSetsTest, SetIsConfinedToItsRow) {
  DenseBitSets bits(3, 130);

  bits.Set(1, 0);
  bits.Set(1, 63);
  bits.Set(1, 64);
  bits.Set(1, 129);

  EXPECT_TRUE(bits.Test(1, 0));
  EXPECT_TRUE(bits.Test(1, 63));
  EXPECT_TRUE(bits.Test(1, 64));
  EXPECT_TRUE(bits.Test(1, 129));
  EXPECT_FALSE(bits.Test(1, 1));
  EXPECT_FALSE(bits.Test(1, 128));
  for (int bit : {0, 63, 64, 129}) {
    EXPECT_FALSE(bits.Test(0, bit));
    EXPECT_FALSE(bits.Test(2, bit));
  }
}

TEST(DenseBitSetsTest, OrAndOrIgnoringBit) {
  DenseBitSets bits(2, 130);
  bits.Set(0, 5);
  bits.Set(1, 5);
  bits.Set(1, 64);
  bits.Set(1, 129);

  bits.Or(0, 1);

  EXPECT_TRUE(bits.Test(0, 5));
  EXPECT_TRUE(bits.Test(0, 64));
  EXPECT_TRUE(bits.Test(0, 129));
  EXPECT_FALSE(bits.Test(0, 6));

  DenseBitSets more(2, 130);
  more.Set(0, 64);
  more.Set(1, 64);
  more.Set(1, 65);
  more.Set(1, 129);

  more.OrIgnoringBit(0, 1, 129);
  more.OrIgnoringBit(0, 1, 64);

  EXPECT_TRUE(more.Test(0, 64));  // Already set in the row; never cleared.
  EXPECT_TRUE(more.Test(0, 65));
  EXPECT_TRUE(more.Test(0, 129));  // Copied by the second call.
  EXPECT_TRUE(more.Test(1, 64));
}

TEST(DenseBitSetsTest, MatchesBruteForceOnRandomOperations) {
  constexpr int kNumInstances = 200;
  constexpr int kNumOps = 300;
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> rows_dist(1, 12);
  std::uniform_int_distribution<int> bits_dist(1, 300);
  std::uniform_int_distribution<int> op_dist(0, 2);

  for (int instance = 0; instance < kNumInstances; ++instance) {
    const int num_rows = rows_dist(rng);
    // Every fourth instance sits on a word boundary.
    const int num_bits =
        instance % 4 == 0 ? 64 * (1 + instance % 3) : bits_dist(rng);
    std::uniform_int_distribution<int> row_dist(0, num_rows - 1);
    std::uniform_int_distribution<int> bit_dist(0, num_bits - 1);
    DenseBitSets bits(num_rows, num_bits);
    BruteForceBitSetRows bf_bits(num_rows, num_bits);

    for (int step = 0; step < kNumOps; ++step) {
      const int row = row_dist(rng);
      const int other = row_dist(rng);
      const int bit = bit_dist(rng);
      switch (op_dist(rng)) {
        case 0:
          bits.Set(row, bit);
          bf_bits.Set(row, bit);
          break;
        case 1:
          bits.Or(row, other);
          bf_bits.Or(row, other);
          break;
        case 2:
          bits.OrIgnoringBit(row, other, bit);
          bf_bits.OrIgnoringBit(row, other, bit);
          break;
      }
      ASSERT_EQ(bits.Test(row, bit), bf_bits.Test(row, bit))
          << "instance " << instance << " step " << step << " row " << row
          << " bit " << bit;
    }
    for (int row = 0; row < num_rows; ++row) {
      for (int bit = 0; bit < num_bits; ++bit) {
        ASSERT_EQ(bits.Test(row, bit), bf_bits.Test(row, bit))
            << "instance " << instance << " row " << row << " bit " << bit;
      }
    }
  }
}

// Benchmarks: the operations TryRemoveDeadWhileParams issues, for rows of 64
// to 65,536 bits (more than twice the largest while tuple seen so far):
// - Or, once per operand edge,
// - OrIgnoringBit, once per body output,
// - Set and Test, once per get-tuple-element.
void BM_DenseBitSetsOr(State& state) {
  const int num_bits = state.range(0);
  DenseBitSets bits(2, num_bits);
  bits.Set(1, num_bits - 1);
  for (auto s : state) {
    bits.Or(0, 1);
    DoNotOptimize(bits);
  }
  state.SetBytesProcessed(state.iterations() * ((num_bits + 63) / 64) * 8);
}
BENCHMARK(BM_DenseBitSetsOr)->Arg(64)->Arg(1024)->Arg(16384)->Arg(65536);

void BM_DenseBitSetsOrIgnoringBit(State& state) {
  const int num_bits = state.range(0);
  DenseBitSets bits(2, num_bits);
  bits.Set(1, num_bits - 1);
  for (auto s : state) {
    bits.OrIgnoringBit(0, 1, num_bits / 2);
    DoNotOptimize(bits);
  }
  state.SetBytesProcessed(state.iterations() * ((num_bits + 63) / 64) * 8);
}
BENCHMARK(BM_DenseBitSetsOrIgnoringBit)
    ->Arg(64)
    ->Arg(1024)
    ->Arg(16384)
    ->Arg(65536);

void BM_DenseBitSetsSetTest(State& state) {
  const int num_bits = state.range(0);
  DenseBitSets bits(1024, num_bits);
  int row = 0;
  int bit = 0;
  bool sink = false;
  for (auto s : state) {
    bits.Set(row, bit);
    sink ^= bits.Test((row + 1) % 1024, bit);
    row = (row + 7) % 1024;
    bit = (bit + 13) % num_bits;
  }
  DoNotOptimize(sink);
}
BENCHMARK(BM_DenseBitSetsSetTest)->Arg(64)->Arg(1024)->Arg(16384)->Arg(65536);

}  // namespace
}  // namespace xla
