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

#include "xla/inlined_bit_set.h"

#include <random>

#include "absl/container/flat_hash_set.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

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

}  // namespace
}  // namespace xla
