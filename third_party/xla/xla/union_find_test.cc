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

#include "xla/union_find.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

TEST(UnionFindTest, MergeKeepsTheCallersValue) {
  UnionFind<int> a(1);
  UnionFind<int> b(2);
  UnionFind<int> c(3);

  a.Merge(&b);
  c.Merge(&a);

  EXPECT_EQ(a.Get(), 3);
  EXPECT_EQ(b.Get(), 3);
  EXPECT_EQ(c.Get(), 3);
  EXPECT_EQ(a.Size(), 3);
}

TEST(DenseUnionFindTest, StartsAsSingletons) {
  DenseUnionFind sets(4);

  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_EQ(sets.Find(i), i);
    EXPECT_EQ(sets.Size(i), 1);
  }
}

TEST(DenseUnionFindTest, MergeJoinsClusters) {
  DenseUnionFind sets(6);

  sets.Merge(0, 1);
  sets.Merge(2, 3);
  sets.Merge(1, 3);
  sets.Merge(4, 4);

  EXPECT_EQ(sets.Find(0), sets.Find(3));
  EXPECT_EQ(sets.Find(1), sets.Find(2));
  EXPECT_NE(sets.Find(0), sets.Find(4));
  EXPECT_NE(sets.Find(4), sets.Find(5));
  EXPECT_EQ(sets.Size(2), 4);
  EXPECT_EQ(sets.Size(4), 1);
}

// Reference: a label per element, relabelled on every merge.
class BruteForceSets {
 public:
  explicit BruteForceSets(int64_t n) : label_(n) {
    for (int64_t i = 0; i < n; ++i) label_[i] = i;
  }
  bool SameSet(int64_t a, int64_t b) const { return label_[a] == label_[b]; }
  int64_t Size(int64_t a) const {
    return std::count(label_.begin(), label_.end(), label_[a]);
  }
  void Merge(int64_t a, int64_t b) {
    const int64_t from = label_[b];
    const int64_t to = label_[a];
    for (int64_t& label : label_) {
      if (label == from) label = to;
    }
  }

 private:
  std::vector<int64_t> label_;
};

TEST(DenseUnionFindTest, MatchesBruteForceOnRandomMerges) {
  constexpr int kNumInstances = 100;
  constexpr int kNumMerges = 200;
  std::mt19937 rng(11);
  std::uniform_int_distribution<int64_t> size_dist(1, 60);

  for (int instance = 0; instance < kNumInstances; ++instance) {
    const int64_t n = size_dist(rng);
    std::uniform_int_distribution<int64_t> element(0, n - 1);
    DenseUnionFind sets(n);
    BruteForceSets reference(n);

    for (int merge = 0; merge < kNumMerges; ++merge) {
      const int64_t a = element(rng);
      const int64_t b = element(rng);
      sets.Merge(a, b);
      reference.Merge(a, b);
      const int64_t x = element(rng);
      const int64_t y = element(rng);
      ASSERT_EQ(sets.Find(x) == sets.Find(y), reference.SameSet(x, y))
          << "instance " << instance << " merge " << merge;
      ASSERT_EQ(sets.Size(x), reference.Size(x))
          << "instance " << instance << " merge " << merge;
    }
    for (int64_t x = 0; x < n; ++x) {
      for (int64_t y = 0; y < n; ++y) {
        ASSERT_EQ(sets.Find(x) == sets.Find(y), reference.SameSet(x, y))
            << "instance " << instance << " pair " << x << ", " << y;
      }
    }
  }
}

// The access pattern of TryRemoveDeadWhileParams: one merge per operand edge
// followed by one find per tuple index, over the instruction slots.
void BM_DenseUnionFindMergeThenFind(::testing::benchmark::State& state) {
  const int64_t n = state.range(0);
  std::mt19937 rng(5);
  std::uniform_int_distribution<int64_t> element(0, n - 1);
  std::vector<int64_t> edges(2 * n);
  for (int64_t& e : edges) e = element(rng);
  for (auto s : state) {
    DenseUnionFind sets(n);
    for (int64_t i = 0; i + 1 < edges.size(); i += 2) {
      sets.Merge(edges[i], edges[i + 1]);
    }
    int64_t sink = 0;
    for (int64_t i = 0; i < n; ++i) sink += sets.Find(i);
    ::benchmark::DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * 2 * n);
}
BENCHMARK(BM_DenseUnionFindMergeThenFind)->Arg(1024)->Arg(100000);

}  // namespace
}  // namespace xla
