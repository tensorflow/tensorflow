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

#include <random>
#include <vector>

#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

using ::benchmark::DoNotOptimize;
using ::testing::benchmark::State;

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

TEST(DenseUnionFindTest, ZeroElements) {
  DenseUnionFind sets(0);
  EXPECT_EQ(sets.num_elements(), 0);
}

TEST(DenseUnionFindTest, StartsAsSingletons) {
  DenseUnionFind sets(4);

  EXPECT_EQ(sets.num_elements(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(sets.Find(i), i);
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
  EXPECT_EQ(sets.Find(4), 4);
  EXPECT_EQ(sets.Find(5), 5);
}

// Union by rank: the root of higher rank represents the merged cluster, and a
// tie keeps the first argument's root and raises its rank.
TEST(DenseUnionFindTest, RepresentativeIsTheRootOfHigherRank) {
  DenseUnionFind sets(5);

  sets.Merge(0, 1);  // Tie: 0 stays the root, now of rank 1.
  EXPECT_EQ(sets.Find(1), 0);
  sets.Merge(2, 0);  // 2 has rank 0, so 0 keeps representing the cluster.
  EXPECT_EQ(sets.Find(2), 0);
  sets.Merge(3, 4);  // Tie: 3 becomes a root of rank 1.
  sets.Merge(3, 0);  // Tie between ranks 1: 3 stays the root, now of rank 2.
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(sets.Find(i), 3);
  }
}

// Reference implementation: one label per element, relabelled on every merge.
class BruteForceSets {
 public:
  explicit BruteForceSets(int n) : label_(n) {
    for (int i = 0; i < n; ++i) {
      label_[i] = i;
    }
  }
  bool SameSet(int a, int b) const { return label_[a] == label_[b]; }
  void Merge(int a, int b) {
    const int from = label_[b];
    const int to = label_[a];
    for (int& label : label_) {
      if (label == from) {
        label = to;
      }
    }
  }

 private:
  std::vector<int> label_;
};

TEST(DenseUnionFindTest, MatchesBruteForceOnRandomMerges) {
  constexpr int kNumInstances = 100;
  constexpr int kNumMerges = 200;
  std::mt19937 rng(11);
  std::uniform_int_distribution<int> size_dist(1, 60);

  for (int instance = 0; instance < kNumInstances; ++instance) {
    const int n = size_dist(rng);
    std::uniform_int_distribution<int> element(0, n - 1);
    DenseUnionFind sets(n);
    BruteForceSets reference(n);

    for (int merge = 0; merge < kNumMerges; ++merge) {
      const int a = element(rng);
      const int b = element(rng);
      sets.Merge(a, b);
      reference.Merge(a, b);
      const int x = element(rng);
      const int y = element(rng);
      ASSERT_EQ(sets.Find(x) == sets.Find(y), reference.SameSet(x, y))
          << "instance " << instance << " merge " << merge;
    }
    for (int x = 0; x < n; ++x) {
      for (int y = 0; y < n; ++y) {
        ASSERT_EQ(sets.Find(x) == sets.Find(y), reference.SameSet(x, y))
            << "instance " << instance << " pair " << x << ", " << y;
      }
    }
  }
}

// The access pattern of TryRemoveDeadWhileParams over the instruction slots of
// a loop, including the construction of the structure:
// - one merge per operand edge,
// - then one find per tuple index.
std::vector<int> RandomEdges(int n) {
  std::mt19937 rng(5);
  std::uniform_int_distribution<int> element(0, n - 1);
  std::vector<int> edges(2 * n);
  for (int& e : edges) {
    e = element(rng);
  }
  return edges;
}

void BM_DenseUnionFindTryRemoveDeadWhileParamsPattern(State& state) {
  const int n = state.range(0);
  const std::vector<int> edges = RandomEdges(n);
  for (auto s : state) {
    DenseUnionFind sets(n);
    for (int i = 0; i + 1 < edges.size(); i += 2) {
      sets.Merge(edges[i], edges[i + 1]);
    }
    int sink = 0;
    for (int i = 0; i < n; ++i) {
      sink += sets.Find(i);
    }
    DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * 2 * n);
}
BENCHMARK(BM_DenseUnionFindTryRemoveDeadWhileParamsPattern)
    ->Arg(1024)
    ->Arg(100000)
    ->Arg(1 << 20)
    ->Arg(1 << 22);

// The same pattern on UnionFind<T>: one node per element, holding its index as
// the value, so a find is a Get.
void BM_UnionFindTryRemoveDeadWhileParamsPattern(State& state) {
  const int n = state.range(0);
  const std::vector<int> edges = RandomEdges(n);
  for (auto s : state) {
    std::vector<UnionFind<int>> sets;
    sets.reserve(n);
    for (int i = 0; i < n; ++i) {
      sets.emplace_back(i);
    }
    for (int i = 0; i + 1 < edges.size(); i += 2) {
      sets[edges[i]].Merge(&sets[edges[i + 1]]);
    }
    int sink = 0;
    for (int i = 0; i < n; ++i) {
      sink += sets[i].Get();
    }
    DoNotOptimize(sink);
  }
  state.SetItemsProcessed(state.iterations() * 2 * n);
}
BENCHMARK(BM_UnionFindTryRemoveDeadWhileParamsPattern)
    ->Arg(1024)
    ->Arg(100000)
    ->Arg(1 << 20)
    ->Arg(1 << 22);

}  // namespace
}  // namespace xla
