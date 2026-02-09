// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_text/core/kernels/disjoint_set_forest.h"

#include <stddef.h>

#include <set>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tensorflow {
namespace text {

// Testing rig.
//
// Template args:
//   Forest: An instantiation of the DisjointSetForest<> template.
template <class Forest>
class DisjointSetForestTest : public ::testing::Test {
 protected:
  using Index = typename Forest::IndexType;

  // Expects that the |expected_sets| and |forest| match.
  void ExpectSets(const std::set<std::set<Index>> &expected_sets,
                  Forest *forest) {
    std::set<std::pair<Index, Index>> expected_pairs;
    for (const auto &expected_set : expected_sets) {
      for (auto it = expected_set.begin(); it != expected_set.end(); ++it) {
        for (auto jt = expected_set.begin(); jt != expected_set.end(); ++jt) {
          expected_pairs.emplace(*it, *jt);
        }
      }
    }

    for (Index lhs = 0; lhs < forest->size(); ++lhs) {
      for (Index rhs = 0; rhs < forest->size(); ++rhs) {
        if (expected_pairs.find({lhs, rhs}) != expected_pairs.end()) {
          EXPECT_EQ(forest->FindRoot(lhs), forest->FindRoot(rhs));
          EXPECT_TRUE(forest->SameSet(lhs, rhs));
        } else {
          EXPECT_NE(forest->FindRoot(lhs), forest->FindRoot(rhs));
          EXPECT_FALSE(forest->SameSet(lhs, rhs));
        }
      }
    }
  }
};

using Forests = ::testing::Types<
    DisjointSetForest<uint8, false>, DisjointSetForest<uint8, true>,
    DisjointSetForest<uint16, false>, DisjointSetForest<uint16, true>,
    DisjointSetForest<uint32, false>, DisjointSetForest<uint32, true>,
    DisjointSetForest<uint64, false>, DisjointSetForest<uint64, true>>;
TYPED_TEST_SUITE(DisjointSetForestTest, Forests);

TYPED_TEST(DisjointSetForestTest, DefaultEmpty) {
  TypeParam forest;
  EXPECT_EQ(0, forest.size());
}

TYPED_TEST(DisjointSetForestTest, InitEmpty) {
  TypeParam forest;
  forest.Init(0);
  EXPECT_EQ(0, forest.size());
}

TYPED_TEST(DisjointSetForestTest, Populated) {
  TypeParam forest;
  forest.Init(5);
  EXPECT_EQ(5, forest.size());
  this->ExpectSets({{0}, {1}, {2}, {3}, {4}}, &forest);

  forest.UnionOfRoots(1, 2);
  this->ExpectSets({{0}, {1, 2}, {3}, {4}}, &forest);

  forest.Union(1, 2);
  this->ExpectSets({{0}, {1, 2}, {3}, {4}}, &forest);

  forest.UnionOfRoots(0, 4);
  this->ExpectSets({{0, 4}, {1, 2}, {3}}, &forest);

  forest.Union(3, 4);
  this->ExpectSets({{0, 3, 4}, {1, 2}}, &forest);

  forest.Union(0, 3);
  this->ExpectSets({{0, 3, 4}, {1, 2}}, &forest);

  forest.Union(2, 0);
  this->ExpectSets({{0, 1, 2, 3, 4}}, &forest);

  forest.Union(1, 3);
  this->ExpectSets({{0, 1, 2, 3, 4}}, &forest);
}

// Testing rig for checking that when union by rank is disabled, the root of a
// merged set can be controlled.
class DisjointSetForestNoUnionByRankTest : public ::testing::Test {
 protected:
  using Forest = DisjointSetForest<uint32, false>;

  // Expects that the roots of the |forest| match |expected_roots|.
  void ExpectRoots(const std::vector<uint32> &expected_roots, Forest *forest) {
    ASSERT_EQ(expected_roots.size(), forest->size());
    for (uint32 i = 0; i < forest->size(); ++i) {
      EXPECT_EQ(expected_roots[i], forest->FindRoot(i));
    }
  }
};

TEST_F(DisjointSetForestNoUnionByRankTest, ManuallySpecifyRoot) {
  Forest forest;
  forest.Init(5);
  ExpectRoots({0, 1, 2, 3, 4}, &forest);

  forest.UnionOfRoots(0, 1);  // 1 is the root
  ExpectRoots({1, 1, 2, 3, 4}, &forest);

  forest.Union(4, 3);  // 3 is the root
  ExpectRoots({1, 1, 2, 3, 3}, &forest);

  forest.Union(0, 2);  // 2 is the root
  ExpectRoots({2, 2, 2, 3, 3}, &forest);

  forest.Union(3, 3);  // no effect
  ExpectRoots({2, 2, 2, 3, 3}, &forest);

  forest.Union(4, 0);  // 2 is the root
  ExpectRoots({2, 2, 2, 2, 2}, &forest);
}

}  // namespace text
}  // namespace tensorflow
