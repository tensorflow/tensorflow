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

#include "tensorflow_text/core/kernels/spanning_tree_iterator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace text {

// Testing rig.  When the bool parameter is true, iterates over spanning forests
// instead of spanning trees.
class SpanningTreeIteratorTest : public ::testing::TestWithParam<bool> {
 protected:
  using SourceList = SpanningTreeIterator::SourceList;

  // Returns |base|^|exponent|.  Computes the value as an integer to avoid
  // rounding issues.
  static int Pow(int base, int exponent) {
    double real_product = 1.0;
    int product = 1;
    for (int i = 0; i < exponent; ++i) {
      product *= base;
      real_product *= base;
    }
    CHECK_EQ(product, real_product) << "Overflow detected.";
    return product;
  }

  // Expects that the number of possible spanning trees for a complete digraph
  // of |num_nodes| nodes is |expected_num_trees|.
  void ExpectNumTrees(int num_nodes, int expected_num_trees) {
    int actual_num_trees = 0;
    iterator_.ForEachTree(
        num_nodes, [&](const SourceList &sources) { ++actual_num_trees; });
    LOG(INFO) << "num_nodes=" << num_nodes
              << " expected_num_trees=" << expected_num_trees
              << " actual_num_trees=" << actual_num_trees;
    EXPECT_EQ(expected_num_trees, actual_num_trees);
  }

  // Expects that the set of possible spanning trees for a complete digraph of
  // |num_nodes| nodes is |expected_trees|.
  void ExpectTrees(int num_nodes, const std::set<SourceList> &expected_trees) {
    std::set<SourceList> actual_trees;
    iterator_.ForEachTree(num_nodes, [&](const SourceList &sources) {
      CHECK(actual_trees.insert(sources).second);
    });
    EXPECT_EQ(expected_trees, actual_trees);
  }

  // Instance for tests.  Shared across assertions in a test to exercise reuse.
  SpanningTreeIterator iterator_{GetParam()};
};

INSTANTIATE_TEST_SUITE_P(AllowForest, SpanningTreeIteratorTest,
                         ::testing::Bool());

TEST_P(SpanningTreeIteratorTest, NumberOfTrees) {
  // According to Cayley's formula, the number of undirected spanning trees on a
  // complete graph of n nodes is n^{n-2}:
  // https://en.wikipedia.org/wiki/Cayley%27s_formula
  //
  // To count the number of directed spanning trees, note that each undirected
  // spanning tree gives rise to n directed spanning trees: choose one of the n
  // nodes as the root, and then orient arcs outwards.  Therefore, the number of
  // directed spanning trees on a complete digraph of n nodes is n^{n-1}.
  //
  // To count the number of directed spanning forests, consider undirected
  // spanning trees on a complete graph of n+1 nodes.  Arbitrarily select one
  // node as the artificial root, orient arcs outwards, and then delete the
  // artificial root and its outbound arcs.  The result is a directed spanning
  // forest on n nodes.  Therefore, the number of directed spanning forests on a
  // complete digraph of n nodes is (n+1)^{n-1}.
  for (int num_nodes = 1; num_nodes <= 7; ++num_nodes) {
    if (GetParam()) {  // forest
      ExpectNumTrees(num_nodes, Pow(num_nodes + 1, num_nodes - 1));
    } else {  // tree
      ExpectNumTrees(num_nodes, Pow(num_nodes, num_nodes - 1));
    }
  }
}

TEST_P(SpanningTreeIteratorTest, OneNodeDigraph) { ExpectTrees(1, {{0}}); }

TEST_P(SpanningTreeIteratorTest, TwoNodeDigraph) {
  if (GetParam()) {                            // forest
    ExpectTrees(2, {{0, 0}, {0, 1}, {1, 1}});  // {0, 1} is two-root structure
  } else {                                     // tree
    ExpectTrees(2, {{0, 0}, {1, 1}});
  }
}

TEST_P(SpanningTreeIteratorTest, ThreeNodeDigraph) {
  if (GetParam()) {  // forest
    ExpectTrees(3, {{0, 0, 0},
                    {0, 0, 1},
                    {0, 0, 2},  // 2-root
                    {0, 1, 0},  // 2-root
                    {0, 1, 1},  // 2-root
                    {0, 1, 2},  // 3-root
                    {0, 2, 0},
                    {0, 2, 2},  // 2-root
                    {1, 1, 0},
                    {1, 1, 1},
                    {1, 1, 2},  // 2-root
                    {1, 2, 2},
                    {2, 0, 2},
                    {2, 1, 1},
                    {2, 1, 2},  // 2-root
                    {2, 2, 2}});
  } else {  // tree
    ExpectTrees(3, {{0, 0, 0},
                    {0, 0, 1},
                    {0, 2, 0},
                    {1, 1, 0},
                    {1, 1, 1},
                    {1, 2, 2},
                    {2, 0, 2},
                    {2, 1, 1},
                    {2, 2, 2}});
  }
}

}  // namespace text
}  // namespace tensorflow
