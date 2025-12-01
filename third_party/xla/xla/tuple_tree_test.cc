/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/tuple_tree.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Pair;
using ::testing::Pointee;

// Test fixture for TupleTree.
class TupleTreeTest : public ::testing::Test {};

namespace {

TEST_F(TupleTreeTest, SingleLeafConstructor) {
  TupleTree<int> tree(42);
  EXPECT_TRUE(tree.IsLeaf({}));
  EXPECT_EQ(tree.element({}), 42);
}

TEST_F(TupleTreeTest, EmptyConstructor) {
  TupleTree<int> tree;
  EXPECT_FALSE(tree.IsLeaf({}));
  EXPECT_THAT(tree.nodes(), ElementsAre(Pair(ShapeIndex({}), 0)));
  EXPECT_THAT(tree.leaves(), ElementsAre());
  EXPECT_EQ(tree.element({}), 0);
}

TEST_F(TupleTreeTest, SingleLeafMoveConstructor) {
  auto val = std::make_unique<int>(42);
  TupleTree<std::unique_ptr<int>> tree(std::move(val));
  EXPECT_TRUE(tree.IsLeaf({}));
  EXPECT_THAT(tree.element({}), Pointee(Eq(42)));
  EXPECT_EQ(val, nullptr);
}

TEST_F(TupleTreeTest, NodeWithValueAndChildren) {
  using Node = TupleTree<int>::Node;

  // Node with value and empty children (empty tuple)
  TupleTree<int> tree1(Node::Tuple(100));
  EXPECT_FALSE(tree1.IsLeaf({}));  // It's a tuple, not a leaf
  EXPECT_THAT(tree1.nodes(), ElementsAre(Pair(ShapeIndex({}), 100)));
  EXPECT_THAT(tree1.leaves(), ElementsAre());

  // Node with value and non-empty children
  TupleTree<int> tree2(Node::Tuple(200, {Node::Leaf(1), Node::Leaf(2)}));
  EXPECT_FALSE(tree2.IsLeaf({}));
  EXPECT_TRUE(tree2.IsLeaf({0}));
  EXPECT_EQ(tree2.element({0}), 1);
  EXPECT_TRUE(tree2.IsLeaf({1}));
  EXPECT_EQ(tree2.element({1}), 2);
  EXPECT_THAT(tree2.nodes(),
              ElementsAre(Pair(ShapeIndex({}), 200), Pair(ShapeIndex({0}), 1),
                          Pair(ShapeIndex({1}), 2)));
  EXPECT_THAT(tree2.leaves(),
              ElementsAre(Pair(ShapeIndex({0}), 1), Pair(ShapeIndex({1}), 2)));

  // Accessing the value of a non-leaf node is not directly supported
  // by element() which is only for leaves.
}

TEST_F(TupleTreeTest, NodeValueAndChildrenConstructors) {
  using Node = TupleTree<int>::Node;

  // Test various constructors
  TupleTree<int> tree1(Node::Tuple(10, {Node::Leaf(1), Node::Leaf(2)}));
  EXPECT_FALSE(tree1.IsLeaf({}));
  EXPECT_EQ(tree1.element({0}), 1);
  EXPECT_EQ(tree1.element({1}), 2);

  std::vector<Node> children;
  children.push_back(Node::Leaf(3));
  children.push_back(Node::Leaf(4));
  TupleTree<int> tree2(Node::Tuple(20, std::move(children)));
  EXPECT_FALSE(tree2.IsLeaf({}));
  EXPECT_EQ(tree2.element({0}), 3);
  EXPECT_EQ(tree2.element({1}), 4);

  int val = 30;
  TupleTree<int> tree3(Node::Tuple(val, {Node::Leaf(5)}));
  EXPECT_FALSE(tree3.IsLeaf({}));
  EXPECT_EQ(tree3.element({0}), 5);

  int val2 = 40;
  std::vector<Node> children2;
  children2.push_back(Node::Leaf(6));
  TupleTree<int> tree4(Node::Tuple(std::move(val2), std::move(children2)));
  EXPECT_FALSE(tree4.IsLeaf({}));
  EXPECT_EQ(tree4.element({0}), 6);
}

TEST_F(TupleTreeTest, DistinguishEmptyTupleFromLeaf) {
  using Node = TupleTree<int>::Node;

  // Leaf node
  TupleTree<int> leaf_tree(Node::Leaf(42));
  EXPECT_TRUE(leaf_tree.IsLeaf({}));
  EXPECT_EQ(leaf_tree.element({}), 42);
  EXPECT_THAT(leaf_tree.nodes(), ElementsAre(Pair(ShapeIndex({}), 42)));
  EXPECT_THAT(leaf_tree.leaves(), ElementsAre(Pair(ShapeIndex({}), 42)));

  // Empty tuple node (with a value)
  TupleTree<int> empty_tuple_tree(Node::Tuple(100));
  EXPECT_FALSE(empty_tuple_tree.IsLeaf({}));
  EXPECT_THAT(empty_tuple_tree.nodes(), ElementsAre(Pair(ShapeIndex({}), 100)));
  EXPECT_THAT(empty_tuple_tree.leaves(), ElementsAre());

  // Empty tuple node (without a value, default constructor)
  TupleTree<int> default_empty_tuple_tree;
  EXPECT_FALSE(default_empty_tuple_tree.IsLeaf({}));
  EXPECT_THAT(default_empty_tuple_tree.nodes(),
              ElementsAre(Pair(ShapeIndex({}), 0)));
  EXPECT_THAT(default_empty_tuple_tree.leaves(), ElementsAre());
}

TEST_F(TupleTreeTest, ElementAccessDeathTest) {
  using Node = TupleTree<int>::Node;
  TupleTree<int> tree(
      {Node::Leaf(1), Node::Tuple({Node::Leaf(2), Node::Leaf(3)})});

  // Index out of bounds
  EXPECT_DEATH(tree.element({5}), "Index out of bounds");

  // Index too deep
  EXPECT_DEATH(tree.element({0, 0}), "Cannot index into a leaf node");
}

TEST_F(TupleTreeTest, MutableElementAccessDeathTest) {
  using Node = TupleTree<int>::Node;
  TupleTree<int> tree(
      {Node::Leaf(1), Node::Tuple({Node::Leaf(2), Node::Leaf(3)})});

  // Index out of bounds
  EXPECT_DEATH(tree.mutable_element({5}), "Index out of bounds");

  // Index too deep
  EXPECT_DEATH(tree.mutable_element({0, 0}), "Cannot index into a leaf node");
}

TEST_F(TupleTreeTest, IsLeafInvalidIndex) {
  TupleTree<int> tree({1, 2});
  EXPECT_FALSE(tree.IsLeaf({5}));     // Out of bounds
  EXPECT_FALSE(tree.IsLeaf({0, 0}));  // Too deep
}

TEST_F(TupleTreeTest, CopySubtreeFromOverwriteLeafWithTuple) {
  TupleTree<int> src_tree({10, 20});
  TupleTree<int> dst_tree(5);  // Dst is initially a leaf

  dst_tree.CopySubtreeFrom(src_tree, {}, {});  // Overwrite root

  EXPECT_FALSE(dst_tree.IsLeaf({}));
  EXPECT_EQ(dst_tree.element({0}), 10);
  EXPECT_EQ(dst_tree.element({1}), 20);
}

TEST_F(TupleTreeTest, CopySubtreeFromOverwriteTupleWithLeaf) {
  TupleTree<int> src_tree(10);
  TupleTree<int> dst_tree({1, 2});  // Dst is initially a tuple

  dst_tree.CopySubtreeFrom(src_tree, {}, {});  // Overwrite root

  EXPECT_TRUE(dst_tree.IsLeaf({}));
  EXPECT_EQ(dst_tree.element({}), 10);
}

TEST_F(TupleTreeTest, CopySubtreeFromCreateNodes) {
  TupleTree<int> src_tree(10);
  TupleTree<int> dst_tree(5);

  // Graft src_tree at {1, 0}, creating node {1}
  dst_tree.CopySubtreeFrom(src_tree, {}, {1, 0});

  EXPECT_TRUE(dst_tree.IsLeaf({0}));
  EXPECT_EQ(dst_tree.element({0}), 5);
  EXPECT_FALSE(dst_tree.IsLeaf({1}));
  EXPECT_TRUE(dst_tree.IsLeaf({1, 0}));
  EXPECT_EQ(dst_tree.element({1, 0}), 10);
}

TEST_F(TupleTreeTest, ElementAccessIndexErrors) {
  TupleTree<int> tree({1, 2});

  // Negative index
  EXPECT_DEATH(tree.element({-1}), "Negative index in ShapeIndex");

  // Index into a leaf
  EXPECT_DEATH(tree.element({0, 0}), "Cannot index into a leaf node");

  // Out of bounds
  EXPECT_DEATH(tree.element({2}), "Index out of bounds");
}

TEST_F(TupleTreeTest, FlatInitializerListConstructor) {
  TupleTree<int> tree({1, 2, 3});

  EXPECT_FALSE(tree.IsLeaf({}));
  EXPECT_TRUE(tree.IsLeaf({0}));
  EXPECT_EQ(tree.element({0}), 1);
  EXPECT_TRUE(tree.IsLeaf({1}));
  EXPECT_EQ(tree.element({1}), 2);
  EXPECT_TRUE(tree.IsLeaf({2}));
  EXPECT_EQ(tree.element({2}), 3);
}

TEST_F(TupleTreeTest, NestedInitializerListConstructor) {
  using Node = TupleTree<int>::Node;
  TupleTree<int> tree({Node::Leaf(1),
                       Node::Tuple({Node::Leaf(2), Node::Leaf(3)}),
                       Node::Leaf(4)});

  EXPECT_FALSE(tree.IsLeaf({}));
  EXPECT_TRUE(tree.IsLeaf({0}));
  EXPECT_EQ(tree.element({0}), 1);

  EXPECT_FALSE(tree.IsLeaf({1}));
  EXPECT_TRUE(tree.IsLeaf({1, 0}));
  EXPECT_EQ(tree.element({1, 0}), 2);
  EXPECT_TRUE(tree.IsLeaf({1, 1}));
  EXPECT_EQ(tree.element({1, 1}), 3);

  EXPECT_TRUE(tree.IsLeaf({2}));
  EXPECT_EQ(tree.element({2}), 4);
}

TEST_F(TupleTreeTest, IndicesAndValuesConstructor) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {
      {{0}, 10}, {{1, 0}, 20}, {{1, 1}, 30}};
  TupleTree<int> tree(absl::MakeSpan(leaves));

  EXPECT_FALSE(tree.IsLeaf({}));
  EXPECT_TRUE(tree.IsLeaf({0}));
  EXPECT_EQ(tree.element({0}), 10);

  EXPECT_FALSE(tree.IsLeaf({1}));
  EXPECT_TRUE(tree.IsLeaf({1, 0}));
  EXPECT_EQ(tree.element({1, 0}), 20);
  EXPECT_TRUE(tree.IsLeaf({1, 1}));
  EXPECT_EQ(tree.element({1, 1}), 30);
}

TEST_F(TupleTreeTest, ElementAccess) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {
      {{0}, 1}, {{1, 0}, 2}, {{1, 1}, 3}};
  TupleTree<int> tree(absl::MakeSpan(leaves));

  EXPECT_EQ(tree.element({0}), 1);
  EXPECT_EQ(tree.element({1, 0}), 2);
  EXPECT_EQ(tree.element({1, 1}), 3);
}

TEST_F(TupleTreeTest, MutableElementAccess) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {
      {{0}, 1}, {{1, 0}, 2}, {{1, 1}, 3}};
  TupleTree<int> tree(absl::MakeSpan(leaves));

  *tree.mutable_element({0}) = 100;
  EXPECT_EQ(tree.element({0}), 100);

  *tree.mutable_element({1, 1}) = 300;
  EXPECT_EQ(tree.element({1, 1}), 300);
}

TEST_F(TupleTreeTest, IsLeaf) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {{{0}, 1}, {{1, 0}, 2}};
  TupleTree<int> tree(absl::MakeSpan(leaves));

  EXPECT_TRUE(tree.IsLeaf({0}));
  EXPECT_FALSE(tree.IsLeaf({1}));
  EXPECT_TRUE(tree.IsLeaf({1, 0}));
}

TEST_F(TupleTreeTest, CopySubtreeFrom) {
  std::vector<std::pair<ShapeIndex, int>> src_leaves = {{{0}, 10}, {{1}, 20}};
  TupleTree<int> src_tree(absl::MakeSpan(src_leaves));

  TupleTree<int> dst_tree(0);  // Single leaf

  // Graft the whole src_tree at {1} in dst_tree
  dst_tree.CopySubtreeFrom(src_tree, {}, {1});

  EXPECT_TRUE(dst_tree.IsLeaf({0}));
  EXPECT_EQ(dst_tree.element({0}), 0);
  EXPECT_FALSE(dst_tree.IsLeaf({1}));
  EXPECT_EQ(dst_tree.element({1, 0}), 10);
  EXPECT_EQ(dst_tree.element({1, 1}), 20);

  // Copy a subtree from src
  TupleTree<int> dst2_tree(0);
  dst2_tree.CopySubtreeFrom(src_tree, {1}, {0});
  EXPECT_FALSE(dst2_tree.IsLeaf({}));
  EXPECT_TRUE(dst2_tree.IsLeaf({0}));
  EXPECT_EQ(dst2_tree.element({0}), 20);
}

TEST_F(TupleTreeTest, CopySubtreeFromRoot) {
  std::vector<std::pair<ShapeIndex, int>> src_leaves = {{{0}, 10}, {{1}, 20}};
  TupleTree<int> src_tree(absl::MakeSpan(src_leaves));

  TupleTree<int> dst_tree(0);  // Single leaf

  // Copy a leaf subtree from src to the root of dst_tree
  dst_tree.CopySubtreeFrom(src_tree, {1}, {});
  EXPECT_TRUE(dst_tree.IsLeaf({}));
  EXPECT_EQ(dst_tree.element({}), 20);

  // Copy a tuple subtree from src to the root of dst_tree
  dst_tree.CopySubtreeFrom(src_tree, {}, {});
  EXPECT_FALSE(dst_tree.IsLeaf({}));
  EXPECT_EQ(dst_tree.element({0}), 10);
  EXPECT_EQ(dst_tree.element({1}), 20);
}

TEST_F(TupleTreeTest, CopyCompatibleSubtreeFromSuccess) {
  using Node = TupleTree<int>::Node;
  TupleTree<int> src_tree(
      Node::Tuple(100, {Node::Leaf(1), Node::Tuple(200, {Node::Leaf(2)})}));
  TupleTree<int> dst_tree(
      Node::Tuple(-1, {Node::Leaf(-2), Node::Tuple(-3, {Node::Leaf(-4)})}));

  // Copy entire tree
  EXPECT_OK(dst_tree.CopyCompatibleSubtreeFrom(src_tree, {}, {}));
  EXPECT_EQ(dst_tree.element({}), 100);
  EXPECT_EQ(dst_tree.element({0}), 1);
  EXPECT_EQ(dst_tree.element({1}), 200);
  EXPECT_EQ(dst_tree.element({1, 0}), 2);

  // Reset dst_tree
  dst_tree = TupleTree<int>(
      Node::Tuple(-1, {Node::Leaf(-2), Node::Tuple(-3, {Node::Leaf(-4)})}));
  // Copy subtree from {1} in src to {1} in dst
  EXPECT_OK(dst_tree.CopyCompatibleSubtreeFrom(src_tree, {1}, {1}));
  EXPECT_EQ(dst_tree.element({}), -1);   // Unchanged
  EXPECT_EQ(dst_tree.element({0}), -2);  // Unchanged
  EXPECT_EQ(dst_tree.element({1}), 200);
  EXPECT_EQ(dst_tree.element({1, 0}), 2);

  // Copy leaf subtree
  EXPECT_OK(dst_tree.CopyCompatibleSubtreeFrom(src_tree, {0}, {0}));
  EXPECT_EQ(dst_tree.element({0}), 1);
}

TEST_F(TupleTreeTest, CopyCompatibleSubtreeFromFailure) {
  using Node = TupleTree<int>::Node;
  TupleTree<int> src_tree(Node::Tuple(100, {Node::Leaf(1), Node::Leaf(2)}));
  TupleTree<int> dst_tree(
      Node::Tuple(-1, {Node::Leaf(-2)}));  // Different child count

  EXPECT_THAT(
      dst_tree.CopyCompatibleSubtreeFrom(src_tree, {}, {}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Subtree structures are incompatible: different number of children"));

  TupleTree<int> dst_tree2(Node::Leaf(-1));  // Dst is leaf, src is tuple
  EXPECT_THAT(
      dst_tree2.CopyCompatibleSubtreeFrom(src_tree, {}, {}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Subtree structures are incompatible: different number of children"));

  TupleTree<int> src_tree2(Node::Leaf(1));
  TupleTree<int> dst_tree3(
      Node::Tuple(-1, {Node::Leaf(-2)}));  // Src is leaf, Dst is tuple
  EXPECT_THAT(
      dst_tree3.CopyCompatibleSubtreeFrom(src_tree2, {}, {}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Subtree structures are incompatible: different number of children"));

  // Compatible at root, but not at deeper level
  TupleTree<int> src_tree3(
      Node::Tuple(100, {Node::Leaf(1), Node::Tuple(200, {Node::Leaf(2)})}));
  TupleTree<int> dst_tree4(
      Node::Tuple(-1, {Node::Leaf(-2), Node::Leaf(-3)}));  // {1} is leaf in dst

  EXPECT_THAT(
      dst_tree4.CopyCompatibleSubtreeFrom(src_tree3, {}, {}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Subtree structures are incompatible: different number of children"));
}

TEST_F(TupleTreeTest, CopyCompatibleSubtreeFromIndexErrors) {
  TupleTree<int> src_tree({1, 2});
  TupleTree<int> dst_tree({3, 4});

  EXPECT_THAT(dst_tree.CopyCompatibleSubtreeFrom(src_tree, {5}, {}),
              StatusIs(absl::StatusCode::kNotFound, "Index out of bounds"));
  EXPECT_THAT(dst_tree.CopyCompatibleSubtreeFrom(src_tree, {}, {5}),
              StatusIs(absl::StatusCode::kNotFound, "Index out of bounds"));
}

TEST_F(TupleTreeTest, SubTree) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {
      {{0}, 10}, {{1, 0}, 20}, {{1, 1}, 30}};
  TupleTree<int> tree(absl::MakeSpan(leaves));

  TF_ASSERT_OK_AND_ASSIGN(TupleTree<int> sub_tree, tree.Subtree({1}));
  EXPECT_FALSE(sub_tree.IsLeaf({}));
  EXPECT_EQ(sub_tree.element({0}), 20);
  EXPECT_EQ(sub_tree.element({1}), 30);

  TF_ASSERT_OK_AND_ASSIGN(TupleTree<int> leaf_sub_tree, tree.Subtree({0}));
  EXPECT_TRUE(leaf_sub_tree.IsLeaf({}));
  EXPECT_EQ(leaf_sub_tree.element({}), 10);

  EXPECT_THAT(tree.Subtree({2}), StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(TupleTreeTest, ForEachElement) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {
      {{0}, 10}, {{1, 0}, 20}, {{1, 1}, 30}};
  TupleTree<int> tree(absl::MakeSpan(leaves));

  std::vector<std::pair<ShapeIndex, int>> visited;
  tree.ForEachElement([&](const ShapeIndex& index, int value) {
    visited.push_back({index, value});
  });

  EXPECT_THAT(
      visited,
      ElementsAre(Pair(ShapeIndex({}), 0), Pair(ShapeIndex({0}), 10),
                  Pair(ShapeIndex({1}), 0), Pair(ShapeIndex({1, 0}), 20),
                  Pair(ShapeIndex({1, 1}), 30)));
}

TEST_F(TupleTreeTest, ForEachMutableElement) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {
      {{0}, 10}, {{1, 0}, 20}, {{1, 1}, 30}};
  TupleTree<int> tree(absl::MakeSpan(leaves));

  int64_t offset = 0;
  tree.ForEachMutableElement(
      [&](const ShapeIndex& index, int* value) { *value += offset++; });

  EXPECT_EQ(tree.element({}), 0);       // 0 + 0
  EXPECT_EQ(tree.element({0}), 11);     // 10 + 1
  EXPECT_EQ(tree.element({1}), 2);      // 0 + 2
  EXPECT_EQ(tree.element({1, 0}), 23);  // 20 + 3
  EXPECT_EQ(tree.element({1, 1}), 34);  // 30 + 4
}

TEST_F(TupleTreeTest, ForEachMutableElementSingleLeaf) {
  TupleTree<int> tree(42);
  tree.ForEachMutableElement([&](const ShapeIndex& index, int* value) {
    EXPECT_TRUE(index.empty());
    *value += 1;
  });
  EXPECT_EQ(tree.element({}), 43);
}

TEST_F(TupleTreeTest, ForEachElementWithStatus) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {{{0}, 10}, {{1, 0}, 20}};
  TupleTree<int> tree(absl::MakeSpan(leaves));

  EXPECT_THAT(
      tree.ForEachElementWithStatus([&](const ShapeIndex& index, int value) {
        if (index == ShapeIndex({1, 0})) {
          return absl::InternalError("Stop here");
        }
        return absl::OkStatus();
      }),
      StatusIs(absl::StatusCode::kInternal));
}

TEST_F(TupleTreeTest, Equality) {
  std::vector<std::pair<ShapeIndex, int>> leaves1 = {{{0}, 10}, {{1, 0}, 20}};
  TupleTree<int> tree1(absl::MakeSpan(leaves1));

  std::vector<std::pair<ShapeIndex, int>> leaves2 = {{{0}, 10}, {{1, 0}, 20}};
  TupleTree<int> tree2(absl::MakeSpan(leaves2));

  std::vector<std::pair<ShapeIndex, int>> leaves3 = {{{0}, 10}, {{1, 1}, 20}};
  TupleTree<int> tree3(absl::MakeSpan(leaves3));

  TupleTree<int> tree4(10);

  EXPECT_EQ(tree1, tree2);
  EXPECT_NE(tree1, tree3);
  EXPECT_NE(tree1, tree4);
}

TEST_F(TupleTreeTest, Iterators) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {
      {{0}, 10}, {{1, 0}, 20}, {{1, 1}, 30}};
  TupleTree<int> tree(absl::MakeSpan(leaves));

  std::vector<std::pair<ShapeIndex, int>> visited;
  for (auto& pair : tree.nodes()) {
    visited.push_back({pair.first, pair.second});
    pair.second += 5;  // Test mutability
  }

  EXPECT_THAT(
      visited,
      ElementsAre(Pair(ShapeIndex({}), 0), Pair(ShapeIndex({0}), 10),
                  Pair(ShapeIndex({1}), 0), Pair(ShapeIndex({1, 0}), 20),
                  Pair(ShapeIndex({1, 1}), 30)));

  // Check leaves() as well
  std::vector<std::pair<ShapeIndex, int>> visited_leaves;
  for (auto& pair : tree.leaves()) {
    visited_leaves.push_back({pair.first, pair.second});
  }
  EXPECT_THAT(visited_leaves, ElementsAre(Pair(ShapeIndex({0}), 15),
                                          Pair(ShapeIndex({1, 0}), 25),
                                          Pair(ShapeIndex({1, 1}), 35)));

  EXPECT_EQ(tree.element({0}), 15);
  EXPECT_EQ(tree.element({1, 0}), 25);
  EXPECT_EQ(tree.element({1, 1}), 35);
}

TEST_F(TupleTreeTest, ConstIterators) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {
      {{0}, 10}, {{1, 0}, 20}, {{1, 1}, 30}};
  const TupleTree<int> tree(absl::MakeSpan(leaves));

  std::vector<std::pair<ShapeIndex, int>> visited;
  for (const auto& pair : tree.nodes()) {
    visited.push_back({pair.first, pair.second});
  }

  EXPECT_THAT(
      visited,
      ElementsAre(Pair(ShapeIndex({}), 0), Pair(ShapeIndex({0}), 10),
                  Pair(ShapeIndex({1}), 0), Pair(ShapeIndex({1, 0}), 20),
                  Pair(ShapeIndex({1, 1}), 30)));

  std::vector<std::pair<ShapeIndex, int>> visited_leaves;
  for (const auto& pair : tree.leaves()) {
    visited_leaves.push_back({pair.first, pair.second});
  }
  EXPECT_THAT(visited_leaves, ElementsAre(Pair(ShapeIndex({0}), 10),
                                          Pair(ShapeIndex({1, 0}), 20),
                                          Pair(ShapeIndex({1, 1}), 30)));
}

TEST_F(TupleTreeTest, LeafIterators) {
  using Node = TupleTree<int>::Node;
  TupleTree<int> tree({Node::Leaf(1),
                       Node::Tuple(100, {Node::Leaf(2), Node::Leaf(3)}),
                       Node::Leaf(4), Node::Tuple(200)});

  // Expected leaves: {0}:1, {1,0}:2, {1,1}:3, {2}:4
  EXPECT_THAT(
      tree.leaves(),
      ElementsAre(Pair(ShapeIndex({0}), 1), Pair(ShapeIndex({1, 0}), 2),
                  Pair(ShapeIndex({1, 1}), 3), Pair(ShapeIndex({2}), 4)));

  // Test const iteration
  const TupleTree<int>& const_tree = tree;
  std::vector<std::pair<ShapeIndex, int>> const_visited;
  for (const auto& pair : const_tree.leaves()) {
    const_visited.push_back(pair);
  }
  EXPECT_THAT(
      const_visited,
      ElementsAre(Pair(ShapeIndex({0}), 1), Pair(ShapeIndex({1, 0}), 2),
                  Pair(ShapeIndex({1, 1}), 3), Pair(ShapeIndex({2}), 4)));

  // Test mutability
  for (auto& pair : tree.leaves()) {
    pair.second *= 10;
  }
  EXPECT_THAT(
      tree.leaves(),
      ElementsAre(Pair(ShapeIndex({0}), 10), Pair(ShapeIndex({1, 0}), 20),
                  Pair(ShapeIndex({1, 1}), 30), Pair(ShapeIndex({2}), 40)));

  EXPECT_EQ(tree.element({0}), 10);
  EXPECT_EQ(tree.element({1, 0}), 20);
  EXPECT_EQ(tree.element({1, 1}), 30);
  EXPECT_EQ(tree.element({2}), 40);
  // Non-leaf elements should be unchanged
  EXPECT_EQ(tree.element({1}), 100);
  EXPECT_EQ(tree.element({3}), 200);
}

TEST_F(TupleTreeTest, LeafIteratorsSingleLeaf) {
  TupleTree<int> tree(42);
  EXPECT_THAT(tree.leaves(), ElementsAre(Pair(ShapeIndex({}), 42)));

  const TupleTree<int>& const_tree = tree;
  EXPECT_THAT(const_tree.leaves(), ElementsAre(Pair(ShapeIndex({}), 42)));

  for (auto& pair : tree.leaves()) {
    pair.second = 100;
  }
  EXPECT_THAT(tree.leaves(), ElementsAre(Pair(ShapeIndex({}), 100)));
}

TEST_F(TupleTreeTest, LeafIteratorsEmptyTuple) {
  TupleTree<int> tree;
  EXPECT_THAT(tree.leaves(), ElementsAre());

  const TupleTree<int>& const_tree = tree;
  EXPECT_THAT(const_tree.leaves(), ElementsAre());
}

TEST_F(TupleTreeTest, Map) {
  using Node = TupleTree<int>::Node;
  TupleTree<int> tree({Node::Leaf(1),
                       Node::Tuple(100, {Node::Leaf(2), Node::Leaf(3)}),
                       Node::Leaf(4), Node::Tuple(200)});

  TupleTree<int> mapped_tree = tree.Map<int>([](int val) { return val * 2; });

  EXPECT_THAT(
      mapped_tree.nodes(),
      ElementsAre(Pair(ShapeIndex({}), 0), Pair(ShapeIndex({0}), 2),
                  Pair(ShapeIndex({1}), 200), Pair(ShapeIndex({1, 0}), 4),
                  Pair(ShapeIndex({1, 1}), 6), Pair(ShapeIndex({2}), 8),
                  Pair(ShapeIndex({3}), 400)));

  // Check that the original tree is unchanged.
  EXPECT_THAT(
      tree.nodes(),
      ElementsAre(Pair(ShapeIndex({}), 0), Pair(ShapeIndex({0}), 1),
                  Pair(ShapeIndex({1}), 100), Pair(ShapeIndex({1, 0}), 2),
                  Pair(ShapeIndex({1, 1}), 3), Pair(ShapeIndex({2}), 4),
                  Pair(ShapeIndex({3}), 200)));
}

TEST_F(TupleTreeTest, MapSingleLeaf) {
  TupleTree<int> tree(42);
  TupleTree<int> mapped_tree = tree.Map<int>([](int val) { return val + 1; });
  EXPECT_THAT(mapped_tree.nodes(), ElementsAre(Pair(ShapeIndex({}), 43)));
}

TEST_F(TupleTreeTest, MapWithStatusSuccess) {
  using Node = TupleTree<int>::Node;
  TupleTree<int> tree({Node::Leaf(1), Node::Leaf(2)});

  TF_ASSERT_OK_AND_ASSIGN(
      TupleTree<int> mapped_tree,
      tree.MapWithStatus<int>(
          [](int val) -> absl::StatusOr<int> { return val * 2; }));

  EXPECT_THAT(mapped_tree.nodes(),
              ElementsAre(Pair(ShapeIndex({}), 0), Pair(ShapeIndex({0}), 2),
                          Pair(ShapeIndex({1}), 4)));
}

TEST_F(TupleTreeTest, MapWithStatusFailure) {
  using Node = TupleTree<int>::Node;
  TupleTree<int> tree({Node::Leaf(1), Node::Leaf(-1), Node::Leaf(2)});

  absl::StatusOr<TupleTree<int>> result =
      tree.MapWithStatus<int>([](int val) -> absl::StatusOr<int> {
        if (val < 0) {
          return absl::InvalidArgumentError("Negative value");
        }
        return val * 2;
      });

  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kInvalidArgument, "Negative value"));
}

TEST_F(TupleTreeTest, IndicesAndValuesConstructorWithDefaultValue) {
  std::vector<std::pair<ShapeIndex, int>> leaves = {
      {{0}, 10}, {{1, 0}, 20}, {{1, 1}, 30}};
  TupleTree<int> tree(absl::MakeSpan(leaves), /*default_value=*/-1);

  EXPECT_FALSE(tree.IsLeaf({}));
  EXPECT_EQ(tree.element({}), -1);  // Root node should have the default value
  EXPECT_TRUE(tree.IsLeaf({0}));
  EXPECT_EQ(tree.element({0}), 10);

  EXPECT_FALSE(tree.IsLeaf({1}));
  EXPECT_EQ(tree.element({1}),
            -1);  // Internal node {1} is value-initialized
  EXPECT_TRUE(tree.IsLeaf({1, 0}));
  EXPECT_EQ(tree.element({1, 0}), 20);
  EXPECT_TRUE(tree.IsLeaf({1, 1}));
  EXPECT_EQ(tree.element({1, 1}), 30);
}

TEST_F(TupleTreeTest, ToNode) {
  using Node = TupleTree<int>::Node;
  TupleTree<int> tree(
      Node::Tuple(100, {Node::Leaf(1),                      // {0}
                        Node::Tuple(200, {Node::Leaf(2),    // {1, 0}
                                          Node::Leaf(3)}),  // {1, 1}
                        Node::Leaf(4)}));                   // {2}

  // ToNode on root
  TF_ASSERT_OK_AND_ASSIGN(Node root_node, tree.ToNode());
  EXPECT_EQ(root_node.value(), 100);
  EXPECT_FALSE(root_node.IsLeaf());
  EXPECT_EQ(root_node.children().size(), 3);

  // ToNode on a leaf node
  TF_ASSERT_OK_AND_ASSIGN(Node leaf_node, tree.ToNode({0}));
  EXPECT_EQ(leaf_node.value(), 1);
  EXPECT_TRUE(leaf_node.IsLeaf());

  // ToNode on a tuple node
  TF_ASSERT_OK_AND_ASSIGN(Node tuple_node, tree.ToNode({1}));
  EXPECT_EQ(tuple_node.value(), 200);
  EXPECT_FALSE(tuple_node.IsLeaf());
  EXPECT_EQ(tuple_node.children().size(), 2);
  EXPECT_EQ(tuple_node.children()[0].value(), 2);
  EXPECT_EQ(tuple_node.children()[1].value(), 3);

  // ToNode on nested leaf node
  TF_ASSERT_OK_AND_ASSIGN(Node nested_leaf_node, tree.ToNode({1, 1}));
  EXPECT_EQ(nested_leaf_node.value(), 3);
  EXPECT_TRUE(nested_leaf_node.IsLeaf());

  // ToNode with invalid index
  EXPECT_THAT(tree.ToNode({5}), StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(tree.ToNode({0, 0}), StatusIs(absl::StatusCode::kInvalidArgument,
                                            "Cannot index into a leaf node"));
}

TEST_F(TupleTreeTest, IsTuple) {
  TupleTree<int> tuple_tree({5});
  TupleTree<int> non_tuple_tree(5);

  EXPECT_TRUE(tuple_tree.IsTuple());
  EXPECT_FALSE(non_tuple_tree.IsTuple());
}

}  // namespace
}  // namespace xla
