// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
// =============================================================================
#include "tensorflow/contrib/boosted_trees/lib/trees/decision_tree.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/batch_features.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace boosted_trees {
namespace trees {
namespace {

class DecisionTreeTest : public ::testing::Test {
 protected:
  DecisionTreeTest() : batch_features_(2) {
    // Create a batch of two examples having one dense float, two sparse float
    // and one sparse int features.
    // The first example is missing the second sparse feature column and the
    // second example is missing the first sparse feature column.
    // This looks like the following:
    // Instance | DenseF1 | SparseF1 | SparseF2 | SparseI1 |
    // 0        |   7     |   -3     |          |    3     |
    // 1        |  -2     |          |   4      |          |
    auto dense_float_matrix = test::AsTensor<float>({7.0f, -2.0f}, {2, 1});
    auto sparse_float_indices1 = test::AsTensor<int64>({0, 0}, {1, 2});
    auto sparse_float_values1 = test::AsTensor<float>({-3.0f});
    auto sparse_float_shape1 = test::AsTensor<int64>({2, 1});
    auto sparse_float_indices2 = test::AsTensor<int64>({1, 0}, {1, 2});
    auto sparse_float_values2 = test::AsTensor<float>({4.0f});
    auto sparse_float_shape2 = test::AsTensor<int64>({2, 1});
    auto sparse_int_indices1 = test::AsTensor<int64>({0, 0}, {1, 2});
    auto sparse_int_values1 = test::AsTensor<int64>({3});
    auto sparse_int_shape1 = test::AsTensor<int64>({2, 1});
    TF_EXPECT_OK(batch_features_.Initialize(
        {dense_float_matrix}, {sparse_float_indices1, sparse_float_indices2},
        {sparse_float_values1, sparse_float_values2},
        {sparse_float_shape1, sparse_float_shape2}, {sparse_int_indices1},
        {sparse_int_values1}, {sparse_int_shape1}));
  }

  template <typename SplitType>
  void TestLinkChildrenBinary(TreeNode* node, SplitType* split) {
    // Verify children were linked.
    DecisionTree::LinkChildren({3, 8}, node);
    EXPECT_EQ(3, split->left_id());
    EXPECT_EQ(8, split->right_id());

    // Invalid cases.
    EXPECT_DEATH(DecisionTree::LinkChildren({}, node),
                 "A binary split node must have exactly two children.");
    EXPECT_DEATH(DecisionTree::LinkChildren({3}, node),
                 "A binary split node must have exactly two children.");
    EXPECT_DEATH(DecisionTree::LinkChildren({1, 2, 3}, node),
                 "A binary split node must have exactly two children.");
  }

  void TestGetChildren(const TreeNode& node,
                       const std::vector<uint32>& expected_children) {
    // Verify children were linked.
    auto children = DecisionTree::GetChildren(node);
    EXPECT_EQ(children.size(), expected_children.size());
    for (size_t idx = 0; idx < children.size(); ++idx) {
      EXPECT_EQ(children[idx], expected_children[idx]);
    }
  }

  utils::BatchFeatures batch_features_;
};

TEST_F(DecisionTreeTest, TraverseEmpty) {
  DecisionTreeConfig tree_config;
  auto example = (*batch_features_.examples_iterable(0, 1).begin());
  EXPECT_EQ(-1, DecisionTree::Traverse(tree_config, 0, example));
}

TEST_F(DecisionTreeTest, TraverseBias) {
  DecisionTreeConfig tree_config;
  tree_config.add_nodes()->mutable_leaf();
  auto example = (*batch_features_.examples_iterable(0, 1).begin());
  EXPECT_EQ(0, DecisionTree::Traverse(tree_config, 0, example));
}

TEST_F(DecisionTreeTest, TraverseInvalidSubRoot) {
  DecisionTreeConfig tree_config;
  tree_config.add_nodes()->mutable_leaf();
  auto example = (*batch_features_.examples_iterable(0, 1).begin());
  EXPECT_EQ(-1, DecisionTree::Traverse(tree_config, 10, example));
}

TEST_F(DecisionTreeTest, TraverseDenseBinarySplit) {
  DecisionTreeConfig tree_config;
  auto* split_node =
      tree_config.add_nodes()->mutable_dense_float_binary_split();
  split_node->set_feature_column(0);
  split_node->set_threshold(0.0f);
  split_node->set_left_id(1);
  split_node->set_right_id(2);
  tree_config.add_nodes()->mutable_leaf();
  tree_config.add_nodes()->mutable_leaf();
  auto example_iterable = batch_features_.examples_iterable(0, 2);

  // Expect right child to be picked as !(7 <= 0);
  auto example_it = example_iterable.begin();
  EXPECT_EQ(2, DecisionTree::Traverse(tree_config, 0, *example_it));

  // Expect left child to be picked as (-2 <= 0);
  EXPECT_EQ(1, DecisionTree::Traverse(tree_config, 0, *++example_it));
}

TEST_F(DecisionTreeTest, TraverseSparseBinarySplit) {
  // Test first sparse feature which is missing for the second example.
  DecisionTreeConfig tree_config1;
  auto* split_node1 = tree_config1.add_nodes()
                          ->mutable_sparse_float_binary_split_default_left()
                          ->mutable_split();
  split_node1->set_feature_column(0);
  split_node1->set_threshold(-20.0f);
  split_node1->set_left_id(1);
  split_node1->set_right_id(2);
  tree_config1.add_nodes()->mutable_leaf();
  tree_config1.add_nodes()->mutable_leaf();
  auto example_iterable = batch_features_.examples_iterable(0, 2);

  // Expect right child to be picked as !(-3 <= -20).
  auto example_it = example_iterable.begin();
  EXPECT_EQ(2, DecisionTree::Traverse(tree_config1, 0, *example_it));

  // Expect left child to be picked as default direction.
  EXPECT_EQ(1, DecisionTree::Traverse(tree_config1, 0, *++example_it));

  // Test second sparse feature which is missing for the first example.
  DecisionTreeConfig tree_config2;
  auto* split_node2 = tree_config2.add_nodes()
                          ->mutable_sparse_float_binary_split_default_right()
                          ->mutable_split();
  split_node2->set_feature_column(1);
  split_node2->set_threshold(4.0f);
  split_node2->set_left_id(1);
  split_node2->set_right_id(2);
  tree_config2.add_nodes()->mutable_leaf();
  tree_config2.add_nodes()->mutable_leaf();

  // Expect right child to be picked as default direction.
  example_it = example_iterable.begin();
  EXPECT_EQ(2, DecisionTree::Traverse(tree_config2, 0, *example_it));

  // Expect left child to be picked as (4 <= 4).
  EXPECT_EQ(1, DecisionTree::Traverse(tree_config2, 0, *++example_it));
}

TEST_F(DecisionTreeTest, TraverseCategoricalIdBinarySplit) {
  DecisionTreeConfig tree_config;
  auto* split_node =
      tree_config.add_nodes()->mutable_categorical_id_binary_split();
  split_node->set_feature_column(0);
  split_node->set_feature_id(3);
  split_node->set_left_id(1);
  split_node->set_right_id(2);
  tree_config.add_nodes()->mutable_leaf();
  tree_config.add_nodes()->mutable_leaf();
  auto example_iterable = batch_features_.examples_iterable(0, 2);

  // Expect left child to be picked as 3 == 3;
  auto example_it = example_iterable.begin();
  EXPECT_EQ(1, DecisionTree::Traverse(tree_config, 0, *example_it));

  // Expect right child to be picked as the feature is missing;
  EXPECT_EQ(2, DecisionTree::Traverse(tree_config, 0, *++example_it));
}

TEST_F(DecisionTreeTest, TraverseCategoricalIdSetMembershipBinarySplit) {
  DecisionTreeConfig tree_config;
  auto* split_node = tree_config.add_nodes()
                         ->mutable_categorical_id_set_membership_binary_split();
  split_node->set_feature_column(0);
  split_node->add_feature_ids(3);
  split_node->set_left_id(1);
  split_node->set_right_id(2);
  tree_config.add_nodes()->mutable_leaf();
  tree_config.add_nodes()->mutable_leaf();
  auto example_iterable = batch_features_.examples_iterable(0, 2);

  // Expect left child to be picked as 3 in {3};
  auto example_it = example_iterable.begin();
  EXPECT_EQ(1, DecisionTree::Traverse(tree_config, 0, *example_it));

  // Expect right child to be picked as the feature is missing;
  EXPECT_EQ(2, DecisionTree::Traverse(tree_config, 0, *++example_it));
}

TEST_F(DecisionTreeTest, TraverseHybridSplits) {
  DecisionTreeConfig tree_config;
  auto* split_node1 =
      tree_config.add_nodes()->mutable_dense_float_binary_split();
  split_node1->set_feature_column(0);
  split_node1->set_threshold(9.0f);
  split_node1->set_left_id(1);   // sparse split.
  split_node1->set_right_id(2);  // leaf
  auto* split_node2 = tree_config.add_nodes()
                          ->mutable_sparse_float_binary_split_default_left()
                          ->mutable_split();
  tree_config.add_nodes()->mutable_leaf();
  split_node2->set_feature_column(0);
  split_node2->set_threshold(-20.0f);
  split_node2->set_left_id(3);
  split_node2->set_right_id(4);
  auto* split_node3 =
      tree_config.add_nodes()->mutable_categorical_id_binary_split();
  split_node3->set_feature_column(0);
  split_node3->set_feature_id(2);
  split_node3->set_left_id(5);
  split_node3->set_right_id(6);
  tree_config.add_nodes()->mutable_leaf();
  tree_config.add_nodes()->mutable_leaf();
  tree_config.add_nodes()->mutable_leaf();
  auto example_iterable = batch_features_.examples_iterable(0, 2);

  // Expect will go left through the first dense split as (7.0f <= 9.0f),
  // then will go right through the sparse split as !(-3 <= -20).
  auto example_it = example_iterable.begin();
  EXPECT_EQ(4, DecisionTree::Traverse(tree_config, 0, *example_it));

  // Expect will go left through the first dense split as (-2.0f <= 9.0f),
  // then will go left the default direction as the sparse feature is missing,
  // then will go right as 2 != 3 on the categorical split.
  EXPECT_EQ(6, DecisionTree::Traverse(tree_config, 0, *++example_it));
}

TEST_F(DecisionTreeTest, LinkChildrenLeaf) {
  // Create leaf node.
  TreeNode node;
  node.mutable_leaf();

  // No-op.
  DecisionTree::LinkChildren({}, &node);

  // Invalid case.
  EXPECT_DEATH(DecisionTree::LinkChildren({1}, &node),
               "A leaf node cannot have children.");
}

TEST_F(DecisionTreeTest, LinkChildrenDenseFloatBinarySplit) {
  TreeNode node;
  auto* split = node.mutable_dense_float_binary_split();
  split->set_left_id(-1);
  split->set_right_id(-1);
  TestLinkChildrenBinary(&node, split);
}

TEST_F(DecisionTreeTest, LinkChildrenSparseFloatBinarySplitDefaultLeft) {
  TreeNode node;
  auto* split =
      node.mutable_sparse_float_binary_split_default_left()->mutable_split();
  split->set_left_id(-1);
  split->set_right_id(-1);
  TestLinkChildrenBinary(&node, split);
}

TEST_F(DecisionTreeTest, LinkChildrenSparseFloatBinarySplitDefaultRight) {
  TreeNode node;
  auto* split =
      node.mutable_sparse_float_binary_split_default_right()->mutable_split();
  split->set_left_id(-1);
  split->set_right_id(-1);
  TestLinkChildrenBinary(&node, split);
}

TEST_F(DecisionTreeTest, LinkChildrenCategoricalSingleIdBinarySplit) {
  TreeNode node;
  auto* split = node.mutable_categorical_id_binary_split();
  split->set_left_id(-1);
  split->set_right_id(-1);
  TestLinkChildrenBinary(&node, split);
}

TEST_F(DecisionTreeTest, LinkChildrenNodeNotSet) {
  // Create unset node.
  TreeNode node;

  // Invalid case.
  EXPECT_DEATH(DecisionTree::LinkChildren({1}, &node),
               "A non-set node cannot have children.");
}

TEST_F(DecisionTreeTest, GetChildrenLeaf) {
  TreeNode node;
  node.mutable_leaf();
  TestGetChildren(node, {});
}

TEST_F(DecisionTreeTest, GetChildrenDenseFloatBinarySplit) {
  TreeNode node;
  auto* split = node.mutable_dense_float_binary_split();
  split->set_left_id(23);
  split->set_right_id(24);
  TestGetChildren(node, {23, 24});
}

TEST_F(DecisionTreeTest, GetChildrenSparseFloatBinarySplitDefaultLeft) {
  TreeNode node;
  auto* split =
      node.mutable_sparse_float_binary_split_default_left()->mutable_split();
  split->set_left_id(12);
  split->set_right_id(13);
  TestGetChildren(node, {12, 13});
}

TEST_F(DecisionTreeTest, GetChildrenSparseFloatBinarySplitDefaultRight) {
  TreeNode node;
  auto* split =
      node.mutable_sparse_float_binary_split_default_right()->mutable_split();
  split->set_left_id(1);
  split->set_right_id(2);
  TestGetChildren(node, {1, 2});
}

TEST_F(DecisionTreeTest, GetChildrenCategoricalSingleIdBinarySplit) {
  TreeNode node;
  auto* split = node.mutable_categorical_id_binary_split();
  split->set_left_id(7);
  split->set_right_id(8);
  TestGetChildren(node, {7, 8});
}

TEST_F(DecisionTreeTest, GetChildrenNodeNotSet) {
  TreeNode node;
  TestGetChildren(node, {});
}

}  // namespace
}  // namespace trees
}  // namespace boosted_trees
}  // namespace tensorflow
