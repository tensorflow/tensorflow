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
#include "tensorflow/contrib/boosted_trees/lib/testutil/random_tree_gen.h"

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace boosted_trees {
namespace testutil {

using tensorflow::boosted_trees::trees::DecisionTreeConfig;
using tensorflow::boosted_trees::trees::TreeNode;
using boosted_trees::trees::DenseFloatBinarySplit;

namespace {

// Append the given nodes to tree with transfer of pointer ownership.
// nodes will not be usable upon return.
template <typename T>
void AppendNodes(DecisionTreeConfig* tree, T* nodes) {
  std::reverse(nodes->pointer_begin(), nodes->pointer_end());
  while (!nodes->empty()) {
    tree->mutable_nodes()->AddAllocated(nodes->ReleaseLast());
  }
}

DenseFloatBinarySplit* GetSplit(TreeNode* node) {
  switch (node->node_case()) {
    case TreeNode::kSparseFloatBinarySplitDefaultLeft:
      return node->mutable_sparse_float_binary_split_default_left()
          ->mutable_split();
    case TreeNode::kSparseFloatBinarySplitDefaultRight:
      return node->mutable_sparse_float_binary_split_default_right()
          ->mutable_split();
    case TreeNode::kDenseFloatBinarySplit:
      return node->mutable_dense_float_binary_split();
    default:
      LOG(FATAL) << "Unknown node type encountered.";
  }
  return nullptr;
}

}  // namespace

RandomTreeGen::RandomTreeGen(tensorflow::random::SimplePhilox* rng,
                             int dense_feature_size, int sparse_feature_size)
    : rng_(rng),
      dense_feature_size_(dense_feature_size),
      sparse_feature_size_(sparse_feature_size) {}

namespace {
void AddWeightAndMetadata(
    boosted_trees::trees::DecisionTreeEnsembleConfig* ret) {
  // Assign the weight of the tree to 1 and say that this weight was updated
  // only once.
  ret->add_tree_weights(1.0);
  auto* meta = ret->add_tree_metadata();
  meta->set_num_tree_weight_updates(1);
}

}  //  namespace

boosted_trees::trees::DecisionTreeEnsembleConfig
RandomTreeGen::GenerateEnsemble(int depth, int tree_count) {
  boosted_trees::trees::DecisionTreeEnsembleConfig ret;
  *(ret.add_trees()) = Generate(depth);
  AddWeightAndMetadata(&ret);
  for (int i = 1; i < tree_count; ++i) {
    *(ret.add_trees()) = Generate(ret.trees(0));
    AddWeightAndMetadata(&ret);
  }
  return ret;
}

DecisionTreeConfig RandomTreeGen::Generate(const DecisionTreeConfig& tree) {
  DecisionTreeConfig ret = tree;
  for (auto& node : *ret.mutable_nodes()) {
    if (node.node_case() == TreeNode::kLeaf) {
      node.mutable_leaf()->mutable_sparse_vector()->set_value(
          0, rng_->RandFloat());
      continue;
    }
    // Original node is a split. Re-generate it's type but retain the split node
    // indices.
    DenseFloatBinarySplit* split = GetSplit(&node);
    const int left_id = split->left_id();
    const int right_id = split->right_id();
    GenerateSplit(&node, left_id, right_id);
  }
  return ret;
}

DecisionTreeConfig RandomTreeGen::Generate(int depth) {
  DecisionTreeConfig ret;
  // Add root,
  TreeNode* node = ret.add_nodes();
  GenerateSplit(node, 1, 2);
  if (depth == 1) {
    // Add left and right leaves.
    TreeNode* left = ret.add_nodes();
    left->mutable_leaf()->mutable_sparse_vector()->add_index(0);
    left->mutable_leaf()->mutable_sparse_vector()->add_value(rng_->RandFloat());
    TreeNode* right = ret.add_nodes();
    right->mutable_leaf()->mutable_sparse_vector()->add_index(0);
    right->mutable_leaf()->mutable_sparse_vector()->add_value(
        rng_->RandFloat());
    return ret;
  } else {
    DecisionTreeConfig left_branch = Generate(depth - 1);
    DecisionTreeConfig right_branch = Generate(depth - 1);
    Combine(&ret, &left_branch, &right_branch);
    return ret;
  }
}

void RandomTreeGen::Combine(DecisionTreeConfig* root,
                            DecisionTreeConfig* left_branch,
                            DecisionTreeConfig* right_branch) {
  const int left_branch_size = left_branch->nodes_size();
  CHECK_EQ(1, root->nodes_size());
  // left_branch starts its index at 1. right_branch starts its index at
  // (left_branch_size + 1).
  auto* root_node = root->mutable_nodes(0);
  DenseFloatBinarySplit* root_split = GetSplit(root_node);
  root_split->set_left_id(1);
  root_split->set_right_id(left_branch_size + 1);
  // Shift left/right branch's indices internally so that everything is
  // consistent.
  ShiftNodeIndex(left_branch, 1);
  ShiftNodeIndex(right_branch, left_branch_size + 1);

  // Complexity O(branch node size). No proto copying though.
  AppendNodes(root, left_branch->mutable_nodes());
  AppendNodes(root, right_branch->mutable_nodes());
}

void RandomTreeGen::ShiftNodeIndex(DecisionTreeConfig* tree, int shift) {
  for (TreeNode& node : *(tree->mutable_nodes())) {
    DenseFloatBinarySplit* split = nullptr;
    switch (node.node_case()) {
      case TreeNode::kLeaf:
        break;
      case TreeNode::kSparseFloatBinarySplitDefaultLeft:
        split = node.mutable_sparse_float_binary_split_default_left()
                    ->mutable_split();
        break;
      case TreeNode::kSparseFloatBinarySplitDefaultRight:
        split = node.mutable_sparse_float_binary_split_default_right()
                    ->mutable_split();
        break;
      case TreeNode::kDenseFloatBinarySplit:
        split = node.mutable_dense_float_binary_split();
        break;
      default:
        LOG(FATAL) << "Unknown node type encountered.";
    }
    if (split) {
      split->set_left_id(shift + split->left_id());
      split->set_right_id(shift + split->right_id());
    }
  }
}

void RandomTreeGen::GenerateSplit(TreeNode* node, int left_id, int right_id) {
  const double denseSplitProb =
      sparse_feature_size_ == 0
          ? 1.0
          : static_cast<double>(dense_feature_size_) /
                (dense_feature_size_ + sparse_feature_size_);
  // Generate the tree such that it has equal probability of going left and
  // right when the feature is missing.
  static constexpr float kLeftProb = 0.5;

  DenseFloatBinarySplit* split;
  int feature_size;
  if (rng_->RandFloat() < denseSplitProb) {
    feature_size = dense_feature_size_;
    split = node->mutable_dense_float_binary_split();
  } else {
    feature_size = sparse_feature_size_;
    if (rng_->RandFloat() < kLeftProb) {
      split = node->mutable_sparse_float_binary_split_default_left()
                  ->mutable_split();
    } else {
      split = node->mutable_sparse_float_binary_split_default_right()
                  ->mutable_split();
    }
  }
  split->set_threshold(rng_->RandFloat());
  split->set_feature_column(rng_->Uniform(feature_size));
  split->set_left_id(left_id);
  split->set_right_id(right_id);
}

}  // namespace testutil
}  // namespace boosted_trees
}  // namespace tensorflow
