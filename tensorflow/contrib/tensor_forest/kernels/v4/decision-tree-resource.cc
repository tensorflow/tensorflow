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
#include "tensorflow/contrib/tensor_forest/kernels/v4/decision-tree-resource.h"

namespace tensorflow {
namespace tensorforest {

using decision_trees::DecisionTree;
using decision_trees::Leaf;
using decision_trees::TreeNode;

DecisionTreeResource::DecisionTreeResource(const TensorForestParams& params)
    : params_(params), decision_tree_(new decision_trees::Model()) {
  model_op_ = LeafModelOperatorFactory::CreateLeafModelOperator(params_);
}

int32 DecisionTreeResource::TraverseTree(
    const std::unique_ptr<TensorDataSet>& input_data, int example,
    int32* leaf_depth, TreePath* path) const {
  const DecisionTree& tree = decision_tree_->decision_tree();
  int32 current_id = 0;
  int32 depth = 0;
  while (true) {
    const TreeNode& current = tree.nodes(current_id);
    if (path != nullptr) {
      *path->add_nodes_visited() = current;
    }
    if (current.has_leaf()) {
      if (leaf_depth != nullptr) {
        *leaf_depth = depth;
      }
      return current_id;
    }
    ++depth;
    const int32 next_id =
        node_evaluators_[current_id]->Decide(input_data, example);
    current_id = tree.nodes(next_id).node_id().value();
  }
}

void DecisionTreeResource::SplitNode(int32 node_id, SplitCandidate* best,
                                     std::vector<int32>* new_children) {
  DecisionTree* tree = decision_tree_->mutable_decision_tree();
  TreeNode* node = tree->mutable_nodes(node_id);
  int32 newid = tree->nodes_size();

  // left
  new_children->push_back(newid);
  TreeNode* new_left = tree->add_nodes();
  new_left->mutable_node_id()->set_value(newid++);
  Leaf* left_leaf = new_left->mutable_leaf();
  model_op_->ExportModel(best->left_stats(), left_leaf);

  // right
  new_children->push_back(newid);
  TreeNode* new_right = tree->add_nodes();
  new_right->mutable_node_id()->set_value(newid);
  Leaf* right_leaf = new_right->mutable_leaf();
  model_op_->ExportModel(best->right_stats(), right_leaf);

  node->clear_leaf();
  node->mutable_binary_node()->Swap(best->mutable_split());
  node->mutable_binary_node()->mutable_left_child_id()->set_value(newid - 1);
  node->mutable_binary_node()->mutable_right_child_id()->set_value(newid);
  while (node_evaluators_.size() <= node_id) {
    node_evaluators_.emplace_back(nullptr);
  }
  node_evaluators_[node_id] = CreateDecisionNodeEvaluator(*node);
}

void DecisionTreeResource::MaybeInitialize() {
  DecisionTree* tree = decision_tree_->mutable_decision_tree();
  if (tree->nodes_size() == 0) {
    model_op_->InitModel(tree->add_nodes()->mutable_leaf());
  } else if (node_evaluators_.empty()) {  // reconstruct evaluators
    for (const auto& node : tree->nodes()) {
      if (node.has_leaf()) {
        node_evaluators_.emplace_back(nullptr);
      } else {
        node_evaluators_.push_back(CreateDecisionNodeEvaluator(node));
      }
    }
  }
}

}  // namespace tensorforest
}  // namespace tensorflow
