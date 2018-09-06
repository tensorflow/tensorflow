/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kerneral/tensor_forest/resources.h"

namespace tensorflow {

void LeafModelResource::MaybeInitialize() {
  DecisionTree* tree = decision_tree_->mutable_decision_tree();
  if (tree->nodes_size() == 0) {
    model_op_->InitModel(tree->add_nodes()->mutable_leaf());
  } else if (node_evaluators_.empty()) {  // reconstruct evaluators
    for (const auto& node : tree->nodes()) {
      if (node.has_leaf()) {
        node_evaluators_.emplace_back(nullptr);
      } else {
        node_evaluators_.push_back(CreateBinaryDecisionNodeEvaluator(node));
      }
    }
  }
}

int32 DecisionTreeResource::TraverseTree(
    const std::unique_ptr<DenseTensorType>& input_data, int example,
    TreePath* path) const {
  const DecisionTree& tree = decision_tree_->decision_tree();
  int32 current_id = 0;
  while (true) {
    const TreeNode& current = tree.nodes(current_id);
    if (path != nullptr) {
      *path->add_nodes_visited() = current;
    }
    if (current.has_leaf()) {
      return current_id;
    }
    const int32 next_id =
        node_evaluators_[current_id]->Decide(input_data, example);
    current_id = tree.nodes(next_id).node_id().value();
  }
}

std::unique_ptr<BinaryDecisionNodeEvaluator> CreateBinaryDecisionNodeEvaluator(
    const decision_trees::TreeNode& node) {
  const decision_trees::BinaryNode& bnode = node.binary_node();
  int32 left = bnode.left_child_id().value();
  int32 right = bnode.right_child_id().value();
  const auto& test = bnode.inequality_left_child_test();
  return std::unique_ptr<BinaryDecisionNodeEvaluator>(
      new BinaryDecisionNodeEvaluator(test, left, right));
}

BinaryDecisionNodeEvaluator::BinaryDecisionNodeEvaluator(
    const decision_trees::InequalityTest& test, int32 left, int32 right)
    : BinaryDecisionNodeEvaluator(left, right) {
  CHECK(safe_strto32(test.feature_id().id().value(), &feature_num_))
      << "Invalid feature ID: [" << test.feature_id().id().value() << "]";
  threshold_ = test.threshold().float_value();
}

int32 BinaryDecisionNodeEvaluator::Decide(
    const std::unique_ptr<DenseTensorType>& dataset, int example) const {
  const float val = (*dataset)(example, feature_num_);
  if (val <= threshold_) {
    return left_child_id_;
  } else {
    return right_child_id_;
  }
}

}  // namespace tensorflow
