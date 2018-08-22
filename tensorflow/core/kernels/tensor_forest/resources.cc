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

#include "tensorflow/core/kernels/tensor_forest/resources.h"
#include "tensorflow/core/kernels/tensor_forest/evaluator.h"
#include "tensorflow/core/kernels/tensor_forest/tensor_forest.pb.h"

namespace tensorflow {

using tensorforest::DecisionTree;

LeafModelResource::LeafModelResource(const int32& leaf_model_type,
                                     const int32& num_output)
    : leaf_model_type_(static_cast<LeafModelType>(leaf_model_type)) {
  model_op_ = LeafModelOperatorFactory::CreateLeafModelOperator(
      leaf_model_type_, num_output);
};

void DecisionTreeResource::MaybeInitialize() {
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
    tensorforest::TreePath* path) const {
  const DecisionTree& tree = decision_tree_->decision_tree();
  int32 current_id = 0;
  while (true) {
    const tensorforest::TreeNode& current = tree.nodes(current_id);
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

}  // namespace tensorflow
