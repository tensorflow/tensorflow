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

#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_RESOURCES_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_RESOURCES_H_

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/tensor_forest/evaluator.h"
#include "tensorflow/core/kernels/tensor_forest/leaf_model.h"
#include "tensorflow/core/kernels/tensor_forest/tensor_forest.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Keep a tree ensemble in memory for efficient evaluation and mutation.
class DecisionTreeResource : public ResourceBase {
 public:
  DecisionTreeResource(const int32& leaf_model_type, const int32& num_output)
      : leaf_model_type_(leaf_model_type) {
    model_op_ =
        LeafModelFactory::CreateLeafModelOperator(leaf_model_type_, num_output);
  };
  string DebugString() override {
    return strings::StrCat("DecisionTree[size=",
                           decision_tree_->decision_tree().nodes_size(), "]");
  }

  mutex* get_mutex() { return &mu_; }

  const tensorforest::Model& decision_tree() const { return *decision_tree_; }

  tensorforest::Model* mutable_decision_tree() { return decision_tree_.get(); }

  const tensorforest::Leaf& get_leaf(int32 id) const {
    return decision_tree_->decision_tree().nodes(id).leaf();
  }

  tensorforest::TreeNode* get_mutable_tree_node(int32 id) {
    return decision_tree_->mutable_decision_tree()->mutable_nodes(id);
  }

  // Resets the resource and frees the proto.
  // Caller needs to hold the mutex lock while calling this.
  void Reset() { decision_tree_.reset(new tensorforest::Model()); }

  void MaybeInitialize();
  int32 TraverseTree(const std::unique_ptr<DenseTensorType>& input_data,
                     int example, tensorforest::TreePath* path) const;

  // void SplitNode(int32 node_id, tensorforest::SplitCandidate* best,
  //                std::vector<int32>* new_children);

 private:
  mutex mu_;
  const int32 leaf_model_type_;
  std::shared_ptr<LeafModelOperator> model_op_;
  std::unique_ptr<tensorforest::Model> decision_tree_;
  std::vector<std::unique_ptr<BinaryDecisionNodeEvaluator>> node_evaluators_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_RESOURCES_H_
