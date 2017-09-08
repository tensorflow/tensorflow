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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_DECISION_TREE_RESOURCE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_DECISION_TREE_RESOURCE_H_

#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/decision_node_evaluator.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/leaf_model_operators.h"
#include "tensorflow/contrib/tensor_forest/proto/fertile_stats.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace tensorforest {


// Keep a tree ensemble in memory for efficient evaluation and mutation.
class DecisionTreeResource : public ResourceBase {
 public:
  // Constructor.
  explicit DecisionTreeResource(const TensorForestParams& params)
      : params_(params), decision_tree_(new decision_trees::Model()) {
    model_op_ = LeafModelOperatorFactory::CreateLeafModelOperator(params_);
  }

  string DebugString() override {
    return strings::StrCat("DecisionTree[size=",
                           decision_tree_->decision_tree().nodes_size(),
                           "]");
  }

  void MaybeInitialize();

  const decision_trees::Model& decision_tree() const {
    return *decision_tree_;
  }

  decision_trees::Model* mutable_decision_tree() {
    return decision_tree_.get();
  }

  const decision_trees::Leaf& get_leaf(int32 id) const {
    return decision_tree_->decision_tree().nodes(id).leaf();
  }

  decision_trees::TreeNode* get_mutable_tree_node(int32 id) {
    return decision_tree_->mutable_decision_tree()->mutable_nodes(id);
  }

  // Resets the resource and frees the proto.
  // Caller needs to hold the mutex lock while calling this.
  void Reset() {
    decision_tree_.reset(new decision_trees::Model());
  }

  mutex* get_mutex() { return &mu_; }

  // Return the TreeNode for the leaf that the example ends up at according
  // to decsion_tree_. Also fill in that leaf's depth if it isn't nullptr.
  int32 TraverseTree(const std::unique_ptr<TensorDataSet>& input_data,
                     int example, int32* depth, TreePath* path) const;

  // Split the given node_id, turning it from a Leaf to a BinaryNode and
  // setting it's split to the given best.  Add new children ids to
  // new_children.
  void SplitNode(int32 node_id, SplitCandidate* best,
                 std::vector<int32>* new_children);

 private:
  mutex mu_;
  const TensorForestParams params_;
  std::unique_ptr<decision_trees::Model> decision_tree_;
  std::shared_ptr<LeafModelOperator> model_op_;
  std::vector<std::unique_ptr<DecisionNodeEvaluator>> node_evaluators_;
};


}  // namespace tensorforest
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_DECISION_TREE_RESOURCE_H_
