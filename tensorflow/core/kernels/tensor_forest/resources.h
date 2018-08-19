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
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kerneral/tensor_forest/leaf_model.h"
#include "tensorflow/core/kerneral/tensor_forest/tensor_forest.pb.h"

namespace tensorflow {


class LeafModelResource : public ResourceBase {
 public:
  LeafModelType(const int32& leaf_model_type) : leaf_model_type_(static_cast<LeafModelType>(leaf_model_type)) {
    model_op_ = LeafModelOperatorFactory::CreateLeafModelOperator(leaf_model_type_);
  };

  void MaybeInitialize();

  mutex* get_mutex() { return &mu_; }
  
  void Reset() {}

 private:
  const LeafModelType& leaf_model_type_;
  mutex mu_;
  std::shared_ptr<LeafModelOperator> model_op_;
};

// Keep a tree ensemble in memory for efficient evaluation and mutation.
class DecisionTreeResource : public LeafModelResource {
 public:

  string DebugString() override {
    return strings::StrCat("DecisionTree[size=",
                           decision_tree_->decision_tree().nodes_size(), "]");
  }

  const tensor_forest::Model& decision_tree() const { return *decision_tree_; }

  tensor_forest::Model* mutable_decision_tree() {
    return decision_tree_.get();
  }

  const tensor_forest::Leaf& get_leaf(int32 id) const {
    return decision_tree_->decision_tree().nodes(id).leaf();
  }

  tensor_forest::TreeNode* get_mutable_tree_node(int32 id) {
    return decision_tree_->mutable_decision_tree()->mutable_nodes(id);
  }

  // Resets the resource and frees the proto.
  // Caller needs to hold the mutex lock while calling this.
  void Reset() override{ decision_tree_.reset(new tensor_forest::Model()); }

  int32 TraverseTree(const Tensor& input_data,
                     int example, int32* depth, TreePath* path) const;

  void SplitNode(int32 node_id, SplitCandidate* best,
                 std::vector<int32>* new_children);

 private:
  std::unique_ptr<tensor_forest::Model> decision_tree_;
  std::vector<std::unique_ptr<DecisionNodeEvaluator>> node_evaluators_;
};

class FertileStatsResource : public LeafModelResource {
 public:

  string DebugString() override { return "FertileStats"; }

  void ExtractFromProto(const FertileStats& stats);

  void PackToProto(FertileStats* stats) const;


  // Reset the stats for a node, but leave the leaf_stats intact.
  void ResetSplitStats(int32 node_id, int32 depth) {
    collection_op_->ClearSlot(node_id);
    collection_op_->InitializeSlot(node_id, depth);
  }

  // Applies the example to the given leaf's statistics. Also applies it to the
  // node's fertile slot's statistics if or initializes a split candidate,
  // where applicable.  Returns if the node is finished or if it's ready to
  // allocate to a fertile slot.
  void AddExampleToStatsAndInitialize(
      const Tensor& input_data,
      const Tensor& target, const std::vector<int>& examples,
      int32 node_id, bool* is_finished);

  // Allocate a fertile slot for each ready node, then new children up to
  // max_fertile_nodes_.
  void Allocate(int32 parent_depth, const std::vector<int32>& new_children);

  // Remove a node's fertile slot.  Should only be called when the node is
  // no longer a leaf.
  void Clear(int32 node);

  // Return the best SplitCandidate for a node, or NULL if no suitable split
  // was found.
  bool BestSplit(int32 node_id, SplitCandidate* best, int32* depth);

 private:
  std::unique_ptr<SplitCollectionOperator> collection_op_;
  void AllocateNode(int32 node_id, int32 depth);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_RESOURCES_H_
