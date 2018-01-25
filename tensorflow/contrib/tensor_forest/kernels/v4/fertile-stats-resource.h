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
#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_FERTILE_STATS_RESOURCE_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_FERTILE_STATS_RESOURCE_H_

#include <vector>

#include "tensorflow/contrib/tensor_forest/kernels/v4/decision_node_evaluator.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_target.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/leaf_model_operators.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/split_collection_operators.h"
#include "tensorflow/contrib/tensor_forest/proto/fertile_stats.pb.h"
#include "tensorflow/contrib/tensor_forest/proto/tensor_forest_params.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace tensorforest {

// Stores a FertileStats proto and implements operations on it.
class FertileStatsResource : public ResourceBase {
 public:
  // Constructor.
  explicit FertileStatsResource(const TensorForestParams& params)
      : params_(params) {
    model_op_ = LeafModelOperatorFactory::CreateLeafModelOperator(params_);
  }

  string DebugString() override {
    return "FertileStats";
  }

  void ExtractFromProto(const FertileStats& stats);

  void PackToProto(FertileStats* stats) const;

  // Resets the resource and frees the proto.
  // Caller needs to hold the mutex lock while calling this.
  void Reset() {
  }

  // Reset the stats for a node, but leave the leaf_stats intact.
  void ResetSplitStats(int32 node_id, int32 depth) {
    collection_op_->ClearSlot(node_id);
    collection_op_->InitializeSlot(node_id, depth);
  }

  mutex* get_mutex() { return &mu_; }

  void MaybeInitialize();

  // Applies the example to the given leaf's statistics. Also applies it to the
  // node's fertile slot's statistics if or initializes a split candidate,
  // where applicable.  Returns if the node is finished or if it's ready to
  // allocate to a fertile slot.
  void AddExampleToStatsAndInitialize(
      const std::unique_ptr<TensorDataSet>& input_data,
      const InputTarget* target, const std::vector<int>& examples,
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
  mutex mu_;
  std::shared_ptr<LeafModelOperator> model_op_;
  std::unique_ptr<SplitCollectionOperator> collection_op_;
  const TensorForestParams params_;

  void AllocateNode(int32 node_id, int32 depth);
};


}  // namespace tensorforest
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_FERTILE_STATS_RESOURCE_H_
