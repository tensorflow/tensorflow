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
#include "tensorflow/contrib/tensor_forest/kernels/v4/fertile-stats-resource.h"

#include <cfloat>

namespace tensorflow {
namespace tensorforest {

void FertileStatsResource::AddExampleToStatsAndInitialize(
    const std::unique_ptr<TensorDataSet>& input_data,
    const InputTarget* target, const std::vector<int>& examples,
    int32 node_id, int32 node_depth, bool* is_finished) {
  // Set leaf's counts for calculating probabilities.
  for (int example : examples) {
    model_op_->UpdateModel(&leaf_stats_[node_id], target, example);
  }

  // Update stats or initialize if needed.
  if (collection_op_->IsInitialized(node_id)) {
    collection_op_->AddExample(input_data, target, examples, node_id);
  } else {
    // This throws away any extra examples, which is more inefficient towards
    // the top but gradually becomes less of an issue as the tree grows.
    for (int example : examples) {
      collection_op_->CreateAndInitializeCandidateWithExample(
          input_data, target, example, node_id);
      if (collection_op_->IsInitialized(node_id)) {
        break;
      }
    }
  }

  *is_finished = collection_op_->IsFinished(node_id);
}

void FertileStatsResource::AllocateNode(int32 node_id, int32 depth) {
  leaf_stats_[node_id] = LeafStat();
  model_op_->InitModel(&leaf_stats_[node_id]);
  collection_op_->InitializeSlot(node_id, depth);
}

void FertileStatsResource::Allocate(int32 parent_depth,
                                    const std::vector<int32>& new_children) {
  const int32 children_depth = parent_depth + 1;
  for (const int32 child : new_children) {
    AllocateNode(child, children_depth);
  }
}

void FertileStatsResource::Clear(int32 node) {
  collection_op_->ClearSlot(node);
  leaf_stats_.erase(node);
}

bool FertileStatsResource::BestSplit(int32 node_id, SplitCandidate* best,
                                     int32* depth) {
  return collection_op_->BestSplit(node_id, best, depth);
}

void FertileStatsResource::MaybeInitialize() {
  if (leaf_stats_.empty()) {
    AllocateNode(0, 0);
  }
}

void FertileStatsResource::ExtractFromProto(const FertileStats& stats) {
  collection_op_ =
      SplitCollectionOperatorFactory::CreateSplitCollectionOperator(params_);
  collection_op_->ExtractFromProto(stats);
  for (int i = 0; i < stats.node_to_slot_size(); ++i) {
    const auto& slot = stats.node_to_slot(i);
    leaf_stats_[slot.node_id()] = slot.leaf_stats();
  }
}

void FertileStatsResource::PackToProto(FertileStats* stats) const {
  for (const auto& entry : leaf_stats_) {
    auto* slot = stats->add_node_to_slot();
    *slot->mutable_leaf_stats() = entry.second;
    slot->set_node_id(entry.first);
  }
  collection_op_->PackToProto(stats);
}
}  // namespace tensorforest
}  // namespace tensorflow
