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
#include "tensorflow/core/kernels/boosted_trees/boosted_trees.pb.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

const float TensorForestTreeResource::get_prediction(
    const int32 id, const int32 dimension) const {
  return decision_tree_->nodes(id).leaf().vector().value(dimension);
};

const int32 TensorForestTreeResource::TraverseTree(
    const TTypes<float>::ConstMatrix* dense_data,
    const int32 example_id) const {
  using boosted_trees::Node;
  using boosted_trees::Tree;
  int32 current_id = 0;
  while (true) {
    const Node& current = decision_tree_->nodes(current_id);
    if (current.has_leaf()) {
      return current_id;
    };
    DCHECK_EQ(current.node_case(), Node::kDenseSplit);
    const auto& split = current.dense_split();

    if ((*dense_data)(example_id, split.feature_id()) <= split.threshold()) {
      current_id = split.left_id();
    } else {
      current_id = split.right_id();
    }
  }
};

bool TensorForestTreeResource::InitFromSerialized(const string& serialized) {
  return ParseProtoUnlimited(decision_tree_, serialized);
}

void TensorForestTreeResource::Reset() {
  arena_.Reset();
  CHECK_EQ(0, arena_.SpaceAllocated());
  decision_tree_ = protobuf::Arena::CreateMessage<boosted_trees::Tree>(&arena_);
}

bool TensorForestFertileStatsResource::InitFromSerialized(
    const string& serialized) {
  return ParseProtoUnlimited(fertile_stats_, serialized);
}
void TensorForestFertileStatsResource::Reset() {
  arena_.Reset();
  CHECK_EQ(0, arena_.SpaceAllocated());
  fertile_stats_ =
      protobuf::Arena::CreateMessage<tensor_forest::FertileStats>(&arena_);
}

const bool TensorForestFertileStatsResource::IsSlotInitialized(
    const int32 node_id) const {
  auto slot = fertile_stats_.find(node_id);
  return slot != fertile_stats_.end();
}

const bool TensorForestFertileStatsResource::IsSlotFinished(
    const int32 node_id) const {
  if (IsSlotInitialized(node_id)) {
    auto slot = fertile_stats_.find(node_id);
    return slot.post_init_leaf_stats().weight_sum() >
           split_nodes_after_samples_;
  }
  return false;
}

void TensorForestFertileStatsResource::UpdateSlotStats(
    const int32 node_id, const int32 example_id,
    const TTypes<float>::ConstMatrix* dense_data,
    const TTypes<float>::ConstMatrix* label) {
  auto slot = fertile_stats_.get(node_id);
  for (auto l : (*label)(example_id)) {
    for (auto candidate : slot.candidates()) {
      if (candidate.split.threshold >=
          (*dense_data)(example_id, candidate.split.feature_id)) {
        candidate.left_split_stats.sum.value(l) += 1;
      }
      candidate.post_init_leaf_stats.sum.value(l) += 1;
      candidate.left_split_stats.sum.value(l) += 1;
    }
    slot.leaf_stats.weight_sum += 1;
  }
};

const bool TensorForestFertileStatsResource::AddPotentialSplitToSlot(
    const int32 node_id, const int32 example_id, const bool is_regression,
    const TTypes<float>::ConstMatrix* dense_data,
    const TTypes<float>::ConstMatrix* label) {
  auto slot = fertile_stats_.get(node_id);
  candidate = new tensor_forest::SplitCandidate();
  auto split = candidate.mutatable_split();
  split.feature_id = rng_.ranomd();
  split.thredhold = (*dense_data)(example_id, featrue_id);

  for (auto l : (*label)(example_id)) {
    candidate.left_split_stats.sum.value(l) += 1;
  }
}
void TensorForestFertileStatsResource::AddExample(
    const int32 example_id, const int32 leaf_id,
    const TTypes<float>::ConstMatrix* dense_data,
    const TTypes<float>::ConstMatrix* label, bool* is_finished) {
  if (IsSlotInitialized(leaf_id)) {
    UpdateSlotStats(leaf_id, example_id, dense_data, label);
  } else {
    AddPotentialSplitToSlot(node_id, example_id, dense_data, label);
  }
}
}  // namespace tensorflow
