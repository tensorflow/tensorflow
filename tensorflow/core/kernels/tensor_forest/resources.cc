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
    const int32 example_id,
    const TTypes<float>::ConstMatrix* dense_data) const {
  using boosted_trees::Node;
  using boosted_trees::Tree;
  int32 current_id = 0;
  while (true) {
    const Node& current = decision_tree_->nodes(current_id);
    if (current.has_leaf()) {
      return current_id;
    };
    DCHECK_EQ(current.node_case(), Node::kDenseSplit)
        << "Only DenseSplit supported";
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
    const int32 node_id, const int32 splits_to_consider) const {
  if (fertile_stats_->node_to_slot().count(node_id) > 0) {
    auto slot = fertile_stats_->node_to_slot().at(node_id);
    return slot.post_init_leaf_stats().weight_sum() > splits_to_consider;
  } else {
    return false;
  }
}

const bool TensorForestFertileStatsResource::IsSlotFinished(
    const int32 node_id, const int32 split_nodes_after_samples,
    const int32 splits_to_consider) const {
  if (IsSlotInitialized(node_id, splits_to_consider)) {
    auto slot = fertile_stats_->node_to_slot().at(node_id);
    return slot.post_init_leaf_stats().weight_sum() > split_nodes_after_samples;
  }
  return false;
}

void TensorForestFertileStatsResource::UpdateSlotStats(
    const bool is_regression, const int32 node_id, const int32 example_id,
    const TTypes<float>::ConstMatrix* dense_feature,
    const TTypes<float>::ConstMatrix* labels, const int32 num_labels = 1) {
  auto slot = fertile_stats_->node_to_slot().at(node_id);
  slot.mutable_leaf_stats()->set_weight_sum(slot.leaf_stats().weight_sum() + 1);
  slot.mutable_post_init_leaf_stats()->set_weight_sum(
      slot.post_init_leaf_stats().weight_sum() + 1);

  for (int i = 0; i < num_labels; i++) {
    auto label = (*labels)(example_id, i);
    /*if (is_regression) {
    slot.mutable_leaf_stats()->mutable_couts_or_sums().set_value(
        i, slot.leaf_stats()->counts_or_sums().value(i) + label);
    }else{}
    */
    slot.mutable_leaf_stats()->mutable_counts_or_sums()->set_value(
        label, slot.leaf_stats().counts_or_sums().value(label) + 1);

    for (auto candidate : slot.candidates()) {
      /* if (is_regression) {
        if (candidate.split().threshold() >=
            (*dense_feature)(example_id, candidate.split().feature_id())) {
          candidate.mutable_left_split_stats()->mutable_sum()->set_value(
              i, candidate.left_split_stats().sum().value(i) + label);
        };
        slot.mutable_post_init_leaf_stats()->set_weight_sum(
            slot.post_init_leaf_stats().weight_sum() + 1);
      } else {

      }
      */
      if (candidate.split().threshold() >=
          (*dense_feature)(example_id, candidate.split().feature_id())) {
        candidate.mutable_left_split_stats()->mutable_sum()->set_value(
            label, candidate.left_split_stats().sum().value(label) + 1);

        float weight =
            candidate.left_leaf_stats().counts_or_sums().value(label);
        float new_weight = weight + 1;
        candidate.mutable_left_leaf_stats()
            ->mutable_counts_or_sums()
            ->set_value(label, weight + 1);

        candidate.mutable_left_split_stats()
            ->mutable_sum_of_square()
            ->set_value(0,
                        candidate.left_split_stats().sum_of_square().value(0) -
                            weight * weight + new_weight * new_weight);
      } else {
        float weight =
            slot.post_init_leaf_stats().counts_or_sums().value(label) -
            candidate.left_leaf_stats().counts_or_sums().value(label);
        float new_weight = weight + 1;
        candidate.mutable_right_split_stats()
            ->mutable_sum_of_square()
            ->set_value(0,
                        candidate.left_split_stats().sum_of_square().value(0) -
                            weight * weight + new_weight * new_weight);
      };
    }
  }
};

const bool TensorForestFertileStatsResource::AddSplitToSlot(
    const int32 node_id, const int32 feature_id, const float threshold) {
  auto slot = fertile_stats_->node_to_slot().at(node_id);
  auto candidate = slot.add_candidates();
  auto split = candidate->mutable_split();
  split->set_threshold(threshold);
  split->set_feature_id(feature_id);
};

}  // namespace tensorflow
