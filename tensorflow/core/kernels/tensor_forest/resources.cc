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

const boosted_trees::Tree& TensorForestTreeResource::decision_tree() const {
  return *decision_tree_;
}

const int32 TensorForestTreeResource::get_size() const {
  return decision_tree_->nodes_size();
}

TensorForestTreeResource::TensorForestTreeResource()
    : decision_tree_(
          protobuf::Arena::CreateMessage<boosted_trees::Tree>(&arena_)) {}

const float TensorForestTreeResource::get_prediction(
    const int32 id, const int32 dimension_id) const {
  return decision_tree_->nodes(id).leaf().vector().value(dimension_id);
}

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
    }
    DCHECK_EQ(current.node_case(), Node::kDenseSplit);
    const auto& split = current.dense_split();

    if ((*dense_data)(example_id, split.feature_id()) <= split.threshold()) {
      current_id = split.left_id();
    } else {
      current_id = split.right_id();
    }
  }
}

bool TensorForestTreeResource::InitFromSerialized(const string& serialized) {
  return ParseProtoUnlimited(decision_tree_, serialized);
}

void TensorForestTreeResource::Reset() {
  arena_.Reset();
  DCHECK_EQ(0, arena_.SpaceAllocated());
  decision_tree_ = protobuf::Arena::CreateMessage<boosted_trees::Tree>(&arena_);
}
void TensorForestTreeResource::SplitNode(const int32 node,
                                         tensor_forest::FertileSlot* slot,
                                         tensor_forest::SplitCandidate* best,
                                         std::vector<int32>* new_children) {
  using boosted_trees::Leaf;
  using boosted_trees::Node;
  auto current_split =
      decision_tree_->mutable_nodes(node)->mutable_dense_split();
  current_split->Swap(best->mutable_split());
  // add left leaf
  auto* new_left = decision_tree_->add_nodes();
  new_left->mutable_leaf()->mutable_vector()->Swap(
      best->mutable_left_leaf_stats()->mutable_counts_or_sums());
  new_children->push_back(decision_tree_->nodes_size());
  // add right leaf
  auto* new_right = decision_tree_->add_nodes();
  for (int i = 0; i <= slot->leaf_stats().counts_or_sums().value_size(); i++) {
    new_right->mutable_leaf()->mutable_vector()->add_value(
        slot->leaf_stats().counts_or_sums().value(i) -
        best->left_leaf_stats().counts_or_sums().value(i));
  }
  new_children->push_back(decision_tree_->nodes_size());
};

bool TensorForestFertileStatsResource::InitFromSerialized(
    const string& serialized) {
  return ParseProtoUnlimited(fertile_stats_, serialized);
};

void TensorForestFertileStatsResource::Reset() {
  arena_.Reset();
  CHECK_EQ(0, arena_.SpaceAllocated());
  fertile_stats_ =
      protobuf::Arena::CreateMessage<tensor_forest::FertileStats>(&arena_);
};

const bool TensorForestFertileStatsResource::IsSlotInitialized(
    const int32 node_id, const int32 splits_to_consider) const {
  if (fertile_stats_->node_to_slot().count(node_id) > 0) {
    auto slot = fertile_stats_->node_to_slot().at(node_id);
    return slot.leaf_stats().weight_sum() > 0 &&
           slot.candidate_size() > splits_to_consider;
  } else {
    return false;
  }
};

const bool TensorForestFertileStatsResource::IsSlotFinished(
    const int32 node_id, const int32 split_nodes_after_samples,
    const int32 splits_to_consider) const {
  if (IsSlotInitialized(node_id, splits_to_consider)) {
    auto slot = fertile_stats_->node_to_slot().at(node_id);
    return slot.leaf_stats().weight_sum() > split_nodes_after_samples;
  }
  return false;
};

void TensorForestFertileStatsResource::UpdateSlotStats(
    const bool is_regression, const int32 node_id, const int32 example_id,
    const int32 num_targets, const TTypes<float>::ConstMatrix* dense_feature,
    const TTypes<float>::ConstMatrix* labels) {
  float incoming_weight = 1.0;
  auto slot = fertile_stats_->node_to_slot().at(node_id);

  auto label = (*labels)(example_id, 0);
  float old_total_weight = slot.leaf_stats().counts_or_sums().value(label);

  slot.mutable_leaf_stats()->set_weight_sum(incoming_weight +
                                            slot->leaf_stats()->weight_sum());
  slot.mutable_leaf_stats()->mutable_counts_or_sums()->set_value(
      label, old_total_weight + incoming_weight);

  for (auto candidate : slot.candidates()) {
    float old_left_weight =
        candidate.left_leaf_stats().counts_or_sums().value(label);

    if ((*dense_feature)(example_id, candidate.split().feature_id()) >=
        candidate.split().threshold()) {
      candidate.mutable_left_split_stats()->mutable_sum()->set_value(
          label,
          candidate.left_split_stats().sum().value(label) + incoming_weight);

      float new_weight = old_left_weight + incoming_weight;

      candidate.mutable_left_split_stats()->mutable_sum_of_square()->set_value(
          0, candidate.left_split_stats().sum_of_square().value(0) -
                 old_left_weight * old_left_weight + new_weight * new_weight);

      candidate.mutable_left_leaf_stats()->set_weight_sum(
          incoming_weight + candidate.left_leaf_stats().weight_sum());

      candidate.mutable_left_leaf_stats()->mutable_counts_or_sums()->set_value(
          label, new_weight);

    } else {
      float old_right_weight = old_total_weight - old_left_weight;

      candidate.mutable_right_split_stats()->mutable_sum()->set_value(
          label,
          candidate.right_split_stats().sum().value(label) + incoming_weight);

      float new_weight = old_right_weight + incoming_weight;

      candidate.mutable_right_split_stats()->mutable_sum_of_square()->set_value(
          0, candidate.right_split_stats().sum_of_square().value(0) -
                 old_right_weight * old_right_weight + new_weight * new_weight);
    }
  }
};

const bool TensorForestFertileStatsResource::BestSplitFromSlot(
    const int32 node_id, tensor_forest::FertileSlot* slot,
    tensor_forest::SplitCandidate* best) {
  auto slot = fertile_stats_->node_to_slot().at(node_id);
  float min_score = FLT_MAX;
  for (auto candidate : slot.candidates()) {
    float left_gini =
        1 - candidate.left_split_stats().sum_of_square().value(0) /
                (candidate.left_split_stats().sum() *
                 candidate.left_split_stats().sum());
    float right_gini =
        1 - candidate.right_split_stats().sum_of_square().value(0) /
                (candidate.right_split_stats().sum() *
                 candidate.right_split_stats().sum());

    if (left_sum > 0 && right_sum > 0 && left_gini + right_gini < min_score) {
      min_score = left_gini + right_gini;
      best = &candidate;
    }
  }
  return min_score != FLT_MAX;
}

const bool TensorForestFertileStatsResource::AddSplitToSlot(
    const int32 node_id, const int32 feature_id, const float threshold,
    const int32 example_id, const int32 num_targets,
    const TTypes<float>::ConstMatrix* dense_feature,
    const TTypes<float>::ConstMatrix* labels) {
  auto slot = fertile_stats_->node_to_slot().at(node_id);
  slot.mutable_leaf_stats()->set_weight_sum(slot.leaf_stats().weight_sum() + 1);
  auto candidate = slot.add_candidates();
  auto split = candidate->mutable_split();
  split->set_threshold(threshold);
  split->set_feature_id(feature_id);
  float incoming_weight = 1.0;
  for (int i = 0; i < num_targets; i++) {
    auto label = (*labels)(example_id, i);

    slot.mutable_leaf_stats()->mutable_counts_or_sums()->set_value(
        label,
        slot.leaf_stats().counts_or_sums().value(label) + incoming_weight);

    auto* mutable_left_split_stats = candidate->mutable_left_split_stats();
    mutable_left_split_stats->mutable_sum()->set_value(
        label,
        candidate->left_split_stats().sum().value(label) + incoming_weight);
    mutable_left_split_stats->mutable_sum_of_square()->set_value(
        0, incoming_weight * incoming_weight);
  }
};

}  // namespace tensorflow
