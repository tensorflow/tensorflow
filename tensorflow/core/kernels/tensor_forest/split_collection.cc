#include "tensorflow/core/kernerls/tensor_forest/split_collection.h"

float RunningGiniScores::sum(int potential_split) const {
  return sum_[potential_split];
}
float RunningGiniScores::square(int potential_split) const {
  return square_[potential_split];
}

void RunningGiniScores::update(int potential_split, float previous_label_count,
                               float incoming_label_count) {
  sum_[potential_split] += incoming_label_count;
  const float new_val = previous_label_count + incoming_label_count;
  square_[potential_split] = square_[potential_split] -
                             previous_label_count * previous_label_count +
                             new_val * new_val;
}

void RunningGiniScores::add_split() {
  sum_.push_back(0);
  square_.push_back(0);
}

void RunningGiniScores::remove_split(int i) {
  sum_.erase(sum_.begin() + i);
  square_.erase(square_.begin() + i);
}

bool SplitCollection::IsInitialized(node_id) {
  auto slot = node_potential_splits_.find(node_id);
  if (slot != node_potential_splits_.end()) {
    if (slot.size() >= num_splits_to_consider_) {
      return true;
    }
  }
  return false;
}

void SplitCollection::AddExample(node_id, example_id) {
  if (!IsInitialized(node_id)) {
    InitializeSlotWithExample(node_id, example_id);
  } else {
    UpdateStats(node_id, example_id);
  }
}

bool SplitCollection::IsFinished(node_id) {
  finish_early_.find(node_id) ||
      (total_samples_seen_.find(node_id) < split_node_after_samples_)
}

void ClassificationSplitCollection::UpdateStats(node_id, example_id, dense_data,
                                                target) {
  auto slot = node_potential_splits_.find(node_id);
  for (auto potential_split : slot) {
    if (dense_data(example_id, potential_split.feature_id) >=
        potential_split.threshold) {
      left_gini.update(left_counts_[class_id], 1);
      left_counts_[class_id] += 1;
    } else {
      right_gini.update(right_counts_[class_id], 1);
      right_counts_[class_id] += 1;
    }
  }
}
