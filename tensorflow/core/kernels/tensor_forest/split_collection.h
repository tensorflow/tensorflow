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
#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_SPLIT_COLLECTION_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_SPLIT_COLLECTION_H_

#include <vector>
#include "tensorflow/core/kernels/boosted_trees/boosted_trees.pb.h"
#include "tensorflow/core/kernels/tensor_forest/tensor_forest.pb.h"

namespace tensorflow {

// Tracks the sum and square of one side of a split for each potential split's
// Gini calculation.
class RunningGiniScores {
 public:
  float sum(int potential_split) const;
  float square(int potential_split) const;
  void update(int potential_split, float old_label_count,
              float incoming_label_count);
  void add_split();
  void remove_split(int potential_split);

 private:
  std::vector<float> sum_;
  std::vector<float> square_;
};

// Class that can initialize and update split collections, and
// report if one is finished and ready to split.
class SplitCollection {
 public:
  explicit SplitCollection(const bool& is_regression,
                           const int32& split_node_after_samples,
                           const int32& num_splits_to_consider)
      : is_regression_(is_regression),
        split_node_after_samples_(split_node_after_samples),
        num_splits_to_consider_(num_splits_to_consider) {}
  virtual ~SplitCollection() {}

  // Initialize from a previously serialized proto.
  virtual void ExtractFromProto(const FertileStats& stats);

  // Serialize contents to the given proto.
  virtual void PackToProto(FertileStats* stats) const;

  virtual void AddExample(const std::unique_ptr<TensorDataSet>& input_data,
                          const InputTarget* target, const int& example_id,
                          int32 node_id) const;

  virtual void InitializeSlotWithExample(int32 node_id, int32 example_id);

  virtual void ClearSlot(const int32 node_id);

  virtual bool IsInitialized(const int32 node_id) const;

  virtual void UpdateStats(const int32 node_id) const;

  virtual bool IsFinished(const int32 node_id) const;

  // Fill in best with the best split that node_id has, return true if this
  // was successful, false if no good split was found.
  virtual bool BestSplit(int32 node_id, SplitCandidate* best,
                         int32* depth) const;

 protected:
  const bool& is_regression_;
  const bool& split_node_after_samples_;
  const bool& num_splits_to_consider_;
  std::unordered_map<int32, bool> finish_early_;

  std::unordered_map<int32, std::vector<boosted_trees::DenseSplit>>
      node_potential_splits_;
};

class ClassificationSplitCollection() : public SplitCollection {
 public:
  void InitializeSlotWithExample(const& int32 node_id, const& int32 example_id);

  void ClearSlot(const& int32 node_id);

  virtual void UpdateStats(const int32 node_id) const;

 private
  std::unordered_map<int32, std::vector<std::unique_ptr<RunningGiniScores>>>
      node_splits_left_gini_;
  std::unordered_map<int32, std::vector<std::unique_ptr<RunningGiniScores>>>
      node_splits_right_gini_;

  // For every slot track total class counts seen at this leaf
  std::unordered_map<int32, std::vector<float>> node_total_counts_;

  // Left-branch taken class counts at this leaf for each split.
  // This is a flat vector for memory-performance reasons.
  // left_counts_[i * num_outputs_ + j] has the j-th class count for split i.
  std::unordered_map<int32, std::vector<float>> node_left_counts_;

  // Also track the number of classes seen for not splitting pure leaves.
  std::unordered_map<int32, int32> total_samples_seen_;
}

class SplitCollectionFactory {
 public:
  static std::unique_ptr<SplitCollection> CreateSplitCollection(
      const bool& is_regression);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_SPLIT_COLLECTION_H_
