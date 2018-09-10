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
#include "tensorflow/core/kernels/boosted_trees/boosted_trees.pb.h"
#include "tensorflow/core/kernels/tensor_forest/tensor_forest.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Keep a tree ensemble in memory for efficient evaluation and mutation.
class TensorForestTreeResource : public ResourceBase {
 public:
  TensorForestTreeResource()
      : decision_tree_(
            protobuf::Arena::CreateMessage<boosted_trees::Tree>(&arena_)){};

  string DebugString() override {
    return strings::StrCat("TensorForestTree[size=", get_size(), "]");
  }

  mutex* get_mutex() { return &mu_; }

  bool InitFromSerialized(const string& serialized);

  // Resets the resource and frees the proto.
  // Caller needs to hold the mutex lock while calling this.
  void Reset();

  const boosted_trees::Tree& decision_tree() const { return *decision_tree_; }

  const int32 get_size() const { return decision_tree_->nodes_size(); }

  const float get_prediction(const int32 id, const int32 dimension) const;

  const int32 TraverseTree(const int32 example_id,
                           const TTypes<float>::ConstMatrix* dense_data) const;

  void SplitNode(node, best, new_children);

  const bool node_has_leaf(const int32 node_id);

 protected:
  mutex mu_;
  protobuf::Arena arena_;
  boosted_trees::Tree* decision_tree_;
};

class TensorForestFertileStatsResource : public ResourceBase {
 public:
  TensorForestFertileStatsResource()
      : fertile_stats_(
            protobuf::Arena::CreateMessage<tensor_forest::FertilStats>(
                &arena_)){};

  string DebugString() override { return "TensorForestFertilStats"; }

  const tensor_forest::FertilStats& fertile_stats() const {
    return *fertile_stats_;
  }
  mutex* get_mutex() { return &mu_; }

  bool InitFromSerialized(const string& serialized);

  // Resets the resource and frees the proto.
  // Caller needs to hold the mutex lock while calling this.
  void Reset();

  void AddExample(const int32 example_id, const int32 leaf_id,
                  const TTypes<float>::ConstMatrix* dense_data,
                  const TTypes<float>::ConstMatrix* label, bool* is_finished);

  const void UpdateSlotStats(const int32 node_id, const int32 example_id,
                             const TTypes<float>::ConstMatrix* dense_data);

  const bool AddPotentialSplitToSlot(
      const int32 node_id, const int32 example_id, const bool is_regression,
      const TTypes<float>::ConstMatrix* dense_data,
      const TTypes<float>::ConstMatrix* label);

  const bool IsSlotFinished(const int32 node_id);

  const bool IsSlotInitialized(const int32 node_id);

  const bool BestSplitFromSlot(const int32 node_id,
                               tensor_forest::SplitCandidate& best);

  void SetRandomSeed(const int32 random_seed) { random_seed_ = random_seed; };

 protected:
  mutex mu_;
  protobuf::Arena arena_;
  tensor_forest::FertileStats* fertile_stats_;
  int32 random_seed_;
}

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_RESOURCES_H_
