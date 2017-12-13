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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_RESOURCES_DECISION_TREE_ENSEMBLE_RESOURCE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_RESOURCES_DECISION_TREE_ENSEMBLE_RESOURCE_H_

#include "tensorflow/contrib/boosted_trees/lib/trees/decision_tree.h"
#include "tensorflow/contrib/boosted_trees/resources/stamped_resource.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace boosted_trees {
namespace models {

// Keep a tree ensemble in memory for efficient evaluation and mutation.
class DecisionTreeEnsembleResource : public StampedResource {
 public:
  // Constructor.
  explicit DecisionTreeEnsembleResource()
      : decision_tree_ensemble_(
            protobuf::Arena::CreateMessage<
                boosted_trees::trees::DecisionTreeEnsembleConfig>(&arena_)) {}

  string DebugString() override {
    return strings::StrCat("GTFlowDecisionTreeEnsemble[size=",
                           decision_tree_ensemble_->trees_size(), "]");
  }

  const boosted_trees::trees::DecisionTreeEnsembleConfig&
  decision_tree_ensemble() const {
    return *decision_tree_ensemble_;
  }

  int32 num_trees() const { return decision_tree_ensemble_->trees_size(); }

  bool InitFromSerialized(const string& serialized, const int64 stamp_token) {
    CHECK_EQ(stamp(), -1) << "Must Reset before Init.";
    if (ParseProtoUnlimited(decision_tree_ensemble_, serialized)) {
      set_stamp(stamp_token);
      return true;
    }
    return false;
  }

  string SerializeAsString() const {
    return decision_tree_ensemble_->SerializeAsString();
  }

  // Increment num_layers_attempted and num_trees_attempted in growing_metadata
  // if the tree is finalized.
  void IncrementAttempts() {
    boosted_trees::trees::GrowingMetadata* const growing_metadata =
        decision_tree_ensemble_->mutable_growing_metadata();
    growing_metadata->set_num_layers_attempted(
        growing_metadata->num_layers_attempted() + 1);
    const int num_trees = decision_tree_ensemble_->trees_size();
    if (num_trees <= 0 || LastTreeMetadata()->is_finalized()) {
      growing_metadata->set_num_trees_attempted(
          growing_metadata->num_trees_attempted() + 1);
    }
  }

  boosted_trees::trees::DecisionTreeConfig* AddNewTree(const float weight) {
    // Adding a tree as well as a weight and a tree_metadata.
    decision_tree_ensemble_->add_tree_weights(weight);
    boosted_trees::trees::DecisionTreeMetadata* const metadata =
        decision_tree_ensemble_->add_tree_metadata();
    metadata->set_num_layers_grown(1);
    return decision_tree_ensemble_->add_trees();
  }

  void RemoveLastTree() {
    QCHECK_GT(decision_tree_ensemble_->trees_size(), 0);
    decision_tree_ensemble_->mutable_trees()->RemoveLast();
    decision_tree_ensemble_->mutable_tree_weights()->RemoveLast();
    decision_tree_ensemble_->mutable_tree_metadata()->RemoveLast();
  }

  boosted_trees::trees::DecisionTreeConfig* LastTree() {
    const int32 tree_size = decision_tree_ensemble_->trees_size();
    QCHECK_GT(tree_size, 0);
    return decision_tree_ensemble_->mutable_trees(tree_size - 1);
  }

  boosted_trees::trees::DecisionTreeMetadata* LastTreeMetadata() {
    const int32 metadata_size = decision_tree_ensemble_->tree_metadata_size();
    QCHECK_GT(metadata_size, 0);
    return decision_tree_ensemble_->mutable_tree_metadata(metadata_size - 1);
  }

  // Retrieves tree weights and returns as a vector.
  std::vector<float> GetTreeWeights() const {
    return {decision_tree_ensemble_->tree_weights().begin(),
            decision_tree_ensemble_->tree_weights().end()};
  }

  float GetTreeWeight(const int32 index) const {
    return decision_tree_ensemble_->tree_weights(index);
  }

  // Sets the weight of i'th tree, and increment num_updates in tree_metadata.
  void SetTreeWeight(const int32 index, const float weight,
                     const int32 increment_num_updates) {
    QCHECK_GE(index, 0);
    QCHECK_LT(index, num_trees());
    decision_tree_ensemble_->set_tree_weights(index, weight);
    if (increment_num_updates != 0) {
      const int32 num_updates = decision_tree_ensemble_->tree_metadata(index)
                                    .num_tree_weight_updates();
      decision_tree_ensemble_->mutable_tree_metadata(index)
          ->set_num_tree_weight_updates(num_updates + increment_num_updates);
    }
  }

  // Resets the resource and frees the protos in arena.
  // Caller needs to hold the mutex lock while calling this.
  virtual void Reset() {
    // Reset stamp.
    set_stamp(-1);

    // Clear tree ensemle.
    arena_.Reset();
    CHECK_EQ(0, arena_.SpaceAllocated());
    decision_tree_ensemble_ = protobuf::Arena::CreateMessage<
        boosted_trees::trees::DecisionTreeEnsembleConfig>(&arena_);
  }

  mutex* get_mutex() { return &mu_; }

 protected:
  protobuf::Arena arena_;
  mutex mu_;
  boosted_trees::trees::DecisionTreeEnsembleConfig* decision_tree_ensemble_;
};

}  // namespace models
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_RESOURCES_DECISION_TREE_ENSEMBLE_RESOURCE_H_
