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

  boosted_trees::trees::DecisionTreeEnsembleConfig*
  mutable_decision_tree_ensemble() {
    return decision_tree_ensemble_;
  }

  // Resets the resource and frees the protos in arena.
  // Caller needs to hold the mutex lock while calling this.
  void Reset() {
    // Reset stamp.
    set_stamp(-1);

    // Clear tree ensemle.
    arena_.Reset();
    CHECK_EQ(0, arena_.SpaceAllocated());
    decision_tree_ensemble_ = protobuf::Arena::CreateMessage<
        boosted_trees::trees::DecisionTreeEnsembleConfig>(&arena_);
  }

  mutex* get_mutex() { return &mu_; }

 private:
  protobuf::Arena arena_;
  mutex mu_;
  boosted_trees::trees::DecisionTreeEnsembleConfig* decision_tree_ensemble_;
};

}  // namespace models
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_RESOURCES_DECISION_TREE_ENSEMBLE_RESOURCE_H_
