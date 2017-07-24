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
#include "tensorflow/contrib/boosted_trees/resources/decision_tree_ensemble_resource.h"

namespace tensorflow {
namespace boosted_trees {
namespace models {

DecisionTreeEnsembleResource::DecisionTreeEnsembleResource()
    : decision_tree_ensemble_(
          protobuf::Arena::CreateMessage<
              boosted_trees::trees::DecisionTreeEnsembleConfig>(&arena_)) {}

string DecisionTreeEnsembleResource::DebugString() {
  return strings::StrCat("GTFlowDecisionTreeEnsemble[size=",
                         decision_tree_ensemble_->trees_size(), "]");
}

void DecisionTreeEnsembleResource::Reset() {
  // Reset stamp.
  set_stamp(-1);

  // Clear tree ensemle.
  arena_.Reset();
  CHECK_EQ(0, arena_.SpaceAllocated());
  decision_tree_ensemble_ = protobuf::Arena::CreateMessage<
      boosted_trees::trees::DecisionTreeEnsembleConfig>(&arena_);
}

}  // namespace models
}  // namespace tensorforest
}  // namespace tensorflow
