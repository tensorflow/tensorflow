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
#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_DROPOUT_UTILS_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_DROPOUT_UTILS_H_

#include <unordered_set>
#include <vector>

#include "tensorflow/contrib/boosted_trees/proto/learner.pb.h"  // NOLINT
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

// Utils for deciding on what trees to be/were dropped when building a new tree.
class DropoutUtils {
 public:
  // This method determines what trees should be dropped and returns their
  // indices and the weights they had when this method ran.
  // seed: random seed to be used
  // config: dropout config, that defines the probability of dropout etc
  // trees_not_to_drop: indices of trees that can't be dropped, for example bias
  // (0) and the last tree in the batch mode.
  // number_of_trees_to_consider: how many trees are currently in the ensemble
  // weights: weights of those trees
  // Returns sorted vector of indices of trees to be dropped and their original
  // weights.
  static tensorflow::Status DropOutTrees(
      const uint64 seed, const learner::LearningRateDropoutDrivenConfig& config,
      const std::unordered_set<int32>& trees_not_to_drop,
      const std::vector<float>& weights, std::vector<int32>* dropped_trees,
      std::vector<float>* original_weights);

  // Recalculates the weights of the trees when the new trees are added to
  // ensemble.
  // dropped_trees: ids of trees that were dropped when trees to add were built.
  // dropped_trees_original_weights: the weight dropped trees had during dropout
  // new_trees_first_index: index of the last tree. If it is already in the
  // ensemble, its weight and num updates are adjusted. Otherwise, its weight
  // and num updates are added as new entries to current_weights and
  // num_updates. num_trees_to_add: how many trees are being added to the
  // ensemble. Returns current_weights: updated vector of the tree weights.
  // Weights of dropped trees are updated. Note that the size of returned vector
  // will be total_num_trees + num_trees_to_add (the last elements are the
  // weights of the new trees to be added) if new_trees_first_index
  // >=current_weights.size num_updates: updated vector with increased number of
  // updates for dropped trees.
  static void GetTreesWeightsForAddingTrees(
      const std::vector<int32>& dropped_trees,
      const std::vector<float>& dropped_trees_original_weights,
      const int32 new_trees_first_index, const int32 num_trees_to_add,
      // Current weights and num_updates will be updated as a result of this
      // func
      std::vector<float>* current_weights,
      // How many weight assignements have been done for each tree already.
      std::vector<int32>* num_updates);
};

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_DROPOUT_UTILS_H_
