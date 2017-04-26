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
#include "tensorflow/contrib/boosted_trees/lib/utils/dropout_utils.h"

#include <iterator>
#include <numeric>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"

using tensorflow::boosted_trees::learner::LearningRateDropoutDrivenConfig;
using tensorflow::random::PhiloxRandom;
using tensorflow::random::SimplePhilox;
using tensorflow::Status;

namespace tensorflow {
namespace boosted_trees {
namespace utils {

Status DropoutUtils::DropOutTrees(const uint64 seed,
                                  const LearningRateDropoutDrivenConfig& config,
                                  const std::vector<float>& weights,
                                  std::vector<int32>* dropped_trees,
                                  std::vector<float>* original_weights) {
  // Verify params.
  if (dropped_trees == nullptr) {
    return errors::Internal("Dropped trees is nullptr.");
  }
  if (original_weights == nullptr) {
    return errors::InvalidArgument("Original weights is nullptr.");
  }
  const float dropout_probability = config.dropout_probability();
  if (dropout_probability < 0 || dropout_probability > 1) {
    return errors::InvalidArgument(
        "Dropout probability must be in [0,1] range");
  }
  const float learning_rate = config.learning_rate();
  if (learning_rate <= 0) {
    return errors::InvalidArgument("Learning rate must be in (0,1] range.");
  }
  const float probability_of_skipping_dropout =
      config.probability_of_skipping_dropout();
  if (probability_of_skipping_dropout < 0 ||
      probability_of_skipping_dropout > 1) {
    return errors::InvalidArgument(
        "Probability of skiping dropout must be in [0,1] range");
  }
  const auto num_trees = weights.size();

  dropped_trees->clear();
  original_weights->clear();

  // If dropout is no op, return.
  if (dropout_probability == 0 || probability_of_skipping_dropout == 1.0) {
    return Status::OK();
  }

  // Roll the dice for each tree.
  PhiloxRandom philox(seed);
  SimplePhilox rng(&philox);

  std::vector<int32> trees_to_keep;

  // What is the probability of skipping dropout altogether.
  if (probability_of_skipping_dropout != 0) {
    // First roll the dice - do we do dropout
    double roll = rng.RandDouble();
    if (roll < probability_of_skipping_dropout) {
      // don't do dropout
      return Status::OK();
    }
  }

  for (int32 i = 0; i < num_trees; ++i) {
    double roll = rng.RandDouble();
    if (roll >= dropout_probability) {
      trees_to_keep.push_back(i);
    } else {
      dropped_trees->push_back(i);
    }
  }

  // Sort the dropped trees indices.
  std::sort(dropped_trees->begin(), dropped_trees->end());
  for (const int32 dropped_tree : *dropped_trees) {
    original_weights->push_back(weights[dropped_tree]);
  }

  return Status::OK();
}

void DropoutUtils::GetTreesWeightsForAddingTrees(
    const std::vector<int32>& dropped_trees,
    const std::vector<float>& dropped_trees_original_weights,
    const int32 num_trees_to_add, std::vector<float>* current_weights,
    std::vector<int32>* num_updates) {
  CHECK(num_updates->size() == current_weights->size());
  // combined weight of trees that were dropped out
  const float dropped_sum =
      std::accumulate(dropped_trees_original_weights.begin(),
                      dropped_trees_original_weights.end(), 0.0);

  const int num_dropped = dropped_trees.size();

  // Allocate additional weight for the new tree
  const float total_new_trees_weight = dropped_sum / (num_dropped + 1);
  for (int i = 0; i < num_trees_to_add; ++i) {
    current_weights->push_back(total_new_trees_weight / num_trees_to_add);
    num_updates->push_back(1);
  }

  for (int32 i = 0; i < dropped_trees.size(); ++i) {
    const int32 dropped = dropped_trees[i];
    const float original_weight = dropped_trees_original_weights[i];
    const float new_weight = original_weight * num_dropped / (num_dropped + 1);
    (*current_weights)[dropped] = new_weight;
    // Update the number of updates per tree.
    ++(*num_updates)[dropped];
  }
}

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
