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
#include "tensorflow/contrib/tensor_forest/kernels/v4/stat_utils.h"
#include <cfloat>

#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"

namespace tensorflow {
namespace tensorforest {

// When using smoothing but only tracking sum and squares, and we're adding
// num_classes for smoothing each class, then Gini looks more like this:
//   Gini = 1 - \sum_i (c_i + 1)^2 / C^2
//   = 1 - (1 / C^2) ( (\sum_i c_i)^2 + 2 (\sum_i c_i) + (\sum_i 1))
//   = 1 - (1 / C^2) ( stats.square() + 2 stats.sum() + #_classes)
//   = 1 - ( stats.square() + 2 stats.sum() + #_classes) / (smoothed_sum *
//                                                          smoothed_sum)
//
//   where
//   smoothed_sum = stats.sum() + #_classes
float GiniImpurity(const LeafStat& stats, int32 num_classes) {
  const float smoothed_sum = num_classes + stats.weight_sum();
  return 1.0 - ((stats.classification().gini().square() +
                 2 * stats.weight_sum() + num_classes) /
                (smoothed_sum * smoothed_sum));
}

float WeightedGiniImpurity(const LeafStat& stats, int32 num_classes) {
  return stats.weight_sum() * GiniImpurity(stats, num_classes);
}

void UpdateGini(LeafStat* stats, float old_val, float weight) {
  stats->set_weight_sum(stats->weight_sum() + weight);
  // Equivalent to stats->square() - old_val * old_val + new_val * new_val,
  // (for new_val = old_val + weight), but more numerically stable.
  stats->mutable_classification()->mutable_gini()->set_square(
      stats->classification().gini().square() + weight * weight +
      2 * old_val * weight);
}

float Variance(const LeafStat& stats, int output) {
  if (stats.weight_sum() == 0) {
    return 0;
  }
  const float e_x =
      stats.regression().mean_output().value(output).float_value() /
      stats.weight_sum();
  const auto e_x2 =
      stats.regression().mean_output_squares().value(output).float_value() /
      stats.weight_sum();
  return e_x2 - e_x * e_x;
}

float TotalVariance(const LeafStat& stats) {
  float sum = 0;
  for (int i = 0; i < stats.regression().mean_output().value_size(); ++i) {
    sum += Variance(stats, i);
  }
  return sum;
}

float SmoothedGini(float sum, float square, int num_classes) {
  // See comments for GiniImpurity above.
  const float smoothed_sum = num_classes + sum;
  return 1.0 - (square + 2 * sum + num_classes) / (smoothed_sum * smoothed_sum);
}

float WeightedSmoothedGini(float sum, float square, int num_classes) {
  return sum * SmoothedGini(sum, square, num_classes);
}

}  // namespace tensorforest
}  // namespace tensorflow
