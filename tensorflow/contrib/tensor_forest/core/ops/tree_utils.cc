// Copyright 2016 Google Inc. All Rights Reserved.
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
#include "tensorflow/contrib/tensor_forest/core/ops/tree_utils.h"

namespace tensorflow {
namespace tensorforest {

using tensorflow::Tensor;

int32 BestFeatureClassification(
    const Tensor& total_counts, const Tensor& split_counts,
    int32 accumulator) {
  int32 best_feature_index = -1;
  // We choose the split with the lowest score.
  float best_score = kint64max;
  const int32 num_splits = static_cast<int32>(split_counts.shape().dim_size(1));
  const int32 num_classes = static_cast<int32>(
      split_counts.shape().dim_size(2));
  // Ideally, Eigen::Tensor::chip would be best to use here but it results
  // in seg faults, so we have to go with flat views of these tensors.  However,
  // it is still pretty efficient because we put off evaluation until the
  // score is actually returned.
  const auto tc = total_counts.Slice(
      accumulator, accumulator + 1).unaligned_flat<float>();
  const auto splits = split_counts.Slice(
      accumulator, accumulator + 1).unaligned_flat<float>();
  Eigen::array<int, 1> bcast;
  bcast[0] = num_splits;
  const auto rights = tc.broadcast(bcast) - splits;

  for (int i = 0; i < num_splits; i++) {
    Eigen::array<int, 1> offsets;
    offsets[0] = i * num_classes;
    Eigen::array<int, 1> extents;
    extents[0] = num_classes;
    float score = WeightedGiniImpurity(splits.slice(offsets, extents)) +
        WeightedGiniImpurity(rights.slice(offsets, extents));

    if (score < best_score) {
      best_score = score;
      best_feature_index = i;
    }
  }
  return best_feature_index;
}

int32 BestFeatureRegression(
    const Tensor& total_sums, const Tensor& total_squares,
    const Tensor& split_sums, const Tensor& split_squares,
    int32 accumulator) {
  int32 best_feature_index = -1;
  // We choose the split with the lowest score.
  float best_score = kint64max;
  const int32 num_splits = static_cast<int32>(split_sums.shape().dim_size(1));
  const int32 num_regression_dims = static_cast<int32>(
      split_sums.shape().dim_size(2));
  // Ideally, Eigen::Tensor::chip would be best to use here but it results
  // in seg faults, so we have to go with flat views of these tensors.  However,
  // it is still pretty efficient because we put off evaluation until the
  // score is actually returned.
  const auto tc_sum = total_sums.Slice(
      accumulator, accumulator + 1).unaligned_flat<float>();
  const auto tc_square = total_squares.Slice(
      accumulator, accumulator + 1).unaligned_flat<float>();
  const auto splits_sum = split_sums.Slice(
      accumulator, accumulator + 1).unaligned_flat<float>();
  const auto splits_square = split_squares.Slice(
      accumulator, accumulator + 1).unaligned_flat<float>();
  // Eigen is infuriating to work with, usually resulting in all kinds of
  // unhelpful compiler errors when trying something that seems sane.  This
  // helps us do a simple thing like access the first element (the counts)
  // of these tensors so we can calculate expected value in Variance().
  const auto splits_count_accessor = split_sums.tensor<float, 3>();
  const auto totals_count_accessor = total_sums.tensor<float, 2>();

  Eigen::array<int, 1> bcast({num_splits});
  const auto right_sums = tc_sum.broadcast(bcast) - splits_sum;
  const auto right_squares = tc_square.broadcast(bcast) - splits_square;

  for (int i = 0; i < num_splits; i++) {
    Eigen::array<int, 1> offsets = {i * num_regression_dims};
    Eigen::array<int, 1> extents = {num_regression_dims};
    float left_count = splits_count_accessor(accumulator, i, 0);
    float right_count = totals_count_accessor(accumulator, 0) - left_count;

    float score = 0;

    // Guard against divide-by-zero.
    if (left_count > 0) {
      score += WeightedVariance(
        splits_sum.slice(offsets, extents),
        splits_square.slice(offsets, extents), left_count);
    }

    if (right_count > 0) {
        score += WeightedVariance(right_sums.slice(offsets, extents),
                                  right_squares.slice(offsets, extents),
                                  right_count);
    }

    if (score < best_score) {
      best_score = score;
      best_feature_index = i;
    }
  }
  return best_feature_index;
}

bool DecideNode(const Tensor& point, int32 feature, float bias) {
  const auto p = point.unaligned_flat<float>();
  CHECK_LT(feature, p.size());
  return p(feature) > bias;
}

bool IsAllInitialized(const Tensor& features) {
  const auto feature_vec = features.unaligned_flat<int32>();
  return feature_vec(feature_vec.size() - 1) >= 0;
}


}  // namespace tensorforest
}  // namespace tensorflow
