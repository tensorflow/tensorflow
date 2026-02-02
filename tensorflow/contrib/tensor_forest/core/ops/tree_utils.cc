// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#include <cfloat>
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace tensorforest {

using tensorflow::Tensor;

DataColumnTypes FeatureSpec(int32 input_feature, const Tensor& spec) {
  const int32 spec_feature =
      (input_feature + 1 < spec.NumElements()) ? input_feature : 0;
  return static_cast<DataColumnTypes>(
      spec.unaligned_flat<int32>()(spec_feature));
}

void GetTwoBest(int max, std::function<float(int)> score_fn, float* best_score,
                int* best_index, float* second_best_score,
                int* second_best_index) {
  *best_index = -1;
  *second_best_index = -1;
  *best_score = FLT_MAX;
  *second_best_score = FLT_MAX;
  for (int i = 0; i < max; i++) {
    float score = score_fn(i);
    if (score < *best_score) {
      *second_best_score = *best_score;
      *second_best_index = *best_index;
      *best_score = score;
      *best_index = i;
    } else if (score < *second_best_score) {
      *second_best_score = score;
      *second_best_index = i;
    }
  }
}

float ClassificationSplitScore(
    const Eigen::Tensor<float, 1, Eigen::RowMajor>& splits,
    const Eigen::Tensor<float, 1, Eigen::RowMajor>& rights,
    int32 num_classes, int i) {
  Eigen::array<int, 1> offsets;
  // Class counts are stored with the total in [0], so the length of each
  // count vector is num_classes + 1.
  offsets[0] = i * (num_classes + 1) + 1;
  Eigen::array<int, 1> extents;
  extents[0] = num_classes;
  return WeightedGiniImpurity(splits.slice(offsets, extents)) +
      WeightedGiniImpurity(rights.slice(offsets, extents));
}

void GetTwoBestClassification(const Tensor& total_counts,
                              const Tensor& split_counts, int32 accumulator,
                              float* best_score, int* best_index,
                              float* second_best_score,
                              int* second_best_index) {
  const int32 num_splits = static_cast<int32>(split_counts.shape().dim_size(1));
  const int32 num_classes =
      static_cast<int32>(split_counts.shape().dim_size(2)) - 1;

  // Ideally, Eigen::Tensor::chip would be best to use here but it results
  // in seg faults, so we have to go with flat views of these tensors.  However,
  // it is still pretty efficient because we put off evaluation until the
  // score is actually returned.
  const auto tc = total_counts.Slice(
      accumulator, accumulator + 1).unaligned_flat<float>();

  // TODO(gilberth): See if we can delay evaluation here by templating the
  // arguments to ClassificationSplitScore.
  const Eigen::Tensor<float, 1, Eigen::RowMajor> splits = split_counts.Slice(
      accumulator, accumulator + 1).unaligned_flat<float>();
  Eigen::array<int, 1> bcast;
  bcast[0] = num_splits;
  const Eigen::Tensor<float, 1, Eigen::RowMajor> rights =
      tc.broadcast(bcast) - splits;

  std::function<float(int)> score_fn = std::bind(
      ClassificationSplitScore, splits, rights, num_classes,
      std::placeholders::_1);

  GetTwoBest(num_splits, score_fn, best_score, best_index, second_best_score,
             second_best_index);
}

int32 BestFeatureClassification(
    const Tensor& total_counts, const Tensor& split_counts,
    int32 accumulator) {
  float best_score;
  float second_best_score;
  int best_feature_index;
  int second_best_index;
  GetTwoBestClassification(total_counts, split_counts, accumulator, &best_score,
                           &best_feature_index, &second_best_score,
                           &second_best_index);
  return best_feature_index;
}

float RegressionSplitScore(
    const Eigen::Tensor<float, 3, Eigen::RowMajor>& splits_count_accessor,
    const Eigen::Tensor<float, 2, Eigen::RowMajor>& totals_count_accessor,
    const Eigen::Tensor<float, 1, Eigen::RowMajor>& splits_sum,
    const Eigen::Tensor<float, 1, Eigen::RowMajor>& splits_square,
    const Eigen::Tensor<float, 1, Eigen::RowMajor>& right_sums,
    const Eigen::Tensor<float, 1, Eigen::RowMajor>& right_squares,
    int32 accumulator,
    int32 num_regression_dims, int i) {
  Eigen::array<int, 1> offsets = {i * num_regression_dims + 1};
  Eigen::array<int, 1> extents = {num_regression_dims - 1};
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
  return score;
}

void GetTwoBestRegression(const Tensor& total_sums, const Tensor& total_squares,
                          const Tensor& split_sums, const Tensor& split_squares,
                          int32 accumulator, float* best_score, int* best_index,
                          float* second_best_score, int* second_best_index) {
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

  Eigen::array<int, 1> bcast;
  bcast[0] = num_splits;
  const auto right_sums = tc_sum.broadcast(bcast) - splits_sum;
  const auto right_squares = tc_square.broadcast(bcast) - splits_square;

  GetTwoBest(num_splits,
             std::bind(RegressionSplitScore, splits_count_accessor,
                       totals_count_accessor, splits_sum, splits_square,
                       right_sums, right_squares, accumulator,
                       num_regression_dims, std::placeholders::_1),
             best_score, best_index, second_best_score, second_best_index);
}

int32 BestFeatureRegression(
    const Tensor& total_sums, const Tensor& total_squares,
    const Tensor& split_sums, const Tensor& split_squares,
    int32 accumulator) {
  float best_score;
  float second_best_score;
  int best_feature_index;
  int second_best_index;
  GetTwoBestRegression(total_sums, total_squares, split_sums, split_squares,
                       accumulator, &best_score, &best_feature_index,
                       &second_best_score, &second_best_index);
  return best_feature_index;
}

bool BestSplitDominatesRegression(
    const Tensor& total_sums, const Tensor& total_squares,
    const Tensor& split_sums, const Tensor& split_squares,
    int32 accumulator) {
  // TODO(thomaswc): Implement this, probably as part of v3.
  return false;
}

bool BestSplitDominatesClassification(
    const Tensor& total_counts,
    const Tensor& split_counts, int32 accumulator,
    float dominate_fraction) {
  float best_score;
  float second_best_score;
  int best_feature_index;
  int second_best_index;
  VLOG(1) << "BSDC for accumulator " << accumulator;
  GetTwoBestClassification(total_counts, split_counts, accumulator, &best_score,
                           &best_feature_index, &second_best_score,
                           &second_best_index);
  VLOG(1) << "Best score = " << best_score;
  VLOG(1) << "2nd best score = " << second_best_score;

  const int32 num_classes =
      static_cast<int32>(split_counts.shape().dim_size(2)) - 1;
  const float n = total_counts.Slice(accumulator, accumulator + 1)
                      .unaligned_flat<float>()(0);

  // Each term in the Gini impurity can range from 0 to 0.5 * 0.5.
  float range = 0.25 * static_cast<float>(num_classes) * n;

  // TODO(thomaswc): The hoeffding bound is actually only valid for linear
  // functions, which the Gini impurity is not.  Come up with a better bound!
  float hoeffding_bound =
      range * sqrt(log(1.0 / (1.0 - dominate_fraction)) / (2.0 * n));

  VLOG(1) << "num_classes = " << num_classes;
  VLOG(1) << "n = " << n;
  VLOG(1) << "range = " << range;
  VLOG(1) << "hoeffding_bound = " << hoeffding_bound;
  return (second_best_score - best_score) > hoeffding_bound;
}

bool DecideNode(const Tensor& point, int32 feature, float bias,
                DataColumnTypes type) {
  const auto p = point.unaligned_flat<float>();
  CHECK_LT(feature, p.size());
  return Decide(p(feature), bias, type);
}


bool Decide(float value, float bias, DataColumnTypes type) {
  switch (type) {
    case kDataFloat:
      return value > bias;

    case kDataCategorical:
      // We arbitrarily define categorical equality as going left.
      return value != bias;

    default:
      LOG(ERROR) << "Got unknown column type: " << type;
      return false;
  }
}


bool IsAllInitialized(const Tensor& features) {
  const auto feature_vec = features.unaligned_flat<int32>();
  return feature_vec(feature_vec.size() - 1) >= 0;
}


}  // namespace tensorforest
}  // namespace tensorflow
