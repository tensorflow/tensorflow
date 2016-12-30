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
#include <algorithm>
#include <cfloat>
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace tensorforest {

using tensorflow::Tensor;

DataColumnTypes FeatureSpec(int32 input_feature, const Tensor& spec) {
  const int32 spec_feature =
      (input_feature + 1 < spec.NumElements()) ? input_feature : 0;
  CHECK(spec_feature >= 0) << "spec feature is not >= than zero: "
                           << spec_feature;
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

// We return the Gini Impurity of the bootstrap sample as an int rather
// than a float, so that we can more easily check for ties.
int BootstrapGini(int n, int s, const random::DistributionSampler& ds,
                  random::SimplePhilox* rand) {
  std::vector<int> counts(s, 0);
  for (int i = 0; i < n; i++) {
    int j = ds.Sample(rand);
    counts[j] += 1;
  }
  int g = 0;
  for (int j = 0; j < s; j++) {
    g += counts[j] * counts[j];
  }
  // The true gini is 1 + (-g) / n^2
  return -g;
}

// Populate *weights with the smoothed per-class frequencies needed to
// initialize a DistributionSampler.  Returns the total number of samples
// seen by this accumulator.
int MakeBootstrapWeights(const Tensor& total_counts, const Tensor& split_counts,
                         int32 accumulator, int index,
                         std::vector<float>* weights) {
  const int32 num_classes =
      static_cast<int32>(split_counts.shape().dim_size(2)) - 1;

  auto tc = total_counts.tensor<float, 2>();
  auto lc = split_counts.tensor<float, 3>();

  int n = tc(accumulator, 0);

  float denom = static_cast<float>(n) + static_cast<float>(num_classes);

  weights->resize(num_classes * 2);
  for (int i = 0; i < num_classes; i++) {
    // Use the Laplace smoothed per-class probabilities when generating the
    // bootstrap samples.
    float left_count = lc(accumulator, index, i + 1);
    (*weights)[i] = (left_count + 1.0) / denom;
    float right_count = tc(accumulator, i + 1) - left_count;
    (*weights)[num_classes + i] = (right_count + 1.0) / denom;
  }

  return n;
}

bool BestSplitDominatesClassificationBootstrap(const Tensor& total_counts,
                                               const Tensor& split_counts,
                                               int32 accumulator,
                                               float dominate_fraction,
                                               random::SimplePhilox* rand) {
  float best_score;
  float second_best_score;
  int best_feature_index;
  int second_best_index;
  GetTwoBestClassification(total_counts, split_counts, accumulator, &best_score,
                           &best_feature_index, &second_best_score,
                           &second_best_index);

  std::vector<float> weights1;
  int n1 = MakeBootstrapWeights(total_counts, split_counts, accumulator,
                                best_feature_index, &weights1);
  random::DistributionSampler ds1(weights1);

  std::vector<float> weights2;
  int n2 = MakeBootstrapWeights(total_counts, split_counts, accumulator,
                                second_best_index, &weights2);
  random::DistributionSampler ds2(weights2);

  const int32 num_classes =
      static_cast<int32>(split_counts.shape().dim_size(2)) - 1;

  float p = 1.0 - dominate_fraction;
  if (p <= 0 || p > 1.0) {
    LOG(FATAL) << "Invalid dominate fraction " << dominate_fraction;
  }

  int bootstrap_samples = 1;
  while (p < 1.0) {
    bootstrap_samples += 1;
    p = p * 2;
  }

  int worst_g1 = 0;
  for (int i = 0; i < bootstrap_samples; i++) {
    int g1 = BootstrapGini(n1, 2 * num_classes, ds1, rand);
    worst_g1 = std::max(worst_g1, g1);
  }

  int best_g2 = 99;
  for (int i = 0; i < bootstrap_samples; i++) {
    int g2 = BootstrapGini(n2, 2 * num_classes, ds2, rand);
    best_g2 = std::min(best_g2, g2);
  }

  return worst_g1 < best_g2;
}

bool BestSplitDominatesClassificationHoeffding(const Tensor& total_counts,
                                               const Tensor& split_counts,
                                               int32 accumulator,
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

  float hoeffding_bound =
      range * sqrt(log(1.0 / (1.0 - dominate_fraction)) / (2.0 * n));

  VLOG(1) << "num_classes = " << num_classes;
  VLOG(1) << "n = " << n;
  VLOG(1) << "range = " << range;
  VLOG(1) << "hoeffding_bound = " << hoeffding_bound;
  return (second_best_score - best_score) > hoeffding_bound;
}

double DirichletCovarianceTrace(const Tensor& total_counts,
                                const Tensor& split_counts, int32 accumulator,
                                int index) {
  const int32 num_classes =
      static_cast<int32>(split_counts.shape().dim_size(2)) - 1;

  auto tc = total_counts.tensor<float, 2>();
  auto lc = split_counts.tensor<float, 3>();

  double leftc = 0.0;
  double leftc2 = 0.0;
  double rightc = 0.0;
  double rightc2 = 0.0;
  for (int i = 1; i <= num_classes; i++) {
    double l = lc(accumulator, index, i) + 1.0;
    leftc += l;
    leftc2 += l * l;

    double r = tc(accumulator, i) - lc(accumulator, index, i) + 1.0;
    rightc += r;
    rightc2 += r * r;
  }

  double left_trace = (1.0 - leftc2 / (leftc * leftc)) / (leftc + 1.0);
  double right_trace = (1.0 - rightc2 / (rightc * rightc)) / (rightc + 1.0);
  return left_trace + right_trace;
}

void getDirichletMean(const Tensor& total_counts, const Tensor& split_counts,
                      int32 accumulator, int index, std::vector<float>* mu) {
  const int32 num_classes =
      static_cast<int32>(split_counts.shape().dim_size(2)) - 1;

  mu->resize(num_classes * 2);
  auto tc = total_counts.tensor<float, 2>();
  auto lc = split_counts.tensor<float, 3>();

  double total = tc(accumulator, 0);

  for (int i = 0; i < num_classes; i++) {
    double l = lc(accumulator, index, i + 1);
    mu->at(i) = (l + 1.0) / (total + num_classes);

    double r = tc(accumulator, i) - l;
    mu->at(i + num_classes) = (r + 1.) / (total + num_classes);
  }
}

// Given lambda3, returns the distance from (mu1, mu2) to the surface.
double getDistanceFromLambda3(double lambda3, const std::vector<float>& mu1,
                              const std::vector<float>& mu2) {
  if (fabs(lambda3) == 1.0) {
    return 0.0;
  }

  int n = mu1.size();
  double lambda1 = -2.0 * lambda3 / n;
  double lambda2 = 2.0 * lambda3 / n;
  // From below,
  //   x = (lambda_1 1 + 2 mu1) / (2 - 2 lambda_3)
  //   y = (lambda_2 1 + 2 mu2) / (2 + 2 lambda_3)
  double dist = 0.0;
  for (size_t i = 0; i < mu1.size(); i++) {
    double diff = (lambda1 + 2.0 * mu1[i]) / (2.0 - 2.0 * lambda3) - mu1[i];
    dist += diff * diff;
    diff = (lambda2 + 2.0 * mu2[i]) / (2.0 + 2.0 * lambda3) - mu2[i];
    dist += diff * diff;
  }
  return dist;
}

// Returns the distance between (mu1, mu2) and (x, y), where (x, y) is the
// nearest point that lies on the surface defined by
// {x dot 1 = 1, y dot 1 = 1, x dot x - y dot y = 0}.
double getChebyshevEpsilon(const std::vector<float>& mu1,
                           const std::vector<float>& mu2) {
  // Math time!!
  // We are trying to minimize d = |mu1 - x|^2 + |mu2 - y|^2 over the surface.
  // Using Langrange multipliers, we get
  //   partial d / partial x = -2 mu1 + 2 x = lambda_1 1 + 2 lambda_3 x
  //   partial d / partial y = -2 mu2 + 2 y = lambda_2 1 - 2 lambda_3 y
  // or
  //   x = (lambda_1 1 + 2 mu1) / (2 - 2 lambda_3)
  //   y = (lambda_2 1 + 2 mu2) / (2 + 2 lambda_3)
  // which implies
  //   2 - 2 lambda_3 = lambda_1 1 dot 1 + 2 mu1 dot 1
  //   2 + 2 lambda_3 = lambda_2 1 dot 1 + 2 mu2 dot 1
  //   |lambda_1 1 + 2 mu1|^2 (2 + 2 lambda_3)^2 =
  //     |lambda_2 1 + 2 mu2|^2 (2 - 2 lambda_3)^2
  // So solving for the lambda's and using the fact that
  // mu1 dot 1 = 1 and mu2 dot 1 = 1,
  //   lambda_1 = -2 lambda_3 / (1 dot 1)
  //   lambda_2 = 2 lambda_3 / (1 dot 1)
  // and (letting n = 1 dot 1)
  //   | - lambda_3 1 + n mu1 |^2 (1 + lambda_3)^2 =
  //   | lambda_3 1 + n mu2 |^2 (1 - lambda_3)^2
  // or
  // (lambda_3^2 n - 2 n lambda_3 + n^2 mu1 dot mu1)(1 + lambda_3)^2 =
  // (lambda_3^2 n + 2 n lambda_3 + n^2 mu2 dot mu2)(1 - lambda_3)^2
  // or
  // (lambda_3^2 - 2 lambda_3 + n mu1 dot mu1)(1 + 2 lambda_3 + lambda_3^2) =
  // (lambda_3^2 + 2 lambda_3 + n mu2 dot mu2)(1 - 2 lambda_3 + lambda_3^2)
  // or
  // lambda_3^2 - 2 lambda_3 + n mu1 dot mu1
  // + 2 lambda_3^3 - 2 lambda_3^2 + 2n lambda_3 mu1 dot mu1
  // + lambda_3^4 - 2 lambda_3^3 + n lambda_3^2 mu1 dot mu1
  // =
  // lambda_3^2 + 2 lambda_3 + n mu2 dot mu2
  // - 2 lambda_3^3 -4 lambda_3^2 - 2n lambda_3 mu2 dot mu2
  // + lambda_3^4 + 2 lambda_3^3 + n lambda_3^2 mu2 dot mu2
  // or
  // - 2 lambda_3 + n mu1 dot mu1
  // - 2 lambda_3^2 + 2n lambda_3 mu1 dot mu1
  // + n lambda_3^2 mu1 dot mu1
  // =
  // + 2 lambda_3 + n mu2 dot mu2
  // -4 lambda_3^2 - 2n lambda_3 mu2 dot mu2
  // + n lambda_3^2 mu2 dot mu2
  // or
  // lambda_3^2 (2 + n mu1 dot mu1 + n mu2 dot mu2)
  // + lambda_3 (2n mu1 dot mu1 + 2n mu2 dot mu2 - 4)
  // + n mu1 dot mu1 - n mu2 dot mu2 = 0
  // which can be solved using the quadratic formula.
  int n = mu1.size();
  double len1 = 0.0;
  for (float m : mu1) {
    len1 += m * m;
  }
  double len2 = 0.0;
  for (float m : mu2) {
    len2 += m * m;
  }
  double a = 2 + n * (len1 + len2);
  double b = 2 * n * (len1 + len2) - 4;
  double c = n * (len1 - len2);
  double discrim = b * b - 4 * a * c;
  if (discrim < 0.0) {
    LOG(WARNING) << "Negative discriminant " << discrim;
    return 0.0;
  }

  double sdiscrim = sqrt(discrim);
  // TODO(thomaswc): Analyze whetever one of these is always closer.
  double v1 = (-b + sdiscrim) / (2 * a);
  double v2 = (-b - sdiscrim) / (2 * a);
  double dist1 = getDistanceFromLambda3(v1, mu1, mu2);
  double dist2 = getDistanceFromLambda3(v2, mu1, mu2);
  return std::min(dist1, dist2);
}

bool BestSplitDominatesClassificationChebyshev(const Tensor& total_counts,
                                               const Tensor& split_counts,
                                               int32 accumulator,
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

  VLOG(1) << "num_classes = " << num_classes;
  VLOG(1) << "n = " << n;
  double trace = DirichletCovarianceTrace(total_counts, split_counts,
                                          accumulator, best_feature_index) +
                 DirichletCovarianceTrace(total_counts, split_counts,
                                          accumulator, second_best_index);

  std::vector<float> mu1;
  getDirichletMean(total_counts, split_counts, accumulator, best_feature_index,
                   &mu1);
  std::vector<float> mu2;
  getDirichletMean(total_counts, split_counts, accumulator, second_best_index,
                   &mu2);
  double epsilon = getChebyshevEpsilon(mu1, mu2);

  if (epsilon == 0.0) {
    return false;
  }

  double dirichlet_bound = 1.0 - trace / (epsilon * epsilon);
  return dirichlet_bound > dominate_fraction;
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

void GetParentWeightedMean(float leaf_sum, const float* leaf_data,
                           float parent_sum, const float* parent_data,
                           float valid_leaf_threshold, int num_outputs,
                           std::vector<float>* mean) {
  float parent_weight = 0.0;
  if (leaf_sum < valid_leaf_threshold && parent_sum >= 0) {
    VLOG(1) << "not enough samples at leaf, including parent counts."
            << "child sum = " << leaf_sum;
    // Weight the parent's counts just enough so that the new sum is
    // valid_leaf_threshold_, but never give any counts a weight of
    // more than 1.
    parent_weight =
        std::min(1.0f, (valid_leaf_threshold - leaf_sum) / parent_sum);
    leaf_sum += parent_weight * parent_sum;
    VLOG(1) << "Sum w/ parent included = " << leaf_sum;
  }

  for (int c = 0; c < num_outputs; c++) {
    float w = leaf_data[c];
    if (parent_weight > 0.0) {
      w += parent_weight * parent_data[c];
    }
    (*mean)[c] = w / leaf_sum;
  }
}

}  // namespace tensorforest
}  // namespace tensorflow
