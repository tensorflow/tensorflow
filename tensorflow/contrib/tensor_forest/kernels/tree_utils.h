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
#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_CORE_OPS_TREE_UTILS_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_CORE_OPS_TREE_UTILS_H_

#include <limits>

#include "tensorflow/contrib/tensor_forest/kernels/data_spec.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensorforest {

// We hide Eigen's hideous types behind a function that returns the (i, j)-th
// entry of a two dimensional tensor; this is that function's type.
using GetFeatureFnType = std::function<float(int32, int32)>;

// TODO(gilberth): Put these in protos so they can be shared by C++ and python.
// Indexes in the tree representation's 2nd dimension for children and features.
const int32 CHILDREN_INDEX = 0;
const int32 FEATURE_INDEX = 1;

// Used in the tree's children sub-tensor to indicate leaf and free nodes.
const int32 LEAF_NODE = -1;
const int32 FREE_NODE = -2;

// Used to indicate column types, e.g. categorical vs. float
enum DataColumnTypes {
  kDataFloat = 0,
  kDataCategorical = 1
};

// Calculates the sum of a tensor.
template<typename T>
T Sum(Tensor counts) {
  Eigen::Tensor<T, 0, Eigen::RowMajor> count_sum =
      counts.unaligned_flat<T>().sum();
  return count_sum(0);
}

// Get the two best scores and their indices among max splits.
void GetTwoBest(int max, const std::function<float(int)>& score_fn,
                float* best_score, int* best_index, float* second_best_score,
                int* second_best_index);

// If the Gini Impurity is g, this returns n^2 (g - 1).  This is an int
// rather than a float, and so can be more easily checked for ties.
int BootstrapGini(int n, int s, const random::DistributionSampler& ds,
                  random::SimplePhilox* rand);

// Get the DataColumnTypes number for the given feature.
DataColumnTypes FindDenseFeatureSpec(
    int32 input_feature, const tensorforest::TensorForestDataSpec& spec);
DataColumnTypes FindSparseFeatureSpec(
    int32 input_feature, const tensorforest::TensorForestDataSpec& spec);

// Given an Eigen::Tensor type, calculate the Gini impurity.
template <typename T>
float RawWeightedGiniImpurity(const T& counts) {
  // Our split score is the Gini impurity times the number of examples
  // seen by the leaf.  If c(i) denotes the i-th class count and c = sum_i c(i)
  // then
  // score = c * (1 - sum_i ( c(i) / c )^2 )
  //       = c - sum_i c(i)^2 / c
  const auto sum = counts.sum();
  const auto sum2 = counts.square().sum();
  Eigen::Tensor<float, 0, Eigen::RowMajor> ret = sum - (sum2 / sum);
  return ret(0);
}

// Given an Eigen::Tensor type, calculate the smoothed Gini impurity, which we
// use to determine the best split (lowest) and which nodes to allocate first
// (highest).
template <typename T>
float WeightedGiniImpurity(const T& counts) {
  const auto smoothed = counts + counts.constant(1.0f);
  return RawWeightedGiniImpurity(smoothed);
}

template<typename T1, typename T2>
float WeightedVariance(const T1& sums, const T2& squares, float count) {
  const auto e_x = sums / count;
  const auto e_x2 = squares / count;
  Eigen::Tensor<float, 0, Eigen::RowMajor> ret = (e_x2 - e_x.square()).sum();
  return count * ret(0);
}

// Returns the best split to use based on the (lowest) Gini impurity.
// Takes in the whole total and per-split count tensors because using
// Tensor::Slice returns a tensor of the same dimensionality, which makes
// things a little awkward.
int32 BestFeatureClassification(const Tensor& total_counts,
                                const Tensor& split_counts, int32 accumulator);

// Returns the best split to use based on the (lowest) variance.
int32 BestFeatureRegression(const Tensor& total_sums,
                            const Tensor& total_squares,
                            const Tensor& split_sums,
                            const Tensor& split_squares, int32 accumulator);

// Returns true if the best split's variance is sufficiently smaller than
// that of the next best split.
bool BestSplitDominatesRegression(
    const Tensor& total_sums, const Tensor& total_squares,
    const Tensor& split_sums, const Tensor& split_squares,
    int32 accumulator);

// Performs booststrap_samples bootstrap samples of the best split's class
// counts and the second best splits's class counts, and returns true if at
// least dominate_fraction of the time, the former has a better (lower)
// Gini impurity.  Does not take over ownership of *rand.
bool BestSplitDominatesClassificationBootstrap(
    const Tensor& total_counts, const Tensor& split_counts, int32 accumulator,
    float dominate_fraction, tensorflow::random::SimplePhilox* rand);

// Returns true if the best split's Gini impurity is sufficiently smaller than
// that of the next best split, as measured by the Hoeffding Tree bound.
bool BestSplitDominatesClassificationHoeffding(const Tensor& total_counts,
                                               const Tensor& split_counts,
                                               int32 accumulator,
                                               float dominate_fraction);

// Returns true if the best split's Gini impurity is sufficiently smaller than
// that of the next best split, as measured by a Chebyshev bound.
bool BestSplitDominatesClassificationChebyshev(const Tensor& total_counts,
                                               const Tensor& split_counts,
                                               int32 accumulator,
                                               float dominate_fraction);

// Initializes everything in the given tensor to the given value.
template <typename T>
void Initialize(Tensor counts, T val = 0) {
  auto flat = counts.unaligned_flat<T>();
  std::fill(flat.data(), flat.data() + flat.size(), val);
}

// Returns a function that accesses the (i,j)-th element of the specified two
// dimensional tensor.
GetFeatureFnType GetDenseFunctor(const Tensor& dense);

// Returns a function that looks for the j-th feature of the i-th example
// in the sparse data, or the default value if not found.  See FindSparseValue.
GetFeatureFnType GetSparseFunctor(const Tensor& sparse_indices,
                                  const Tensor& sparse_values);

// Returns true if the point falls to the right (i.e., the selected feature
// of the input point is greater than the bias threshold), and false if it
// falls to the left.
// Even though our input data is forced into float Tensors, it could have
// originally been something else (e.g. categorical string data) which
// we treat differently.
bool DecideNode(const GetFeatureFnType& get_dense,
                const GetFeatureFnType& get_sparse, int32 i, int32 feature,
                float bias, const tensorforest::TensorForestDataSpec& spec);

// If T is a sparse float matrix represented by sparse_input_indices and
// sparse_input_values, FindSparseValue returns T(i,j), or 0.0 if (i,j)
// isn't present in sparse_input_indices.  sparse_input_indices is assumed
// to be sorted.
template <typename T1, typename T2>
float FindSparseValue(
    const T1& sparse_input_indices,
    const T2& sparse_input_values,
    int32 i, int32 j) {
  int32 low = 0;
  int32 high = sparse_input_values.dimension(0);
  while (low < high) {
    int32 mid = (low + high) / 2;
    int64 midi = internal::SubtleMustCopy(sparse_input_indices(mid, 0));
    int64 midj = internal::SubtleMustCopy(sparse_input_indices(mid, 1));
    if (midi == i) {
      if (midj == j) {
        return sparse_input_values(mid);
      }
      if (midj < j) {
        low = mid + 1;
      } else {
        high = mid;
      }
      continue;
    }
    if (midi < i) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return 0.0;
}

// Returns the number of sparse values for example input_index.
// Also returns the index where those features start in sparse_input_start
// if any were found.
// Assumes that the first column in indices is ordered.
template <typename T1>
int32 GetNumSparseFeatures(const T1& indices, int32 input_index,
                           int64* sparse_input_start) {
  // Binary search for input_index.
  // TODO(gilberth): Consider using std::lower_bound, std::upper_bound
  // for a simpler but possibly slower solution, or searching for
  // input_start and input_end simultaneously.
  const int64 num_total = indices.dimension(0);
  int64 index;
  int64 low = 0;
  int64 high = num_total;
  *sparse_input_start = -1;  // Easy error checking.

  while (true) {
    if (low == high) {
      return 0;
    }
    index = low + (high - low) / 2;
    const int64 feature_index = indices(index, 0);
    if (feature_index == input_index) {
      // found it.
      break;
    } else if (feature_index < input_index) {
      // Correct for the implicit floor in the index assignment.
      if (low == index) {
        return 0;
      }
      low = index;
    } else {
      high = index;
    }
  }

  // Scan for the start and end of the input_index range.
  int64 input_start = index;
  int64 val = indices(input_start, 0);
  while (val == input_index) {
    --input_start;
    if (input_start < 0) {
      break;
    }
    val = indices(input_start, 0);
  }
  *sparse_input_start = input_start + 1;
  int32 input_end = index;
  val = indices(input_end, 0);
  while (val == input_index) {
    ++input_end;
    if (input_end >= num_total) {
      break;
    }
    val = indices(input_end, 0);
  }
  return input_end - input_start - 1;
}

// Returns left/right decision between the input value and the threshold bias.
// For floating point types, the decision is value > bias, but for
// categorical data, it is value != bias.
bool Decide(float value, float bias, DataColumnTypes type = kDataFloat);


// Returns true if all the splits are initialized. Since they get initialized
// in order, we can simply infer this from the last split.
// This should only be called for a single allocator's candidate features
// (i.e. candidate_split_features.Slice(accumulator, accumulator + 1) ).
template <typename EigenType>
bool IsAllInitialized(const EigenType& features, int32 accumulator,
                      int32 num_splits) {
  return features(accumulator, num_splits - 1) >= 0;
}

// Tensorforest currently only allows tensors up to 2^31 elements.  Return false
// if any dimension is greater than that, true otherwise.
inline bool CheckTensorBounds(OpKernelContext* context, const Tensor& tensor) {
  for (int i = 0; i < (tensor).dims(); ++i) {
    if (!TF_PREDICT_TRUE(tensor.shape().dim_size(i) <
                         std::numeric_limits<int32>::max())) {
      context->CtxFailure((errors::InvalidArgument(
          strings::StrCat("Tensor has a dimension that is greater than 2^31: ",
                          tensor.DebugString()))));
      return false;
    }
  }
  return true;
}

void GetParentWeightedMean(float leaf_sum, const float* leaf_data,
                           float parent_sum, const float* parent_data,
                           float valid_leaf_threshold, int num_outputs,
                           std::vector<float>* mean);

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_CORE_OPS_TREE_UTILS_H_
