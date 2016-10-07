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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_CORE_OPS_TREE_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_CORE_OPS_TREE_UTILS_H_

#include <limits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensorforest {

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

// Get the DataColumnTypes number for the given feature.  The default type
// is stored in index 0, individual feature types start at index 1.
DataColumnTypes FeatureSpec(int32 input_feature, const Tensor& spec);

// Given an Eigen::Tensor type, calculate the Gini impurity, which we use
// to determine the best split (lowest) and which nodes to allocate first
// (highest).
template<typename T>
float WeightedGiniImpurity(const T& counts) {
  // Our split score is the Gini impurity times the number of examples
  // seen by the leaf.  If c(i) denotes the i-th class count and c = sum_i c(i)
  // then
  // score = c * (1 - sum_i ( c(i) / c )^2 )
  //       = c - sum_i c(i)^2 / c
  const auto smoothed = counts + counts.constant(1.0f);
  const auto sum = smoothed.sum();
  const auto sum2 = smoothed.square().sum();
  Eigen::Tensor<float, 0, Eigen::RowMajor> ret = sum - (sum2 / sum);
  return ret(0);
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

// Returns true if the best split's Gini impurity is sufficiently smaller than
// that of the next best split.
bool BestSplitDominatesClassification(
    const Tensor& total_counts,
    const Tensor& split_counts, int32 accumulator,
    float dominate_fraction);

// Initializes everything in the given tensor to the given value.
template <typename T>
void Initialize(Tensor counts, T val = 0) {
  auto flat = counts.unaligned_flat<T>();
  std::fill(flat.data(), flat.data() + flat.size(), val);
}

// Returns true if the point falls to the right (i.e., the selected feature
// of the input point is greater than the bias threshold), and false if it
// falls to the left.
// Even though our input data is forced into float Tensors, it could have
// originally been something else (e.g. categorical string data) which
// we treat differently.
bool DecideNode(const Tensor& point, int32 feature, float bias,
                DataColumnTypes type = kDataFloat);

// Returns input_data(i, feature) > bias.
template <typename T>
bool DecideDenseNode(const T& input_data,
                     int32 i, int32 feature, float bias,
                     DataColumnTypes type = kDataFloat) {
    CHECK_LT(i, input_data.dimensions()[0]);
    CHECK_LT(feature, input_data.dimensions()[1]);
    return Decide(input_data(i, feature), bias, type);
}

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

// Returns t(i, feature) > bias, where t is the sparse tensor represented by
// sparse_input_indices and sparse_input_values.
template <typename T1, typename T2>
bool DecideSparseNode(
    const T1& sparse_input_indices,
    const T2& sparse_input_values,
    int32 i, int32 feature, float bias,
    DataColumnTypes type = kDataFloat) {
  return Decide(
      FindSparseValue(sparse_input_indices, sparse_input_values, i, feature),
      bias, type);
}

// Returns left/right decision between the input value and the threshold bias.
// For floating point types, the decision is value > bias, but for
// categorical data, it is value != bias.
bool Decide(float value, float bias, DataColumnTypes type = kDataFloat);

// Returns true if all the splits are initialized. Since they get initialized
// in order, we can simply infer this from the last split.
// This should only be called for a single allocator's candidate features
// (i.e. candidate_split_features.Slice(accumulator, accumulator + 1) ).
bool IsAllInitialized(const Tensor& features);

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

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_CORE_OPS_TREE_UTILS_H_
