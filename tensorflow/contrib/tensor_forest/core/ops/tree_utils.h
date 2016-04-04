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
#ifndef LEARNING_LIB_TENSOR_FOREST_V2_TREE_UTILS_H_
#define LEARNING_LIB_TENSOR_FOREST_V2_TREE_UTILS_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensorforest {

// Indexes in the tree representation's 2nd dimension for children and features.
const int32 CHILDREN_INDEX = 0;
const int32 FEATURE_INDEX = 1;

// Used in the tree's children sub-tensor to indicate leaf and free nodes.
const int32 LEAF_NODE = -1;
const int32 FREE_NODE = -2;

// Calculates the sum of a tensor.
template<typename T>
T Sum(tensorflow::Tensor counts) {
  Eigen::Tensor<T, 0, Eigen::RowMajor> count_sum =
      counts.unaligned_flat<T>().sum();
  return count_sum(0);
}

// Given an Eigen::Tensor type, calculate the Gini impurity, which we use
// to determine the best split (lowest) and which nodes to allocate first
// (highest).
template<typename T>
int32 WeightedGiniImpurity(const T& counts) {
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

// Returns the best split to use based on the (lowest) Gini impurity.
// Takes in the whole total and per-split count tensors because using
// Tensor::Slice returns a tensor of the same dimensionality, which makes
// things a little awkward.
// TODO(gilberth): Currently test_util.BestFeatureToSplit doesn't work with
// this code because the shapes of the incoming tensors are different than
// in v1.  Try to make it work for both versions?
int32 BestFeature(const tensorflow::Tensor& total_counts,
                  const tensorflow::Tensor& split_counts,
                  int32 accumulator);

// Initializes everything in the given tensor to the given value.
template <typename T>
void Initialize(tensorflow::Tensor counts, T val = 0) {
  auto flat = counts.unaligned_flat<T>();
  std::fill(flat.data(), flat.data() + flat.size(), val);
}

// Returns true if the point falls to the right (i.e., the selected feature
// of the input point is greater than the bias threshold), and false if it
// falls to the left.
bool DecideNode(const tensorflow::Tensor& point, int32 feature, float bias);

// Returns true if all the splits are initialized. Since they get initialized
// in order, we can simply infer this from the last split.
// This should only be called for a single allocator's candidate features
// (i.e. candidate_split_features.Slice(accumulator, accumulator + 1) ).
bool IsAllInitialized(const tensorflow::Tensor& features);

}  // namespace tensorforest
} // namespace tensorflow

#endif  // LEARNING_LIB_TENSOR_FOREST_V2_TREE_UTILS_H_
