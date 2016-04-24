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

int32 BestFeature(const Tensor& total_counts, const Tensor& split_counts,
                  int32 accumulator) {
  int32 best_feature_index = -1;
  // We choose the split with the lowest score.
  float best_score = kint64max;
  const int32 num_splits = split_counts.shape().dim_size(1);
  const int32 num_classes = split_counts.shape().dim_size(2);
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

bool DecideNode(const Tensor& point, int32 feature, float bias) {
  const auto p = point.unaligned_flat<float>();
  return p(feature) > bias;
}

bool IsAllInitialized(const Tensor& features) {
  const auto feature_vec = features.unaligned_flat<int32>();
  return feature_vec(feature_vec.size() - 1) >= 0;
}


}  // namespace tensorforest
}  // namespace tensorflow
