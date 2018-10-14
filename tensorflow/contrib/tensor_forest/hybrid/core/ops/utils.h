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

#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_HYBRID_CORE_OPS_UTILS_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_HYBRID_CORE_OPS_UTILS_H_
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensorforest {

// Returns the probability that the point falls to the left.
float LeftProbability(const Tensor& point, const Tensor& weight, float bias,
                      int num_features);

float LeftProbabilityK(const Tensor& point, std::vector<int32> feature_set,
                       const Tensor& weight, float bias, int num_features,
                       int k);

// Returns a random set of num_features_to_pick features in the
// range [0, num_features).  Must return the same set of
// features for subsequent calls with the same tree_num, node_num, and
// random_seed.  This allows us to calculate feature sets between calls to ops
// without having to store their values.
void GetFeatureSet(int32 tree_num, int32 node_num, int32 random_seed,
                   int32 num_features, int32 num_features_to_pick,
                   std::vector<int32>* features);

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_HYBRID_CORE_OPS_UTILS_H_
