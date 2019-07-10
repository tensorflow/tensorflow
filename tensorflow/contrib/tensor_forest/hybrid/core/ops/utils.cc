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
#include "tensorflow/contrib/tensor_forest/hybrid/core/ops/utils.h"

#include <math.h>
#include <cmath>
#include <vector>

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace tensorforest {

using tensorflow::Tensor;

float LeftProbability(const Tensor& point, const Tensor& weight, float bias,
                      int num_features) {
  const auto p = point.unaligned_flat<float>();
  const auto w = weight.unaligned_flat<float>();
  float dot_product = 0.0;
  for (int i = 0; i < num_features; i++) {
    dot_product += w(i) * p(i);
  }

  // TODO(thomaswc): At some point we should consider
  // //learning/logistic/logodds-to-prob.h
  return 1.0 / (1.0 + std::exp(-dot_product + bias));
}

float LeftProbabilityK(const Tensor& point, std::vector<int32> feature_set,
                       const Tensor& weight, float bias, int num_features,
                       int k) {
  const auto p = point.unaligned_flat<float>();
  const auto w = weight.unaligned_flat<float>();

  float dot_product = 0.0;

  for (int32 i = 0; i < k; i++) {
    CHECK_LT(feature_set[i], num_features);
    dot_product += p(feature_set[i]) * w(i);
  }

  // TODO(thomaswc): At some point we should consider
  // //learning/logistic/logodds-to-prob.h
  return 1.0 / (1.0 + std::exp(-dot_product + bias));
}

void GetFeatureSet(int32 tree_num, int32 node_num, int32 random_seed,
                   int32 num_features, int32 num_features_to_pick,
                   std::vector<int32>* features) {
  features->clear();
  uint64 seed = node_num ^ (tree_num << 16) ^ random_seed;
  random::PhiloxRandom rng(seed);
  for (int i = 0; i < num_features_to_pick; ++i) {
    // PhiloxRandom returns an array of int32's
    const random::PhiloxRandom::ResultType rand = rng();
    const int32 feature = (rand[0] + rand[1]) % num_features;
    features->push_back(feature);
  }
}

}  // namespace tensorforest
}  // namespace tensorflow
