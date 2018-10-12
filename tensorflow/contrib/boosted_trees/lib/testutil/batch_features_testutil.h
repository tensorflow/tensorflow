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
#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_TESTUTIL_BATCH_FEATURES_TESTUTIL_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_TESTUTIL_BATCH_FEATURES_TESTUTIL_H_

#include "tensorflow/contrib/boosted_trees/lib/utils/batch_features.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace boosted_trees {
namespace testutil {

// This method calls Initialize on the given 'batch_features', which will be
// populated with randomly generated feature values when the call returns.
// 'tensors' returns a vector of all tensors used in the initialization,
// because they must outlive 'batch_features'.
//
// All float features will be either missing or uniformly randomly chosen
// from [0, 1). For sparse (float) features, a sparsity is uniformly randomly
// chosen from ['sparsity_lo', 'sparsity_hi') per feature, and each instance
// will have a probability of sparsity of missing that feature, in other words,
// sparsity = 1 - density.
void RandomlyInitializeBatchFeatures(
    tensorflow::random::SimplePhilox* rng, uint32 num_dense_float_features,
    uint32 num_sparse_float_features, double sparsity_lo, double sparsity_hi,
    boosted_trees::utils::BatchFeatures* batch_features);

}  // namespace testutil
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_TESTUTIL_BATCH_FEATURES_TESTUTIL_H_
