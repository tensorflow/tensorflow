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
#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_STAT_UTILS_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_STAT_UTILS_H_
#include "tensorflow/contrib/tensor_forest/proto/fertile_stats.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensorforest {

// Returns the smoothed, unweighted Gini impurity.
float GiniImpurity(const LeafStat& stats, int32 num_classes);

// Returns the smoothed, weighted Gini impurity
float WeightedGiniImpurity(const LeafStat& stats, int32 num_classes);

// Updates the GiniStats given the old and new values of a class count that
// was updated.
void UpdateGini(LeafStat* stats, float old_val, float weight);

// Returns the variance in stats for the given output.
float Variance(const LeafStat& stats, int output);

// Returns the variance sum for all outputs.
float TotalVariance(const LeafStat& stats);

// ------- functions used by C++ stats classes  -------- //
// Returns the smoothed gini score given the sum and sum of the squares of the
// class counts.
float SmoothedGini(float sum, float square, int num_classes);

// Returns the smoothed gini score weighted by the sum.
float WeightedSmoothedGini(float sum, float square, int num_classes);

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_STAT_UTILS_H_
