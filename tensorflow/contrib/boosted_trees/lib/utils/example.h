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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLE_H_

#include <unordered_set>
#include <vector>
#include "tensorflow/contrib/boosted_trees/lib/utils/optional_value.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

// Holds data for one example and enables lookup by feature column.
struct Example {
  // Default constructor creates an empty example.
  Example() : example_idx(-1) {}

  // Example index.
  int64 example_idx;

  // Dense and sparse float features indexed by feature column.
  // TODO(salehay): figure out a design to support multivalent float features.
  std::vector<float> dense_float_features;
  std::vector<OptionalValue<float>> sparse_float_features;

  // Sparse integer features indexed by feature column.
  // Note that all integer features are assumed to be categorical, i.e. will
  // never be compared by order. Also these features can be multivalent.
  std::vector<std::unordered_set<int64>> sparse_int_features;
};

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLE_H_
