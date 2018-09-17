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
#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_RANDOM_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_RANDOM_H_

#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

// Generates a poisson distributed number with mean 1 for use in bootstrapping.
inline int32 PoissonBootstrap(random::SimplePhilox* rng) {
  // Knuth, special cased for lambda = 1.0 for efficiency.
  static const float lbound = exp(-1.0f);
  int32 n = 0;
  for (float r = 1; r > lbound; r *= rng->RandFloat()) {
    ++n;
  }
  return n - 1;
}

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_RANDOM_H_
