/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LIB_RANDOM_PHILOX_RANDOM_TEST_UTILS_H_
#define TENSORFLOW_LIB_RANDOM_PHILOX_RANDOM_TEST_UTILS_H_

#include <algorithm>

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace random {

// Return a random seed.
inline uint64 GetTestSeed() { return New64(); }

// A utility function to fill the given array with samples from the given
// distribution.
template <class Distribution>
void FillRandoms(PhiloxRandom gen, typename Distribution::ResultElementType* p,
                 int64 size) {
  const int granularity = Distribution::kResultElementCount;

  CHECK(size % granularity == 0) << " size: " << size
                                 << " granularity: " << granularity;

  Distribution dist;
  for (int i = 0; i < size; i += granularity) {
    const auto sample = dist(&gen);
    std::copy(&sample[0], &sample[0] + granularity, &p[i]);
  }
}

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_RANDOM_PHILOX_RANDOM_TEST_UTILS_H_
