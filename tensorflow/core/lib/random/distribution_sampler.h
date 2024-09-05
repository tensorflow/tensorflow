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

// DistributionSampler allows generating a discrete random variable with a given
// distribution.
// The values taken by the variable are [0, N) and relative weights for each
// value are specified using a vector of size N.
//
// The Algorithm takes O(N) time to precompute data at construction time and
// takes O(1) time (2 random number generation, 2 lookups) for each sample.
// The data structure takes O(N) memory.
//
// In contrast, util/random/weighted-picker.h provides O(lg N) sampling.
// The advantage of that implementation is that weights can be adjusted
// dynamically, while DistributionSampler doesn't allow weight adjustment.
//
// The algorithm used is Walker's Aliasing algorithm, described in Knuth, Vol 2.

#ifndef TENSORFLOW_CORE_LIB_RANDOM_DISTRIBUTION_SAMPLER_H_
#define TENSORFLOW_CORE_LIB_RANDOM_DISTRIBUTION_SAMPLER_H_

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/lib/random/distribution_sampler.h"

namespace tensorflow {
namespace random {
using tsl::random::DistributionSampler;  // NOLINT(misc-unused-using-decls)
}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_RANDOM_DISTRIBUTION_SAMPLER_H_
