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

#ifndef XLA_TSL_LIB_RANDOM_DISTRIBUTION_SAMPLER_H_
#define XLA_TSL_LIB_RANDOM_DISTRIBUTION_SAMPLER_H_

#include <memory>
#include <utility>

#include "absl/types/span.h"
#include "xla/tsl/lib/random/simple_philox.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"

namespace tsl {
namespace random {

class DistributionSampler {
 public:
  explicit DistributionSampler(const absl::Span<const float> weights);

  ~DistributionSampler() {}

  int Sample(SimplePhilox* rand) const {
    float r = rand->RandFloat();
    // Since n is typically low, we don't bother with UnbiasedUniform.
    int idx = rand->Uniform(num_);
    if (r < prob(idx)) return idx;
    // else pick alt from that bucket.
    DCHECK_NE(-1, alt(idx));
    return alt(idx);
  }

  int num() const { return num_; }

 private:
  float prob(int idx) const {
    DCHECK_LT(idx, num_);
    return data_[idx].first;
  }

  int alt(int idx) const {
    DCHECK_LT(idx, num_);
    return data_[idx].second;
  }

  void set_prob(int idx, float f) {
    DCHECK_LT(idx, num_);
    data_[idx].first = f;
  }

  void set_alt(int idx, int val) {
    DCHECK_LT(idx, num_);
    data_[idx].second = val;
  }

  int num_;
  std::unique_ptr<std::pair<float, int>[]> data_;

  DistributionSampler(const DistributionSampler&) = delete;
  void operator=(const DistributionSampler&) = delete;
};

}  // namespace random
}  // namespace tsl

#endif  // XLA_TSL_LIB_RANDOM_DISTRIBUTION_SAMPLER_H_
