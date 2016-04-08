/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_LIB_RANDOM_DISTRIBUTION_SAMPLER_H_
#define TENSORFLOW_LIB_RANDOM_DISTRIBUTION_SAMPLER_H_

#include <memory>
#include <utility>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {

class DistributionSampler {
 public:
  explicit DistributionSampler(const gtl::ArraySlice<float>& weights);

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

  TF_DISALLOW_COPY_AND_ASSIGN(DistributionSampler);
};

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_RANDOM_DISTRIBUTION_SAMPLER_H_
