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

#include "tsl/lib/random/distribution_sampler.h"

#include <memory>
#include <vector>

#include "absl/types/span.h"

namespace tsl {
namespace random {

DistributionSampler::DistributionSampler(
    const absl::Span<const float> weights) {
  DCHECK(!weights.empty());
  int n = weights.size();
  num_ = n;
  data_.reset(new std::pair<float, int>[n]);

  std::unique_ptr<double[]> pr(new double[n]);

  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += weights[i];
    set_alt(i, -1);
  }

  // These are long/short items - called high/low because of reserved keywords.
  std::vector<int> high;
  high.reserve(n);
  std::vector<int> low;
  low.reserve(n);

  // compute proportional weights
  for (int i = 0; i < n; i++) {
    double p = (weights[i] * n) / sum;
    pr[i] = p;
    if (p < 1.0) {
      low.push_back(i);
    } else {
      high.push_back(i);
    }
  }

  // Now pair high with low.
  while (!high.empty() && !low.empty()) {
    int l = low.back();
    low.pop_back();
    int h = high.back();
    high.pop_back();

    set_alt(l, h);
    DCHECK_GE(pr[h], 1.0);
    double remaining = pr[h] - (1.0 - pr[l]);
    pr[h] = remaining;

    if (remaining < 1.0) {
      low.push_back(h);
    } else {
      high.push_back(h);
    }
  }
  // Transfer pr to prob with rounding errors.
  for (int i = 0; i < n; i++) {
    set_prob(i, pr[i]);
  }
  // Because of rounding errors, both high and low may have elements, that are
  // close to 1.0 prob.
  for (size_t i = 0; i < high.size(); i++) {
    int idx = high[i];
    set_prob(idx, 1.0);
    // set alt to self to prevent rounding errors returning 0
    set_alt(idx, idx);
  }
  for (size_t i = 0; i < low.size(); i++) {
    int idx = low[i];
    set_prob(idx, 1.0);
    // set alt to self to prevent rounding errors returning 0
    set_alt(idx, idx);
  }
}

}  // namespace random
}  // namespace tsl
