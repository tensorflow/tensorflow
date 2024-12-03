/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/lib/monitoring/percentile_sampler.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "xla/tsl/lib/monitoring/types.h"
#include "tsl/platform/env_time.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/types.h"

// We replace this implementation with a null implementation for mobile
// platforms.
#ifdef IS_MOBILE_PLATFORM
// Do nothing.
#else

namespace tsl {
namespace monitoring {

void PercentileSamplerCell::Add(double sample) {
  uint64 nstime = EnvTime::NowNanos();
  mutex_lock l(mu_);
  samples_[next_position_] = {nstime, sample};
  ++next_position_;
  if (TF_PREDICT_FALSE(next_position_ >= samples_.size())) {
    next_position_ = 0;
  }
  if (TF_PREDICT_FALSE(num_samples_ < samples_.size())) {
    ++num_samples_;
  }
  ++total_samples_;
  accumulator_ += sample;
}

Percentiles PercentileSamplerCell::value() const {
  Percentiles pct_samples;
  pct_samples.unit_of_measure = unit_of_measure_;
  size_t total_samples;
  long double accumulator;
  std::vector<Sample> samples = GetSamples(&total_samples, &accumulator);
  if (!samples.empty()) {
    pct_samples.num_samples = samples.size();
    pct_samples.total_samples = total_samples;
    pct_samples.accumulator = accumulator;
    pct_samples.start_nstime = samples.front().nstime;
    pct_samples.end_nstime = samples.back().nstime;

    long double total = 0.0;
    for (auto& sample : samples) {
      total += sample.value;
    }
    pct_samples.mean = total / pct_samples.num_samples;
    long double total_sigma = 0.0;
    for (auto& sample : samples) {
      double delta = sample.value - pct_samples.mean;
      total_sigma += delta * delta;
    }
    pct_samples.stddev = std::sqrt(total_sigma / pct_samples.num_samples);

    std::sort(samples.begin(), samples.end());
    pct_samples.min_value = samples.front().value;
    pct_samples.max_value = samples.back().value;
    for (auto percentile : percentiles_) {
      size_t index = std::min<size_t>(
          static_cast<size_t>(percentile * pct_samples.num_samples / 100.0),
          pct_samples.num_samples - 1);
      PercentilePoint pct = {percentile, samples[index].value};
      pct_samples.points.push_back(pct);
    }
  }
  return pct_samples;
}

std::vector<PercentileSamplerCell::Sample> PercentileSamplerCell::GetSamples(
    size_t* total_samples, long double* accumulator) const {
  mutex_lock l(mu_);
  std::vector<Sample> samples;
  if (num_samples_ == samples_.size()) {
    samples.insert(samples.end(), samples_.begin() + next_position_,
                   samples_.end());
  }
  samples.insert(samples.end(), samples_.begin(),
                 samples_.begin() + next_position_);
  *total_samples = total_samples_;
  *accumulator = accumulator_;
  return samples;
}

}  // namespace monitoring
}  // namespace tsl

#endif  // IS_MOBILE_PLATFORM
