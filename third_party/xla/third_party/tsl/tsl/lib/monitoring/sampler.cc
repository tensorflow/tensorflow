/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/lib/monitoring/sampler.h"

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tsl/platform/platform.h"
// clang-format on

// We replace this implementation with a null implementation for mobile
// platforms.
#ifdef IS_MOBILE_PLATFORM
// Do nothing.
#else

namespace tsl {
namespace monitoring {
namespace {

class ExplicitBuckets : public Buckets {
 public:
  ~ExplicitBuckets() override = default;

  explicit ExplicitBuckets(std::vector<double> bucket_limits)
      : bucket_limits_(std::move(bucket_limits)) {
    CHECK_GT(bucket_limits_.size(), 0);
    // Verify that the bucket boundaries are strictly increasing
    for (size_t i = 1; i < bucket_limits_.size(); i++) {
      CHECK_GT(bucket_limits_[i], bucket_limits_[i - 1]);
    }
    // We augment the bucket limits so that all boundaries are within [-DBL_MAX,
    // DBL_MAX].
    //
    // Since we use ThreadSafeHistogram, we don't have to explicitly add
    // -DBL_MAX, because it uses these limits as upper-bounds, so
    // bucket_count[0] is always the number of elements in
    // [-DBL_MAX, bucket_limits[0]).
    if (bucket_limits_.back() != DBL_MAX) {
      bucket_limits_.push_back(DBL_MAX);
    }
  }

  const std::vector<double>& explicit_bounds() const override {
    return bucket_limits_;
  }

 private:
  std::vector<double> bucket_limits_;

  ExplicitBuckets(const ExplicitBuckets&) = delete;
  void operator=(const ExplicitBuckets&) = delete;
};

class ExponentialBuckets : public Buckets {
 public:
  ~ExponentialBuckets() override = default;

  ExponentialBuckets(double scale, double growth_factor, int bucket_count)
      : explicit_buckets_(
            ComputeBucketLimits(scale, growth_factor, bucket_count)) {}

  const std::vector<double>& explicit_bounds() const override {
    return explicit_buckets_.explicit_bounds();
  }

 private:
  static std::vector<double> ComputeBucketLimits(double scale,
                                                 double growth_factor,
                                                 int bucket_count) {
    CHECK_GT(bucket_count, 0);
    std::vector<double> bucket_limits;
    double bound = scale;
    for (int i = 0; i < bucket_count; i++) {
      bucket_limits.push_back(bound);
      bound *= growth_factor;
    }
    return bucket_limits;
  }

  ExplicitBuckets explicit_buckets_;

  ExponentialBuckets(const ExponentialBuckets&) = delete;
  void operator=(const ExponentialBuckets&) = delete;
};

}  // namespace

// static
std::unique_ptr<Buckets> Buckets::Explicit(std::vector<double> bucket_limits) {
  return std::unique_ptr<Buckets>(
      new ExplicitBuckets(std::move(bucket_limits)));
}

// static
std::unique_ptr<Buckets> Buckets::Explicit(
    std::initializer_list<double> bucket_limits) {
  return std::unique_ptr<Buckets>(new ExplicitBuckets(bucket_limits));
}

// static
std::unique_ptr<Buckets> Buckets::Exponential(double scale,
                                              double growth_factor,
                                              int bucket_count) {
  return std::unique_ptr<Buckets>(
      new ExponentialBuckets(scale, growth_factor, bucket_count));
}

}  // namespace monitoring
}  // namespace tsl

#endif  // IS_MOBILE_PLATFORM
