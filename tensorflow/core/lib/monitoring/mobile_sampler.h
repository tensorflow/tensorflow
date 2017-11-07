/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Null implementation of the Sampler metric for mobile platforms.

#ifndef THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_MOBILE_SAMPLER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_MOBILE_SAMPLER_H_

#include <memory>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace monitoring {

// SamplerCell which has a null implementation.
class SamplerCell {
 public:
  SamplerCell() {}
  ~SamplerCell() {}

  void Add(double value) {}
  HistogramProto value() const { return HistogramProto(); }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SamplerCell);
};

// Buckets which has a null implementation.
class Buckets {
 public:
  Buckets() = default;
  ~Buckets() = default;

  static std::unique_ptr<Buckets> Explicit(
      std::initializer_list<double> bucket_limits) {
    return std::unique_ptr<Buckets>(new Buckets());
  }

  static std::unique_ptr<Buckets> Exponential(double scale,
                                              double growth_factor,
                                              int bucket_count) {
    return std::unique_ptr<Buckets>(new Buckets());
  }

  const std::vector<double>& explicit_bounds() const {
    return explicit_bounds_;
  }

 private:
  std::vector<double> explicit_bounds_;

  TF_DISALLOW_COPY_AND_ASSIGN(Buckets);
};

// Sampler which has a null implementation.
template <int NumLabels>
class Sampler {
 public:
  ~Sampler() {}

  template <typename... MetricDefArgs>
  static Sampler* New(const MetricDef<MetricKind::kCumulative, HistogramProto,
                                      NumLabels>& metric_def,
                      std::unique_ptr<Buckets> buckets) {
    return new Sampler<NumLabels>(std::move(buckets));
  }

  template <typename... Labels>
  SamplerCell* GetCell(const Labels&... labels) {
    return &default_sampler_cell_;
  }

 private:
  Sampler(std::unique_ptr<Buckets> buckets) : buckets_(std::move(buckets)) {}

  SamplerCell default_sampler_cell_;
  std::unique_ptr<Buckets> buckets_;

  TF_DISALLOW_COPY_AND_ASSIGN(Sampler);
};

}  // namespace monitoring
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_MOBILE_SAMPLER_H_
