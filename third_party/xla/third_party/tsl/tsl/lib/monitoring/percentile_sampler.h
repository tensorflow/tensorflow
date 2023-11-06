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

#ifndef TENSORFLOW_TSL_LIB_MONITORING_PERCENTILE_SAMPLER_H_
#define TENSORFLOW_TSL_LIB_MONITORING_PERCENTILE_SAMPLER_H_

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tsl/platform/platform.h"
// clang-format on

// We replace this implementation with a null implementation for mobile
// platforms.
#ifdef IS_MOBILE_PLATFORM

#include "tsl/platform/status.h"
#include "tsl/lib/monitoring/collection_registry.h"
#include "tsl/lib/monitoring/metric_def.h"
#include "tsl/lib/monitoring/types.h"
#include "tsl/platform/macros.h"

namespace tsl {
namespace monitoring {

class PercentileSamplerCell {
 public:
  void Add(double sample) {}

  Percentiles value() const { return Percentiles(); }
};

template <int NumLabels>
class PercentileSampler {
 public:
  static PercentileSampler* New(
      const MetricDef<MetricKind::kCumulative, Percentiles, NumLabels>&
          metric_def,
      std::vector<double> percentiles, size_t max_samples,
      UnitOfMeasure unit_of_measure);

  template <typename... Labels>
  PercentileSamplerCell* GetCell(const Labels&... labels) {
    return &default_cell_;
  }

  Status GetStatus() { return tsl::OkStatus(); }

 private:
  PercentileSamplerCell default_cell_;

  PercentileSampler() = default;

  PercentileSampler(const PercentileSampler&) = delete;
  void operator=(const PercentileSampler&) = delete;
};

template <int NumLabels>
PercentileSampler<NumLabels>* PercentileSampler<NumLabels>::New(
    const MetricDef<MetricKind::kCumulative, Percentiles, NumLabels>&
    /* metric_def */,
    std::vector<double> /* percentiles */, size_t /* max_samples */,
    UnitOfMeasure /* unit_of_measure */) {
  return new PercentileSampler<NumLabels>();
}

}  // namespace monitoring
}  // namespace tsl

#else  // IS_MOBILE_PLATFORM

#include <cmath>
#include <map>

#include "tsl/platform/status.h"
#include "tsl/lib/monitoring/collection_registry.h"
#include "tsl/lib/monitoring/metric_def.h"
#include "tsl/lib/monitoring/types.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {
namespace monitoring {

// PercentileSamplerCell stores each value of an PercentileSampler.
// The class uses a circular buffer to maintain a window of samples.
//
// This class is thread-safe.
class PercentileSamplerCell {
 public:
  PercentileSamplerCell(UnitOfMeasure unit_of_measure,
                        std::vector<double> percentiles, size_t max_samples)
      : unit_of_measure_(unit_of_measure),
        percentiles_(std::move(percentiles)),
        samples_(max_samples),
        num_samples_(0),
        next_position_(0),
        total_samples_(0),
        accumulator_(0.0) {}

  // Atomically adds a sample.
  void Add(double sample);

  Percentiles value() const;

 private:
  struct Sample {
    bool operator<(const Sample& rhs) const { return value < rhs.value; }

    uint64 nstime = 0;
    double value = NAN;
  };

  std::vector<Sample> GetSamples(size_t* total_samples,
                                 long double* accumulator) const;

  mutable mutex mu_;
  UnitOfMeasure unit_of_measure_;
  const std::vector<double> percentiles_;
  std::vector<Sample> samples_ TF_GUARDED_BY(mu_);
  size_t num_samples_ TF_GUARDED_BY(mu_);
  size_t next_position_ TF_GUARDED_BY(mu_);
  size_t total_samples_ TF_GUARDED_BY(mu_);
  long double accumulator_ TF_GUARDED_BY(mu_);

  PercentileSamplerCell(const PercentileSamplerCell&) = delete;
  void operator=(const PercentileSamplerCell&) = delete;
};

// A stateful class for updating a cumulative percentile sampled metric.
//
// This class stores, in each cell, up to max_samples values in a circular
// buffer, and returns the percentiles information as cell value.
//
// PercentileSampler allocates storage and maintains a cell for each value. You
// can retrieve an individual cell using a label-tuple and update it separately.
// This improves performance since operations related to retrieval, like
// map-indexing and locking, are avoided.
//
// This class is thread-safe.
template <int NumLabels>
class PercentileSampler {
 public:
  ~PercentileSampler() {
    // Deleted here, before the metric_def is destroyed.
    registration_handle_.reset();
  }

  // Creates the metric based on the metric-definition arguments and buckets.
  //
  // Example;
  // auto* sampler_with_label =
  // PercentileSampler<1>::New({"/tensorflow/sampler",
  //   "Tensorflow sampler", "MyLabelName"}, {10.0, 20.0, 30.0}, 1024,
  //   UnitOfMeasure::kTime);
  static PercentileSampler* New(
      const MetricDef<MetricKind::kCumulative, Percentiles, NumLabels>&
          metric_def,
      std::vector<double> percentiles, size_t max_samples,
      UnitOfMeasure unit_of_measure);

  // Retrieves the cell for the specified labels, creating it on demand if
  // not already present.
  template <typename... Labels>
  PercentileSamplerCell* GetCell(const Labels&... labels)
      TF_LOCKS_EXCLUDED(mu_);

  Status GetStatus() { return status_; }

 private:
  friend class PercentileSamplerCell;

  PercentileSampler(const MetricDef<MetricKind::kCumulative, Percentiles,
                                    NumLabels>& metric_def,
                    std::vector<double> percentiles, size_t max_samples,
                    UnitOfMeasure unit_of_measure)
      : metric_def_(metric_def),
        unit_of_measure_(unit_of_measure),
        percentiles_(std::move(percentiles)),
        max_samples_(max_samples),
        registration_handle_(CollectionRegistry::Default()->Register(
            &metric_def_, [&](MetricCollectorGetter getter) {
              auto metric_collector = getter.Get(&metric_def_);
              mutex_lock l(mu_);
              for (const auto& cell : cells_) {
                metric_collector.CollectValue(cell.first, cell.second.value());
              }
            })) {
    if (registration_handle_) {
      for (size_t i = 0; i < percentiles_.size(); ++i) {
        if (percentiles_[i] < 0.0 || percentiles_[i] > 100.0) {
          status_ = Status(absl::StatusCode::kInvalidArgument,
                           "Percentile values must be in [0, 100] range.");
          break;
        }
        if (i + 1 < percentiles_.size() &&
            percentiles_[i] >= percentiles_[i + 1]) {
          status_ =
              Status(absl::StatusCode::kInvalidArgument,
                     "Percentile values must be in strictly ascending order.");
          break;
        }
      }
    } else {
      status_ = Status(absl::StatusCode::kAlreadyExists,
                       "Another metric with the same name already exists.");
    }
  }

  mutable mutex mu_;

  Status status_;

  using LabelArray = std::array<string, NumLabels>;
  // we need a container here that guarantees pointer stability of the value,
  // namely, the pointer of the value should remain valid even after more cells
  // are inserted.
  std::map<LabelArray, PercentileSamplerCell> cells_ TF_GUARDED_BY(mu_);

  // The metric definition. This will be used to identify the metric when we
  // register it for collection.
  const MetricDef<MetricKind::kCumulative, Percentiles, NumLabels> metric_def_;

  UnitOfMeasure unit_of_measure_ = UnitOfMeasure::kNumber;

  // The percentiles samples required for this metric.
  const std::vector<double> percentiles_;

  // The maximum size of the samples colected by the PercentileSamplerCell cell.
  const size_t max_samples_ = 0;

  // Registration handle with the CollectionRegistry.
  std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

  PercentileSampler(const PercentileSampler&) = delete;
  void operator=(const PercentileSampler&) = delete;
};

template <int NumLabels>
PercentileSampler<NumLabels>* PercentileSampler<NumLabels>::New(
    const MetricDef<MetricKind::kCumulative, Percentiles, NumLabels>&
        metric_def,
    std::vector<double> percentiles, size_t max_samples,
    UnitOfMeasure unit_of_measure) {
  return new PercentileSampler<NumLabels>(metric_def, std::move(percentiles),
                                          max_samples, unit_of_measure);
}

template <int NumLabels>
template <typename... Labels>
PercentileSamplerCell* PercentileSampler<NumLabels>::GetCell(
    const Labels&... labels) TF_LOCKS_EXCLUDED(mu_) {
  // Provides a more informative error message than the one during array
  // construction below.
  static_assert(
      sizeof...(Labels) == NumLabels,
      "Mismatch between PercentileSampler<NumLabels> and number of labels "
      "provided in GetCell(...).");

  const LabelArray& label_array = {{labels...}};
  mutex_lock l(mu_);
  const auto found_it = cells_.find(label_array);
  if (found_it != cells_.end()) {
    return &(found_it->second);
  }
  return &(cells_
               .emplace(std::piecewise_construct,
                        std::forward_as_tuple(label_array),
                        std::forward_as_tuple(unit_of_measure_, percentiles_,
                                              max_samples_))
               .first->second);
}

}  // namespace monitoring
}  // namespace tsl

#endif  // IS_MOBILE_PLATFORM
#endif  // TENSORFLOW_TSL_LIB_MONITORING_PERCENTILE_SAMPLER_H_
