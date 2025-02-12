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

#ifndef XLA_TSL_LIB_MONITORING_SAMPLER_H_
#define XLA_TSL_LIB_MONITORING_SAMPLER_H_

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "absl/status/status.h"
#include "tsl/platform/platform.h"
// clang-format on

// We replace this implementation with a null implementation for mobile
// platforms.
#ifdef IS_MOBILE_PLATFORM

#include <memory>

#include "xla/tsl/lib/monitoring/metric_def.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/protobuf/histogram.pb.h"

namespace tsl {
namespace monitoring {
using tensorflow::HistogramProto;
// SamplerCell which has a null implementation.
class SamplerCell {
 public:
  SamplerCell() {}
  ~SamplerCell() {}

  void Add(double value) {}
  HistogramProto value() const { return HistogramProto(); }

 private:
  SamplerCell(const SamplerCell&) = delete;
  void operator=(const SamplerCell&) = delete;
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

  Buckets(const Buckets&) = delete;
  void operator=(const Buckets&) = delete;
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

  Status GetStatus() { return OkStatus(); }

 private:
  Sampler(std::unique_ptr<Buckets> buckets) : buckets_(std::move(buckets)) {}

  SamplerCell default_sampler_cell_;
  std::unique_ptr<Buckets> buckets_;

  Sampler(const Sampler&) = delete;
  void operator=(const Sampler&) = delete;
};

}  // namespace monitoring
}  // namespace tsl

#else  // IS_MOBILE_PLATFORM

#include <float.h>

#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "xla/tsl/lib/histogram/histogram.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "xla/tsl/lib/monitoring/metric_def.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/protobuf/histogram.pb.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {
namespace monitoring {
using tensorflow::HistogramProto;

// SamplerCell stores each value of an Sampler.
//
// A cell can be passed off to a module which may repeatedly update it without
// needing further map-indexing computations. This improves both encapsulation
// (separate modules can own a cell each, without needing to know about the map
// to which both cells belong) and performance (since map indexing and
// associated locking are both avoided).
//
// This class is thread-safe.
class SamplerCell {
 public:
  explicit SamplerCell(const std::vector<double>& bucket_limits)
      : histogram_(bucket_limits) {}

  ~SamplerCell() {}

  // Atomically adds a sample.
  void Add(double sample);

  // Returns the current histogram value as a proto.
  HistogramProto value() const;

 private:
  histogram::ThreadSafeHistogram histogram_;

  SamplerCell(const SamplerCell&) = delete;
  void operator=(const SamplerCell&) = delete;
};

// Bucketing strategies for the samplers.
//
// We automatically add -DBL_MAX and DBL_MAX to the ranges, so that no sample
// goes out of bounds.
//
// WARNING: If you are changing the interface here, please do change the same in
// mobile_sampler.h.
class Buckets {
 public:
  virtual ~Buckets() = default;

  // Sets up buckets of the form:
  // [-DBL_MAX, ..., scale * growth^i,
  //   scale * growth_factor^(i + 1), ..., DBL_MAX].
  //
  // So for powers of 2 with a bucket count of 10, you would say (1, 2, 10)
  static std::unique_ptr<Buckets> Exponential(double scale,
                                              double growth_factor,
                                              int bucket_count);

  // Sets up buckets of the form:
  // [-DBL_MAX, ..., bucket_limits[i], bucket_limits[i + 1], ..., DBL_MAX].
  static std::unique_ptr<Buckets> Explicit(
      std::initializer_list<double> bucket_limits);

  // This alternative Explicit Buckets factory method is primarily meant to be
  // used by the CLIF layer code paths that are incompatible with
  // initialize_lists.
  static std::unique_ptr<Buckets> Explicit(std::vector<double> bucket_limits);

  virtual const std::vector<double>& explicit_bounds() const = 0;
};

// A stateful class for updating a cumulative histogram metric.
//
// This class encapsulates a set of histograms (or a single histogram for a
// label-less metric) configured with a list of increasing bucket boundaries.
// Each histogram is identified by a tuple of labels. The class allows the
// user to add a sample to each histogram value.
//
// Sampler allocates storage and maintains a cell for each value. You can
// retrieve an individual cell using a label-tuple and update it separately.
// This improves performance since operations related to retrieval, like
// map-indexing and locking, are avoided.
//
// This class is thread-safe.
template <int NumLabels>
class Sampler {
 public:
  ~Sampler() {
    // Deleted here, before the metric_def is destroyed.
    registration_handle_.reset();
  }

  // Creates the metric based on the metric-definition arguments and buckets.
  //
  // Example;
  // auto* sampler_with_label = Sampler<1>::New({"/tensorflow/sampler",
  //   "Tensorflow sampler", "MyLabelName"}, {10.0, 20.0, 30.0});
  static Sampler* New(const MetricDef<MetricKind::kCumulative, HistogramProto,
                                      NumLabels>& metric_def,
                      std::unique_ptr<Buckets> buckets);

  // Retrieves the cell for the specified labels, creating it on demand if
  // not already present.
  template <typename... Labels>
  SamplerCell* GetCell(const Labels&... labels) TF_LOCKS_EXCLUDED(mu_);

  absl::Status GetStatus() { return status_; }

 private:
  friend class SamplerCell;

  Sampler(const MetricDef<MetricKind::kCumulative, HistogramProto, NumLabels>&
              metric_def,
          std::unique_ptr<Buckets> buckets)
      : metric_def_(metric_def),
        buckets_(std::move(buckets)),
        registration_handle_(CollectionRegistry::Default()->Register(
            &metric_def_, [&](MetricCollectorGetter getter) {
              auto metric_collector = getter.Get(&metric_def_);

              tf_shared_lock l(mu_);
              for (const auto& cell : cells_) {
                metric_collector.CollectValue(cell.first, cell.second.value());
              }
            })) {
    if (registration_handle_) {
      status_ = absl::OkStatus();
    } else {
      status_ =
          absl::Status(absl::StatusCode::kAlreadyExists,
                       "Another metric with the same name already exists.");
    }
  }

  mutable mutex mu_;

  absl::Status status_;

  using LabelArray = std::array<string, NumLabels>;
  // we need a container here that guarantees pointer stability of the value,
  // namely, the pointer of the value should remain valid even after more cells
  // are inserted.
  std::map<LabelArray, SamplerCell> cells_ TF_GUARDED_BY(mu_);

  // The metric definition. This will be used to identify the metric when we
  // register it for collection.
  const MetricDef<MetricKind::kCumulative, HistogramProto, NumLabels>
      metric_def_;

  // Bucket limits for the histograms in the cells.
  std::unique_ptr<Buckets> buckets_;

  // Registration handle with the CollectionRegistry.
  std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

  Sampler(const Sampler&) = delete;
  void operator=(const Sampler&) = delete;
};

////
//  Implementation details follow. API readers may skip.
////

inline void SamplerCell::Add(const double sample) { histogram_.Add(sample); }

inline HistogramProto SamplerCell::value() const {
  HistogramProto pb;
  histogram_.EncodeToProto(&pb, true /* preserve_zero_buckets */);
  return pb;
}

template <int NumLabels>
Sampler<NumLabels>* Sampler<NumLabels>::New(
    const MetricDef<MetricKind::kCumulative, HistogramProto, NumLabels>&
        metric_def,
    std::unique_ptr<Buckets> buckets) {
  return new Sampler<NumLabels>(metric_def, std::move(buckets));
}

template <int NumLabels>
template <typename... Labels>
SamplerCell* Sampler<NumLabels>::GetCell(const Labels&... labels)
    TF_LOCKS_EXCLUDED(mu_) {
  // Provides a more informative error message than the one during array
  // construction below.
  static_assert(sizeof...(Labels) == NumLabels,
                "Mismatch between Sampler<NumLabels> and number of labels "
                "provided in GetCell(...).");

  const LabelArray& label_array = {{labels...}};
  {
    tf_shared_lock l(mu_);
    const auto found_it = cells_.find(label_array);
    if (found_it != cells_.end()) {
      return &(found_it->second);
    }
  }
  mutex_lock l(mu_);
  return &(cells_
               .emplace(std::piecewise_construct,
                        std::forward_as_tuple(label_array),
                        std::forward_as_tuple(buckets_->explicit_bounds()))
               .first->second);
}

}  // namespace monitoring
}  // namespace tsl

#endif  // IS_MOBILE_PLATFORM
#endif  // XLA_TSL_LIB_MONITORING_SAMPLER_H_
