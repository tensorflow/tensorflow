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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_SAMPLER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_SAMPLER_H_

// We replace this implementation with a null implementation for mobile
// platforms.
#include "tensorflow/core/platform/platform.h"
#ifdef IS_MOBILE_PLATFORM
#include "tensorflow/core/lib/monitoring/mobile_sampler.h"
#else

#include <float.h>
#include <map>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace monitoring {

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
  SamplerCell(const std::vector<double>& bucket_limits)
      : histogram_(bucket_limits) {}

  ~SamplerCell() {}

  // Atomically adds a sample.
  void Add(double sample);

  // Returns the current histogram value as a proto.
  HistogramProto value() const;

 private:
  histogram::ThreadSafeHistogram histogram_;

  TF_DISALLOW_COPY_AND_ASSIGN(SamplerCell);
};

// A stateful class for updating a cumulative histogram metric.
//
// This class encapsulates a set of histograms (or a single histogram for a
// label-less metric) configured with a list of increasing bucket boundaries.
// Each histogram is identified by a tuple of labels. The class allows the user
// to add a sample to each histogram value.
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

  // Creates the metric based on the metric-definition arguments.
  //
  // Example;
  // auto* sampler_with_label = Sampler<1>::New({"/tensorflow/sampler",
  //   "Tensorflow sampler", "MyLabelName"}, {10.0, 20.0, 30.0});
  //
  // We automatically add -DBL_MAX and DBL_MAX to the list of bucket limits, so
  // that no sample goes out of bounds. So for the above example, the ranges end
  // up being: [-DBL_Max, 10.0, 20.0, 30.0, DBL_MAX]
  //
  // REQUIRES: bucket_limits[i] values are monotonically increasing.
  // REQUIRES: bucket_limits is not empty().
  static Sampler* New(const MetricDef<MetricKind::kCumulative, HistogramProto,
                                      NumLabels>& metric_def,
                      const std::vector<double>& bucket_limits);

  // Retrieves the cell for the specified labels, creating it on demand if
  // not already present.
  template <typename... Labels>
  SamplerCell* GetCell(const Labels&... labels) LOCKS_EXCLUDED(mu_);

 private:
  friend class SamplerCell;

  Sampler(const MetricDef<MetricKind::kCumulative, HistogramProto, NumLabels>&
              metric_def,
          const std::vector<double>& bucket_limits)
      : metric_def_(metric_def),
        bucket_limits_(bucket_limits),
        registration_handle_(CollectionRegistry::Default()->Register(
            &metric_def_, [&](MetricCollectorGetter getter) {
              auto metric_collector = getter.Get(&metric_def_);

              mutex_lock l(mu_);
              for (const auto& cell : cells_) {
                metric_collector.CollectValue(cell.first, cell.second.value());
              }
            })) {}

  mutable mutex mu_;

  // The metric definition. This will be used to identify the metric when we
  // register it for collection.
  const MetricDef<MetricKind::kCumulative, HistogramProto, NumLabels>
      metric_def_;

  // Bucket limits for the histograms in the cells.
  const std::vector<double> bucket_limits_;

  // Registration handle with the CollectionRegistry.
  std::unique_ptr<CollectionRegistry::RegistrationHandle> registration_handle_;

  // We use a std::map here because we give out pointers to the SamplerCells,
  // which need to remain valid even after more cells.
  using LabelArray = std::array<string, NumLabels>;
  std::map<LabelArray, SamplerCell> cells_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(Sampler);
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
    const std::vector<double>& bucket_limits) {
  CHECK_GT(bucket_limits.size(), 0);
  // Verify that the bucket boundaries are strictly increasing
  for (size_t i = 1; i < bucket_limits.size(); i++) {
    CHECK_GT(bucket_limits[i], bucket_limits[i - 1]);
  }
  std::vector<double> augmented_bucket_limits(bucket_limits);
  // We add DBL_MAX to the end so that all boundaries are within [-DBL_MAX,
  // DBL_MAX].
  if (bucket_limits.back() != DBL_MAX) {
    augmented_bucket_limits.push_back(DBL_MAX);
  }
  return new Sampler<NumLabels>(metric_def, augmented_bucket_limits);
}

template <int NumLabels>
template <typename... Labels>
SamplerCell* Sampler<NumLabels>::GetCell(const Labels&... labels)
    LOCKS_EXCLUDED(mu_) {
  // Provides a more informative error message than the one during array
  // construction below.
  static_assert(sizeof...(Labels) == NumLabels,
                "Mismatch between Sampler<NumLabels> and number of labels "
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
                        std::forward_as_tuple(bucket_limits_))
               .first->second);
}

}  // namespace monitoring
}  // namespace tensorflow

#endif  // IS_MOBILE_PLATFORM
#endif  // THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_SAMPLER_H_
