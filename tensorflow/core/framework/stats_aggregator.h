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
#ifndef TENSORFLOW_CORE_FRAMEWORK_STATS_AGGREGATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_STATS_AGGREGATOR_H_

#include <memory>
#include <string>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

class Summary;
class SummaryWriterInterface;
namespace data {

// A `StatsAggregator` accumulates statistics incrementally. A
// `StatsAggregator` can accumulate multiple different statistics, distinguished
// by a string name.
//
// The class currently supports accumulating `Histogram`, `scalar` objects and
// tfstreamz metrics, and we expect to add other methods in future.
//
// NOTE(mrry): `StatsAggregator` is a virtual interface because we anticipate
// that many different implementations will have the same interface. For
// example, we have different implementations in "stats_aggregator_ops.cc" for
// simple in-memory implementation that integrates with the pull-based summary
// API, and for the push-based `SummaryWriterInterface`, and we may add
// implementations that work well with other custom monitoring services.
class StatsAggregator {
 public:
  virtual ~StatsAggregator() {}

  // Add the given `values` to the histogram with the given `name`. Each
  // element of `values` will be treated as a separate sample in the histogram.
  virtual void AddToHistogram(const string& name,
                              gtl::ArraySlice<double> values,
                              int64_t global_step) = 0;

  // TODO(shivaniagrawal): consistency in double and float usage.
  // Add the given `value` as Scalar with the given `name`.
  virtual void AddScalar(const string& name, float value,
                         int64_t global_step) = 0;

  // Stores a protocol buffer representation of the aggregator state in the
  // given `out_summary`.
  virtual void EncodeToProto(Summary* out_summary) = 0;

  // Sets a `summary_writer` with this stats_aggregator.
  virtual Status SetSummaryWriter(SummaryWriterInterface* summary_writer) = 0;

  // Increment the `label` cell of metrics mapped with `name` by given `value`.
  virtual void IncrementCounter(const string& name, const string& label,
                                int64_t val) = 0;
};

// A `StatsAggregatorResource` wraps a sharable `StatsAggregator` as a resource
// in the TensorFlow resource manager.
//
// NOTE(mrry): This class is separate from `StatsAggregator` in order to
// simplify the memory management of the shared object. Most users of
// `StatsAggregator` interact with a `std::shared_ptr<StatsAggregator>` whereas
// the `ResourceBase` API requires explicit reference counting.
class StatsAggregatorResource : public ResourceBase {
 public:
  // Creates a new resource from the given `stats_aggregator`.
  StatsAggregatorResource(std::unique_ptr<StatsAggregator> stats_aggregator)
      : stats_aggregator_(stats_aggregator.release()) {}

  // Returns the wrapped `StatsAggregator`.
  std::shared_ptr<StatsAggregator> stats_aggregator() const {
    return stats_aggregator_;
  }

  string DebugString() const override { return "StatsAggregatorResource"; }

 private:
  const std::shared_ptr<StatsAggregator> stats_aggregator_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_STATS_AGGREGATOR_H_
