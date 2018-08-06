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

// A `StatsAggregator` accumulates statistics incrementally. A
// `StatsAggregator` can accumulate multiple different statistics, distinguished
// by a string name.
//
// The class currently supports accumulating `Histogram` objects, and we expect
// to add other methods in future.
//
// NOTE(mrry): `StatsAggregator` is a virtual interface because we anticipate
// that many different implementations will the same interface. For example, the
// current implementation in "stats_aggregator_ops.cc" is a simple in-memory
// implementation that integrates with the pull-based summary API, and we may
// add implementations that work with the push-based `SummaryWriterInterface`,
// as well as custom monitoring services.
class StatsAggregator {
 public:
  virtual ~StatsAggregator() {}

  // Add the given `values` to the histogram with the given `name`. Each
  // element of `values` will be treated as a separate sample in the histogram.
  virtual void AddToHistogram(const string& name,
                              gtl::ArraySlice<double> values) = 0;

  // TODO(shivaniagarawal): consistency in double and float usage.
  // Add the given `value` as Scalar with the given `name`.
  virtual void AddScalar(const string& name, float value) = 0;

  // Stores a protocol buffer representation of the aggregator state in the
  // given `out_summary`.
  // TODO(mrry): Consider separating this method from the `StatsAggregator`
  // interface. It is possible that not all implementations will support
  // encoding their state as a protocol buffer.
  virtual void EncodeToProto(Summary* out_summary) = 0;

  // Increment the `label` cell of metrics mapped with `name` by given `value`.
  virtual void IncrementCounter(const string& name, const string& label,
                                int64 val) = 0;
};

// A `StatsAggregatorResource` wraps a shareable `StatsAggregator` as a resource
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

  string DebugString() { return "StatsAggregatorResource"; }

 private:
  const std::shared_ptr<StatsAggregator> stats_aggregator_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_STATS_AGGREGATOR_H_
