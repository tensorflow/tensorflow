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

#ifndef TENSORFLOW_GRAPPLER_COSTS_COST_ESTIMATOR_H_
#define TENSORFLOW_GRAPPLER_COSTS_COST_ESTIMATOR_H_

#include <chrono>
#include <unordered_map>
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class GraphDef;
class CostGraphDef;

namespace grappler {
struct GrapplerItem;

constexpr int64 kMemoryUnknown = -1ll;
constexpr int64 kZeroMemory = 0ll;

// Holds the set of things we might want to estimate or measure in Grappler.
// Always produce execution time. Other fields are optional depending on the
// estimator being used.
struct Costs {
  // Returns a Costs structure with default values for all of the fields.
  inline Costs();

  // Builds a Costs structure with all zero values, rather than unknowns.
  static inline Costs ZeroCosts();

  struct MicroSeconds : std::chrono::microseconds {
    MicroSeconds() : std::chrono::microseconds(0) {}
    MicroSeconds(double d) : std::chrono::microseconds(static_cast<int64>(d)) {}
    MicroSeconds(const std::chrono::microseconds& d)
        : std::chrono::microseconds(d) {}
    MicroSeconds& operator=(const std::chrono::microseconds& d) {
      std::chrono::microseconds::operator=(d);
      return *this;
    }
  };
  struct NanoSeconds : std::chrono::nanoseconds {
    NanoSeconds() : std::chrono::nanoseconds(0) {}
    NanoSeconds(double d) : std::chrono::nanoseconds(static_cast<int64>(d)) {}
    NanoSeconds(const std::chrono::nanoseconds& d)
        : std::chrono::nanoseconds(d) {}
    NanoSeconds& operator=(const std::chrono::nanoseconds& d) {
      std::chrono::nanoseconds::operator=(d);
      return *this;
    }
    MicroSeconds asMicroSeconds() const {
      std::chrono::microseconds us =
          std::chrono::duration_cast<std::chrono::microseconds>(*this);
      return MicroSeconds(us);
    }
  };

  // We store all our times in nanoseconds. If needs be, we can always switch to
  // picoseconds in the future by updating this typedef.
  typedef NanoSeconds Duration;

  // Overall cost of running the graph; latency.
  // Mean
  Duration execution_time;
  Duration min_execution_time;
  Duration max_execution_time;

  // Computation cost of running the graph.
  Duration compute_time;

  // Memory access cost of running the graph.
  Duration memory_time;

  // This field can be a very pessimistic estimate of the main memory
  // requirements of a graph. For example, it might assume that all activations
  // are live for all of a graph's execution.
  int64 max_memory;  // Maximum main memory requirement in bytes over all ops.

  // These fields are used for TPU-related estimations. They are per-op
  // maximums, so each op is evaluated independently, but we want the maximum of
  // the value over all ops.
  int64 max_per_op_buffers;    // Sum of all buffers used by the ops.
  int64 max_per_op_streaming;  // Ignore largest input buffer, assuming it
                               // streams from main memory.
  // If the time estimation is inaccurate.
  bool inaccurate = false;

  // Max possible memory usage per device.
  std::unordered_map<string, uint64> estimated_max_memory_per_device;
};

inline std::ostream& operator<<(std::ostream& os, const Costs::MicroSeconds d) {
  os << d.count() << "us";
  return os;
}
inline std::ostream& operator<<(std::ostream& os, const Costs::NanoSeconds d) {
  os << d.count() << "ns";
  return os;
}

Costs::Costs() {
  execution_time = Duration::zero();
  compute_time = Duration::zero();
  memory_time = Duration::zero();
  max_memory = kMemoryUnknown;
  max_per_op_buffers = kMemoryUnknown;
  max_per_op_streaming = kMemoryUnknown;
}

Costs Costs::ZeroCosts() {
  Costs costs;
  costs.execution_time = Duration::zero();
  costs.compute_time = Duration::zero();
  costs.memory_time = Duration::zero();
  costs.max_memory = kZeroMemory;
  costs.max_per_op_buffers = kZeroMemory;
  costs.max_per_op_streaming = kZeroMemory;
  return costs;
}

// Given a GrapperItem and an optimized implementation of the corresponding
// TensorFlow graph, the CostEstimator attempts to predicts the actual cost of
// running the graph.
class CostEstimator {
 public:
  virtual ~CostEstimator() {}

  // Initializes the estimator for the specified grappler item.
  // The estimator shouldn't be used if this function returns any status other
  // that OK.
  virtual Status Initialize(const GrapplerItem& item) = 0;

  // Predicts the cost of running the given optimized version of the grappler
  // item.
  // If a CostGraphDef is passed, it will be populated with detailed information
  // about the cost of running each operation of the optimized graph.
  // if a double value is passed, it will be set to a value that reflects the
  // overall cost of running the graph (e.g. the latency of the computation).
  // Returns a status that indicate is the performance could be estimated or
  // not.
  virtual Status PredictCosts(const GraphDef& optimized_graph,
                              CostGraphDef* cost_graph, Costs* cost) const = 0;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_COSTS_COST_ESTIMATOR_H_
