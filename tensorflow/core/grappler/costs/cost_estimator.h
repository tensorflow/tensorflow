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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_COST_ESTIMATOR_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_COST_ESTIMATOR_H_

#include <cmath>
#include <unordered_map>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
class GraphDef;
class CostGraphDef;

namespace grappler {
struct GrapplerItem;

constexpr int64 kMemoryUnknown = -1ll;
constexpr int64 kZeroMemory = 0ll;

struct DeviceInfo {
  // Billions of operations executed per second.
  double gigaops;

  // Bandwidth to main memory in GB per second.
  double gb_per_sec;

  // Read bandwidth to intermediate memory in GB per second.
  double intermediate_read_gb_per_sec;

  // Write bandwidth to intermediate memory in GB per second.
  double intermediate_write_gb_per_sec;

  DeviceInfo()
      : gigaops(INFINITY),
        gb_per_sec(INFINITY),
        intermediate_read_gb_per_sec(INFINITY),
        intermediate_write_gb_per_sec(INFINITY) {}

  DeviceInfo(const DeviceInfo& input)
      : gigaops(input.gigaops),
        gb_per_sec(input.gb_per_sec),
        intermediate_read_gb_per_sec(input.intermediate_read_gb_per_sec),
        intermediate_write_gb_per_sec(input.intermediate_write_gb_per_sec) {}

  DeviceInfo(double gigaops, double gb_per_sec,
             double intermediate_read_gb_per_sec = INFINITY,
             double intermediate_write_gb_per_sec = INFINITY)
      : gigaops(gigaops),
        gb_per_sec(gb_per_sec),
        intermediate_read_gb_per_sec(intermediate_read_gb_per_sec),
        intermediate_write_gb_per_sec(intermediate_write_gb_per_sec) {}
};

// Holds the set of things we might want to estimate or measure in Grappler.
// Always produce execution time. Other fields are optional depending on the
// estimator being used.
struct Costs {
  // Returns a Costs structure with default values for all of the fields.
  inline Costs();

  // Builds a Costs structure with all zero values, rather than unknowns.
  static inline Costs ZeroCosts();

  struct MilliSeconds : std::chrono::milliseconds {
    MilliSeconds() : std::chrono::milliseconds(0) {}
    MilliSeconds(double d) : std::chrono::milliseconds(static_cast<int64>(d)) {}
    MilliSeconds(const std::chrono::milliseconds& d)
        : std::chrono::milliseconds(d) {}
    MilliSeconds& operator=(const std::chrono::milliseconds& d) {
      std::chrono::milliseconds::operator=(d);
      return *this;
    }
  };
  struct MicroSeconds : std::chrono::microseconds {
    MicroSeconds() : std::chrono::microseconds(0) {}
    MicroSeconds(double d) : std::chrono::microseconds(static_cast<int64>(d)) {}
    MicroSeconds(const std::chrono::microseconds& d)
        : std::chrono::microseconds(d) {}
    MicroSeconds& operator=(const std::chrono::microseconds& d) {
      std::chrono::microseconds::operator=(d);
      return *this;
    }
    MilliSeconds asMilliSeconds() const {
      return std::chrono::duration_cast<std::chrono::milliseconds>(*this);
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
      return std::chrono::duration_cast<std::chrono::microseconds>(*this);
    }
    MilliSeconds asMilliSeconds() const {
      return std::chrono::duration_cast<std::chrono::milliseconds>(*this);
    }
    static NanoSeconds infinity() {
      return NanoSeconds(std::chrono::nanoseconds::max());
    }
  };

  // We store all our times in nanoseconds. If needs be, we can always switch to
  // picoseconds in the future by updating this typedef.
  typedef NanoSeconds Duration;

  // Overall cost of running the graph; latency.
  Duration execution_time;

  // Computation cost of running the graph.
  Duration compute_time;

  // Memory access cost of running the graph.
  Duration memory_time;

  // Intermediate memory access cost of running the graph
  Duration intermediate_memory_time;
  Duration intermediate_memory_read_time;   // Intermediate memory read cost.
  Duration intermediate_memory_write_time;  // Intermediate memory write cost.

  // This field can be a very pessimistic estimate of the main memory
  // requirements of a graph. For example, it might assume that all activations
  // are live for all of a graph's execution.
  int64 max_memory;  // Maximum main memory requirement in bytes over all ops.
  int64 persistent_memory;
  int64 temporary_memory;

  // These fields are used for TPU-related estimations. They are per-op
  // maximums, so each op is evaluated independently, but we want the maximum of
  // the value over all ops.
  int64 max_per_op_buffers;    // Sum of all buffers used by the ops.
  int64 max_per_op_streaming;  // Ignore largest input buffer, assuming it
                               // streams from main memory.

  // Number of ops included in this Costs in total.
  // Default initialized to be one.
  int64 num_ops_total = 1;
  // If the time estimation is inaccurate.
  bool inaccurate = false;
  // Number of ops that are estimated with unknown shapes.
  int64 num_ops_with_unknown_shapes = 0;
  // TODO(pcma): include a counter for total inaccurate ops and counters for
  // other reasons causing the inaccuracy

  // Max possible memory usage per device.
  std::unordered_map<string, uint64> estimated_max_memory_per_device;
};

inline std::ostream& operator<<(std::ostream& os, const Costs::MilliSeconds d) {
  os << d.count() << "ms";
  return os;
}
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
  intermediate_memory_time = Duration::zero();
  max_memory = kMemoryUnknown;
  persistent_memory = kMemoryUnknown;
  temporary_memory = kMemoryUnknown;
  max_per_op_buffers = kMemoryUnknown;
  max_per_op_streaming = kMemoryUnknown;
}

Costs Costs::ZeroCosts() {
  Costs costs;
  costs.execution_time = Duration::zero();
  costs.compute_time = Duration::zero();
  costs.memory_time = Duration::zero();
  costs.intermediate_memory_time = Duration::zero();
  costs.max_memory = kZeroMemory;
  costs.persistent_memory = kZeroMemory;
  costs.temporary_memory = kZeroMemory;
  costs.max_per_op_buffers = kZeroMemory;
  costs.max_per_op_streaming = kZeroMemory;
  return costs;
}

Costs CombineCosts(const Costs& left, const Costs& right);

// Multiplies Costs by a scalar.
// Equivalent to applying CombineCosts "multiplier" times.
Costs MultiplyCosts(const Costs& costs, int multiplier);

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
  // If a RunMetadata is passed, it will be populated with detailed information
  // about the cost of running each operation of the optimized graph.
  // if a double value is passed, it will be set to a value that reflects the
  // overall cost of running the graph (e.g. the latency of the computation).
  // Returns a status that indicate is the performance could be estimated or
  // not.
  virtual Status PredictCosts(const GraphDef& optimized_graph,
                              RunMetadata* run_metadata, Costs* cost) const = 0;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_COST_ESTIMATOR_H_
