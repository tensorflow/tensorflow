/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "tsl/profiler/utils/xplane_builder.h"

namespace xla {
namespace profiler {

using tsl::profiler::XSpace;

struct MemcpyDetails {
  // The amount of data copied for memcpy events.
  size_t num_bytes;
  // The destination device for peer-2-peer communication (memcpy). The source
  // device is implicit: it's the current device.
  uint32_t destination;
  // Whether or not the memcpy is asynchronous.
  bool async;
};

struct MemAllocDetails {
  // The amount of data requested for cudaMalloc events.
  uint64_t num_bytes;
};

struct MemsetDetails {
  // The number of memory elements getting set
  size_t num_bytes;
  // Whether or not the memset is asynchronous.
  bool async;
};

struct KernelDetails {
  // The number of registers used in this kernel.
  uint32_t registers_per_thread;
  // The amount of shared memory space used by a thread block.
  uint32_t static_shared_memory_usage;
  // The amount of dynamic memory space used by a thread block.
  uint32_t dynamic_shared_memory_usage;
  // X-dimension of a thread block.
  uint32_t block_x;
  // Y-dimension of a thread block.
  uint32_t block_y;
  // Z-dimension of a thread block.
  uint32_t block_z;
  // X-dimension of a grid.
  uint32_t grid_x;
  // Y-dimension of a grid.
  uint32_t grid_y;
  // Z-dimension of a grid.
  uint32_t grid_z;

  // kernel address. Used for calculating core occupancy
  void* func_ptr;
};

inline std::string ToXStat(const KernelDetails& kernel_info,
                           double occupancy_pct) {
  return absl::StrCat(
      "regs:", kernel_info.registers_per_thread,
      " static_shared:", kernel_info.static_shared_memory_usage,
      " dynamic_shared:", kernel_info.dynamic_shared_memory_usage,
      " grid:", kernel_info.grid_x, ",", kernel_info.grid_y, ",",
      kernel_info.grid_z, " block:", kernel_info.block_x, ",",
      kernel_info.block_y, ",", kernel_info.block_z,
      " occ_pct:", occupancy_pct);
}

enum class RocmTracerEventType {
  Unsupported = 0,
  Kernel,
  MemcpyH2D,
  MemcpyD2H,
  MemcpyD2D,
  MemcpyP2P,
  MemcpyOther,
  MemoryAlloc,
  MemoryFree,
  Memset,
  Synchronization,
  Generic,
};

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type);

enum class RocmTracerEventSource {
  Invalid = 0,
  ApiCallback,
  Activity,
};

const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source);

enum class RocmTracerEventDomain {
  InvalidDomain = 0,
  HIP_API,
  HIP_OPS,
};
const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain);
// RocmTracerSyncTypes forward decleration
enum class RocmTracerSyncTypes;

struct SynchronizationDetails {
  RocmTracerSyncTypes sync_type;
};

struct RocmTracerEvent {
  static constexpr uint32_t kInvalidDeviceId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kInvalidThreadId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kInvalidCorrelationId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64_t kInvalidStreamId =
      std::numeric_limits<uint64_t>::max();
  RocmTracerEventType type;
  RocmTracerEventSource source = RocmTracerEventSource::Invalid;
  RocmTracerEventDomain domain;
  std::string name;
  // This points to strings in AnnotationMap, which should outlive the point
  // where serialization happens.
  absl::string_view annotation;
  absl::string_view roctx_range;
  uint64_t start_time_ns = 0;
  uint64_t end_time_ns = 0;
  uint32_t device_id = kInvalidDeviceId;
  uint32_t correlation_id = kInvalidCorrelationId;
  uint32_t thread_id = kInvalidThreadId;
  int64_t stream_id = kInvalidStreamId;
  union {
    MemcpyDetails memcpy_info;                    // If type == Memcpy*
    MemsetDetails memset_info;                    // If type == Memset*
    MemAllocDetails memalloc_info;                // If type == MemoryAlloc
    KernelDetails kernel_info;                    // If type == Kernel
    SynchronizationDetails synchronization_info;  // If type == Synchronization
  };
};

struct RocmTraceCollectorOptions {
  // Maximum number of events to collect from callback API; if -1, no limit.
  // if 0, the callback API is enabled to build a correlation map, but no
  // events are collected.
  uint64_t max_callback_api_events;
  // Maximum number of events to collect from activity API; if -1, no limit.
  uint64_t max_activity_api_events;
  // Maximum number of annotation strings that we can accommodate.
  uint64_t max_annotation_strings;
  // Number of GPUs involved.
  uint32_t num_gpus;
};

class AnnotationMap {
 public:
  explicit AnnotationMap(uint64_t max_size) : max_size_(max_size) {}
  void Add(uint32_t correlation_id, const std::string& annotation);
  absl::string_view LookUp(uint32_t correlation_id);

 private:
  struct AnnotationMapImpl {
    // The population/consumption of annotations might happen from multiple
    // callback/activity api related threads.
    absl::Mutex mutex;
    // Annotation tends to be repetitive, use a hash_set to store the strings,
    // an use the reference to the string in the map.
    absl::node_hash_set<std::string> annotations;
    absl::flat_hash_map<uint32_t, absl::string_view> correlation_map;
  };
  const uint64_t max_size_;
  AnnotationMapImpl map_;

 public:
  // Disable copy and move.
  AnnotationMap(const AnnotationMap&) = delete;
  AnnotationMap& operator=(const AnnotationMap&) = delete;
};

class RocmTraceCollector {
 public:
  explicit RocmTraceCollector(const RocmTraceCollectorOptions& options)
      : options_(options), annotation_map_(options.max_annotation_strings) {}
  virtual ~RocmTraceCollector() {}

  virtual void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint32_t num_events) = 0;
  virtual void Flush() = 0;
  virtual void Export(XSpace* space) = 0;

  AnnotationMap* annotation_map() { return &annotation_map_; }

 protected:
  RocmTraceCollectorOptions options_;

 private:
  AnnotationMap annotation_map_;

 public:
  // Disable copy and move.
  RocmTraceCollector(const RocmTraceCollector&) = delete;
  RocmTraceCollector& operator=(const RocmTraceCollector&) = delete;
};

std::unique_ptr<RocmTraceCollector> CreateRocmCollector(
    const RocmTraceCollectorOptions& options, const uint64_t start_walltime_ns,
    const uint64_t start_gputime_ns);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_
