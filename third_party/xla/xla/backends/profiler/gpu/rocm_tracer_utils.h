/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_UTILS_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace xla {
namespace profiler {

// Mirrors the identical typedef in cupti_buffer_events.h for the CUPTI path.
// Both resolve to the same type; ROCm and CUPTI are never compiled together.
using ScopeRangeIdTree = absl::flat_hash_map<int64_t, int64_t>;

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
  // Total dispatch-time private-segment (scratch) bytes per work-item.
  uint32_t private_segment_size;
  // Total dispatch-time group-segment (LDS) bytes per workgroup. Includes
  // static and dynamic LDS allocation.
  uint32_t group_segment_size;
  // Architecture and accumulator VGPRs allocated per work-item.
  uint32_t registers_per_work_item;
  // Static group-segment (LDS) bytes per workgroup from the kernel symbol.
  uint32_t static_group_segment_size;
  // X-dimension of a workgroup (grid.x*block.x)
  uint32_t workgroup_x;
  // Y-dimension of a workgroup (grid.x*block.x)
  uint32_t workgroup_y;
  // Z-dimension of a workgroup (grid.x*block.x)
  uint32_t workgroup_z;
  // X-dimension of a grid.
  uint32_t grid_x;
  // Y-dimension of a grid.
  uint32_t grid_y;
  // Z-dimension of a grid.
  uint32_t grid_z;

  // kernel address. Used for calculating core occupancy
  void* func_ptr;
};

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

// RocmTracerSyncTypes forward declaration
enum class RocmTracerSyncTypes;

struct SynchronizationDetails {
  RocmTracerSyncTypes sync_type;
};

struct RocmTracerEvent {
  static constexpr uint32_t kInvalidDeviceId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64_t kInvalidThreadId =
      std::numeric_limits<uint64_t>::max();
  // Matches rocprofiler_correlation_id_t::internal, which is 64-bit.
  static constexpr uint64_t kInvalidCorrelationId =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint64_t kInvalidStreamId =
      std::numeric_limits<uint64_t>::max();
  RocmTracerEventType type;
  RocmTracerEventSource source = RocmTracerEventSource::Invalid;
  RocmTracerEventDomain domain;
  std::string name;
  // This points to strings in AnnotationMap, which should outlive the point
  // where serialization happens.
  absl::string_view annotation;
  // Set only for kernel/HIP-API events: the ROCTX label that was active on
  // the dispatching thread at call time, stored via AnnotationMap::Add and
  // retrieved by AnnotationMap::LookUpRoctxRange. Empty for Generic (marker)
  // events, which carry their label in `name` instead.
  // The view points into AnnotationMap's interning pool, which is session-
  // scoped. Export() runs within the same session, so the lifetime is safe.
  absl::string_view roctx_range;
  uint64_t start_time_ns = 0;
  uint64_t end_time_ns = 0;
  uint32_t device_id = kInvalidDeviceId;
  uint64_t correlation_id = kInvalidCorrelationId;
  uint64_t thread_id = kInvalidThreadId;
  uint64_t stream_id = kInvalidStreamId;
  uint64_t queue_id = 0;  // HSA queue handle, preserved for debugging/rocprof
                          // correlation. Not exported to XEvents yet.
  int64_t scope_range_id = 0;

  union {
    MemcpyDetails memcpy_info;                    // If type == Memcpy*
    MemsetDetails memset_info;                    // If type == Memset*
    MemAllocDetails memalloc_info;                // If type == MemoryAlloc
    KernelDetails kernel_info;                    // If type == Kernel
    SynchronizationDetails synchronization_info;  // If type == Synchronization
  };
};

// Represents one pending ROCTX range pushed via roctxRangePushA. Stored on
// a per-thread stack in RocmTracer and consumed when roctxRangePop fires.
struct RoctxFrame {
  // TODO(rocm-profiler): carry a reference into AnnotationMap's intern pool
  // instead of owning a copy. Blocked on lifetime, not on the generation
  // check: the generation guards *emission*, but Enable() calls
  // annotation_map_.Clear() while frames pushed before it are still live on
  // some other thread's stack, so a reference would dangle even though the
  // frame is correctly dropped at pop. Needs the pool to outlive the session
  // (or a per-frame refcount) before the copy can go.
  std::string message;  // the range label (owned here for lifetime safety)
  uint64_t start_ns;    // timestamp captured at push time
  // Profiling session this frame was pushed in. Frames live on a thread_local
  // stack that no session boundary can reach, so a range pushed before
  // Enable() and popped after it would otherwise emit an event with a
  // previous session's start timestamp into the new session's collector.
  // Enable() bumps the generation; a pop whose frame predates it is dropped.
  uint64_t generation;
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
  void Add(uint64_t correlation_id, const std::string& annotation,
           absl::string_view roctx_range = {},
           absl::Span<const int64_t> scope_range_ids = {});
  absl::string_view LookUp(uint64_t correlation_id);
  absl::string_view LookUpRoctxRange(uint64_t correlation_id);
  int64_t LookUpScopeRangeId(uint64_t correlation_id);
  ScopeRangeIdTree TakeScopeRangeIdTree();
  void Clear();

 private:
  struct AnnotationMapImpl {
    // The population/consumption of annotations might happen from multiple
    // callback/activity api related threads.
    absl::Mutex mutex;
    // Annotation tends to be repetitive, use a hash_set to store the strings,
    // and use a reference_wrapper into the set in the maps. node_hash_set
    // guarantees pointer and reference stability on rehash, so the stored
    // references remain valid for the lifetime of the set.
    absl::node_hash_set<std::string> annotations ABSL_GUARDED_BY(mutex);
    absl::flat_hash_map<uint64_t, std::reference_wrapper<const std::string>>
        correlation_map ABSL_GUARDED_BY(mutex);
    absl::flat_hash_map<uint64_t, std::reference_wrapper<const std::string>>
        roctx_range_map ABSL_GUARDED_BY(mutex);
    absl::flat_hash_map<uint64_t, int64_t> scope_range_id_map
        ABSL_GUARDED_BY(mutex);
    ScopeRangeIdTree scope_range_id_tree ABSL_GUARDED_BY(mutex);
  };
  const uint64_t max_size_;
  AnnotationMapImpl map_;

 public:
  // Disable copy and move.
  AnnotationMap(const AnnotationMap&) = delete;
  AnnotationMap& operator=(const AnnotationMap&) = delete;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_UTILS_H_
