/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_BUFFER_EVENTS_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_BUFFER_EVENTS_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/profiler/utils/buffer_pool.h"
#include "xla/tsl/profiler/utils/lock_free_queue.h"
#include "tsl/platform/thread_annotations.h"

namespace xla {
namespace profiler {

struct MemcpyDetails {
  // The amount of data copied for memcpy events.
  size_t num_bytes;
  // The destination device for peer-2-peer communication (memcpy). The source
  // device is implicit: it's the current device.
  uint32_t destination;
  // Whether or not the memcpy is asynchronous.
  bool async;
  // This contains CUpti_ActivityMemcpyKind for activity event (on device).
  // For events from other CuptiTracerEventSource, it is always 0.
  int8_t copy_kind;
  // CUpti_ActivityMemoryKind of source.
  int8_t src_mem_kind;
  // CUpti_ActivityMemoryKind of destination.
  int8_t dst_mem_kind;

  // ID of the hardware channel on which this operation ran.
  uint32_t channel_id = static_cast<uint32_t>(-1);
  // CUpti_ChannelType of the channel above.
  int8_t channel_type = 0;  // CUPTI_CHANNEL_TYPE_INVALID
};

struct MemAllocDetails {
  // Size of memory to be written over in bytes.
  size_t num_bytes;
  // The CUpti_ActivityMemoryKind value for this activity event.
  int8_t mem_kind;
  // The virtual address of allocation. 0 if it is a free operation.
  uint64_t address;
};

using MemFreeDetails = MemAllocDetails;

// Memory residency contains details read from CUpti_ActivityMemory type. This
// is populated in the CUPTI tracer encounters a CUPTI_ACTIVITY_KIND_MEMORY
// event. The start of this even corresponse to a cudaMalloc, and the end
// corresponds to a cudaFree.
using MemoryResidencyDetails = MemAllocDetails;

// cudaHostRegister
struct HostRegisterDetails {
  size_t num_bytes;
  uint64_t address;
  unsigned int flags;
};

// cudaHostUnregister
struct HostUnregisterDetails {
  uint64_t address;
};

struct MemsetDetails {
  // Size of memory to be written over in bytes.
  size_t num_bytes;
  // The CUpti_ActivityMemoryKind value for this activity event.
  int8_t mem_kind;
  // Whether or not the memset is asynchronous.
  bool async;

  // ID of the hardware channel on which this operation ran.
  uint32_t channel_id = -1;
  // CUpti_ChannelType of the channel above.
  int8_t channel_type = 0;  // CUPTI_CHANNEL_TYPE_INVALID
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

  // ID of the hardware channel on which this operation ran.
  uint32_t channel_id = -1;
  // CUpti_ChannelType of the channel above.
  int8_t channel_type = 0;  // CUPTI_CHANNEL_TYPE_INVALID
};

struct GenericDetails {
  uint32_t cbid;
};

struct CudaGraphDetails {
  uint32_t cbid;  // 0 for activity events, otherwise the cbid of the callback
  uint32_t orig_graph_id;  // The original graph from which new graph is
                           // instantiated. Note graph_id is put into general
                           // fields as if trace in node mode, many activity
                           // events will contains graph id.
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

// Gets the name of the CUpti_ActivityMemoryKind value.
absl::string_view GetMemoryKindName(int8_t memory_kind);

enum class CuptiTracerEventType {
  Unsupported = 0,
  Kernel = 1,
  MemcpyH2D = 2,
  MemcpyD2H = 3,
  MemcpyD2D = 4,
  MemcpyP2P = 5,
  MemcpyOther = 6,
  MemoryAlloc = 7,
  Overhead = 8,
  UnifiedMemory = 9,
  MemoryFree = 10,
  Memset = 11,
  MemoryResidency = 12,
  HostRegister = 13,
  HostUnregister = 14,
  CudaGraph = 15,
  ThreadMarkerRange = 16,
  ThreadMarkerStart = 17,
  ThreadMarkerEnd = 18,
  Generic = 100,
};

const char* GetTraceEventTypeName(const CuptiTracerEventType& type);

enum class CuptiTracerEventSource {
  Invalid = 0,
  DriverCallback = 1,
  Activity = 2,
  // Maybe consider adding runtime callback and metric api in the future.
};

struct CuptiTracerEvent {
  static constexpr uint64_t kInvalidThreadId =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint32_t kInvalidCorrelationId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64_t kInvalidContextId =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint64_t kInvalidStreamId =
      std::numeric_limits<uint64_t>::max();
  CuptiTracerEventType type = CuptiTracerEventType::Unsupported;
  CuptiTracerEventSource source = CuptiTracerEventSource::Invalid;
  // Although CUpti_CallbackData::functionName is persistent, however
  // CUpti_ActivityKernel4::name is not persistent, therefore we need a copy of
  // it.
  std::string name;
  // This points to strings in AnnotationMap, which should outlive the point
  // where serialization happens.
  absl::string_view annotation;
  absl::string_view nvtx_range;
  uint64_t start_time_ns = 0;
  uint64_t end_time_ns = 0;
  uint32_t device_id = 0;
  uint32_t correlation_id = kInvalidCorrelationId;
  uint64_t thread_id = kInvalidThreadId;
  int64_t context_id = kInvalidContextId;
  int64_t stream_id = kInvalidStreamId;
  uint32_t graph_id = 0;
  int64_t scope_range_id = 0;
  union {
    // For Memcpy API and activities. `type` must be Memcpy*.
    MemcpyDetails memcpy_info;
    // Used for MemAlloc API. `type` must be MemoryAlloc.
    MemAllocDetails memalloc_info;
    // Used for kernel activities. `type` must be Kernel.
    KernelDetails kernel_info;
    // Used for MemFree activities. `type` must be MemoryFree.
    MemFreeDetails memfree_info;
    // Used for cuMemHostRegister.  `type` must be HostRegister.
    HostRegisterDetails host_register_info;
    // Used for cuMemHostUnregister.  `type` must be HostUnregister.
    HostUnregisterDetails host_unregister_info;
    // Used for Memset API and activities. `type` must be Memset.
    MemsetDetails memset_info;
    // Used for Memory residency activities. `type` must be MemoryResidency.
    MemoryResidencyDetails memory_residency_info;
    // Used for `source` DriverCallback, `type` must be Generic.
    GenericDetails generic_info;
    // Used for `source` DriverCallback, `type` must be CudaGraph.
    CudaGraphDetails cuda_graph_info;
  };
};

// As annotation and nvtx range strings are of large duplication, it is worth
// to keep single copy of different strings to save memory footprint. This class
// will construct a string when deduping unseen string_view input, and return
// the string_view on the newly created string. If the input str is contains in
// its internal data, it just return it's internal copy's string_view. All
// returned string_view will keep valid as the object of this class is alive.
class StringDeduper {
 public:
  void Clear() { strings_.clear(); }

  // max_unique_count is not put into data member to make it consistent with
  // existing logic.
  absl::string_view Dedup(absl::string_view str, size_t max_unique_count = 0);

  size_t Size() const { return strings_.size(); }

 private:
  absl::node_hash_set<std::string> strings_;
};

// AnnotationMap keep the map from a correlation id to its corresponding
// annotation and nvtx_range. During Add(), unseen input string view will
// cause new internal string constructed. This annotation map also controls
// per-device annotation string count.
class AnnotationMap {
 public:
  struct AnnotationInfo {
    absl::string_view annotation;
    absl::string_view nvtx_range;
    int64_t scope_range_id = 0;
  };

  explicit AnnotationMap(uint64_t max_size, uint32_t num_gpus)
      : max_size_(max_size), per_device_map_(num_gpus) {}

  void Add(uint32_t device_id, uint32_t correlation_id,
           absl::string_view annotation, absl::string_view nvtx_range,
           int64_t scope_range_id = 0);

  AnnotationInfo LookUp(uint32_t device_id, uint32_t correlation_id) const
      ABSL_ATTRIBUTE_LIFETIME_BOUND;

 private:
  struct PerDeviceAnnotationMap {
    StringDeduper annotation_deduper;
    StringDeduper nvtx_range_deduper;
    absl::flat_hash_map<uint32_t, AnnotationInfo> correlation_map;
  };
  const uint64_t max_size_;
  absl::FixedArray<PerDeviceAnnotationMap> per_device_map_;

  AnnotationMap(const AnnotationMap&) = delete;
  void operator=(const AnnotationMap&) = delete;
};

struct CuptiEventCollectorDelegate {
  AnnotationMap& annotation_map;
  std::function<void(CuptiTracerEvent&&)> receive;

  explicit CuptiEventCollectorDelegate(
      AnnotationMap& p_annotation_map,
      std::function<void(CuptiTracerEvent&&)> p_receive)
      : annotation_map(p_annotation_map), receive(std::move(p_receive)) {}
};

// A tree of scope range ids which map child_id ==> parent_id
typedef absl::flat_hash_map<int64_t, int64_t> ScopeRangeIdTree;

class CuptiActivityBufferManager {
 public:
  struct ActivityBufferAndSize {
    std::unique_ptr<uint8_t, std::function<void(uint8_t*)>> buffer;
    size_t size;  // size in bytes for the events filled by CUPTI.
    explicit ActivityBufferAndSize(uint8_t* p = nullptr, size_t sz = 0);
  };

  explicit CuptiActivityBufferManager(size_t buffer_size_in_bytes)
      : buffer_pool_(buffer_size_in_bytes) {}

  size_t GetBufferSizeInBytes() { return buffer_pool_.GetBufferSizeInBytes(); }

  uint8_t* GetOrCreateBuffer() { return buffer_pool_.GetOrCreateBuffer(); }

  void ReclaimBuffer(uint8_t* p) { buffer_pool_.ReclaimBuffer(p); }

  void CacheCuptiFilledActivityBuffer(uint8_t* p, size_t sz) {
    absl::MutexLock lock(&buffer_mutex_);
    cached_buffers_.emplace_back(p, sz);
  }

  std::list<ActivityBufferAndSize> PopCachedBuffers() {
    std::list<ActivityBufferAndSize> result;
    absl::MutexLock lock(&buffer_mutex_);
    std::swap(result, cached_buffers_);
    return result;
  }

 private:
  tsl::profiler::BufferPool buffer_pool_;
  absl::Mutex buffer_mutex_;
  std::list<ActivityBufferAndSize> cached_buffers_ TF_GUARDED_BY(buffer_mutex_);
};

void AddActivityBufferListEventsTo(
    CuptiEventCollectorDelegate& receiver,
    std::list<CuptiActivityBufferManager::ActivityBufferAndSize>& buffer_list,
    size_t max_activity_event_count, size_t& dropped_activity_event_count);

class CallbackAnnotationsAndEvents {
 public:
  static constexpr size_t kQueueBlockSize = 256 * 1024;
  using EventQueue =
      tsl::profiler::BlockedQueue<CuptiTracerEvent, kQueueBlockSize>;

  CallbackAnnotationsAndEvents() = default;

  CallbackAnnotationsAndEvents(CallbackAnnotationsAndEvents&& another);

  CallbackAnnotationsAndEvents& operator=(
      CallbackAnnotationsAndEvents&& another);

  void Clear();

  size_t NumAnnotations() const { return annotations_.Size(); }

  absl::string_view DedupAnnotation(absl::string_view str) {
    return annotations_.Dedup(str);
  }

  absl::string_view DedupNvtxRange(absl::string_view str) {
    return nvtx_ranges_.Dedup(str);
  }

  EventQueue& event_queue() { return event_queue_; }

  ScopeRangeIdTree& scope_range_id_tree() { return scope_range_id_tree_; }

  size_t NumDroppedEvents() const { return num_dropped_events_; }

  void IncNumDroppedEvents() { ++num_dropped_events_; }

 private:
  // Annotation tends to be repetitive, use a hash_set to store the strings,
  // and use the reference to the string in the hash_set.
  StringDeduper annotations_;
  StringDeduper nvtx_ranges_;
  size_t num_dropped_events_ = 0;
  EventQueue event_queue_;
  ScopeRangeIdTree scope_range_id_tree_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_BUFFER_EVENTS_H_
