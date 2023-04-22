/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_COLLECTOR_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_COLLECTOR_H_

#include <memory>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

struct MemcpyDetails {
  // The amount of data copied for memcpy events.
  size_t num_bytes;
  // The destination device for peer-2-peer communication (memcpy). The source
  // device is implicit: it's the current device.
  uint32 destination;
  // Whether or not the memcpy is asynchronous.
  bool async;
  // This contains CUpti_ActivityMemcpyKind for activity event (on device).
  // For events from other CuptiTracerEventSource, it is always 0.
  int8 copy_kind;
  // CUpti_ActivityMemoryKind of source.
  int8 src_mem_kind;
  // CUpti_ActivityMemoryKind of destination.
  int8 dst_mem_kind;
};

struct MemAllocDetails {
  // Size of memory to be written over in bytes.
  size_t num_bytes;
  // The CUpti_ActivityMemoryKind value for this activity event.
  int8 mem_kind;
  // The virtual address of allocation. 0 if it is a free operation.
  uint64 address;
};

using MemFreeDetails = MemAllocDetails;

// Memory residency contains details read from CUpti_ActivityMemory type. This
// is populated in the CUPTI tracer encounters a CUPTI_ACTIVITY_KIND_MEMORY
// event. The start of this even corresponse to a cudaMalloc, and the end
// corresponds to a cudaFree.
using MemoryResidencyDetails = MemAllocDetails;

struct MemsetDetails {
  // Size of memory to be written over in bytes.
  size_t num_bytes;
  // The CUpti_ActivityMemoryKind value for this activity event.
  int8 mem_kind;
  // Whether or not the memset is asynchronous.
  bool async;
};

struct KernelDetails {
  // The number of registers used in this kernel.
  uint32 registers_per_thread;
  // The amount of shared memory space used by a thread block.
  uint32 static_shared_memory_usage;
  // The amount of dynamic memory space used by a thread block.
  uint32 dynamic_shared_memory_usage;
  // X-dimension of a thread block.
  uint32 block_x;
  // Y-dimension of a thread block.
  uint32 block_y;
  // Z-dimension of a thread block.
  uint32 block_z;
  // X-dimension of a grid.
  uint32 grid_x;
  // Y-dimension of a grid.
  uint32 grid_y;
  // Z-dimension of a grid.
  uint32 grid_z;
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
absl::string_view GetMemoryKindName(int8 memory_kind);

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
  static constexpr uint32 kInvalidThreadId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint32 kInvalidCorrelationId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64 kInvalidContextId =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint64 kInvalidStreamId =
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
  uint64 start_time_ns = 0;
  uint64 end_time_ns = 0;
  uint32 device_id = 0;
  uint32 correlation_id = kInvalidCorrelationId;
  uint32 thread_id = kInvalidThreadId;
  int64 context_id = kInvalidContextId;
  int64 stream_id = kInvalidStreamId;
  union {
    // For Memcpy API and activities. `type` must be Memcpy*.
    MemcpyDetails memcpy_info;
    // Used for MemAlloc API. `type` must be MemoryAlloc.
    MemAllocDetails memalloc_info;
    // Used for kernel activities. `type` must be Kernel.
    KernelDetails kernel_info;
    // Used for MemFree activities. `type` must be MemoryFree.
    MemFreeDetails memfree_info;
    // Used for Memset API and activities. `type` must be Memset.
    MemsetDetails memset_info;
    // Used for Memory residency activities. `type` must be MemoryResidency.
    MemoryResidencyDetails memory_residency_info;
  };
};

struct CuptiTracerCollectorOptions {
  // Maximum number of events to collect from callback API; if -1, no limit.
  // if 0, the callback API is enabled to build a correlation map, but no
  // events are collected.
  uint64 max_callback_api_events = 2 * 1024 * 1024;
  // Maximum number of events to collect from activity API; if -1, no limit.
  uint64 max_activity_api_events = 2 * 1024 * 1024;
  // Maximum number of annotation strings that we can accommodate.
  uint64 max_annotation_strings = 1024 * 1024;
  // Number of GPUs involved.
  uint32 num_gpus;
};

class AnnotationMap {
 public:
  struct AnnotationInfo {
    absl::string_view annotation;
    absl::string_view nvtx_range;
  };

  explicit AnnotationMap(uint64 max_size, uint32 num_gpus)
      : max_size_(max_size), per_device_map_(num_gpus) {}
  void Add(uint32 device_id, uint32 correlation_id,
           const absl::string_view annotation,
           const absl::string_view nvtx_range);
  AnnotationInfo LookUp(uint32 device_id, uint32 correlation_id);

 private:
  struct PerDeviceAnnotationMap {
    // The population/consumption of annotations might happen from multiple
    // callback/activity api related threads.
    absl::Mutex mutex;
    // Annotation tends to be repetitive, use a hash_set to store the strings,
    // an use the reference to the string in the map.
    absl::node_hash_set<std::string> annotations;
    absl::node_hash_set<std::string> nvtx_ranges;
    absl::flat_hash_map<uint32, AnnotationInfo> correlation_map;
  };
  const uint64 max_size_;
  absl::FixedArray<PerDeviceAnnotationMap> per_device_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(AnnotationMap);
};

class CuptiTraceCollector {
 public:
  explicit CuptiTraceCollector(const CuptiTracerCollectorOptions& options)
      : options_(options),
        annotation_map_(options.max_annotation_strings, options.num_gpus) {}
  virtual ~CuptiTraceCollector() {}

  // Producer side functions (i.e. called by CuptiTracer).
  virtual void AddEvent(CuptiTracerEvent&& event) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint32 num_events) = 0;
  virtual void Flush() = 0;

  // Consumer side functions (i.e. called by GPU tracer);
  virtual void Export(StepStats* step_stats) {}
  virtual bool Export(XSpace* space, uint64 end_gpu_ns) { return true; }
  virtual std::string ReportNumEventsIfDropped() { return ""; }

  AnnotationMap* annotation_map() { return &annotation_map_; }

 protected:
  CuptiTracerCollectorOptions options_;

 private:
  AnnotationMap annotation_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(CuptiTraceCollector);
};

std::unique_ptr<CuptiTraceCollector> CreateCuptiCollector(
    const CuptiTracerCollectorOptions& options, const uint64 start_walltime_ns,
    const uint64 start_gputime_ns);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_CUPTI_COLLECTOR_H_
