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

#include "xla/backends/profiler/gpu/cupti_buffer_events.h"

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/mem.h"

namespace xla {
namespace profiler {

namespace {

using absl::StatusCode;

template <typename CuptiActivity>
struct CuptiActivityHasGraphId {
  static constexpr bool value = false;
};

// CUPTI from CUDA 11.6 adds information about the hardware channel that ops
// run on; this makes its way into the channel_id and channel_type fields in the
// structs we export.
//
// Define some type aliases so we can access the hardware channel id if it's
// available.
#if CUDA_VERSION >= 12000  // CUDA 12.0
#define TF_CUPTI_HAS_CHANNEL_ID 1
using CuptiActivityKernelTy = CUpti_ActivityKernel9;
using CuptiActivityMemcpyTy = CUpti_ActivityMemcpy5;
using CuptiActivityMemcpyP2PTy = CUpti_ActivityMemcpyPtoP4;
using CuptiActivityMemsetTy = CUpti_ActivityMemset4;

template <>
struct CuptiActivityHasGraphId<CuptiActivityKernelTy> {
  static constexpr bool value = true;
};
template <>
struct CuptiActivityHasGraphId<CuptiActivityMemcpyTy> {
  static constexpr bool value = true;
};
template <>
struct CuptiActivityHasGraphId<CuptiActivityMemcpyP2PTy> {
  static constexpr bool value = true;
};
template <>
struct CuptiActivityHasGraphId<CuptiActivityMemsetTy> {
  static constexpr bool value = true;
};
#elif CUDA_VERSION >= 11060  // CUDA 11.6
#define TF_CUPTI_HAS_CHANNEL_ID 1
using CuptiActivityKernelTy = CUpti_ActivityKernel7;
using CuptiActivityMemcpyTy = CUpti_ActivityMemcpy5;
using CuptiActivityMemcpyP2PTy = CUpti_ActivityMemcpyPtoP4;
using CuptiActivityMemsetTy = CUpti_ActivityMemset4;

template <>
struct CuptiActivityHasGraphId<CuptiActivityKernelTy> {
  static constexpr bool value = true;
};
template <>
struct CuptiActivityHasGraphId<CuptiActivityMemcpyTy> {
  static constexpr bool value = true;
};
template <>
struct CuptiActivityHasGraphId<CuptiActivityMemcpyP2PTy> {
  static constexpr bool value = true;
};
template <>
struct CuptiActivityHasGraphId<CuptiActivityMemsetTy> {
  static constexpr bool value = true;
};
#else
using CuptiActivityKernelTy = CUpti_ActivityKernel4;
using CuptiActivityMemcpyTy = CUpti_ActivityMemcpy;
using CuptiActivityMemcpyP2PTy = CUpti_ActivityMemcpy2;
using CuptiActivityMemsetTy = CUpti_ActivityMemset;
#endif

// TODO: (b/350105610), Using Cupti_ActivityGraphTrace2 for CUDA 12.3 and later
#if CUDA_VERSION >= 11070
using CuptiActivityGraphTraceTy = CUpti_ActivityGraphTrace;
#endif  // CUDA_VERSION >= 11070

#if CUDA_VERSION >= 8000
using CuptiActivityMarkerTy = CUpti_ActivityMarker2;
constexpr int kCuptiActivityMarkerVersion = 2;
#else
using CuptiActivityMarkerTy = CUpti_ActivityMarker;
constexpr int kCuptiActivityMarkerVersion = 1;
#endif  // CUDA_VERSION >= 11070

// Maps an OverheadKind enum to a const string.
const char *getActivityOverheadKindString(CUpti_ActivityOverheadKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
      return "COMPILER";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
      return "BUFFER_FLUSH";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
      return "INSTRUMENTATION";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
      return "RESOURCE";
    default:
      break;
  }
  return "<UNKNOWN>";
}

const char *getActivityUnifiedMemoryKindString(
    CUpti_ActivityUnifiedMemoryCounterKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
      return "UM_BYTES_TRANSFER_HTOD";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
      return "UM_BYTES_TRANSFER_DTOH";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
      return "UM_CPU_PAGE_FAULT";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
      return "UM_GPU_PAGE_FAULT";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING:
      return "UM_THRASHING";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING:
      return "UM_THROTTLING";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP:
      return "UM_REMOTE_MAP";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD:
      return "UM_BYTES_TRANSFER_DTOD";
    default:
      break;
  }
  return "<UNKNOWN>";
}

template <typename CuptiActivity>
void SetEventGraphId(CuptiTracerEvent &event,
                     const CuptiActivity *cupti_activity) {
  if constexpr (CuptiActivityHasGraphId<CuptiActivity>::value) {
    event.graph_id = cupti_activity->graphId;
  }
}

template <bool cupti_has_channel_id, typename CuptiActivityKernel>
void AddKernelActivityEvent(CuptiEventCollectorDelegate &collector,
                            const CuptiActivityKernel *kernel) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Kernel;
  event.source = CuptiTracerEventSource::Activity;
  event.name = kernel->name;
  event.start_time_ns = kernel->start;
  event.end_time_ns = kernel->end;
  event.device_id = kernel->deviceId;
  event.context_id = kernel->contextId;
  event.stream_id = kernel->streamId;
  event.correlation_id = kernel->correlationId;
  AnnotationMap::AnnotationInfo info =
      collector.annotation_map.LookUp(event.device_id, event.correlation_id);
  event.annotation = info.annotation;
  event.nvtx_range = info.nvtx_range;
  event.scope_range_id = info.scope_range_id;
  SetEventGraphId(event, kernel);
  event.kernel_info.registers_per_thread = kernel->registersPerThread;
  event.kernel_info.static_shared_memory_usage = kernel->staticSharedMemory;
  event.kernel_info.dynamic_shared_memory_usage = kernel->dynamicSharedMemory;
  event.kernel_info.block_x = kernel->blockX;
  event.kernel_info.block_y = kernel->blockY;
  event.kernel_info.block_z = kernel->blockZ;
  event.kernel_info.grid_x = kernel->gridX;
  event.kernel_info.grid_y = kernel->gridY;
  event.kernel_info.grid_z = kernel->gridZ;
  if constexpr (cupti_has_channel_id) {
    event.kernel_info.channel_id = kernel->channelID;
    event.kernel_info.channel_type = kernel->channelType;
  }
  collector.receive(std::move(event));
}

void AddGraphTraceActivityEvent(CuptiEventCollectorDelegate &collector,
                                CuptiActivityGraphTraceTy *graph_trace) {
  AnnotationMap::AnnotationInfo info = collector.annotation_map.LookUp(
      graph_trace->deviceId, graph_trace->correlationId);
  collector.receive(CuptiTracerEvent{
      /* .type = */ CuptiTracerEventType::CudaGraph,
      /* .source = */ CuptiTracerEventSource::Activity,
      /* .name = */ absl::StrCat("CudaGraphExec:", graph_trace->graphId),
      /* .annotation = */ info.annotation,
      /* .nvtx_range = */ info.nvtx_range,
      /* .start_time_ns = */ graph_trace->start,
      /* .end_time_ns = */ graph_trace->end,
      /* .device_id = */ graph_trace->deviceId,
      /* .correlation_id = */ graph_trace->correlationId,
      // This is device event where thread_id is meaningless, using its default
      // value kInvalidThreadId here.
      /* .thread_id = */ CuptiTracerEvent::kInvalidThreadId,
      /* .context_id = */ graph_trace->contextId,
      /* .stream_id = */ graph_trace->streamId,
      /* .graph_id = */ graph_trace->graphId,
      /* .scope_range_id = */ info.scope_range_id,
  });
}

template <int CuptiActivityMarkerVersion>
const char *GetActivityMarkerDomain(const CuptiActivityMarkerTy *marker_trace) {
  if constexpr (CuptiActivityMarkerVersion == 1) {
    return "";
  } else {
    return marker_trace->domain;
  }
}

void AddMarkerActivityEvent(CuptiEventCollectorDelegate &collector,
                            CuptiActivityMarkerTy *marker_trace) {
  // Currently only support thread marker (i.e., nvtx range push/pop)
  if (marker_trace->objectKind != CUPTI_ACTIVITY_OBJECT_THREAD) return;
  if (marker_trace->flags == CUPTI_ACTIVITY_FLAG_MARKER_START) {
    collector.receive(CuptiTracerEvent{
        /* .type = */ CuptiTracerEventType::ThreadMarkerStart,
        /* .source = */ CuptiTracerEventSource::Activity,
        /* .name = */ marker_trace->name,
        /* .annotation = */ "",
        /* .nvtx_range = */
        GetActivityMarkerDomain<kCuptiActivityMarkerVersion>(marker_trace),
        /* .start_time_ns = */ marker_trace->timestamp,
        /* .end_time_ns = */ marker_trace->timestamp,
        /* .device_id = */ 0,
        /* .correlation_id = */ 0,
        /* .thread_id = */ marker_trace->objectId.pt.threadId,
        /* .context_id = */ 0,
        /* .stream_id = */ 0,
        /* .graph_id = */ marker_trace->id,
    });
  } else if (marker_trace->flags == CUPTI_ACTIVITY_FLAG_MARKER_END) {
    collector.receive(CuptiTracerEvent{
        /* .type = */ CuptiTracerEventType::ThreadMarkerEnd,
        /* .source = */ CuptiTracerEventSource::Activity,
        /* .name = */ "",
        /* .annotation = */ "",
        /* .nvtx_range = */ "",
        /* .start_time_ns = */ marker_trace->timestamp,
        /* .end_time_ns = */ marker_trace->timestamp,
        /* .device_id = */ 0,
        /* .correlation_id = */ 0,
        /* .thread_id = */ marker_trace->objectId.pt.threadId,
        /* .context_id = */ 0,
        /* .stream_id = */ 0,
        /* .graph_id = */ marker_trace->id,
    });
  }
}

void AddMemcpyActivityEvent(CuptiEventCollectorDelegate &collector,
                            const CuptiActivityMemcpyTy *memcpy) {
  CuptiTracerEvent event{};
  switch (memcpy->copyKind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      event.type = CuptiTracerEventType::MemcpyH2D;
      event.name = "MemcpyH2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      event.type = CuptiTracerEventType::MemcpyD2H;
      event.name = "MemcpyD2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      event.type = CuptiTracerEventType::MemcpyD2D;
      event.name = "MemcpyD2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      event.type = CuptiTracerEventType::MemcpyP2P;
      event.name = "MemcpyP2P";
      break;
    default:
      event.type = CuptiTracerEventType::MemcpyOther;
      event.name = "MemcpyOther";
      break;
  }

  event.source = CuptiTracerEventSource::Activity;
  event.start_time_ns = memcpy->start;
  event.end_time_ns = memcpy->end;
  event.device_id = memcpy->deviceId;
  event.context_id = memcpy->contextId;
  event.stream_id = memcpy->streamId;
  event.correlation_id = memcpy->correlationId;
  AnnotationMap::AnnotationInfo info =
      collector.annotation_map.LookUp(event.device_id, event.correlation_id);
  event.annotation = info.annotation;
  event.nvtx_range = info.nvtx_range;
  event.scope_range_id = info.scope_range_id;
  SetEventGraphId(event, memcpy);
  event.memcpy_info.copy_kind = memcpy->copyKind;
  event.memcpy_info.num_bytes = memcpy->bytes;
  event.memcpy_info.destination = memcpy->deviceId;
  event.memcpy_info.async = memcpy->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC;
  event.memcpy_info.src_mem_kind = memcpy->srcKind;
  event.memcpy_info.dst_mem_kind = memcpy->dstKind;
#if TF_CUPTI_HAS_CHANNEL_ID
  event.memcpy_info.channel_id = memcpy->channelID;
  event.memcpy_info.channel_type = memcpy->channelType;
#endif
  collector.receive(std::move(event));
}

// Invokes callback upon peer-2-peer memcpy between different GPU devices.
void AddMemcpyP2PActivityEvent(CuptiEventCollectorDelegate &collector,
                               const CuptiActivityMemcpyP2PTy *memcpy) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::MemcpyP2P;
  event.name = "MemcpyP2P";
  event.source = CuptiTracerEventSource::Activity;
  event.start_time_ns = memcpy->start;
  event.end_time_ns = memcpy->end;
  event.device_id = memcpy->srcDeviceId;
  event.context_id = memcpy->contextId;
  event.stream_id = memcpy->streamId;
  event.correlation_id = memcpy->correlationId;
  AnnotationMap::AnnotationInfo info =
      collector.annotation_map.LookUp(event.device_id, event.correlation_id);
  event.annotation = info.annotation;
  event.nvtx_range = info.nvtx_range;
  event.scope_range_id = info.scope_range_id;
  SetEventGraphId(event, memcpy);
  event.memcpy_info.copy_kind = CUPTI_ACTIVITY_MEMCPY_KIND_PTOP;
  event.memcpy_info.num_bytes = memcpy->bytes;
  event.memcpy_info.destination = memcpy->dstDeviceId;
  event.memcpy_info.async = memcpy->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC;
  event.memcpy_info.src_mem_kind = memcpy->srcKind;
  event.memcpy_info.dst_mem_kind = memcpy->dstKind;
#if TF_CUPTI_HAS_CHANNEL_ID
  event.memcpy_info.channel_id = memcpy->channelID;
  event.memcpy_info.channel_type = memcpy->channelType;
#endif
  collector.receive(std::move(event));
}

void AddCuptiOverheadActivityEvent(CuptiEventCollectorDelegate &collector,
                                   const CUpti_ActivityOverhead *overhead) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Overhead;
  event.name = getActivityOverheadKindString(overhead->overheadKind);
  event.source = CuptiTracerEventSource::Activity;
  event.start_time_ns = overhead->start;
  event.end_time_ns = overhead->end;
  // If the overhead is not related to a device, we assign it to device 0.
  event.device_id = 0;
  // NOTE: no correlation id.
  switch (overhead->objectKind) {
    case CUPTI_ACTIVITY_OBJECT_UNKNOWN:
      // Don't know how to deal with such activities because of we need either
      // attribute it to a GPU stream or a CPU thread.
      return;

    case CUPTI_ACTIVITY_OBJECT_THREAD:
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
      event.thread_id = overhead->objectId.pt.threadId;
      break;
    case CUPTI_ACTIVITY_OBJECT_STREAM:
      event.stream_id = overhead->objectId.dcs.streamId;
      TF_FALLTHROUGH_INTENDED;
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
      event.device_id = overhead->objectId.dcs.deviceId;
      break;
    default:
      LOG(ERROR) << "Unexpected object kind: " << overhead->objectKind;
      return;
  }
  collector.receive(std::move(event));
}

void AddUnifiedMemoryActivityEvent(
    CuptiEventCollectorDelegate &collector,
    const CUpti_ActivityUnifiedMemoryCounter2 *record) {
  VLOG(3) << "Cuda Unified Memory Activity, kind: " << record->counterKind
          << " src: " << record->srcId << " dst: " << record->dstId;
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::UnifiedMemory;
  event.name = getActivityUnifiedMemoryKindString(record->counterKind);
  event.source = CuptiTracerEventSource::Activity;
  event.start_time_ns = record->start;
  if (record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT ||
      record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING ||
      record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP ||
      record->end <= record->start) {
    // If the end time is not valid, trim it so that it can be shown on the UI.
    event.end_time_ns = record->start + 1;
  } else {
    event.end_time_ns = record->end;
  }
  event.device_id = record->srcId;
  // NOTE: not context id and correlation id.

  // For visualization purpose, we assign a pseudo stream id for each
  // record->counterKind of unified memory related events.
  constexpr int kPseudoStreamId = 0x10000000;
  event.stream_id = kPseudoStreamId + record->counterKind;
  event.memcpy_info.copy_kind = CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN;
  // Check whether the activity is byte transfer.
  if (record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD ||
      record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH ||
      record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD) {
    event.memcpy_info.num_bytes = record->value;
  } else {
    event.memcpy_info.num_bytes = 0;
  }
  event.memcpy_info.destination = record->dstId;
  event.memcpy_info.async = false;
  collector.receive(std::move(event));
}

void AddMemoryActivityEvent(CuptiEventCollectorDelegate &collector,
                            const CUpti_ActivityMemory *memory) {
  CuptiTracerEvent event{};
  event.name = absl::StrCat("Memory ", GetMemoryKindName(memory->memoryKind));
  event.type = CuptiTracerEventType::MemoryResidency;
  event.source = CuptiTracerEventSource::Activity;
  event.start_time_ns = memory->start;
  event.end_time_ns = std::max(memory->end, memory->start + 1);
  event.device_id = memory->deviceId;
  event.context_id = memory->contextId;
  // Assign to default stream (0) so that event is included during Flush().
  event.stream_id = 0;
  event.memory_residency_info.num_bytes = memory->bytes;
  event.memory_residency_info.mem_kind = memory->memoryKind;
  event.memory_residency_info.address = memory->address;
  VLOG(5) << "Cuda activity " << event.name
          << " addr: " << reinterpret_cast<void *>(memory->address)
          << " bytes: " << memory->bytes;
  collector.receive(std::move(event));
}

void AddMemsetActivityEvent(CuptiEventCollectorDelegate &collector,
                            const CuptiActivityMemsetTy *memset) {
  auto mem_kind = memset->memoryKind;
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Memset;
  event.source = CuptiTracerEventSource::Activity;
  event.name = absl::StrCat("Memset ", mem_kind);
  event.start_time_ns = memset->start;
  event.end_time_ns = std::max(memset->end, memset->start + 1);
  event.device_id = memset->deviceId;
  event.correlation_id = memset->correlationId;
  event.context_id = memset->contextId;
  event.stream_id = memset->streamId;
  SetEventGraphId(event, memset);
  event.memset_info.num_bytes = memset->bytes;
  event.memset_info.mem_kind = mem_kind;
  event.memset_info.async = (memset->flags & CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC);
#if TF_CUPTI_HAS_CHANNEL_ID
  event.memset_info.channel_id = memset->channelID;
  event.memset_info.channel_type = memset->channelType;
#endif
  VLOG(5) << "Cuda activity " << event.name << " bytes: " << memset->bytes
          << " async: " << event.memset_info.async;
  collector.receive(std::move(event));
}

void AddSynchronizationActivityEvent(
    CuptiEventCollectorDelegate &collector,
    const CUpti_ActivitySynchronization *sync) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Generic;
  event.source = CuptiTracerEventSource::Activity;
  switch (sync->type) {
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE:
      event.name = "cuEventSynchronize";
      break;
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT:
      event.name = "cuStreamWaitEvent";
      break;
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE:
      event.name = "cuStreamSynchronize";
      break;
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE:
      event.name = "cuCtxSynchronize";
      break;
    default:
      event.name = "unknown synchronization event";
      break;
  }
  event.start_time_ns = sync->start;
  event.end_time_ns = std::max(sync->end, sync->start + 1);
  event.correlation_id = sync->correlationId;
  event.context_id = sync->contextId;
  VLOG(5) << "Cuda activity " << event.name;
  collector.receive(std::move(event));
}

static absl::Status ConvertActivityBuffer(
    CuptiEventCollectorDelegate &collector, uint8_t *buffer, const size_t size,
    const size_t max_activity_event_count, size_t &total_activity_event_count,
    size_t &dropped_activity_event_count) {
  CuptiInterface *cupti_interface = GetCuptiInterface();
  CUpti_Activity *record = nullptr;
  while (true) {
    CUptiResult status =
        cupti_interface->ActivityGetNextRecord(buffer, size, &record);
    if (status == CUPTI_SUCCESS) {
      if (total_activity_event_count >= max_activity_event_count) {
        dropped_activity_event_count++;
        continue;
      }
      total_activity_event_count++;
      switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:  // sequential
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
          AddKernelActivityEvent<TF_CUPTI_HAS_CHANNEL_ID>(
              collector, reinterpret_cast<CuptiActivityKernelTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_CDP_KERNEL:
          AddKernelActivityEvent<false>(
              collector, reinterpret_cast<CUpti_ActivityCdpKernel *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_MEMCPY:
          AddMemcpyActivityEvent(
              collector, reinterpret_cast<CuptiActivityMemcpyTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_MEMCPY2:
          AddMemcpyP2PActivityEvent(
              collector, reinterpret_cast<CuptiActivityMemcpyP2PTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_OVERHEAD:
          AddCuptiOverheadActivityEvent(
              collector, reinterpret_cast<CUpti_ActivityOverhead *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
          AddUnifiedMemoryActivityEvent(
              collector,
              reinterpret_cast<CUpti_ActivityUnifiedMemoryCounter2 *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_MEMORY: {
          AddMemoryActivityEvent(
              collector, reinterpret_cast<CUpti_ActivityMemory *>(record));
        } break;
        case CUPTI_ACTIVITY_KIND_MEMSET:
          AddMemsetActivityEvent(
              collector, reinterpret_cast<CuptiActivityMemsetTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
          AddSynchronizationActivityEvent(
              collector,
              reinterpret_cast<CUpti_ActivitySynchronization *>(record));
          break;
#if CUDA_VERSION >= 11070
        case CUPTI_ACTIVITY_KIND_GRAPH_TRACE:
          AddGraphTraceActivityEvent(
              collector, reinterpret_cast<CuptiActivityGraphTraceTy *>(record));
          break;
#endif
        case CUPTI_ACTIVITY_KIND_MARKER:
          AddMarkerActivityEvent(
              collector, reinterpret_cast<CuptiActivityMarkerTy *>(record));
          break;
        default:
          VLOG(3) << "Activity type " << record->kind << " is not supported.";
          break;
      }
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      // Normal, just reach the end of the valid activity events.
      break;
    } else if (status == CUPTI_ERROR_INVALID_KIND) {
      VLOG(3) << "CUPTI parse ACTIVITY buffer got CUPTI_ERROR_INVALID_KIND";
      break;
    } else {
      LOG(WARNING) << "CUPTI parse ACTIVITY buffer error: " << status;
      return absl::Status(StatusCode::kInternal,
                          "Parse cupti activity buffer error.");
    }
  }
  VLOG(3) << "CUPTI tracer post-process one ACTIVITY buffer of size: " << size
          << ", total events count:" << total_activity_event_count;
  return absl::OkStatus();
}

}  // namespace

absl::string_view StringDeduper::Dedup(absl::string_view str,
                                       size_t max_unique_count) {
  if (str.empty()) return absl::string_view();
  auto it = strings_.find(str);
  if (it != strings_.end()) return *it;
  if (max_unique_count == 0 || strings_.size() < max_unique_count)
    return *strings_.emplace(str).first;
  return absl::string_view();
}

void AnnotationMap::Add(uint32_t device_id, uint32_t correlation_id,
                        const absl::string_view annotation,
                        const absl::string_view nvtx_range,
                        int64_t scope_range_id) {
  if (annotation.empty() && nvtx_range.empty()) return;
  VLOG(3) << "Add annotation: device_id: " << device_id
          << " correlation_id: " << correlation_id
          << " annotation: " << annotation;
  if (device_id >= per_device_map_.size()) return;
  auto &per_device_map = per_device_map_[device_id];
  if (per_device_map.annotation_deduper.Size() < max_size_) {
    AnnotationInfo info;
    info.annotation = per_device_map.annotation_deduper.Dedup(annotation);
    info.nvtx_range = per_device_map.nvtx_range_deduper.Dedup(nvtx_range);
    info.scope_range_id = scope_range_id;
    per_device_map.correlation_map.emplace(correlation_id, info);
  }
}

AnnotationMap::AnnotationInfo AnnotationMap::LookUp(
    uint32_t device_id, uint32_t correlation_id) const {
  if (device_id >= per_device_map_.size()) return AnnotationInfo();
  auto &per_device_map = per_device_map_[device_id];
  auto it = per_device_map.correlation_map.find(correlation_id);
  return it != per_device_map.correlation_map.end() ? it->second
                                                    : AnnotationInfo();
}

CuptiActivityBufferManager::ActivityBufferAndSize::ActivityBufferAndSize(
    uint8_t *p, size_t sz)
    : buffer(p,
             [](uint8_t *p) {
               if (p != nullptr) tsl::port::AlignedFree(p);
             }),
      size(sz) {}

void AddActivityBufferListEventsTo(
    CuptiEventCollectorDelegate &collector,
    std::list<CuptiActivityBufferManager::ActivityBufferAndSize> &buffer_list,
    size_t max_activity_event_count, size_t &dropped_activity_event_count) {
  dropped_activity_event_count = 0;
  size_t total_activity_event_count = 0;
  while (!buffer_list.empty()) {
    CuptiActivityBufferManager::ActivityBufferAndSize buffer_and_size(
        std::move(buffer_list.front()));
    buffer_list.pop_front();
    ConvertActivityBuffer(collector, buffer_and_size.buffer.get(),
                          buffer_and_size.size, max_activity_event_count,
                          total_activity_event_count,
                          dropped_activity_event_count)
        .IgnoreError();
  }
}

CallbackAnnotationsAndEvents::CallbackAnnotationsAndEvents(
    CallbackAnnotationsAndEvents &&another) {
  *this = std::move(another);
}

CallbackAnnotationsAndEvents &CallbackAnnotationsAndEvents::operator=(
    CallbackAnnotationsAndEvents &&another) {
  annotations_ = std::move(another.annotations_);
  nvtx_ranges_ = std::move(another.nvtx_ranges_);
  num_dropped_events_ = another.num_dropped_events_;
  event_queue_ = std::move(another.event_queue_);
  scope_range_id_tree_ = std::move(another.scope_range_id_tree_);
  another.Clear();
  return *this;
}

void CallbackAnnotationsAndEvents::Clear() {
  annotations_.Clear();
  nvtx_ranges_.Clear();
  num_dropped_events_ = 0;
  event_queue_.Clear();
  scope_range_id_tree_.clear();
}

}  // namespace profiler
}  // namespace xla
