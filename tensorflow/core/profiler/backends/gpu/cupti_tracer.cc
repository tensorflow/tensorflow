/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/backends/gpu/cupti_tracer.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/generated_nvtx_meta.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/backends/cpu/annotation_stack.h"
#include "tensorflow/core/profiler/backends/gpu/cupti_collector.h"
#include "tensorflow/core/profiler/backends/gpu/nvtx_utils.h"

namespace tensorflow {
namespace profiler {

namespace {

// CUPTI from CUDA 11.6 adds information about the hardware channel that ops
// run on; this makes its way into the channel_id and channel_type fields in the
// structs we export.
//
// Define some type aliases so we can access the hardware channel id if it's
// available.
#if CUDA_VERSION >= 11060  // CUDA 11.6
#define TF_CUPTI_HAS_CHANNEL_ID 1
using CuptiActivityKernelTy = CUpti_ActivityKernel7;
using CuptiActivityMemcpyTy = CUpti_ActivityMemcpy5;
using CuptiActivityMemcpyP2PTy = CUpti_ActivityMemcpyPtoP4;
using CuptiActivityMemsetTy = CUpti_ActivityMemset4;
#else
using CuptiActivityKernelTy = CUpti_ActivityKernel4;
using CuptiActivityMemcpyTy = CUpti_ActivityMemcpy;
using CuptiActivityMemcpyP2PTy = CUpti_ActivityMemcpy2;
using CuptiActivityMemsetTy = CUpti_ActivityMemset;
#endif

static thread_local int internalCuCall = 0;

// Temporary disable cupti api tracing for this thread during the life scope of
// this class. Used for the API calls that initiated by us.
class CuptiApiTracingDisabler {
 public:
  CuptiApiTracingDisabler() { internalCuCall++; }
  ~CuptiApiTracingDisabler() { internalCuCall--; }
};

Status ToStatus(CUptiResult result) {
  if (result == CUPTI_SUCCESS) {
    return OkStatus();
  }
  const char *str = nullptr;
  cuptiGetResultString(result, &str);
  return errors::Unavailable("CUPTI error: ", str ? str : "<unknown>");
}

Status ToStatus(CUresult result) {
  if (result == CUDA_SUCCESS) {
    return OkStatus();
  }
  const char *str = nullptr;
  cuGetErrorName(result, &str);
  return errors::Unavailable("CUDA error: ", str ? str : "<unknown>");
}

inline void LogIfError(const Status &status) {
  if (status.ok()) return;
  LOG(ERROR) << status.error_message();
}

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

// CUPTI_ERROR_INSUFFICIENT_PRIVILEGES is introduced at CUDA 10.1.
#if CUDA_VERSION <= 10000
#define CUPTI_ERROR_INSUFFICIENT_PRIVILEGES 35
#endif

#define RETURN_IF_CUPTI_ERROR(expr)                                         \
  do {                                                                      \
    CUptiResult status = expr;                                              \
    if (ABSL_PREDICT_FALSE(status != CUPTI_SUCCESS)) {                      \
      const char *errstr = "";                                              \
      cupti_interface_->GetResultString(status, &errstr);                   \
      LOG(ERROR) << "function " << #expr << "failed with error " << errstr; \
      if (status == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) {                  \
        return errors::PermissionDenied("CUPTI need root access!");         \
      } else {                                                              \
        return errors::Internal("CUPTI call error", errstr);                \
      }                                                                     \
    }                                                                       \
  } while (false)

size_t Bytes2D(const CUDA_MEMCPY2D *p) { return p->Height * p->WidthInBytes; }

size_t Bytes3D(const CUDA_MEMCPY3D *p) {
  return p->Depth * p->Height * p->WidthInBytes;
}

template <typename CudaMemcpy>
CuptiTracerEventType MemcpyKind(const CudaMemcpy *p) {
  if (p->srcMemoryType == CU_MEMORYTYPE_HOST &&
      p->dstMemoryType == CU_MEMORYTYPE_DEVICE) {
    return CuptiTracerEventType::MemcpyH2D;
  }
  if (p->srcMemoryType == CU_MEMORYTYPE_DEVICE &&
      p->dstMemoryType == CU_MEMORYTYPE_HOST) {
    return CuptiTracerEventType::MemcpyD2H;
  }
  if (p->srcMemoryType == CU_MEMORYTYPE_DEVICE &&
      p->dstMemoryType == CU_MEMORYTYPE_DEVICE) {
    return CuptiTracerEventType::MemcpyD2D;
  }
  return CuptiTracerEventType::Unsupported;
}

std::tuple<size_t /*bytes*/, CuptiTracerEventType, bool /*async*/>
DecodeDriverMemcpy(CUpti_CallbackId cbid, const void *params) {
  switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2: {
      const auto *p = reinterpret_cast<const cuMemcpyHtoD_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyH2D,
                             false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2: {
      const auto *p =
          reinterpret_cast<const cuMemcpyHtoDAsync_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyH2D,
                             true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2: {
      const auto *p = reinterpret_cast<const cuMemcpyDtoH_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyD2H,
                             false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2: {
      const auto *p =
          reinterpret_cast<const cuMemcpyDtoHAsync_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyD2H,
                             true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2: {
      const auto *p = reinterpret_cast<const cuMemcpyDtoD_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyD2D,
                             false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2: {
      const auto *p =
          reinterpret_cast<const cuMemcpyDtoDAsync_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyD2D,
                             true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy: {
      const auto *p = reinterpret_cast<const cuMemcpy_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyOther,
                             false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync: {
      const auto *p = reinterpret_cast<const cuMemcpyAsync_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyOther,
                             true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2: {
      const auto *p = reinterpret_cast<const cuMemcpy2D_v2_params *>(params);
      return std::make_tuple(Bytes2D(p->pCopy), MemcpyKind(p->pCopy), false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2: {
      const auto *p =
          reinterpret_cast<const cuMemcpy2DAsync_v2_params *>(params);
      return std::make_tuple(Bytes2D(p->pCopy), MemcpyKind(p->pCopy), true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2: {
      const auto *p = reinterpret_cast<const cuMemcpy3D_v2_params *>(params);
      return std::make_tuple(Bytes3D(p->pCopy), MemcpyKind(p->pCopy), true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2: {
      const auto *p =
          reinterpret_cast<const cuMemcpy3DAsync_v2_params *>(params);
      return std::make_tuple(Bytes3D(p->pCopy), MemcpyKind(p->pCopy), true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer: {
      const auto *p2p_params =
          reinterpret_cast<const cuMemcpyPeer_params *>(params);
      return std::make_tuple(p2p_params->ByteCount,
                             CuptiTracerEventType::MemcpyP2P, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync: {
      const auto *p2p_params =
          reinterpret_cast<const cuMemcpyPeerAsync_params *>(params);
      return std::make_tuple(p2p_params->ByteCount,
                             CuptiTracerEventType::MemcpyP2P, true);
    }
    default: {
      LOG(ERROR) << "Unsupported memcpy activity observed: " << cbid;
      return std::make_tuple(0, CuptiTracerEventType::Unsupported, false);
    }
  }
}

std::tuple<size_t /*bytes*/, CuptiTracerEventType, bool /*async*/>
DecodeDriverMemset(CUpti_CallbackId cbid, const void *params) {
  switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD8_v2_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD16_v2_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD32_v2_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD2D8_v2_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD2D16_v2_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD2D32_v2_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async: {
      const auto *p = reinterpret_cast<const cuMemsetD8Async_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async: {
      const auto *p = reinterpret_cast<const cuMemsetD16Async_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async: {
      const auto *p = reinterpret_cast<const cuMemsetD32Async_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async: {
      const auto *p =
          reinterpret_cast<const cuMemsetD2D8Async_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async: {
      const auto *p =
          reinterpret_cast<const cuMemsetD2D16Async_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async: {
      const auto *p =
          reinterpret_cast<const cuMemsetD2D32Async_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, true);
    }
    default: {
      LOG(ERROR) << "Unsupported memset activity observed: " << cbid;
      return std::make_tuple(0, CuptiTracerEventType::Unsupported, false);
    }
  }
}

// Cupti callback corresponding to a driver or runtime API. This global function
// is invoked twice for each API: at entry and at exit. The cbdata
// parameter is guaranteed by Cupti to be thread-safe. Most invocations are
// dropped to the floor and entry/exit is tracked for the APIs we deem
// performance-relevant.
void CUPTIAPI ApiCallback(void *user_data, CUpti_CallbackDomain domain,
                          CUpti_CallbackId cbid,
                          const CUpti_CallbackData *cbdata) {
  CuptiTracer *tracer = reinterpret_cast<CuptiTracer *>(user_data);
  tracer->HandleCallback(domain, cbid, cbdata).IgnoreError();
}

// Callback which is invoked when an empty buffer is requested by CUPTI.
// Allocates an empty aligned-memory buffer. The buffer is used by CUPTI as a
// ring buffer where device maintains activity profiles that have been
// collected.
void CUPTIAPI RequestCuptiActivityBuffer(uint8_t **buffer, size_t *size,
                                         size_t *maxNumRecords) {
  CuptiTracer::GetCuptiTracerSingleton()->RequestActivityBuffer(buffer, size);
  VLOG(3) << "Requested CUPTI Buffer, buffer=" << std::hex
          << reinterpret_cast<uintptr_t>(*buffer) << std::dec
          << " size=" << *size;
  // Request CUPTI to fill as many records as possible in the buffer.
  *maxNumRecords = 0;
}

// Callback which is invoked when a buffer containing activity records is
// available from CUPTI. Processes the buffer after reading activity records
// from it.
void CUPTIAPI ProcessCuptiActivityBuffer(CUcontext context, uint32_t stream_id,
                                         uint8_t *buffer, size_t size,
                                         size_t valid_size) {
  VLOG(3) << "Processing CUPTI Buffer, buffer:" << std::hex
          << reinterpret_cast<uintptr_t>(buffer) << std::dec
          << " size: " << size << " valid_size: " << valid_size;
  VLOG(3) << "Activity profile for stream " << stream_id;

  Status status = CuptiTracer::GetCuptiTracerSingleton()->ProcessActivityBuffer(
      context, stream_id, buffer, valid_size);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
}

void AddKernelEventUponApiExit(CuptiTraceCollector *collector, uint32 device_id,
                               const CUpti_CallbackData *cbdata,
                               uint64 start_time, uint64 end_time) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Kernel;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->symbolName ? cbdata->symbolName : cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  VLOG(3) << "Cuda Kernel launch API exit. name=" << event.name;
  collector->AddEvent(std::move(event));
}

// Performs the actual callback for both normal and P2P memcpy operations.
CuptiTracerEvent PopulateMemcpyCallbackEvent(
    CuptiTracerEventType type, const CUpti_CallbackData *cbdata,
    size_t num_bytes, uint32 src_device, uint32 dst_device, bool async,
    uint64 start_time, uint64 end_time) {
  CuptiTracerEvent event{};
  event.type = type;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = src_device;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memcpy_info.num_bytes = num_bytes;
  event.memcpy_info.destination = dst_device;
  event.memcpy_info.async = async;
  // These are not populated during callback for API activities.
  event.memcpy_info.copy_kind = CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN;
  event.memcpy_info.dst_mem_kind = CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN;
  event.memcpy_info.src_mem_kind = CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN;
  return event;
}

void AddNormalMemcpyEventUponApiExit(CuptiTraceCollector *collector,
                                     uint32 device_id, CUpti_CallbackId cbid,
                                     const CUpti_CallbackData *cbdata,
                                     uint64 start_time, uint64 end_time) {
  size_t num_bytes;
  CuptiTracerEventType type;
  bool async;
  std::tie(num_bytes, type, async) =
      DecodeDriverMemcpy(cbid, cbdata->functionParams);

  VLOG(3) << "Cuda Memcpy API exit. sz=" << num_bytes;
  CuptiTracerEvent event =
      PopulateMemcpyCallbackEvent(type, cbdata, num_bytes, device_id, device_id,
                                  async, start_time, end_time);
  collector->AddEvent(std::move(event));
}

void AddCuMemsetEventUponApiExit(CuptiTraceCollector *collector,
                                 uint32 device_id, CUpti_CallbackId cbid,
                                 const CUpti_CallbackData *cbdata,
                                 uint64 start_time, uint64 end_time) {
  // We are casting all variants of cuMemset to cuMemsetD8 for accessing the
  // first member attribute, a CUdeviceptr.
  const auto *params =
      static_cast<const cuMemsetD8_v2_params *>(cbdata->functionParams);
  size_t num_bytes;
  bool async;
  CuptiTracerEventType type;
  std::tie(num_bytes, type, async) =
      DecodeDriverMemset(cbid, cbdata->functionParams);

  CuptiTracerEvent event{};
  event.type = type;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memset_info.num_bytes = num_bytes;
  // memset_info.kind cannot be determined from API.
  event.memset_info.async = async;
  VLOG(3) << "Cuda Memset API exit."
          << " dptr=" << reinterpret_cast<void *>(params->dstDevice)
          << " sz=" << num_bytes;
  collector->AddEvent(std::move(event));
}

void AddP2PMemcpyEventUponApiExit(CuptiTraceCollector *collector,
                                  CuptiInterface *cupti_interface,
                                  uint32 device_id, CUpti_CallbackId cbid,
                                  const CUpti_CallbackData *cbdata,
                                  uint64 start_time, uint64 end_time) {
  size_t num_bytes;
  CuptiTracerEventType type;
  bool async;
  std::tie(num_bytes, type, async) =
      DecodeDriverMemcpy(cbid, cbdata->functionParams);

  uint32 dst_device = -1, src_device = -1;
  const auto *p2p_params =
      static_cast<const cuMemcpyPeer_params *>(cbdata->functionParams);
  cupti_interface->GetDeviceId(p2p_params->srcContext, &src_device);
  cupti_interface->GetDeviceId(p2p_params->dstContext, &dst_device);
  VLOG(3) << "Cuda P2P Memcpy API exit, src: " << src_device
          << " dst: " << dst_device << " size:" << num_bytes;
  CuptiTracerEvent event =
      PopulateMemcpyCallbackEvent(type, cbdata, num_bytes, src_device,
                                  dst_device, async, start_time, end_time);
  collector->AddEvent(std::move(event));
}

void AddCuMemAllocEventUponApiExit(CuptiTraceCollector *collector,
                                   uint32 device_id, CUpti_CallbackId cbid,
                                   const CUpti_CallbackData *cbdata,
                                   uint64 start_time, uint64 end_time) {
  const auto *params =
      static_cast<const cuMemAlloc_v2_params *>(cbdata->functionParams);
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::MemoryAlloc;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memalloc_info.num_bytes = params->bytesize;
  VLOG(3) << "Cuda MemAlloc API exit."
          << " dptr=" << reinterpret_cast<void *>(*params->dptr)
          << " sz=" << params->bytesize;
  collector->AddEvent(std::move(event));
}

void AddCuMemAllocPitchEventUponApiExit(CuptiTraceCollector *collector,
                                        uint32 device_id, CUpti_CallbackId cbid,
                                        const CUpti_CallbackData *cbdata,
                                        uint64 start_time, uint64 end_time) {
  const auto *params =
      static_cast<const cuMemAllocPitch_v2_params *>(cbdata->functionParams);
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::MemoryAlloc;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  const size_t size_in_bytes = *params->pPitch * params->Height;
  event.memalloc_info.num_bytes = size_in_bytes;
  VLOG(3) << "Cuda MemAllocPitch API exit."
          << " dptr=" << reinterpret_cast<void *>(*params->dptr)
          << " sz=" << size_in_bytes;
  collector->AddEvent(std::move(event));
}

void AddCuMemFreeEventUponApiExit(CuptiTraceCollector *collector,
                                  uint32 device_id, CUpti_CallbackId cbid,
                                  const CUpti_CallbackData *cbdata,
                                  uint64 start_time, uint64 end_time) {
  const auto *params =
      static_cast<const cuMemFree_v2_params *>(cbdata->functionParams);
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::MemoryFree;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  VLOG(3) << "Cuda MemFree API exit."
          << " dptr=" << reinterpret_cast<void *>(params->dptr);
  collector->AddEvent(std::move(event));
}

void AddGenericEventUponApiExit(CuptiTraceCollector *collector,
                                uint32 device_id, CUpti_CallbackId cbid,
                                const CUpti_CallbackData *cbdata,
                                uint64 start_time, uint64 end_time) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Generic;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  VLOG(3) << "Observed generic API exit."
          << " name=" << cbdata->functionName;
  collector->AddEvent(std::move(event));
}

void AddKernelActivityEvent(CuptiTraceCollector *collector,
                            const CuptiActivityKernelTy *kernel) {
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
  AnnotationMap::AnnotationInfo info = collector->annotation_map()->LookUp(
      event.device_id, event.correlation_id);
  event.annotation = info.annotation;
  event.nvtx_range = info.nvtx_range;
  event.kernel_info.registers_per_thread = kernel->registersPerThread;
  event.kernel_info.static_shared_memory_usage = kernel->staticSharedMemory;
  event.kernel_info.dynamic_shared_memory_usage = kernel->dynamicSharedMemory;
  event.kernel_info.block_x = kernel->blockX;
  event.kernel_info.block_y = kernel->blockY;
  event.kernel_info.block_z = kernel->blockZ;
  event.kernel_info.grid_x = kernel->gridX;
  event.kernel_info.grid_y = kernel->gridY;
  event.kernel_info.grid_z = kernel->gridZ;
#if TF_CUPTI_HAS_CHANNEL_ID
  event.kernel_info.channel_id = kernel->channelID;
  event.kernel_info.channel_type = kernel->channelType;
#endif
  collector->AddEvent(std::move(event));
}

void AddMemcpyActivityEvent(CuptiTraceCollector *collector,
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
  AnnotationMap::AnnotationInfo info = collector->annotation_map()->LookUp(
      event.device_id, event.correlation_id);
  event.annotation = info.annotation;
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
  collector->AddEvent(std::move(event));
}

// Invokes callback upon peer-2-peer memcpy between different GPU devices.
void AddMemcpyP2PActivityEvent(CuptiTraceCollector *collector,
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
  AnnotationMap::AnnotationInfo info = collector->annotation_map()->LookUp(
      event.device_id, event.correlation_id);
  event.annotation = info.annotation;
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
  collector->AddEvent(std::move(event));
}

void AddCuptiOverheadActivityEvent(CuptiTraceCollector *collector,
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
  collector->AddEvent(std::move(event));
}

void AddUnifiedMemoryActivityEvent(
    CuptiTraceCollector *collector,
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
  collector->AddEvent(std::move(event));
}

void AddMemoryActivityEvent(CuptiTraceCollector *collector,
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
  collector->AddEvent(std::move(event));
}

void AddMemsetActivityEvent(CuptiTraceCollector *collector,
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
  event.memset_info.num_bytes = memset->bytes;
  event.memset_info.mem_kind = mem_kind;
  event.memset_info.async = (memset->flags & CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC);
#if TF_CUPTI_HAS_CHANNEL_ID
  event.memset_info.channel_id = memset->channelID;
  event.memset_info.channel_type = memset->channelType;
#endif
  VLOG(5) << "Cuda activity " << event.name << " bytes: " << memset->bytes
          << " async: " << event.memset_info.async;
  collector->AddEvent(std::move(event));
}

void AddSynchronizationActivityEvent(
    CuptiTraceCollector *collector, const CUpti_ActivitySynchronization *sync) {
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
  collector->AddEvent(std::move(event));
}

// This hook uses cupti activity api to measure device side activities.
class CuptiDriverApiHookWithActivityApi : public CuptiDriverApiHook {
 public:
  CuptiDriverApiHookWithActivityApi(const CuptiTracerOptions &option,
                                    CuptiInterface *cupti_interface,
                                    CuptiTraceCollector *collector)
      : option_(option),
        cupti_interface_(cupti_interface),
        collector_(collector) {}

  Status OnDriverApiEnter(int device_id, CUpti_CallbackDomain domain,
                          CUpti_CallbackId cbid,
                          const CUpti_CallbackData *cbdata) override {
    // Stash away the current Cupti timestamp into cbdata.
    *cbdata->correlationData =
        option_.required_callback_api_events ? CuptiTracer::GetTimestamp() : 0;
    return OkStatus();
  }
  Status OnDriverApiExit(int device_id, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbid,
                         const CUpti_CallbackData *cbdata) override {
    // If we are not collecting CPU events from Callback API, we can return now.
    if (!option_.required_callback_api_events) {
      return OkStatus();
    }

    // Grab timestamp for API exit. API entry timestamp saved in cbdata.
    uint64 end_tsc = CuptiTracer::GetTimestamp();
    uint64 start_tsc = *cbdata->correlationData;
    TrackContext(cbid, cbdata->context);
    return AddDriverApiCallbackEvent(collector_, cupti_interface_, device_id,
                                     start_tsc, end_tsc, domain, cbid, cbdata);
  }
  Status SyncAndFlush() override {
    if (option_.sync_devices_before_stop) {
      CuptiApiTracingDisabler disabler;
      absl::MutexLock lock(&mutex_);
      for (auto &ctx : contexts_) {
        cuCtxPushCurrent(ctx);
        cuCtxSynchronize();  // Ignore error here for best effort.
        CUcontext current;
        cuCtxPopCurrent(&current);
      }
    }
    return OkStatus();
  }

 private:
  void TrackContext(CUpti_CallbackId cbid, CUcontext ctx) {
    if (!option_.sync_devices_before_stop) return;
    if (ctx == nullptr) return;
    absl::MutexLock lock(&mutex_);
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy_v2 ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy) {
      contexts_.erase(ctx);
    } else {
      contexts_.emplace(ctx);
    }
  }

  const CuptiTracerOptions option_;
  CuptiInterface *cupti_interface_;
  CuptiTraceCollector *collector_;
  absl::Mutex mutex_;
  absl::flat_hash_set<CUcontext> contexts_ TF_GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(CuptiDriverApiHookWithActivityApi);
};

struct KernelRecord {
  const char *kernel_name;
  // TODO(csigg): cuStreamGetCtx introduced in CUDA 9.2 would allow us to only
  // record the stream and infer the context during collection.
  CUcontext context;
  CUstream stream;
  uint32 correlation_id;
  CUevent start_event;
  CUevent stop_event;
  KernelDetails details;
  uint64 start_timestamp;
};

struct MemcpyRecord {
  CuptiTracerEventType type;
  size_t size_bytes;
  CUcontext context;
  CUstream stream;
  uint32 correlation_id;
  bool async;
  CUevent start_event;
  CUevent stop_event;
  uint64 start_timestamp;
};

Status CreateAndRecordEvent(CUevent *event, CUstream stream) {
  CuptiApiTracingDisabler disabler;
  TF_RETURN_IF_ERROR(ToStatus(cuEventCreate(event, CU_EVENT_DEFAULT)));
  return ToStatus(cuEventRecord(*event, stream));
}

#if CUDA_VERSION >= 10000
// Maintain and restore current thread's CUDA context.
// Note: cuStreamGetCtx only available after CUDA 9.2.
class ScopedCudaContext {
 public:
  explicit ScopedCudaContext(CUstream stream) : stream_(stream) {
    CuptiApiTracingDisabler disabler;  // don't trace cuda call in this func.
    CUcontext context;
    if (cuStreamGetCtx(stream, &context) != CUDA_SUCCESS) return;
    context_ = context;
    uint32 device_ordinal;
    if (cuptiGetDeviceId(context, &device_ordinal) != CUPTI_SUCCESS) return;
    device_ordinal_ = device_ordinal;
    context_pushed_ = cuCtxPushCurrent(context) == CUDA_SUCCESS;
  }
  ~ScopedCudaContext() {
    if (!context_pushed_) return;
    CuptiApiTracingDisabler disabler;  // don't trace cuda call in this func.
    cuCtxPopCurrent(&*context_);
  }

  // If successful, return the device ordinal of the relevant cuda stream.
  // Otherwise absl::nullopt;
  absl::optional<uint32> GetDeviceOrdinal() { return device_ordinal_; }

  // If successful, return the cuda context of the relevant cuda stream.
  // Otherwise absl::nullopt;
  absl::optional<CUcontext> GetContext() { return context_; }

 private:
  CUstream stream_;
  absl::optional<CUcontext> context_;
  absl::optional<uint32> device_ordinal_;
  bool context_pushed_ = false;
};
#endif

// Stores a series of kernel and memcpy records.
class CudaEventRecorder {
 public:
  CudaEventRecorder(CuptiInterface *cupti_interface,
                    CuptiTraceCollector *collector, int ordinal)
      : cupti_interface_(cupti_interface),
        collector_(collector),
        ordinal_(ordinal) {
    device_name_ = absl::StrCat("gpu ", ordinal);  // default.
    CUdevice device;
    if (cuDeviceGet(&device, ordinal) == CUDA_SUCCESS) {
      char name[100];
      if (cuDeviceGetName(name, sizeof(name), device) == CUDA_SUCCESS) {
        device_name_ = name;
      }
    }
  }

  // Registers the start of a kernel launch. The returned index should be passed
  // to StopKernel() after the kernel launch has completed.
  template <typename T>
  size_t StartKernel(const char *kernel_name, CUcontext context,
                     uint32 correlation_id, const T *params) {
    CUstream stream = params->hStream;
    KernelRecord record = {kernel_name, context, stream, correlation_id};
    record.details.registers_per_thread = 0;  // unknown.
    record.details.static_shared_memory_usage = params->sharedMemBytes;
    record.details.dynamic_shared_memory_usage = 0;  // unknown
    record.details.block_x = params->blockDimX;
    record.details.block_y = params->blockDimY;
    record.details.block_z = params->blockDimZ;
    record.details.grid_x = params->gridDimX;
    record.details.grid_y = params->gridDimY;
    record.details.grid_z = params->gridDimZ;
    record.start_timestamp = CuptiTracer::GetTimestamp();
    LogIfError(CreateAndRecordEvent(&record.start_event, stream));
    absl::MutexLock lock(&mutex_);
    if (stopped_) return -1;
    kernel_records_.push_back(record);
    return kernel_records_.size() - 1;
  }
  uint64 StopKernel(size_t index) {
    absl::MutexLock lock(&mutex_);
    if (index >= kernel_records_.size()) return 0;
    auto &record = kernel_records_[index];
    LogIfError(CreateAndRecordEvent(&record.stop_event, record.stream));
    return record.start_timestamp;
  }

  // Registers the start of a copy operation. The returned index should be
  // passed to StopMemcpy() after the memcpy has completed.
  size_t StartMemcpy(CuptiTracerEventType type, size_t size_bytes,
                     CUcontext context, CUstream stream, uint32 correlation_id,
                     bool async) {
    MemcpyRecord record = {type,   size_bytes,     context,
                           stream, correlation_id, async};
    record.start_timestamp = CuptiTracer::GetTimestamp();
    LogIfError(CreateAndRecordEvent(&record.start_event, stream));
    absl::MutexLock lock(&mutex_);
    if (stopped_) return -1;
    memcpy_records_.push_back(record);
    return memcpy_records_.size() - 1;
  }
  uint64 StopMemcpy(size_t index) {
    absl::MutexLock lock(&mutex_);
    if (index >= memcpy_records_.size()) return 0;
    auto &record = memcpy_records_[index];
    LogIfError(CreateAndRecordEvent(&record.stop_event, record.stream));
    return record.start_timestamp;
  }

  Status Stop() {
    {
      absl::MutexLock lock(&mutex_);
      stopped_ = true;
      LOG(INFO) << "Collecting " << kernel_records_.size()
                << " kernel records, " << memcpy_records_.size()
                << " memcpy records.";

      // Gather all profiled streams and contexts.
      for (const auto &record : kernel_records_) {
        TF_RETURN_IF_ERROR(
            AddStreamInfo(record.context, record.stream, "Kernel"));
      }
      for (const auto &record : memcpy_records_) {
        TF_RETURN_IF_ERROR(AddStreamInfo(record.context, record.stream,
                                         GetTraceEventTypeName(record.type)));
      }
    }

    // Synchronize all contexts, record end events, synchronize again.
    // This scheme is an unreliable measure to associate a event with the wall
    // time. There are chances that other threads might enque kernels which
    // delay the second synchronization.
    TF_RETURN_IF_ERROR(Synchronize());
    for (auto &pair : context_infos_) {
      TF_RETURN_IF_ERROR(ToStatus(cuCtxSetCurrent(pair.first)));
      TF_RETURN_IF_ERROR(CreateAndRecordEvent(&pair.second.end_event, nullptr));
    }

    TF_RETURN_IF_ERROR(Synchronize());
    end_walltime_us_ = Env::Default()->NowMicros();
    return OkStatus();
  }

  Status Flush(AnnotationMap *annotation_map) {
    auto kernel_records = ConsumeKernelRecords();
    auto memcpy_records = ConsumeMemcpyRecords();
    for (const auto &record : kernel_records) {
      TF_RETURN_IF_ERROR(SaveRecord(record, annotation_map));
    }
    for (const auto &record : memcpy_records) {
      TF_RETURN_IF_ERROR(SaveRecord(record, annotation_map));
    }
    return OkStatus();
  }

  std::vector<KernelRecord> ConsumeKernelRecords() {
    absl::MutexLock lock(&mutex_);
    return std::move(kernel_records_);
  }
  std::vector<MemcpyRecord> ConsumeMemcpyRecords() {
    absl::MutexLock lock(&mutex_);
    return std::move(memcpy_records_);
  }

 private:
  struct ContextInfo {
    uint32 context_id = 0;
    int num_streams = 0;
    CUevent end_event;
  };

  struct StreamInfo {
    uint32 stream_id = 0;
    std::string name;
    int index;  // 0 is reserved for null stream.
    const ContextInfo *ctx_info;
  };

  // Synchronizes all contexts.
  Status Synchronize() const {
    CuptiApiTracingDisabler disabler;
    for (const auto &pair : context_infos_) {
      TF_RETURN_IF_ERROR(ToStatus(cuCtxSetCurrent(pair.first)));
      TF_RETURN_IF_ERROR(ToStatus(cuCtxSynchronize()));
    }
    return OkStatus();
  }

  // Returns element from context_infos_, adding it if not yet present.
  Status GetContextInfo(CUcontext context, ContextInfo **ctx_info_ptr) {
    auto it = context_infos_.find(context);

    if (it == context_infos_.end()) {
      uint32 context_id = 0;
      RETURN_IF_CUPTI_ERROR(
          cupti_interface_->GetContextId(context, &context_id));
      ContextInfo ctx_info = {context_id};
      it = context_infos_.emplace(context, ctx_info).first;
    }

    *ctx_info_ptr = &it->second;
    return OkStatus();
  }

  // Adds element to stream_infos_ if not yet present. If present, clear name
  // if it doesn't match parameter.
  Status AddStreamInfo(CUcontext context, CUstream stream,
                       absl::string_view name) {
    StreamKey key(context, stream);
    auto it = stream_infos_.find(key);
    if (it != stream_infos_.end()) {
      if (it->second.name != name) {
        it->second.name.clear();  // Stream with inconsistent names, clear it.
      }
      return OkStatus();
    }

    ContextInfo *ctx_info;
    TF_RETURN_IF_ERROR(GetContextInfo(context, &ctx_info));
    int index = stream ? ++ctx_info->num_streams : 0;
    uint32 stream_id = 0;
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
    RETURN_IF_CUPTI_ERROR(
        cupti_interface_->GetStreamIdEx(context, stream, 1, &stream_id));
#else
    RETURN_IF_CUPTI_ERROR(
        cupti_interface_->GetStreamIdEx(context, stream, 0, &stream_id));
#endif

    StreamInfo stream_info = {stream_id, static_cast<std::string>(name), index,
                              ctx_info};
    stream_infos_.emplace(key, stream_info);
    return OkStatus();
  }

  // Returns time in microseconds between events recorded on the GPU.
  static uint64_t GetElapsedTimeUs(CUevent start, CUevent stop) {
    CuptiApiTracingDisabler disabler;
    float elapsed_ms = 0.0f;
    LogIfError(ToStatus(cuEventElapsedTime(&elapsed_ms, start, stop)));
    return static_cast<uint64>(
        std::llroundf(1000 * std::max(elapsed_ms, 0.0f)));
  }

  Status SaveRecord(const KernelRecord &record,
                    AnnotationMap *annotation_map) const {
    if (!record.start_event || !record.stop_event) {
      return OkStatus();
    }
    const auto &stream_info =
        stream_infos_.at(StreamKey(record.context, record.stream));
    auto start_us =
        GetElapsedTimeUs(record.start_event, stream_info.ctx_info->end_event);
    auto elapsed_us = GetElapsedTimeUs(record.start_event, record.stop_event);

    std::string annotation;

    CuptiTracerEvent event{};
    event.type = CuptiTracerEventType::Kernel;
    event.source = CuptiTracerEventSource::Activity;  // on gpu device.
    event.name = record.kernel_name;
    event.start_time_ns = (end_walltime_us_ - start_us) * 1000;
    event.end_time_ns = event.start_time_ns + elapsed_us * 1000;
    event.device_id = ordinal_;
    event.context_id = stream_info.ctx_info->context_id;
    event.stream_id = stream_info.stream_id;
    event.correlation_id = record.correlation_id;
    AnnotationMap::AnnotationInfo info = collector_->annotation_map()->LookUp(
        event.device_id, event.correlation_id);
    event.annotation = info.annotation;
    event.kernel_info = record.details;
    collector_->AddEvent(std::move(event));
    return OkStatus();
  }

  Status SaveRecord(const MemcpyRecord &record,
                    AnnotationMap *annotation_map) const {
    if (!record.start_event || !record.stop_event) {
      return OkStatus();
    }
    const auto &stream_info =
        stream_infos_.at(StreamKey(record.context, record.stream));
    auto start_us =
        GetElapsedTimeUs(record.start_event, stream_info.ctx_info->end_event);
    auto elapsed_us = GetElapsedTimeUs(record.start_event, record.stop_event);

    CuptiTracerEvent event{};
    event.type = record.type;
    event.name = GetTraceEventTypeName(event.type);
    event.source = CuptiTracerEventSource::Activity;
    event.start_time_ns = (end_walltime_us_ - start_us) * 1000;
    event.end_time_ns = event.start_time_ns + elapsed_us * 1000;
    event.device_id = ordinal_;
    event.context_id = stream_info.ctx_info->context_id;
    event.stream_id = stream_info.stream_id;
    event.correlation_id = record.correlation_id;
    AnnotationMap::AnnotationInfo info = collector_->annotation_map()->LookUp(
        event.device_id, event.correlation_id);
    event.annotation = info.annotation;
    event.memcpy_info.num_bytes = record.size_bytes;
    // TODO: support MemcpyD2D where destination != source;
    event.memcpy_info.destination = ordinal_;
    event.memcpy_info.async = record.async;
    // TODO: set src_mem_kind and dst_mem_kind.
    collector_->AddEvent(std::move(event));
    return OkStatus();
  }

  absl::Mutex mutex_;
  bool stopped_ TF_GUARDED_BY(mutex_) = false;
  std::vector<KernelRecord> kernel_records_ TF_GUARDED_BY(mutex_);
  std::vector<MemcpyRecord> memcpy_records_ TF_GUARDED_BY(mutex_);

  CuptiInterface *cupti_interface_;
  CuptiTraceCollector *collector_;
  const int ordinal_;
  std::string device_name_;
  uint64 end_walltime_us_;
  // Include context in key to distinguish null streams.
  using StreamKey = std::pair<CUcontext, CUstream>;

  absl::node_hash_map<CUcontext, ContextInfo> context_infos_;
  absl::flat_hash_map<StreamKey, StreamInfo> stream_infos_;
};

// This hook uses cuda events to measure device side activities.
class CuptiDriverApiHookWithCudaEvent : public CuptiDriverApiHook {
 public:
  CuptiDriverApiHookWithCudaEvent(const CuptiTracerOptions &option,
                                  CuptiInterface *cupti_interface,
                                  CuptiTraceCollector *collector)
      : option_(option),
        cupti_interface_(cupti_interface),
        collector_(collector) {
    int num_gpus = CuptiTracer::NumGpus();
    cuda_event_recorders_.reserve(num_gpus);
    for (int i = 0; i < num_gpus; ++i) {
      cuda_event_recorders_.emplace_back(
          absl::make_unique<CudaEventRecorder>(cupti_interface, collector, i));
    }
  }
  ~CuptiDriverApiHookWithCudaEvent() {
    for (auto *callback_context : callback_contexts_) delete callback_context;
  }

  Status OnDriverApiEnter(int device_id, CUpti_CallbackDomain domain,
                          CUpti_CallbackId cbid,
                          const CUpti_CallbackData *cbdata) override {
    auto *recorder = cuda_event_recorders_[device_id].get();
    switch (cbid) {
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: {
        DCHECK_NE(cbdata->symbolName, nullptr);
        const auto *params =
            static_cast<const cuLaunchKernel_params *>(cbdata->functionParams);
        *cbdata->correlationData = recorder->StartKernel<cuLaunchKernel_params>(
            cbdata->symbolName, cbdata->context, cbdata->correlationId, params);
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel: {
        DCHECK_NE(cbdata->symbolName, nullptr);
        const auto *params =
            static_cast<const cuLaunchCooperativeKernel_params *>(
                cbdata->functionParams);
        *cbdata->correlationData =
            recorder->StartKernel<cuLaunchCooperativeKernel_params>(
                cbdata->symbolName, cbdata->context, cbdata->correlationId,
                params);
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice: {
#if CUDA_VERSION >= 10000
        const auto *params =
            static_cast<const cuLaunchCooperativeKernelMultiDevice_params *>(
                cbdata->functionParams);
        std::vector<uint32> record_indices;
        record_indices.reserve(params->numDevices);
        *cbdata->correlationData = -1;  // Invalid value.
        const auto &annotation = AnnotationStack::Get();
        for (int i = 0; i < params->numDevices; ++i) {
          CUstream stream = params->launchParamsList[i].hStream;
          ScopedCudaContext scoped_cuda_context(stream);
          auto dev_id = scoped_cuda_context.GetDeviceOrdinal();
          auto context = scoped_cuda_context.GetContext();
          if (!dev_id) return errors::Internal("Invalid CUDA stream");
          // Because annotation are per device, therefore we need to populate
          // annotation for each device involved.
          collector_->annotation_map()->Add(*dev_id, cbdata->correlationId,
                                            annotation, "");
          record_indices.push_back(
              cuda_event_recorders_[*dev_id]->StartKernel<CUDA_LAUNCH_PARAMS>(
                  "CooperativeKernelMultiDevice", *context,
                  cbdata->correlationId, &(params->launchParamsList[i])));
        }
        auto *callback_context =
            new CuptiApiCallbackContext(std::move(record_indices));
        callback_contexts_.insert(callback_context);
        *cbdata->correlationData = reinterpret_cast<uint64>(callback_context);
#else
        VLOG(1) << "Unhandled cuLaunchCooperativeKernelMultiDevice.";
#endif
      } break;
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy: {
        const auto *params =
            static_cast<const cuMemcpy_params *>(cbdata->functionParams);
        StartMemcpy<cuMemcpy_params>(GetMemcpyType(params->src, params->dst),
                                     cbdata, recorder);
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync: {
        const auto *params =
            static_cast<const cuMemcpyAsync_params *>(cbdata->functionParams);
        StartMemcpyAsync<cuMemcpyAsync_params>(
            GetMemcpyType(params->src, params->dst), cbdata, recorder);
        break;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
        StartMemcpy<cuMemcpyHtoD_v2_params>(CuptiTracerEventType::MemcpyH2D,
                                            cbdata, recorder);
        break;
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
        StartMemcpyAsync<cuMemcpyHtoDAsync_v2_params>(
            CuptiTracerEventType::MemcpyH2D, cbdata, recorder);
        break;
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
        StartMemcpy<cuMemcpyDtoH_v2_params>(CuptiTracerEventType::MemcpyD2H,
                                            cbdata, recorder);
        break;
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
        StartMemcpyAsync<cuMemcpyDtoHAsync_v2_params>(
            CuptiTracerEventType::MemcpyD2H, cbdata, recorder);
        break;
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
        StartMemcpy<cuMemcpyDtoD_v2_params>(CuptiTracerEventType::MemcpyD2D,
                                            cbdata, recorder);
        break;
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
        StartMemcpyAsync<cuMemcpyDtoDAsync_v2_params>(
            CuptiTracerEventType::MemcpyD2D, cbdata, recorder);
        break;
      default:
        VLOG(1) << "Unexpected callback id: " << cbid;
        break;
    }
    return OkStatus();
  }

  Status OnDriverApiExit(int device_id, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbid,
                         const CUpti_CallbackData *cbdata) override {
    auto *recorder = cuda_event_recorders_[device_id].get();
    if (*cbdata->correlationData == static_cast<size_t>(-1)) return OkStatus();
    uint64 start_tsc = 0;
    switch (cbid) {
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
        start_tsc = recorder->StopKernel(*cbdata->correlationData);
        break;
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice: {
#if CUDA_VERSION >= 10000
        auto *callback_context = reinterpret_cast<CuptiApiCallbackContext *>(
            *cbdata->correlationData);
        callback_contexts_.erase(callback_context);
        auto record_indices = std::move(callback_context->record_indices);
        delete callback_context;
        const auto *params =
            static_cast<const cuLaunchCooperativeKernelMultiDevice_params *>(
                cbdata->functionParams);
        if (record_indices.size() != params->numDevices)
          return errors::Internal("Invalid correlation data");
        for (int i = 0; i < params->numDevices; ++i) {
          CUstream stream = params->launchParamsList[i].hStream;
          ScopedCudaContext scoped_cuda_context(stream);
          auto dev_id = scoped_cuda_context.GetDeviceOrdinal();
          if (!dev_id) return errors::Internal("Invalid CUDA stream");
          start_tsc =
              cuda_event_recorders_[*dev_id]->StopKernel(record_indices[i]);
        }
#endif
      } break;
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpy:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
      case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
        start_tsc = recorder->StopMemcpy(*cbdata->correlationData);
        break;
      default:
        VLOG(1) << "Unexpected callback id: " << cbid;
        // TODO: figure out how to get start timestamp in this case.
        return OkStatus();
    }
    // If we are not collecting CPU events from Callback API, we can return now.
    if (!option_.required_callback_api_events) {
      return OkStatus();
    }

    // Grab timestamp for API exit. API entry timestamp saved in cbdata.
    uint64 end_tsc = CuptiTracer::GetTimestamp();
    return AddDriverApiCallbackEvent(collector_, cupti_interface_, device_id,
                                     start_tsc, end_tsc, domain, cbid, cbdata);
  }
  Status SyncAndFlush() override {
    for (auto &recorder : cuda_event_recorders_) {
      TF_RETURN_IF_ERROR(recorder->Stop());
    }
    for (auto &recorder : cuda_event_recorders_) {
      TF_RETURN_IF_ERROR(recorder->Flush(collector_->annotation_map()));
    }
    return OkStatus();
  }

 private:
  template <typename T>
  static void StartMemcpy(CuptiTracerEventType type,
                          const CUpti_CallbackData *cbdata,
                          CudaEventRecorder *recorder) {
    const auto *params = static_cast<const T *>(cbdata->functionParams);
    *cbdata->correlationData =
        recorder->StartMemcpy(type, params->ByteCount, cbdata->context, nullptr,
                              cbdata->correlationId, /*async*/ false);
  }

  template <typename T>
  static void StartMemcpyAsync(CuptiTracerEventType type,
                               const CUpti_CallbackData *cbdata,
                               CudaEventRecorder *recorder) {
    const auto *params = static_cast<const T *>(cbdata->functionParams);
    *cbdata->correlationData = recorder->StartMemcpy(
        type, params->ByteCount, cbdata->context, params->hStream,
        cbdata->correlationId, /*async*/ true);
  }

  static CUmemorytype GetMemoryType(CUdeviceptr ptr) {
    CuptiApiTracingDisabler disabler;
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    auto status =
        cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr);
    if (status == CUDA_ERROR_INVALID_VALUE) {
      // Pointer not registered with CUDA, must be host memory.
      return CU_MEMORYTYPE_HOST;
    }
    LogIfError(ToStatus(status));
    return mem_type;
  }

  static CuptiTracerEventType GetMemcpyType(CUdeviceptr src, CUdeviceptr dst) {
    CUmemorytype src_type = GetMemoryType(src);
    CUmemorytype dst_type = GetMemoryType(dst);
    // TODO: handle CU_MEMORYTYPE_ARRAY case
    if (src_type == CU_MEMORYTYPE_HOST && dst_type == CU_MEMORYTYPE_DEVICE) {
      return CuptiTracerEventType::MemcpyH2D;
    } else if (src_type == CU_MEMORYTYPE_DEVICE &&
               dst_type == CU_MEMORYTYPE_HOST) {
      return CuptiTracerEventType::MemcpyD2H;
    } else if (src_type == CU_MEMORYTYPE_DEVICE &&
               dst_type == CU_MEMORYTYPE_DEVICE) {
      return CuptiTracerEventType::MemcpyD2D;
    }
    return CuptiTracerEventType::MemcpyOther;
  }

  // Each cuLaunchCooperativeKernelMultiDevice will need to add an entry in
  // each corresponding device, therefore we need to keep records of all
  // the record indices in each device's record array.
  // We allocate such data structure during API entry and free during API exit.
  // However there is no guarantee that we receive such callbacks in pairs, we
  // maintain a on-going API calls to make sure no memory leaks.
  struct CuptiApiCallbackContext {
    explicit CuptiApiCallbackContext(std::vector<uint32> &&r)
        : record_indices(std::move(r)) {}
    std::vector<uint32> record_indices;
  };

  const CuptiTracerOptions option_;
  CuptiInterface *cupti_interface_;
  CuptiTraceCollector *collector_;
  absl::node_hash_set<CuptiApiCallbackContext *> callback_contexts_;
  std::vector<std::unique_ptr<CudaEventRecorder>> cuda_event_recorders_;
  TF_DISALLOW_COPY_AND_ASSIGN(CuptiDriverApiHookWithCudaEvent);
};

/*static*/ std::string ErrorWithHostname(absl::string_view error_message) {
  return absl::StrCat(port::Hostname(), ": ", error_message);
}

}  // namespace

/*static*/ Status CuptiDriverApiHook::AddDriverApiCallbackEvent(
    CuptiTraceCollector *collector, CuptiInterface *cupti_interface,
    int device_id, uint64 start_tsc, uint64 end_tsc,
    CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
    const CUpti_CallbackData *cbdata) {
  switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
      AddKernelEventUponApiExit(collector, device_id, cbdata, start_tsc,
                                end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2:
      // This would be the place to populate the memcpy API activity's src and
      // dst memory kind by casting cbdata->functionParams. However, we are not
      // doing that because that will incur significant overhead to get the
      // memory aperture of each argument.
      AddNormalMemcpyEventUponApiExit(collector, device_id, cbid, cbdata,
                                      start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync:
      AddP2PMemcpyEventUponApiExit(collector, cupti_interface, device_id, cbid,
                                   cbdata, start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2:
      AddCuMemAllocEventUponApiExit(collector, device_id, cbid, cbdata,
                                    start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2:
      AddCuMemAllocPitchEventUponApiExit(collector, device_id, cbid, cbdata,
                                         start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2:
      AddCuMemFreeEventUponApiExit(collector, device_id, cbid, cbdata,
                                   start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async:
      AddCuMemsetEventUponApiExit(collector, device_id, cbid, cbdata, start_tsc,
                                  end_tsc);
      break;
    default:
      AddGenericEventUponApiExit(collector, device_id, cbid, cbdata, start_tsc,
                                 end_tsc);
      break;
  }
  return OkStatus();
}

const char *GetTraceEventTypeName(const CuptiTracerEventType &type) {
  // Do not use a default so that this gives a build error when
  // CuptiTracerEventType is extended but this is not.
  switch (type) {
    case CuptiTracerEventType::MemcpyH2D:
      return "MemcpyH2D";
    case CuptiTracerEventType::MemcpyD2H:
      return "MemcpyD2H";
    case CuptiTracerEventType::MemcpyD2D:
      return "MemcpyD2D";
    case CuptiTracerEventType::MemcpyP2P:
      return "MemcpyP2P";
    case CuptiTracerEventType::MemcpyOther:
      return "MemcpyOther";
    case CuptiTracerEventType::Kernel:
      return "Compute";
    case CuptiTracerEventType::MemoryAlloc:
      return "MemoryAlloc";
    case CuptiTracerEventType::MemoryFree:
      return "MemoryFree";
    case CuptiTracerEventType::Memset:
      return "Memset";
    case CuptiTracerEventType::Overhead:
      return "Overhead";
    case CuptiTracerEventType::UnifiedMemory:
      return "UnifiedMemory";
    case CuptiTracerEventType::Generic:
      return "Generic";
    case CuptiTracerEventType::MemoryResidency:
      return "MemoryResidency";
    case CuptiTracerEventType::Unsupported:
      return "";
  }
}

CuptiTracer::CuptiTracer(CuptiInterface *cupti_interface)
    : num_gpus_(NumGpus()),
      cupti_interface_(cupti_interface),
      buffer_pool_(kBufferSizeInBytes) {}

/* static */ CuptiTracer *CuptiTracer::GetCuptiTracerSingleton() {
  static auto *singleton = new CuptiTracer(GetCuptiInterface());
  return singleton;
}

bool CuptiTracer::IsAvailable() const {
  return NumGpus() && !activity_tracing_enabled_ && !api_tracing_enabled_;
}

int CuptiTracer::NumGpus() {
  static int num_gpus = []() -> int {
    if (cuInit(0) != CUDA_SUCCESS) {
      return 0;
    }
    int gpu_count;
    if (cuDeviceGetCount(&gpu_count) != CUDA_SUCCESS) {
      return 0;
    }
    LOG(INFO) << "Profiler found " << gpu_count << " GPUs";
    return gpu_count;
  }();
  return num_gpus;
}

void CuptiTracer::Enable(const CuptiTracerOptions &option,
                         CuptiTraceCollector *collector) {
  option_ = option;
  collector_ = collector;
  if (option_->enable_event_based_activity) {
    option_->enable_activity_api = false;
    cupti_driver_api_hook_.reset(new CuptiDriverApiHookWithCudaEvent(
        option, cupti_interface_, collector));
  } else {
    cupti_driver_api_hook_.reset(new CuptiDriverApiHookWithActivityApi(
        option, cupti_interface_, collector));
  }

  Status status = EnableApiTracing();
  need_root_access_ |= status.code() == error::PERMISSION_DENIED;
  if (!status.ok()) return;

  if (option_->enable_activity_api) {
    EnableActivityTracing().IgnoreError();
  }
  tensorflow::profiler::AnnotationStack::Enable(true);
}

void CuptiTracer::Disable() {
  DisableApiTracing().IgnoreError();
  if (option_->enable_activity_api) {
    DisableActivityTracing().IgnoreError();
  }
  cupti_interface_->CleanUp();
  Finalize().IgnoreError();
  cupti_driver_api_hook_->SyncAndFlush().IgnoreError();
  collector_->Flush();
  collector_ = nullptr;
  option_.reset();
  cupti_driver_api_hook_.reset();
  tensorflow::profiler::AnnotationStack::Enable(false);
}

Status CuptiTracer::EnableApiTracing() {
  if (api_tracing_enabled_) return OkStatus();

  VLOG(1) << "Enable subscriber";
  // Subscribe can return CUPTI_ERROR_MAX_LIMIT_REACHED.
  // The application which calls CUPTI APIs cannot be used with Nvidia tools
  // like nvprof, Nvidia Visual Profiler, Nsight Compute, Nsight Systems.
  RETURN_IF_CUPTI_ERROR(cupti_interface_->Subscribe(
      &subscriber_, (CUpti_CallbackFunc)ApiCallback, this));
  api_tracing_enabled_ = true;

  if (!option_->cbids_selected.empty()) {
    for (auto cbid : option_->cbids_selected) {
      RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableCallback(
          1 /* ENABLE */, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid));
    }
  } else {  // select all callback ids.
    RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableDomain(
        1 /* ENABLE */, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API));
  }

  if (option_->enable_nvtx_tracking) {
    RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableDomain(
        1 /* ENABLE */, subscriber_, CUPTI_CB_DOMAIN_NVTX));
  }
  return OkStatus();
}

Status CuptiTracer::DisableApiTracing() {
  if (!api_tracing_enabled_) return OkStatus();

  api_tracing_enabled_ = false;

  if (!option_->cbids_selected.empty()) {
    for (auto cbid : option_->cbids_selected) {
      RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableCallback(
          0 /* DISABLE */, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid));
    }
  } else {
    RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableDomain(
        0 /* DISABLE */, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API));
  }

  if (option_->enable_nvtx_tracking) {
    RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableDomain(
        0 /* DISABLE */, subscriber_, CUPTI_CB_DOMAIN_NVTX));
  }

  VLOG(1) << "Disable subscriber";
  RETURN_IF_CUPTI_ERROR(cupti_interface_->Unsubscribe(subscriber_));
  return OkStatus();
}

Status CuptiTracer::EnableActivityTracing() {
  if (!option_->activities_selected.empty()) {
    // Initialize callback functions for Cupti Activity API.
    VLOG(1) << "Registering CUPTI activity callbacks";
    RETURN_IF_CUPTI_ERROR(cupti_interface_->ActivityRegisterCallbacks(
        RequestCuptiActivityBuffer, ProcessCuptiActivityBuffer));

    VLOG(1) << "Enabling activity tracing for "
            << option_->activities_selected.size() << " activities";
    for (auto activity : option_->activities_selected) {
      VLOG(1) << "Enabling activity tracing for: " << activity;
      if (activity == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) {
        ConfigureActivityUnifiedMemoryCounter(true);
      }
      RETURN_IF_CUPTI_ERROR(cupti_interface_->ActivityEnable(activity));
    }
  }
  activity_tracing_enabled_ = true;
  return OkStatus();
}

Status CuptiTracer::DisableActivityTracing() {
  if (activity_tracing_enabled_) {
    VLOG(1) << "Disabling activity tracing for "
            << option_->activities_selected.size() << " activities";
    for (auto activity : option_->activities_selected) {
      VLOG(1) << "Disabling activity tracing for: " << activity;
      if (activity == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) {
        ConfigureActivityUnifiedMemoryCounter(false);
      }
      RETURN_IF_CUPTI_ERROR(cupti_interface_->ActivityDisable(activity));
    }
    option_->activities_selected.clear();

    VLOG(1) << "Flushing CUPTI activity buffer";
    RETURN_IF_CUPTI_ERROR(
        cupti_interface_->ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    LOG(INFO) << "CUPTI activity buffer flushed";
  }
  activity_tracing_enabled_ = false;
  return OkStatus();
}

Status CuptiTracer::Finalize() {
  if (option_->cupti_finalize) {
    VLOG(1) << "CuptiFinalize";
    RETURN_IF_CUPTI_ERROR(cupti_interface_->Finalize());
  }
  return OkStatus();
}

/*static*/ uint64 CuptiTracer::GetTimestamp() {
  uint64_t tsc;
  CuptiInterface *cupti_interface = GetCuptiInterface();
  if (cupti_interface && cupti_interface->GetTimestamp(&tsc) == CUPTI_SUCCESS) {
    return tsc;
  }
  // Return 0 on error. If an activity timestamp is 0, the activity will be
  // dropped during time normalization.
  return 0;
}

Status CuptiTracer::HandleNVTXCallback(CUpti_CallbackId cbid,
                                       const CUpti_CallbackData *cbdata) {
  const CUpti_NvtxData *pdata =
      reinterpret_cast<const CUpti_NvtxData *>(cbdata);
  if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangePushEx) {
    const nvtxDomainRangePushEx_params *params =
        reinterpret_cast<const nvtxDomainRangePushEx_params *>(
            pdata->functionParams);
    // TODO(profiler): The messageType is actually NVTX_MESSAGE_TYPE_REGISTERED
    // (which is 3), However it seems to me that we can not get the registered
    // string from nvtxDomainRegisterStringA_params. If we reinterpret the
    // payload as ascii, it happen to work.
    NVTXRangeTracker::EnterRange(params->core.eventAttrib->message.ascii);
  } else if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangePop) {
    NVTXRangeTracker::ExitRange();
  }
  return OkStatus();
}

Status CuptiTracer::HandleCallback(CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid,
                                   const CUpti_CallbackData *cbdata) {
  if (!api_tracing_enabled_) return OkStatus();    // already unsubscribed.
  if (!cupti_driver_api_hook_) return OkStatus();  // already unsubscribed.
  if (domain == CUPTI_CB_DOMAIN_NVTX) return HandleNVTXCallback(cbid, cbdata);
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return OkStatus();
  if (internalCuCall) return OkStatus();

  if (cbdata->context == nullptr) {
    // API callback is called before any CUDA context is created.
    // This is expected to be rare, and we ignore this case.
    VLOG(3) << "API callback received before creation of CUDA context\n";
    return errors::Internal("cutpi callback without context");
  }

  // Grab a correct device ID.
  uint32 device_id = -1;
  RETURN_IF_CUPTI_ERROR(
      cupti_interface_->GetDeviceId(cbdata->context, &device_id));
  if (device_id >= num_gpus_) {
    return errors::Internal("Invalid device id:", device_id);
  }

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    TF_RETURN_IF_ERROR(cupti_driver_api_hook_->OnDriverApiEnter(
        device_id, domain, cbid, cbdata));
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    // Set up the map from correlation id to annotation string.
    const auto &annotation = AnnotationStack::Get();
    if (!annotation.empty()) {
      if (cbid ==
          CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice) {
        // Kernels are launched on different devices by this API call, therefore
        // we need to populate per device annotation map respectively.
        for (int i = 0; i < num_gpus_; ++i) {
          collector_->annotation_map()->Add(i, cbdata->correlationId,
                                            annotation, "");
        }
      } else {
        absl::string_view nvtx_range = NVTXRangeTracker::CurrentRange();
        collector_->annotation_map()->Add(device_id, cbdata->correlationId,
                                          annotation, nvtx_range);
      }
    }

    TF_RETURN_IF_ERROR(cupti_driver_api_hook_->OnDriverApiExit(
        device_id, domain, cbid, cbdata));
  }
  return OkStatus();
}

void CuptiTracer::ConfigureActivityUnifiedMemoryCounter(bool enable) {
  CUpti_ActivityUnifiedMemoryCounterConfig config[2];
  // By experiments, currently only measurements from these two activities are
  // trustworthy. Others like GPU page fault may be problematic.
  config[0].kind =
      CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
  config[1].kind =
      CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;

  for (size_t i = 0; i < 2; i++) {
    config[i].enable = enable;
  }

  CUptiResult res;

  res = cupti_interface_->ActivityConfigureUnifiedMemoryCounter(config, 2);
  if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED) {
    LOG(ERROR) << "Unified memory is not supported on the "
                  "underlying platform.\n";
  } else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE) {
    LOG(ERROR) << "Unified memory is not supported on the device.\n";
  } else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES) {
    LOG(ERROR) << "Unified memory is not supported on the "
                  "non-P2P multi-gpu setup.\n";
  } else if (res != CUPTI_SUCCESS) {
    const char *errstr = "";
    cuptiGetResultString(res, &errstr);
    LOG(ERROR) << "Error while enabling unified memory profiling: " << errstr;
  } else {
    VLOG(1) << "Configuring Unified memory profiling: " << res;
  }
}

void CuptiTracer::RequestActivityBuffer(uint8_t **buffer, size_t *size) {
  *buffer = buffer_pool_.GetOrCreateBuffer();
  if (*buffer == nullptr) {
    LOG(WARNING)
        << "CUPTI Buffer not allocated, activity records will be dropped";
    *size = 0;
    return;
  }
  *size = buffer_pool_.GetBufferSizeInBytes();
}

Status CuptiTracer::ProcessActivityBuffer(CUcontext context, uint32_t stream_id,
                                          uint8_t *buffer, size_t size) {
  auto buffer_cleanup =
      gtl::MakeCleanup([&]() { buffer_pool_.ReclaimBuffer(buffer); });
  if (size == 0) {
    return OkStatus();
  }
  if (!activity_tracing_enabled_) {
    LOG(WARNING) << "CUPTI activity buffer is reclaimed after flush.";
    return OkStatus();
  }
  if (cupti_interface_->Disabled()) return errors::Internal("Disabled.");

  CUpti_Activity *record = nullptr;
  while (true) {
    CUptiResult status =
        cupti_interface_->ActivityGetNextRecord(buffer, size, &record);
    if (status == CUPTI_SUCCESS) {
      switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:  // sequential
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
          AddKernelActivityEvent(
              collector_, reinterpret_cast<CuptiActivityKernelTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_MEMCPY:
          AddMemcpyActivityEvent(
              collector_, reinterpret_cast<CuptiActivityMemcpyTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_MEMCPY2:
          AddMemcpyP2PActivityEvent(
              collector_, reinterpret_cast<CuptiActivityMemcpyP2PTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_OVERHEAD:
          AddCuptiOverheadActivityEvent(
              collector_, reinterpret_cast<CUpti_ActivityOverhead *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
          AddUnifiedMemoryActivityEvent(
              collector_,
              reinterpret_cast<CUpti_ActivityUnifiedMemoryCounter2 *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_MEMORY: {
          AddMemoryActivityEvent(
              collector_, reinterpret_cast<CUpti_ActivityMemory *>(record));
        } break;
        case CUPTI_ACTIVITY_KIND_MEMSET:
          AddMemsetActivityEvent(
              collector_, reinterpret_cast<CuptiActivityMemsetTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
          AddSynchronizationActivityEvent(
              collector_,
              reinterpret_cast<CUpti_ActivitySynchronization *>(record));
          break;
        default:
          VLOG(3) << "Activity type " << record->kind << " is not supported.";
          break;
      }
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      return errors::Internal("Parse cupti activity buffer error.");
    }
  }

  // Report dropped records.
  size_t dropped;
  RETURN_IF_CUPTI_ERROR(cupti_interface_->ActivityGetNumDroppedRecords(
      context, stream_id, &dropped));
  if (dropped != 0) {
    uint32 device_id = -1;
    RETURN_IF_CUPTI_ERROR(cupti_interface_->GetDeviceId(context, &device_id));
    collector_->OnEventsDropped("cupti activity buffer full", dropped);
  }
  return OkStatus();
}

/*static*/ std::string CuptiTracer::ErrorIfAny() {
  if (CuptiTracer::NumGpus() == 0) {
    return ErrorWithHostname("No GPU detected.");
  } else if (CuptiTracer::GetCuptiTracerSingleton()->NeedRootAccess()) {
    return ErrorWithHostname(
        "Insufficient privilege to run libcupti (you need root permission).");
  } else if (CuptiTracer::GetTimestamp() == 0) {
    return ErrorWithHostname(
        "Failed to load libcupti (is it installed and accessible?)");
  }
  return "";
}

}  // namespace profiler
}  // namespace tensorflow
