/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/gpu/rocm_tracer.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "rocm/rocm_config.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/profiler/internal/cpu/annotation_stack.h"
#include "tensorflow/core/profiler/utils/time_utils.h"

namespace tensorflow {
namespace profiler {

constexpr uint32_t RocmTracerEvent::kInvalidDeviceId;

#define RETURN_IF_ROCTRACER_ERROR(expr)                                      \
  do {                                                                       \
    roctracer_status_t status = expr;                                        \
    if (status != ROCTRACER_STATUS_SUCCESS) {                                \
      const char* errstr = wrap::roctracer_error_string();                   \
      LOG(ERROR) << "function " << #expr << "failed with error " << errstr;  \
      return errors::Internal(absl::StrCat("roctracer call error", errstr)); \
    }                                                                        \
  } while (false)

namespace {

// GetCachedTID() caches the thread ID in thread-local storage (which is a
// userspace construct) to avoid unnecessary system calls. Without this caching,
// it can take roughly 98ns, while it takes roughly 1ns with this caching.
int32_t GetCachedTID() {
  static thread_local int32_t current_thread_id =
      Env::Default()->GetCurrentThreadId();
  return current_thread_id;
}

const char* GetActivityDomainName(uint32_t domain) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API:
      return "HSA API";
    case ACTIVITY_DOMAIN_HSA_OPS:
      return "HSA OPS";
    // case ACTIVITY_DOMAIN_HIP_OPS:
    //   return "HIP OPS";
    case ACTIVITY_DOMAIN_HCC_OPS:
      return "HCC OPS";
    // case ACTIVITY_DOMAIN_HIP_VDI:
    //   return "HIP VDI";
    case ACTIVITY_DOMAIN_HIP_API:
      return "HIP API";
    case ACTIVITY_DOMAIN_KFD_API:
      return "KFD API";
    case ACTIVITY_DOMAIN_EXT_API:
      return "EXT API";
    case ACTIVITY_DOMAIN_ROCTX:
      return "ROCTX";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

std::string GetActivityDomainOpName(uint32_t domain, uint32_t op) {
  std::ostringstream oss;
  oss << GetActivityDomainName(domain) << " - ";
  switch (domain) {
    case ACTIVITY_DOMAIN_HIP_API:
      oss << hip_api_name(op);
      break;
    default:
      oss << op;
      break;
  }
  return oss.str();
}

const char* GetActivityPhaseName(uint32_t phase) {
  switch (phase) {
    case ACTIVITY_API_PHASE_ENTER:
      return "ENTER";
    case ACTIVITY_API_PHASE_EXIT:
      return "EXIT";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

inline void DumpApiCallbackData(uint32_t domain, uint32_t cbid,
                                const void* cbdata) {
  std::ostringstream oss;
  oss << "API callback for " << GetActivityDomainName(domain);
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t* data =
        reinterpret_cast<const hip_api_data_t*>(cbdata);
    oss << " - " << hip_api_name(cbid);
    oss << ", correlation_id=" << data->correlation_id;
    oss << ", phase=" << GetActivityPhaseName(data->phase);
    switch (cbid) {
      case HIP_API_ID_hipModuleLaunchKernel:
      case HIP_API_ID_hipExtModuleLaunchKernel:
      case HIP_API_ID_hipHccModuleLaunchKernel:
      case HIP_API_ID_hipLaunchKernel:
        break;
      case HIP_API_ID_hipMemcpyDtoH:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoH.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoHAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoHAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyHtoD:
        oss << ", sizeBytes=" << data->args.hipMemcpyHtoD.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyHtoDAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyHtoDAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoD:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoD.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoDAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoDAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemsetD32:
        oss << ", value=" << data->args.hipMemsetD32.value;
        oss << ", count=" << data->args.hipMemsetD32.count;
        break;
      case HIP_API_ID_hipMemsetD32Async:
        oss << ", value=" << data->args.hipMemsetD32Async.value;
        oss << ", count=" << data->args.hipMemsetD32Async.count;
        break;
      case HIP_API_ID_hipMemsetD8:
        oss << ", value=" << data->args.hipMemsetD8.value;
        oss << ", count=" << data->args.hipMemsetD8.count;
        break;
      case HIP_API_ID_hipMemsetD8Async:
        oss << ", value=" << data->args.hipMemsetD8Async.value;
        oss << ", count=" << data->args.hipMemsetD8Async.count;
        break;
      case HIP_API_ID_hipMalloc:
        oss << ", size=" << data->args.hipMalloc.size;
        break;
      case HIP_API_ID_hipFree:
        oss << ", ptr=" << data->args.hipFree.ptr;
        break;
      case HIP_API_ID_hipStreamSynchronize:
        break;
      default:
        DCHECK(false);
        break;
    }
  } else {
    oss << ": " << cbid;
  }
  VLOG(3) << oss.str();
}

void DumpActivityRecord(const roctracer_record_t* record) {
  std::ostringstream oss;
  oss << "Activity callback for " << GetActivityDomainName(record->domain);
  oss << wrap::roctracer_op_string(record->domain, record->op, record->kind);
  oss << ", correlation_id=" << record->correlation_id;
  oss << ", begin_ns=" << record->begin_ns;
  oss << ", end_ns=" << record->end_ns;
  oss << ", device_id=" << record->device_id;
  oss << ", queue_id=" << record->queue_id;
  oss << ", process_id=" << record->process_id;
  oss << ", thread_id=" << record->thread_id;
  oss << ", external_id=" << record->external_id;
  oss << ", bytes=" << record->bytes;
  VLOG(3) << oss.str();
}

}  // namespace

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type) {
  switch (type) {
    case RocmTracerEventType::MemcpyH2D:
      return "MemcpyH2D";
    case RocmTracerEventType::MemcpyD2H:
      return "MemcpyD2H";
    case RocmTracerEventType::MemcpyD2D:
      return "MemcpyD2D";
    case RocmTracerEventType::MemcpyP2P:
      return "MemcpyP2P";
    case RocmTracerEventType::MemcpyOther:
      return "MemcpyOther";
    case RocmTracerEventType::Kernel:
      return "Kernel";
    case RocmTracerEventType::MemoryAlloc:
      return "MemoryAlloc";
    case RocmTracerEventType::Generic:
      return "Generic";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source) {
  switch (source) {
    case RocmTracerEventSource::ApiCallback:
      return "ApiCallback";
      break;
    case RocmTracerEventSource::Activity:
      return "Activity";
      break;
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain) {
  switch (domain) {
    case RocmTracerEventDomain::HIP_API:
      return "HIP_API";
      break;
    case RocmTracerEventDomain::HCC_OPS:
      return "HCC_OPS";
      break;
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

void DumpRocmTracerEvent(const RocmTracerEvent& event,
                         uint64_t start_walltime_ns,
                         uint64_t start_gputime_ns) {
  std::ostringstream oss;

  oss << "correlation_id=" << event.correlation_id;
  oss << ",type=" << GetRocmTracerEventTypeName(event.type);
  oss << ",source=" << GetRocmTracerEventSourceName(event.source);
  oss << ",domain=" << GetRocmTracerEventDomainName(event.domain);
  oss << ",name=" << event.name;
  oss << ",annotation=" << event.annotation;

  // oss << ",start_time_ns=" << event.start_time_ns;
  // oss << ",end_time_ns=" << event.end_time_ns;

  oss << ",start_time_us="
      << (start_walltime_ns + (start_gputime_ns - event.start_time_ns)) / 1000;
  oss << ",duration=" << (event.end_time_ns - event.start_time_ns) / 1000;

  oss << ",device_id=" << event.device_id;
  oss << ",thread_id=" << event.thread_id;
  oss << ",stream_id=" << event.stream_id;

  switch (event.type) {
    case RocmTracerEventType::Kernel:
      break;
    case RocmTracerEventType::MemcpyD2H:
    case RocmTracerEventType::MemcpyH2D:
    case RocmTracerEventType::MemcpyD2D:
    case RocmTracerEventType::MemcpyP2P:
      oss << ",num_bytes=" << event.memcpy_info.num_bytes;
      oss << ",destination=" << event.memcpy_info.destination;
      oss << ",async=" << event.memcpy_info.async;
      break;
    case RocmTracerEventType::MemoryAlloc:
      oss << ",num_bytes=" << event.memalloc_info.num_bytes;
      break;
    case RocmTracerEventType::StreamSynchronize:
      break;
    case RocmTracerEventType::Generic:
      break;
    default:
      DCHECK(false);
      break;
  }
  VLOG(3) << oss.str();
}

Status RocmApiCallbackImpl::operator()(uint32_t domain, uint32_t cbid,
                                       const void* cbdata) {
  // DumpApiCallbackData(domain, cbid, cbdata);

  if (domain != ACTIVITY_DOMAIN_HIP_API) return Status::OK();

  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(cbdata);

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    // Nothing to do here
  } else if (data->phase == ACTIVITY_API_PHASE_EXIT) {
    // Set up the map from correlation id to annotation string.
    const std::string& annotation = AnnotationStack::Get();
    if (!annotation.empty()) {
      collector_->annotation_map()->Add(data->correlation_id, annotation);
    }

    DumpApiCallbackData(domain, cbid, cbdata);

    switch (cbid) {
      case HIP_API_ID_hipModuleLaunchKernel:
      case HIP_API_ID_hipExtModuleLaunchKernel:
      case HIP_API_ID_hipHccModuleLaunchKernel:
      case HIP_API_ID_hipLaunchKernel:
        AddKernelEventUponApiExit(cbid, data);
        // Add the correlation_ids for these events to the pending set
        // so that we can explicitly wait for their corresponding
        // HIP runtime activity records, before exporting the trace data
        tracer_->AddToPendingActivityRecords(data->correlation_id);
        break;
      case HIP_API_ID_hipMemcpyDtoH:
      case HIP_API_ID_hipMemcpyDtoHAsync:
      case HIP_API_ID_hipMemcpyHtoD:
      case HIP_API_ID_hipMemcpyHtoDAsync:
      case HIP_API_ID_hipMemcpyDtoD:
      case HIP_API_ID_hipMemcpyDtoDAsync:
      case HIP_API_ID_hipMemcpyAsync:
        AddMemcpyEventUponApiExit(cbid, data);
        break;
      case HIP_API_ID_hipMemsetD32:
      case HIP_API_ID_hipMemsetD32Async:
      case HIP_API_ID_hipMemsetD8:
      case HIP_API_ID_hipMemsetD8Async:
        AddMemsetEventUponApiExit(cbid, data);
        break;
      case HIP_API_ID_hipMalloc:
      case HIP_API_ID_hipFree:
        AddMallocEventUponApiExit(cbid, data);
        break;
      case HIP_API_ID_hipStreamSynchronize:
        AddStreamSynchronizeEventUponApiExit(cbid, data);
        break;
      default:
        AddGenericEventUponApiExit(cbid, data);
        break;
    }
  }
  return Status::OK();
}

void RocmApiCallbackImpl::AddKernelEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Kernel;
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;
  switch (cbid) {
    case HIP_API_ID_hipModuleLaunchKernel: {
      const hipFunction_t kernelFunc = data->args.hipModuleLaunchKernel.f;
      if (kernelFunc != nullptr) event.name = hipKernelNameRef(kernelFunc);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipModuleLaunchKernel.sharedMemBytes;
      event.kernel_info.block_x = data->args.hipModuleLaunchKernel.blockDimX;
      event.kernel_info.block_y = data->args.hipModuleLaunchKernel.blockDimY;
      event.kernel_info.block_z = data->args.hipModuleLaunchKernel.blockDimZ;
      event.kernel_info.grid_x = data->args.hipModuleLaunchKernel.gridDimX;
      event.kernel_info.grid_y = data->args.hipModuleLaunchKernel.gridDimY;
      event.kernel_info.grid_z = data->args.hipModuleLaunchKernel.gridDimZ;
    } break;
    case HIP_API_ID_hipExtModuleLaunchKernel: {
      const hipFunction_t kernelFunc = data->args.hipExtModuleLaunchKernel.f;
      if (kernelFunc != nullptr) event.name = hipKernelNameRef(kernelFunc);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipExtModuleLaunchKernel.sharedMemBytes;
      unsigned int blockDimX =
          data->args.hipExtModuleLaunchKernel.localWorkSizeX;
      unsigned int blockDimY =
          data->args.hipExtModuleLaunchKernel.localWorkSizeY;
      unsigned int blockDimZ =
          data->args.hipExtModuleLaunchKernel.localWorkSizeZ;

      event.kernel_info.block_x = blockDimX;
      event.kernel_info.block_y = blockDimY;
      event.kernel_info.block_z = blockDimZ;
      event.kernel_info.grid_x =
          data->args.hipExtModuleLaunchKernel.globalWorkSizeX / blockDimX;
      event.kernel_info.grid_y =
          data->args.hipExtModuleLaunchKernel.globalWorkSizeY / blockDimY;
      event.kernel_info.grid_z =
          data->args.hipExtModuleLaunchKernel.globalWorkSizeZ / blockDimZ;
    } break;
    case HIP_API_ID_hipHccModuleLaunchKernel: {
      const hipFunction_t kernelFunc = data->args.hipHccModuleLaunchKernel.f;
      if (kernelFunc != nullptr) event.name = hipKernelNameRef(kernelFunc);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipHccModuleLaunchKernel.sharedMemBytes;
      event.kernel_info.block_x = data->args.hipHccModuleLaunchKernel.blockDimX;
      event.kernel_info.block_y = data->args.hipHccModuleLaunchKernel.blockDimY;
      event.kernel_info.block_z = data->args.hipHccModuleLaunchKernel.blockDimZ;
      event.kernel_info.grid_x =
          data->args.hipHccModuleLaunchKernel.globalWorkSizeX /
          event.kernel_info.block_x;
      event.kernel_info.grid_y =
          data->args.hipHccModuleLaunchKernel.globalWorkSizeY /
          event.kernel_info.block_y;
      event.kernel_info.grid_z =
          data->args.hipHccModuleLaunchKernel.globalWorkSizeZ /
          event.kernel_info.block_z;
      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipHccModuleLaunchKernel.sharedMemBytes;
    } break;
    case HIP_API_ID_hipLaunchKernel: {
      const void* func_addr = data->args.hipLaunchKernel.function_address;
      hipStream_t stream = data->args.hipLaunchKernel.stream;
      if (func_addr != nullptr)
        event.name = hipKernelNameRefByPtr(func_addr, stream);

      event.kernel_info.dynamic_shared_memory_usage =
          data->args.hipLaunchKernel.sharedMemBytes;
      event.kernel_info.block_x = data->args.hipLaunchKernel.dimBlocks.x;
      event.kernel_info.block_y = data->args.hipLaunchKernel.dimBlocks.y;
      event.kernel_info.block_z = data->args.hipLaunchKernel.dimBlocks.z;
      event.kernel_info.grid_x = data->args.hipLaunchKernel.numBlocks.x;
      event.kernel_info.grid_y = data->args.hipLaunchKernel.numBlocks.y;
      event.kernel_info.grid_z = data->args.hipLaunchKernel.numBlocks.z;
    } break;
  }
  collector_->AddEvent(std::move(event));
}

void RocmApiCallbackImpl::AddMemcpyEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.name = wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;

  // TODO(rocm): figure out a way to properly populate this field.
  event.memcpy_info.destination = 0;
  switch (cbid) {
    case HIP_API_ID_hipMemcpyDtoH:
      event.type = RocmTracerEventType::MemcpyD2H;
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoH.sizeBytes;
      event.memcpy_info.async = false;
      break;
    case HIP_API_ID_hipMemcpyDtoHAsync:
      event.type = RocmTracerEventType::MemcpyD2H;
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoHAsync.sizeBytes;
      event.memcpy_info.async = true;
      break;
    case HIP_API_ID_hipMemcpyHtoD:
      event.type = RocmTracerEventType::MemcpyH2D;
      event.memcpy_info.num_bytes = data->args.hipMemcpyHtoD.sizeBytes;
      event.memcpy_info.async = false;
      break;
    case HIP_API_ID_hipMemcpyHtoDAsync:
      event.type = RocmTracerEventType::MemcpyH2D;
      event.memcpy_info.num_bytes = data->args.hipMemcpyHtoDAsync.sizeBytes;
      event.memcpy_info.async = true;
      break;
    case HIP_API_ID_hipMemcpyDtoD:
      event.type = RocmTracerEventType::MemcpyD2D;
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoD.sizeBytes;
      event.memcpy_info.async = false;
      break;
    case HIP_API_ID_hipMemcpyDtoDAsync:
      event.type = RocmTracerEventType::MemcpyD2D;
      event.memcpy_info.num_bytes = data->args.hipMemcpyDtoDAsync.sizeBytes;
      event.memcpy_info.async = true;
      break;
    case HIP_API_ID_hipMemcpyAsync:
      event.type = RocmTracerEventType::MemcpyOther;
      event.memcpy_info.num_bytes = data->args.hipMemcpyAsync.sizeBytes;
      event.memcpy_info.async = true;
      break;
    default:
      LOG(ERROR) << "Unsupported memcpy activity observed: " << cbid;
      break;
  }
  collector_->AddEvent(std::move(event));
}

void RocmApiCallbackImpl::AddMemsetEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.name = wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.source = RocmTracerEventSource::ApiCallback;
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;

  // ROCM TODO: figure out a way to properly populate this field.
  event.memcpy_info.destination = 0;
  switch (cbid) {
    case HIP_API_ID_hipMemsetD8:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_elements = data->args.hipMemsetD8.count;
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD8Async:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_elements = data->args.hipMemsetD8Async.count;
      event.memset_info.async = true;
      break;
    case HIP_API_ID_hipMemsetD32:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_elements = data->args.hipMemsetD32.count;
      event.memset_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD32Async:
      event.type = RocmTracerEventType::Memset;
      event.memset_info.num_elements = data->args.hipMemsetD32Async.count;
      event.memset_info.async = true;
      break;
    default:
      LOG(ERROR) << "Unsupported memset activity observed: " << cbid;
      break;
  }
  collector_->AddEvent(std::move(event));
}

void RocmApiCallbackImpl::AddMallocEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::MemoryAlloc;
  event.source = RocmTracerEventSource::ApiCallback;
  event.name = wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;

  switch (cbid) {
    case HIP_API_ID_hipMalloc:
      event.memalloc_info.num_bytes = data->args.hipMalloc.size;
      break;
    case HIP_API_ID_hipFree:
      event.memalloc_info.num_bytes = 0;
      break;
  }
  collector_->AddEvent(std::move(event));
}

void RocmApiCallbackImpl::AddStreamSynchronizeEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::StreamSynchronize;
  event.source = RocmTracerEventSource::ApiCallback;
  event.name = wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;

  collector_->AddEvent(std::move(event));
}

void RocmApiCallbackImpl::AddGenericEventUponApiExit(
    uint32_t cbid, const hip_api_data_t* data) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Generic;
  event.source = RocmTracerEventSource::ApiCallback;
  event.name = wrap::roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
  event.thread_id = GetCachedTID();
  event.correlation_id = data->correlation_id;

  collector_->AddEvent(std::move(event));
}

Status RocmActivityCallbackImpl::operator()(const char* begin,
                                            const char* end) {
  const roctracer_record_t* record =
      reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record =
      reinterpret_cast<const roctracer_record_t*>(end);

  while (record < end_record) {
    // DumpActivityRecord(record);

    switch (record->domain) {
      // HIP API activities.
      case ACTIVITY_DOMAIN_HIP_API:
        switch (record->op) {
          case HIP_API_ID_hipModuleLaunchKernel:
          case HIP_API_ID_hipExtModuleLaunchKernel:
          case HIP_API_ID_hipHccModuleLaunchKernel:
          case HIP_API_ID_hipLaunchKernel:
            DumpActivityRecord(record);
            AddHipKernelActivityEvent(record);
            break;

          case HIP_API_ID_hipMemcpyDtoH:
          case HIP_API_ID_hipMemcpyHtoD:
          case HIP_API_ID_hipMemcpyDtoD:
          case HIP_API_ID_hipMemcpyDtoHAsync:
          case HIP_API_ID_hipMemcpyHtoDAsync:
          case HIP_API_ID_hipMemcpyDtoDAsync:
          case HIP_API_ID_hipMemcpyAsync:
            DumpActivityRecord(record);
            AddHipMemcpyActivityEvent(record);
            break;

          case HIP_API_ID_hipMemsetD32:
          case HIP_API_ID_hipMemsetD32Async:
          case HIP_API_ID_hipMemsetD8:
          case HIP_API_ID_hipMemsetD8Async:
            DumpActivityRecord(record);
            AddHipMemsetActivityEvent(record);
            break;

          case HIP_API_ID_hipMalloc:
          case HIP_API_ID_hipFree:
            DumpActivityRecord(record);
            AddHipMallocEvent(record);
            break;

          case HIP_API_ID_hipStreamSynchronize:
            DumpActivityRecord(record);
            AddHipStreamSynchronizeEvent(record);
            break;

          default:
            // DumpActivityRecord(record);
            break;
        }  // switch (record->op).
        break;

      // HCC ops activities.
      case ACTIVITY_DOMAIN_HCC_OPS:
        switch (record->op) {
          case HIP_OP_ID_DISPATCH:
            DumpActivityRecord(record);
            AddHccKernelActivityEvent(record);
            tracer_->RemoveFromPendingActivityRecords(record->correlation_id);
            break;
          case HIP_OP_ID_COPY:
            DumpActivityRecord(record);
            AddHccMemcpyActivityEvent(record);
            break;
          default:
            // DumpActivityRecord(record);
            break;
        }  // switch (record->op).
        break;
    }

    RETURN_IF_ROCTRACER_ERROR(static_cast<roctracer_status_t>(
        roctracer_next_record(record, &record)));
  }

  return Status::OK();
}

void RocmActivityCallbackImpl::AddHipKernelActivityEvent(
    const roctracer_record_t* record) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::Kernel;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;

  collector_->AddEvent(std::move(event));
}

void RocmActivityCallbackImpl::AddHipMemcpyActivityEvent(
    const roctracer_record_t* record) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.memcpy_info.num_bytes = record->bytes;
  event.memcpy_info.destination = record->device_id;

  switch (record->op) {
    case HIP_API_ID_hipMemcpyDtoH:
      event.type = RocmTracerEventType::MemcpyD2H;
      event.memcpy_info.async = false;
      break;
    case HIP_API_ID_hipMemcpyDtoHAsync:
      event.type = RocmTracerEventType::MemcpyD2H;
      event.memcpy_info.async = true;
      break;
    case HIP_API_ID_hipMemcpyHtoD:
      event.type = RocmTracerEventType::MemcpyH2D;
      event.memcpy_info.async = false;
      break;
    case HIP_API_ID_hipMemcpyHtoDAsync:
      event.type = RocmTracerEventType::MemcpyH2D;
      event.memcpy_info.async = true;
      break;
    case HIP_API_ID_hipMemcpyDtoD:
      event.type = RocmTracerEventType::MemcpyD2D;
      event.memcpy_info.async = false;
      // ROCM TODO: figure out a way to properly populate this field.
      event.memcpy_info.destination = record->device_id;
      break;
    case HIP_API_ID_hipMemcpyDtoDAsync:
      event.type = RocmTracerEventType::MemcpyD2D;
      event.memcpy_info.async = true;
      // ROCM TODO: figure out a way to properly populate this field.
      event.memcpy_info.destination = record->device_id;
      break;
    case HIP_API_ID_hipMemcpyAsync:
      event.type = RocmTracerEventType::MemcpyOther;
      event.memcpy_info.async = true;
      // ROCM TODO: figure out a way to properly populate this field.
      event.memcpy_info.destination = record->device_id;
      break;
    default:
      event.type = RocmTracerEventType::MemcpyOther;
      event.memcpy_info.async = false;
      event.memcpy_info.destination = record->device_id;
      break;
  }

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;

  collector_->AddEvent(std::move(event));
}

void RocmActivityCallbackImpl::AddHipMemsetActivityEvent(
    const roctracer_record_t* record) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.type = RocmTracerEventType::Memset;

  switch (record->op) {
    case HIP_API_ID_hipMemsetD8:
      event.memset_info.num_elements = record->bytes;
      event.memcpy_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD8Async:
      event.memset_info.num_elements = record->bytes;
      event.memcpy_info.async = true;
      break;
    case HIP_API_ID_hipMemsetD32:
      event.memset_info.num_elements = record->bytes / 4;
      event.memcpy_info.async = false;
      break;
    case HIP_API_ID_hipMemsetD32Async:
      event.memset_info.num_elements = record->bytes / 4;
      event.memcpy_info.async = true;
      break;
  }

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;

  collector_->AddEvent(std::move(event));
}

void RocmActivityCallbackImpl::AddHipMallocEvent(
    const roctracer_record_t* record) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::MemoryAlloc;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;

  collector_->AddEvent(std::move(event));
}

void RocmActivityCallbackImpl::AddHipStreamSynchronizeEvent(
    const roctracer_record_t* record) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HIP_API;
  event.type = RocmTracerEventType::StreamSynchronize;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;

  collector_->AddEvent(std::move(event));
}

void RocmActivityCallbackImpl::AddHccKernelActivityEvent(
    const roctracer_record_t* record) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HCC_OPS;
  event.type = RocmTracerEventType::Kernel;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;
  event.device_id = record->device_id;
  event.stream_id = record->queue_id;

  collector_->AddEvent(std::move(event));
}

void RocmActivityCallbackImpl::AddHccMemcpyActivityEvent(
    const roctracer_record_t* record) {
  RocmTracerEvent event;
  event.domain = RocmTracerEventDomain::HCC_OPS;
  // Set MemcpyOther here. The field won't really be used when we aggregate
  // with other RocmTracerEvent instances coming from API callbacks.
  event.type = RocmTracerEventType::MemcpyOther;
  event.source = RocmTracerEventSource::Activity;
  event.name =
      wrap::roctracer_op_string(record->domain, record->op, record->kind);
  event.correlation_id = record->correlation_id;
  event.annotation = collector_->annotation_map()->LookUp(event.correlation_id);

  event.start_time_ns = record->begin_ns;
  event.end_time_ns = record->end_ns;
  event.device_id = record->device_id;
  event.stream_id = record->queue_id;

  collector_->AddEvent(std::move(event));
}

void AnnotationMap::Add(uint32_t correlation_id,
                        const std::string& annotation) {
  if (annotation.empty()) return;
  VLOG(3) << "Add annotation: "
          << " correlation_id=" << correlation_id
          << ", annotation: " << annotation;
  absl::MutexLock lock(&map_.mutex);
  if (map_.annotations.size() < max_size_) {
    absl::string_view annotation_str =
        *map_.annotations.insert(annotation).first;
    map_.correlation_map.emplace(correlation_id, annotation_str);
  }
}

absl::string_view AnnotationMap::LookUp(uint32_t correlation_id) {
  absl::MutexLock lock(&map_.mutex);
  auto it = map_.correlation_map.find(correlation_id);
  return it != map_.correlation_map.end() ? it->second : absl::string_view();
}

/* static */ RocmTracer* RocmTracer::GetRocmTracerSingleton() {
  static auto* singleton = new RocmTracer();
  return singleton;
}

bool RocmTracer::IsAvailable() const {
  return !activity_tracing_enabled_ && !api_tracing_enabled_;
}

int RocmTracer::NumGpus() {
  static int num_gpus = []() -> int {
    if (hipInit(0) != hipSuccess) {
      return 0;
    }
    int gpu_count;
    if (hipGetDeviceCount(&gpu_count) != hipSuccess) {
      return 0;
    }
    LOG(INFO) << "Profiler found " << gpu_count << " GPUs";
    return gpu_count;
  }();
  return num_gpus;
}

void RocmTracer::Enable(const RocmTracerOptions& options,
                        RocmTraceCollector* collector) {
  options_ = options;
  collector_ = collector;
  api_cb_impl_ = new RocmApiCallbackImpl(options, this, collector);
  activity_cb_impl_ = new RocmActivityCallbackImpl(options, this, collector);

  // From ROCm 3.5 onwards, the following call is required.
  // don't quite know what it does (no documentation!), only that without it
  // the call to enable api/activity tracing will run into a segfault
  wrap::roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);

  EnableApiTracing().IgnoreError();
  EnableActivityTracing().IgnoreError();
  LOG(INFO) << "GpuTracer started";
}

void RocmTracer::Disable() {
  DisableApiTracing().IgnoreError();
  DisableActivityTracing().IgnoreError();
  delete api_cb_impl_;
  delete activity_cb_impl_;
  collector_->Flush();
  collector_ = nullptr;
  options_.reset();
  LOG(INFO) << "GpuTracer stopped";
}

void ApiCallback(uint32_t domain, uint32_t cbid, const void* cbdata,
                 void* user_data) {
  RocmTracer* tracer = reinterpret_cast<RocmTracer*>(user_data);
  tracer->ApiCallbackHandler(domain, cbid, cbdata);
}

void RocmTracer::ApiCallbackHandler(uint32_t domain, uint32_t cbid,
                                    const void* cbdata) {
  if (api_tracing_enabled_) (*api_cb_impl_)(domain, cbid, cbdata);
}

Status RocmTracer::EnableApiTracing() {
  if (api_tracing_enabled_) return Status::OK();
  api_tracing_enabled_ = true;

  for (auto& iter : options_->api_callbacks) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Enabling API tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          wrap::roctracer_enable_domain_callback(domain, ApiCallback, this));
    } else {
      VLOG(3) << "Enabling API tracing for " << ops.size() << " ops in domain "
              << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Enabling API tracing for "
                << GetActivityDomainOpName(domain, op);
        RETURN_IF_ROCTRACER_ERROR(
            wrap::roctracer_enable_op_callback(domain, op, ApiCallback, this));
      }
    }
  }
  return Status::OK();
}

Status RocmTracer::DisableApiTracing() {
  if (!api_tracing_enabled_) return Status::OK();
  api_tracing_enabled_ = false;

  for (auto& iter : options_->api_callbacks) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Disabling API tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          wrap::roctracer_disable_domain_callback(domain));
    } else {
      VLOG(3) << "Disabling API tracing for " << ops.size() << " ops in domain "
              << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Disabling API tracing for "
                << GetActivityDomainOpName(domain, op);
        RETURN_IF_ROCTRACER_ERROR(
            wrap::roctracer_disable_op_callback(domain, op));
      }
    }
  }
  return Status::OK();
}

void ActivityCallback(const char* begin, const char* end, void* user_data) {
  RocmTracer* tracer = reinterpret_cast<RocmTracer*>(user_data);
  tracer->ActivityCallbackHandler(begin, end);
}

void RocmTracer::ActivityCallbackHandler(const char* begin, const char* end) {
  if (activity_tracing_enabled_) {
    (*activity_cb_impl_)(begin, end);
  } else {
    LOG(WARNING) << "ActivityCallbackHandler called when "
                    "activity_tracing_enabled_ is false";

    VLOG(3) << "Dropped Activity Records Start";
    const roctracer_record_t* record =
        reinterpret_cast<const roctracer_record_t*>(begin);
    const roctracer_record_t* end_record =
        reinterpret_cast<const roctracer_record_t*>(end);
    while (record < end_record) {
      DumpActivityRecord(record);
      roctracer_next_record(record, &record);
    }
    VLOG(3) << "Dropped Activity Records End";
  }
}

Status RocmTracer::EnableActivityTracing() {
  if (activity_tracing_enabled_) return Status::OK();
  activity_tracing_enabled_ = true;

  if (!options_->activity_tracing.empty()) {
    // Creat the memory pool to store activity records in
    if (wrap::roctracer_default_pool_expl(nullptr) == NULL) {
      roctracer_properties_t properties{};
      properties.buffer_size = 0x1000;
      properties.buffer_callback_fun = ActivityCallback;
      properties.buffer_callback_arg = this;
      VLOG(3) << "Creating roctracer activity buffer";
      RETURN_IF_ROCTRACER_ERROR(
          wrap::roctracer_open_pool_expl(&properties, nullptr));
    }
  }

  for (auto& iter : options_->activity_tracing) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Enabling Activity tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          wrap::roctracer_enable_domain_activity_expl(domain, nullptr));
    } else {
      VLOG(3) << "Enabling Activity tracing for " << ops.size()
              << " ops in domain " << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Enabling Activity tracing for "
                << GetActivityDomainOpName(domain, op);
        RETURN_IF_ROCTRACER_ERROR(
            wrap::roctracer_enable_op_activity(domain, op));
      }
    }
  }

  return Status::OK();
}

Status RocmTracer::DisableActivityTracing() {
  if (!activity_tracing_enabled_) return Status::OK();

  for (auto& iter : options_->activity_tracing) {
    activity_domain_t domain = iter.first;
    std::vector<uint32_t>& ops = iter.second;
    if (ops.size() == 0) {
      VLOG(3) << "Disabling Activity tracing for domain "
              << GetActivityDomainName(domain);
      RETURN_IF_ROCTRACER_ERROR(
          wrap::roctracer_disable_domain_activity(domain));
    } else {
      VLOG(3) << "Disabling Activity tracing for " << ops.size()
              << " ops in domain " << GetActivityDomainName(domain);
      for (auto& op : ops) {
        VLOG(3) << "Disabling Activity tracing for "
                << GetActivityDomainOpName(domain, op);
        RETURN_IF_ROCTRACER_ERROR(
            wrap::roctracer_disable_op_activity(domain, op));
      }
    }
  }

  // Flush the activity buffer BEFORE setting the activity_tracing_enable_
  // flag to FALSE. This is because the activity record callback routine is
  // gated by the same flag
  VLOG(3) << "Flushing roctracer activity buffer";
  RETURN_IF_ROCTRACER_ERROR(wrap::roctracer_flush_activity_expl(nullptr));

  // Explicitly wait for (almost) all pending acitivity records
  // The choice of all of the following is based what seemed to work
  // best when enabling tracing on a large testcase (BERT)
  // * 100 ms as the initial sleep duration AND
  // * 1 as the initial threshold value
  // * 6 as the maximum number of iterations
  int duration_ms = 100;
  size_t threshold = 1;
  for (int i = 0; i < 6; i++, duration_ms *= 2, threshold *= 2) {
    if (GetPendingActivityRecordsCount() < threshold) break;
    VLOG(3) << "Wait for pending activity records :"
            << " Pending count = " << GetPendingActivityRecordsCount()
            << ", Threshold = " << threshold;
    VLOG(3) << "Wait for pending activity records : sleep for " << duration_ms
            << " ms";
    tensorflow::profiler::SleepForMillis(duration_ms);
  }
  ClearPendingActivityRecordsCount();

  activity_tracing_enabled_ = false;

  return Status::OK();
}

/*static*/ uint64_t RocmTracer::GetTimestamp() {
  uint64_t ts;
  if (wrap::roctracer_get_timestamp(&ts) != ROCTRACER_STATUS_SUCCESS) {
    const char* errstr = wrap::roctracer_error_string();
    LOG(ERROR) << "function roctracer_get_timestamp failed with error "
               << errstr;
    // Return 0 on error.
    return 0;
  }
  return ts;
}

}  // namespace profiler
}  // namespace tensorflow
