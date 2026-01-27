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

// This translation unit is **self‑contained**: it provides minimal stub
// implementations for the rocprofiler callbacks that XLA needs to register
// (toolInit / toolFinialize / code_object_callback).  They do nothing except
// keep the compiler and linker happy.  Once real logging is implemented, you
// can replace the stubs with the actual logic.

#include "xla/backends/profiler/gpu/rocm_tracer.h"

#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/rocprofiler-sdk/agent.h"
#include "rocm/include/rocprofiler-sdk/buffer.h"
#include "rocm/include/rocprofiler-sdk/buffer_tracing.h"
#include "rocm/include/rocprofiler-sdk/callback_tracing.h"
#include "rocm/include/rocprofiler-sdk/context.h"
#include "rocm/include/rocprofiler-sdk/cxx/details/name_info.hpp"
#include "rocm/include/rocprofiler-sdk/fwd.h"
#include "rocm/include/rocprofiler-sdk/hip/runtime_api_id.h"
#include "rocm/include/rocprofiler-sdk/internal_threading.h"
#include "rocm/include/rocprofiler-sdk/registration.h"
#include "rocm/include/rocprofiler-sdk/rocprofiler.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/platform/abi.h"

// for rocprofiler-sdk
namespace xla {
namespace profiler {

using tsl::profiler::AnnotationStack;

// represents an invalid or uninitialized device ID used in RocmTracer events.
constexpr uint32_t RocmTracerEvent::kInvalidDeviceId;

inline auto GetCallbackTracingNames() {
  return rocprofiler::sdk::get_callback_tracing_names();
}

std::vector<rocprofiler_agent_v0_t> GetGpuDeviceAgents();

//-----------------------------------------------------------------------------
// copy api calls
bool isCopyApi(uint32_t id) {
  switch (id) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArrayAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArrayAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoH:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbolAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoA:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoD:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeer:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbolAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream:
      return true;
    default: {
    };
  }
  return false;
}

// ----------------------------------------------------------------------------
// Stub implementations for RocmTracer static functions expected by
// rocprofiler-sdk.
// ----------------------------------------------------------------------------
RocmTracer& RocmTracer::GetRocmTracerSingleton() {
  static RocmTracer obj;
  return obj;
}

bool RocmTracer::IsAvailable() const {
  return !activity_tracing_enabled_ && !api_tracing_enabled_;  // &&NumGpus()
}

/*static*/ uint64_t RocmTracer::GetTimestamp() {
  uint64_t ts;
  if (rocprofiler_get_timestamp(&ts) != ROCPROFILER_STATUS_SUCCESS) {
    LOG(ERROR) << "function rocprofiler_get_timestamp failed with error ";
    return 0;
  }
  return ts;
}

void RocmTracer::Enable(const RocmTracerOptions& options,
                        RocmTraceCollector* collector) {
  absl::MutexLock lock(collector_mutex_);
  if (collector_ != nullptr) {
    LOG(WARNING) << "ROCM tracer is already running!";
    return;
  }
  options_ = options;
  collector_ = collector;
  api_tracing_enabled_ = true;
  activity_tracing_enabled_ = true;
  rocprofiler_start_context(context_);
  LOG(INFO) << "GpuTracer started with number of GPUs = " << NumGpus();
}

void RocmTracer::HipApiEvent(const rocprofiler_record_header_t* hdr,
                             RocmTracerEvent* trace_event) {
  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_hip_api_record_t*>(
          hdr->payload);

  trace_event->type = RocmTracerEventType::Kernel;
  trace_event->source = RocmTracerEventSource::ApiCallback;
  trace_event->domain = RocmTracerEventDomain::HIP_API;
  trace_event->name = "??";
  trace_event->start_time_ns = rec.start_timestamp;
  trace_event->end_time_ns = rec.end_timestamp;
  trace_event->device_id = RocmTracerEvent::kInvalidDeviceId;
  trace_event->correlation_id = rec.correlation_id.internal;
  trace_event->annotation =
      annotation_map()->LookUp(trace_event->correlation_id);
  trace_event->thread_id = rec.thread_id;
  trace_event->stream_id = RocmTracerEvent::kInvalidStreamId;
  trace_event->kernel_info = KernelDetails{};

  {
    // bounds-check name table: kind and operation
    absl::MutexLock lock(kernel_lock_);
    const size_t kind = static_cast<size_t>(rec.kind);
    if (kind < name_info_.size()) {
      const auto& vec = name_info_[kind];
      const size_t op = static_cast<size_t>(rec.operation);
      if (op < vec.operations.size()) {
        trace_event->name = vec[op];
      } else {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
          LOG(ERROR) << "HIP op OOB: kind " << kind << " op = " << op
                     << " vec.size() = " << vec.operations.size();
        }
        trace_event->name = "HIP_UNKNOWN_OP";
      }
    } else {
      static std::atomic<int> once{0};
      if (once.fetch_add(1) == 0) {
        LOG(ERROR) << "HIP kind OOB: kind = " << kind
                   << " name_info_.size() = " << name_info_.size();
      }
      trace_event->name = "HIP_UNKNOWN_KIND";
    }
  }

  if (isCopyApi(rec.operation)) {
    // actually one needs to set the real type
    trace_event->type = RocmTracerEventType::MemcpyOther;
  }
}

void RocmTracer::MemcpyEvent(const rocprofiler_record_header_t* hdr,
                             RocmTracerEvent* trace_event) {
  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_memory_copy_record_t*>(
          hdr->payload);

#define OO(src, target)                              \
  case ROCPROFILER_MEMORY_COPY_##src:                \
    trace_event->type = RocmTracerEventType::target; \
    trace_event->name = #target;                     \
    break;

  switch (rec.operation) {
    OO(NONE, MemcpyOther)
    OO(HOST_TO_HOST, MemcpyOther)
    OO(HOST_TO_DEVICE, MemcpyH2D)
    OO(DEVICE_TO_HOST, MemcpyD2H)
    OO(DEVICE_TO_DEVICE, MemcpyD2D)
    default:
      LOG(WARNING) << "Unexpected memcopy operation " << rec.operation;
      trace_event->type = RocmTracerEventType::MemcpyOther;
  }
#undef OO
  const auto &src_gpu = agents_[static_cast<uint32_t>(rec.src_agent_id.handle)],
             &dst_gpu = agents_[static_cast<uint32_t>(rec.dst_agent_id.handle)];

  // Assign device_id based on copy direction
  if (trace_event->type == RocmTracerEventType::MemcpyH2D &&
      dst_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
    trace_event->device_id = dst_gpu.id.handle;  // Destination is GPU
  } else if (trace_event->type == RocmTracerEventType::MemcpyD2H &&
             src_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
    trace_event->device_id = src_gpu.id.handle;  // Source is GPU
  } else if (trace_event->type == RocmTracerEventType::MemcpyD2D) {
    // Prefer destination GPU for D2D
    trace_event->device_id = dst_gpu.id.handle;
  } else {
    // Fallback for MemcpyOther or HOST_TO_HOST
    if (dst_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
      trace_event->device_id = dst_gpu.id.handle;
    } else if (src_gpu.type == ROCPROFILER_AGENT_TYPE_GPU) {
      trace_event->device_id = src_gpu.id.handle;
    } else {
      LOG(WARNING) << "No GPU ID available for memory copy operation: "
                   << trace_event->name << ", src_agent_type=" << src_gpu.type
                   << ", dst_agent_type=" << dst_gpu.type;
      trace_event->device_id = 0;  // Invalid ID or default
    }
  }

  trace_event->source = RocmTracerEventSource::Activity;
  trace_event->domain = RocmTracerEventDomain::HIP_OPS;
  trace_event->start_time_ns = rec.start_timestamp;
  trace_event->end_time_ns = rec.end_timestamp;
  trace_event->correlation_id = rec.correlation_id.internal;
  trace_event->annotation =
      annotation_map()->LookUp(trace_event->correlation_id);
  trace_event->thread_id = rec.thread_id;
  // we do not know valid stream ID for memcpy
  // rec.stream_id.handle;
  trace_event->stream_id = RocmTracerEvent::kInvalidStreamId;
  trace_event->memcpy_info = MemcpyDetails{
      .num_bytes = rec.bytes,
      .destination = static_cast<uint32_t>(dst_gpu.id.handle),
      .async = false,
  };

  VLOG(2) << "copy bytes: " << trace_event->memcpy_info.num_bytes
          << " stream: " << trace_event->stream_id << " src_id "
          << trace_event->device_id << " dst_id "
          << trace_event->memcpy_info.destination;
}

void RocmTracer::KernelEvent(const rocprofiler_record_header_t* hdr,
                             RocmTracerEvent* trace_event) {
  const auto& rec =
      *static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
          hdr->payload);

  const auto& kinfo = rec.dispatch_info;
  trace_event->type = RocmTracerEventType::Kernel;
  trace_event->source = RocmTracerEventSource::Activity;
  trace_event->domain = RocmTracerEventDomain::HIP_OPS;
  trace_event->name = "??";
  trace_event->start_time_ns = rec.start_timestamp;
  trace_event->end_time_ns = rec.end_timestamp;
  trace_event->device_id = agents_[kinfo.agent_id.handle].id.handle;
  trace_event->correlation_id = rec.correlation_id.internal;
  trace_event->annotation =
      annotation_map()->LookUp(trace_event->correlation_id);
  trace_event->thread_id = rec.thread_id;
  trace_event->stream_id = kinfo.queue_id.handle;
  trace_event->kernel_info = KernelDetails{
      .private_segment_size = kinfo.private_segment_size,
      .group_segment_size = kinfo.group_segment_size,
      .workgroup_x = kinfo.workgroup_size.x,
      .workgroup_y = kinfo.workgroup_size.y,
      .workgroup_z = kinfo.workgroup_size.z,
      .grid_x = kinfo.grid_size.x,
      .grid_y = kinfo.grid_size.y,
      .grid_z = kinfo.grid_size.z,
      .func_ptr = nullptr,
  };

  auto it = kernel_info_.find(kinfo.kernel_id);
  if (it != kernel_info_.end()) trace_event->name = it->second.name;
}

void RocmTracer::TracingCallback(rocprofiler_context_id_t context,
                                 rocprofiler_buffer_id_t buffer_id,
                                 rocprofiler_record_header_t** headers,
                                 size_t num_headers, uint64_t drop_count) {
  if (collector() == nullptr) {
    return;
  }
  if (num_headers == 0) {
    return;
  }
  assert(drop_count == 0 && "drop count should be zero for lossless policy");

  if (headers == nullptr) {
    LOG(ERROR)
        << "rocprofiler invoked a buffer callback with a null pointer to the "
           "array of headers. this should never happen";
    return;
  }

  for (size_t i = 0; i < num_headers; i++) {
    RocmTracerEvent event;
    auto header = headers[i];

    if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;

    switch (header->kind) {
      case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API:
        HipApiEvent(header, &event);
        break;

      case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
        KernelEvent(header, &event);
        break;

      case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
        MemcpyEvent(header, &event);
        break;

      default:
        continue;
    }  // switch

    absl::MutexLock lock(collector_mutex_);
    if (collector()) {
      collector()->AddEvent(std::move(event), false);
    }
  }  // for
}

void RocmTracer::CodeObjectCallback(
    rocprofiler_callback_tracing_record_t record, void* callback_data) {
  if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
      record.operation == ROCPROFILER_CODE_OBJECT_LOAD) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // mainly for debugging
      LOG(WARNING)
          << "Callback phase unload without registering kernel names ...";
    }
  } else if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
             record.operation ==
                 ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
      absl::MutexLock lock(kernel_lock_);
      kernel_info_.emplace(
          data->kernel_id,
          ProfilerKernelInfo{tsl::port::MaybeAbiDemangle(data->kernel_name),
                             *data});
    } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // FIXME: clear these?  At minimum need kernel names at shutdown, async
      // completion We don't erase it just in case a buffer callback still needs
      // this kernel_info_.erase(data->kernel_id);
    }
  }
}

static void code_object_callback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t* user_data,
                                 void* callback_data) {
  RocmTracer::GetRocmTracerSingleton().CodeObjectCallback(record,
                                                          callback_data);
}

static void tool_tracing_callback(rocprofiler_context_id_t context,
                                  rocprofiler_buffer_id_t buffer_id,
                                  rocprofiler_record_header_t** headers,
                                  size_t num_headers, void* user_data,
                                  uint64_t drop_count) {
  RocmTracer::GetRocmTracerSingleton().TracingCallback(
      context, buffer_id, headers, num_headers, drop_count);
}

int RocmTracer::toolInit(rocprofiler_client_finalize_t fini_func,
                         void* tool_data) {
  // Gather API names
  name_info_ = GetCallbackTracingNames();

  // Gather agent info
  num_gpus_ = 0;
  for (const auto& agent : GetGpuDeviceAgents()) {
    LOG(INFO) << "agent id = " << agent.id.handle
              << ", dev = " << agent.device_id
              << ", name = " << (agent.name ? agent.name : "null");
    agents_[agent.id.handle] = agent;
    if (agent.type == ROCPROFILER_AGENT_TYPE_GPU) {
      num_gpus_++;
    }
  }

  // Utility context to gather code‑object info
  rocprofiler_create_context(&utility_context_);

  // buffered tracing
  auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

  rocprofiler_configure_callback_tracing_service(
      utility_context_, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
      code_object_ops.data(), code_object_ops.size(), code_object_callback,
      nullptr);

  rocprofiler_start_context(utility_context_);
  LOG(INFO) << "rocprofiler start utilityContext";

  // a multiple of the page size, and the gap allows the buffer to absorb bursts
  // of GPU events
  constexpr auto buffer_size_bytes = 100 * 4096;
  constexpr auto buffer_watermark_bytes = 40 * 4096;

  // Utility context to gather code‑object info
  rocprofiler_create_context(&context_);

  rocprofiler_create_buffer(context_, buffer_size_bytes, buffer_watermark_bytes,
                            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                            tool_tracing_callback, tool_data, &buffer_);

  rocprofiler_configure_buffer_tracing_service(
      context_, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0,
      buffer_);

  rocprofiler_configure_buffer_tracing_service(
      context_, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0,
      buffer_);

  rocprofiler_configure_buffer_tracing_service(
      context_, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, buffer_);

  {
    // for annotations
    const rocprofiler_tracing_operation_t* hip_ops = nullptr;
    size_t hip_ops_count = 0;

    rocprofiler_configure_callback_tracing_service(
        context_, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, hip_ops,
        hip_ops_count,
        [](rocprofiler_callback_tracing_record_t record,
           rocprofiler_user_data_t*, void*) {
          if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
            const std::string& annotation =
                tsl::profiler::AnnotationStack::Get();
            if (!annotation.empty()) {
              RocmTracer::GetRocmTracerSingleton().annotation_map()->Add(
                  record.correlation_id.internal, annotation);
            }
          }
        },
        nullptr);
  }

  auto client_thread = rocprofiler_callback_thread_t{};
  rocprofiler_create_callback_thread(&client_thread);
  rocprofiler_assign_callback_thread(buffer_, client_thread);

  int isValid = 0;
  rocprofiler_context_is_valid(context_, &isValid);
  if (isValid == 0) {
    context_.handle = 0;  // Leak on failure.
    return -1;
  }

  return 0;
}

void RocmTracer::toolFinalize(void* tool_data) {
  auto& obj = RocmTracer::GetRocmTracerSingleton();
  LOG(INFO) << "Calling toolFinalize!";
  rocprofiler_stop_context(obj.utility_context_);
  obj.utility_context_.handle = 0;
  rocprofiler_stop_context(obj.context_);
  // flush buffer here or in disable?
  obj.context_.handle = 0;
}

void RocmTracer::Disable() {
  rocprofiler_status_t status = rocprofiler_flush_buffer(buffer_);
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    LOG(WARNING) << "rocprofiler_flush_buffer failed with error " << status;
  }
  absl::MutexLock lock(collector_mutex_);
  collector_->Flush();
  collector_ = nullptr;
  api_tracing_enabled_ = false;
  activity_tracing_enabled_ = false;
  LOG(INFO) << "GpuTracer stopped";
}

// ----------------------------------------------------------------------------
// Helper that returns all device agents (GPU + CPU for now).
// ----------------------------------------------------------------------------
std::vector<rocprofiler_agent_v0_t> GetGpuDeviceAgents() {
  std::vector<rocprofiler_agent_v0_t> agents;

  rocprofiler_query_available_agents_cb_t iterate_cb =
      [](rocprofiler_agent_version_t agents_ver, const void** agents_arr,
         size_t num_agents, void* udata) {
        if (agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0) {
          LOG(ERROR) << "unexpected rocprofiler agent version: " << agents_ver;
          return ROCPROFILER_STATUS_ERROR;
        }
        auto* agents_vec =
            static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for (size_t i = 0; i < num_agents; ++i) {
          const auto* agent =
              static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
          agents_vec->push_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
      };

  rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                     iterate_cb, sizeof(rocprofiler_agent_t),
                                     static_cast<void*>(&agents));
  return agents;
}

static int toolInitStatic(rocprofiler_client_finalize_t finalize_func,
                          void* tool_data) {
  return RocmTracer::GetRocmTracerSingleton().toolInit(finalize_func,
                                                       tool_data);
}

// ----------------------------------------------------------------------------
// C‑linkage entry‑point expected by rocprofiler-sdk.
// ----------------------------------------------------------------------------
extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    uint32_t version, const char* runtime_version, uint32_t priority,
    rocprofiler_client_id_t* id) {
  auto& obj = RocmTracer::GetRocmTracerSingleton();  // Ensure constructed,
                                                     // critical for tracing.

  id->name = "XLA-with-rocprofiler-sdk";
  obj.client_id_ = id;

  LOG(INFO) << "Configure rocprofiler-sdk...";

  const uint32_t major = version / 10000;
  const uint32_t minor = (version % 10000) / 100;
  const uint32_t patch = version % 100;

  LOG(INFO) << absl::StrFormat(
      "%s Configure XLA with rocprofv3... (priority=%u) is using "
      "rocprofiler-sdk v%u.%u.%u (%s)",
      id->name, static_cast<unsigned>(priority), static_cast<unsigned>(major),
      static_cast<unsigned>(minor), static_cast<unsigned>(patch),
      runtime_version ? runtime_version : "unknown");

  static rocprofiler_tool_configure_result_t cfg{
      sizeof(rocprofiler_tool_configure_result_t), &toolInitStatic,
      &RocmTracer::toolFinalize, nullptr};

  return &cfg;
}

}  // namespace profiler
}  // namespace xla

void __attribute__((constructor)) init_rocm_lib() {
  rocprofiler_force_configure(xla::profiler::rocprofiler_configure);
}
