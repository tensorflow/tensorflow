/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_driver.h"

#include <stdint.h>
#include <stdlib.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/const_init.h"
#include "absl/container/inlined_vector.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/gpu/gpu_diagnostics.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/platform/status.h"
#include "tsl/platform/threadpool.h"

namespace stream_executor {
namespace gpu {

absl::Mutex CreatedContexts::mu_{absl::kConstInit};

namespace {

// Returns the current context and checks that it is in the set of CUDA contexts
// created by StreamExecutor (to ensure that the CUDA runtime didn't create a
// context behind our backs).
CUcontext CurrentContext() {
  CUcontext current = cuda::CurrentContextOrDie();
  if (current != nullptr && !CreatedContexts::Has(current)) {
    LOG(FATAL) << "current context was not created by the StreamExecutor "
                  "cuda_driver API: "
               << current
               << "; a CUDA runtime call "
                  "was likely performed without using a StreamExecutor context";
  }
  return current;
}

// CUDA driver routines may require a large amount of stack (particularly
// cuModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
tsl::thread::ThreadPool* GetDriverExecutor() {
  static tsl::thread::ThreadPool* thread_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), tsl::ThreadOptions(), "cuda_driver", 1);
  return thread_pool;
}

}  // namespace

namespace {

thread_local struct ThreadLocalData {
  GpuContext* context;
  int device_ordinal;
  int depth;
} tls_data = {};

}  // namespace

ScopedActivateContext::ScopedActivateContext(GpuContext* cuda_context) {
  auto* tls = &tls_data;

  // If this is an outermost scope, we must not assume that the CUDA context has
  // been left in the same state we left it. Other code may have run on this
  // thread and altered the context.
  if (tls->depth == 0) {
    VLOG(3) << "ScopedActivateContext switching to "
            << cuda_context->device_ordinal();
    TF_CHECK_OK(cuda::ToStatus(cuCtxSetCurrent(cuda_context->context()),
                               "Failed setting context"));
    tls->depth = 1;
    tls->device_ordinal = cuda_context->device_ordinal();
    tls->context = cuda_context;
    to_restore_ = nullptr;
    return;
  }

  tls->depth++;
  if (tls->device_ordinal == cuda_context->device_ordinal()) {
    DCHECK_EQ(CurrentContext(), cuda_context->context());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching context from "
          << tls->device_ordinal << " to " << cuda_context->device_ordinal();

  to_restore_ = tls->context;
  // Set the context and update thread local.
  TF_CHECK_OK(cuda::ToStatus(cuCtxSetCurrent(cuda_context->context()),
                             "Failed setting context"));
  tls->device_ordinal = cuda_context->device_ordinal();
  tls->context = cuda_context;
}

ScopedActivateContext::~ScopedActivateContext() {
  auto* tls = &tls_data;

  tls->depth--;
  DCHECK_GE(tls->depth, 0);
  if (to_restore_ == nullptr) {
    // Leave context, tls->device_ordinal, and tls->context set.
    return;
  }

  // Set context and update thread local.
  TF_CHECK_OK(cuda::ToStatus(cuCtxSetCurrent(to_restore_->context()),
                             "Failed setting context"));
  tls->device_ordinal = to_restore_->device_ordinal();
  tls->context = to_restore_;
}

namespace {

// Returns a stringified device number associated with pointer, primarily for
// logging purposes. Returns "?" if the device could not be successfully
// queried.
std::string CUDAPointerToDeviceString(CUdeviceptr pointer) {
  auto value = GpuDriver::GetPointerDevice(pointer);
  if (value.ok()) {
    return absl::StrCat(value.value());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified memory space associated with pointer, primarily for
// logging purposes. Returns "?" if the memory space could not be successfully
// queried.
std::string CUDAPointerToMemorySpaceString(CUdeviceptr pointer) {
  auto value = GpuDriver::GetPointerMemorySpace(pointer);
  if (value.ok()) {
    return MemoryTypeString(value.value());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified representation of whether or not peer access is
// permitted between the "from" and "to" pointers' associated contexts,
// primarily for logging purposes. Returns "error" if an error is encountered
// in the process of querying.
std::string CUDAPointersToCanAccessString(CUdeviceptr from, CUdeviceptr to) {
  auto from_context = GpuDriver::GetPointerContext(from);
  if (!from_context.ok()) {
    LOG(ERROR) << "could not retrieve source pointer's context: "
               << from_context.status();
    return "source ptr error";
  }
  auto to_context = GpuDriver::GetPointerContext(to);
  if (!to_context.ok()) {
    LOG(ERROR) << "could not retrieve destination pointer's context: "
               << to_context.status();
    return "destination ptr error";
  }
  return GpuDriver::CanEnablePeerAccess(from_context.value(),
                                        to_context.value())
             ? "true"
             : "false";
}

// Actually performs the work of CUDA initialization. Wrapped up in one-time
// execution guard.
static absl::Status InternalInit() {
  absl::Status status =
      cuda::ToStatus(cuInit(0 /* = flags */), "Failed call to cuInit");
  if (status.ok()) {
    return status;
  }

  LOG(ERROR) << "failed call to cuInit: " << status;

  Diagnostician::LogDiagnosticInformation();
  return status;
}

// Synchronize with spinlocks.
const char kScheduleSpinString[] = "spin";
// Synchronize with spinlocks that also call CPU yield instructions.
const char kScheduleYieldString[] = "yield";
// Synchronize with a "synchronization primitive" (e.g. mutex).
const char kScheduleBlockingSyncString[] = "blocking_sync";

int GetFlagsFromEnv() {
  const char* gpu_schedule_string =
      std::getenv("TF_CUDA_PLATFORM_GPU_DEVICE_SCHEDULE");

  if (gpu_schedule_string == nullptr) {
    return 0;
  }

  unsigned device_flags = 0;
  if (strcmp(kScheduleSpinString, gpu_schedule_string) == 0) {
    device_flags = CU_CTX_SCHED_SPIN;
  } else if (strcmp(kScheduleYieldString, gpu_schedule_string) == 0) {
    device_flags = CU_CTX_SCHED_YIELD;
  } else if (strcmp(kScheduleBlockingSyncString, gpu_schedule_string) == 0) {
    device_flags = CU_CTX_SCHED_BLOCKING_SYNC;
  } else {
    LOG(QFATAL) << "Unknown option for environment variable "
                   "TF_CUDA_PLATFORM_GPU_DEVICE_SCHEDULE "
                << gpu_schedule_string << " should be one of {"
                << kScheduleBlockingSyncString << ", " << kScheduleSpinString
                << ", " << kScheduleYieldString << "}";
  }

  return device_flags;
}

}  // namespace

absl::StatusOr<CUresult> QueryEvent(GpuContext* context, CUevent event) {
  ScopedActivateContext activated{context};
  CUresult res = cuEventQuery(event);
  if (res != CUDA_SUCCESS && res != CUDA_ERROR_NOT_READY) {
    return cuda::ToStatus(res, ("failed to query event"));
  }

  return res;
}

absl::Status GpuDriver::Init() {
  // Cached return value from calling InternalInit(), as cuInit need only be
  // called once, but GpuDriver::Init may be called many times.
  static absl::Status* init_retval = [] {
    return new absl::Status(InternalInit());
  }();
  return *init_retval;
}

absl::Status GpuDriver::GetDevice(int device_ordinal, CUdevice* device) {
  return cuda::ToStatus(cuDeviceGet(device, device_ordinal),
                        "Failed call to cuDeviceGet");
}

absl::Status GpuDriver::GetDeviceName(CUdevice device,
                                      std::string* device_name) {
  static const size_t kCharLimit = 64;
  absl::InlinedVector<char, 4> chars(kCharLimit);
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGetName(chars.begin(), kCharLimit - 1, device),
                     "Failed to get device name"));
  chars[kCharLimit - 1] = '\0';
  *device_name = chars.begin();
  return absl::OkStatus();
}

absl::Status GpuDriver::CreateContext(int device_ordinal, CUdevice device,
                                      GpuContext** context) {
  *context = nullptr;

  int flags = GetFlagsFromEnv();

  unsigned int former_primary_context_flags;
  int former_primary_context_is_active;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDevicePrimaryCtxGetState(device, &former_primary_context_flags,
                                 &former_primary_context_is_active)));
  if (former_primary_context_flags != flags) {
    if (former_primary_context_is_active) {
      LOG(ERROR)
          << "The primary context is active and has a different flag set ("
          << former_primary_context_flags << ") than the desired flag set ("
          << flags << ").";
    } else {
      TF_RETURN_IF_ERROR(
          cuda::ToStatus(cuDevicePrimaryCtxSetFlags(device, flags)));
    }
  }

  CUcontext former_context = cuda::CurrentContextOrDie();
  CUcontext new_context;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuDevicePrimaryCtxRetain(&new_context, device)));
  if (former_context != nullptr) {
    CUdevice former_device;
    if (cuCtxGetDevice(&former_device) == CUDA_SUCCESS) {
      if (former_device == device) {
        if (former_context == new_context) {
          VLOG(2) << "The primary context " << former_context << " for device "
                  << device
                  << " exists before initializing the StreamExecutor.";
        } else {
          LOG(WARNING) << "A non-primary context " << former_context
                       << " for device " << device
                       << " exists before initializing the StreamExecutor. The "
                       << "primary context is now " << new_context << ". We "
                       << "haven't verified StreamExecutor works with that.";
        }
      }
    } else {
      LOG(ERROR) << "Failed to get the device of the current context "
                 << former_context;
    }
  }
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuCtxSetCurrent(former_context)));

  *context = CreatedContexts::Add(new_context, device_ordinal);
  CHECK(*context != nullptr)
      << "success in this call must entail non-null result";
  VLOG(2) << "created or reused context " << new_context << " for this thread";
  return absl::OkStatus();
}

void GpuDriver::DestroyContext(GpuContext* context) {
  if (context == nullptr) {
    return;
  }
  auto status = cuda::ToStatus(cuCtxPushCurrent(context->context()));
  if (!status.ok()) {
    LOG(ERROR) << "failed to Push CUDA context; leaking: " << status;
  }
  CUdevice device;
  cuCtxGetDevice(&device);
  cuCtxPopCurrent(nullptr);

  status = cuda::ToStatus(cuDevicePrimaryCtxRelease(device));

  if (!status.ok()) {
    LOG(ERROR) << "failed to release CUDA context; leaking: " << status;
  }

  CreatedContexts::Remove(context->context());
}

absl::Status GpuDriver::FuncGetAttribute(CUfunction_attribute attribute,
                                         CUfunction func,
                                         int* attribute_value) {
  return cuda::ToStatus(
      cuFuncGetAttribute(attribute_value, attribute, func),
      absl::StrCat("Failed to query kernel attribute: ", attribute));
}

absl::Status GpuDriver::FuncSetCacheConfig(CUfunction function,
                                           CUfunc_cache cache_config) {
  return cuda::ToStatus(cuFuncSetCacheConfig(function, cache_config),
                        "Failed to set CUDA kernel cache config");
}

absl::StatusOr<CUsharedconfig> GpuDriver::ContextGetSharedMemConfig(
    GpuContext* context) {
  CUsharedconfig shared_mem_config;
  ScopedActivateContext activation(context);
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuCtxGetSharedMemConfig(&shared_mem_config),
                                    "Failed to get shared memory config"));
  return shared_mem_config;
}

absl::Status GpuDriver::ContextSetSharedMemConfig(
    GpuContext* context, CUsharedconfig shared_mem_config) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuCtxSetSharedMemConfig(shared_mem_config),
                        "Failed to set shared memory config");
}

absl::Status GpuDriver::CreateGraph(CUgraph* graph) {
  VLOG(2) << "Create new CUDA graph";
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuGraphCreate(graph, /*flags=*/0),
                                    "Failed to create CUDA graph"));
  VLOG(2) << "Created CUDA graph " << *graph;
  return absl::OkStatus();
}

absl::Status GpuDriver::DestroyGraph(CUgraph graph) {
  VLOG(2) << "Destroy CUDA graph " << graph;
  return cuda::ToStatus(cuGraphDestroy(graph), "Failed to destroy CUDA graph");
}

static std::string_view StreamCaptureModeToString(
    GpuDriver::StreamCaptureMode mode) {
  switch (mode) {
    case GpuDriver::StreamCaptureMode::kGlobal:
      return "global";
    case GpuDriver::StreamCaptureMode::kThreadLocal:
      return "threadlocal";
    case GpuDriver::StreamCaptureMode::kRelaxed:
      return "relaxed";
  }
}

absl::Status GpuDriver::StreamBeginCapture(CUstream stream,
                                           StreamCaptureMode mode) {
  CUstreamCaptureMode cu_mode;
  switch (mode) {
    case StreamCaptureMode::kGlobal:
      cu_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
      break;
    case StreamCaptureMode::kThreadLocal:
      cu_mode = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
      break;
    case StreamCaptureMode::kRelaxed:
      cu_mode = CU_STREAM_CAPTURE_MODE_RELAXED;
      break;
  }

  VLOG(2) << "Beginning stream " << stream << " capture in "
          << StreamCaptureModeToString(mode) << " mode";
  return cuda::ToStatus(cuStreamBeginCapture(stream, cu_mode),
                        "Failed to begin stream capture");
}

absl::Status GpuDriver::StreamBeginCaptureToGraph(CUstream stream,
                                                  CUgraph graph,
                                                  StreamCaptureMode mode) {
  CUstreamCaptureMode cu_mode;
  switch (mode) {
    case StreamCaptureMode::kGlobal:
      cu_mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
      break;
    case StreamCaptureMode::kThreadLocal:
      cu_mode = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
      break;
    case StreamCaptureMode::kRelaxed:
      cu_mode = CU_STREAM_CAPTURE_MODE_RELAXED;
      break;
  }

#if CUDA_VERSION >= 12030
  VLOG(2) << "Beginning stream " << stream << " capture in "
          << StreamCaptureModeToString(mode) << " mode to graph " << graph;
  return cuda::ToStatus(
      cuStreamBeginCaptureToGraph(stream, graph,
                                  /*dependencies=*/nullptr,
                                  /*dependencyData=*/nullptr,
                                  /*numDependencies=*/0, cu_mode),
      "Failed to begin stream capture to graph");
#else
  return absl::UnimplementedError(
      "StreamBeginCaptureToGraph is not implemented");
#endif  // CUDA_VERSION >= 12030
}

absl::Status GpuDriver::StreamEndCapture(CUstream stream, CUgraph* graph) {
  VLOG(2) << "End stream " << stream << " capture";

  return cuda::ToStatus(cuStreamEndCapture(stream, graph),
                        "Failed to end stream capture");
}

absl::Status GpuDriver::GraphInstantiate(CUgraphExec* exec, CUgraph graph,
                                         const GraphInstantiateFlags& flags) {
  VLOG(2) << "Instantiate CUDA executable graph from graph " << graph << " ("
          << "auto_free_on_launch=" << flags.auto_free_on_launch << ", "
          << "device_launch=" << flags.device_launch << ", "
          << "use_node_priority=" << flags.use_node_prirotiy << ", "
          << "upload=" << flags.upload << ")";

#if CUDA_VERSION >= 12000
  uint64_t cu_flags = 0;
  if (flags.auto_free_on_launch)
    cu_flags |= CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
  if (flags.use_node_prirotiy)
    cu_flags |= CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
  if (flags.device_launch)
    cu_flags |= CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH;
  if (flags.upload) cu_flags |= CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD;

  return cuda::ToStatus(cuGraphInstantiate(exec, graph, cu_flags),
                        "Failed to instantiate CUDA graph");
#else
  return cuda::ToStatus(cuGraphInstantiate(exec, graph, nullptr, nullptr, 0),
                        "Failed to instantiate CUDA graph");
#endif  // CUDA_VERSION >= 12000
}

absl::Status GpuDriver::GraphLaunch(CUgraphExec exec, CUstream stream) {
  VLOG(2) << "Launching CUDA executable graph " << exec << " on a stream "
          << stream;
  return cuda::ToStatus(cuGraphLaunch(exec, stream),
                        "Failed to launch CUDA graph");
}

absl::Status GpuDriver::GraphNodeSetEnabled(CUgraphExec exec, CUgraphNode node,
                                            bool enabled) {
  // Node is enabled if value != 0, otherwise the node is disabled.
  unsigned value = enabled ? 1 : 0;
  VLOG(2) << "Set CUDA executable graph " << exec << " node " << node
          << " enabled flag to " << value;
  return cuda::ToStatus(cuGraphNodeSetEnabled(exec, node, value),
                        "Failed to set CUDA graph node enabled flag");
}

absl::Status GpuDriver::GraphExecUpdate(CUgraphExec exec, CUgraph graph,
                                        GraphExecUpdateResultInfo* result) {
  VLOG(2) << "Update CUDA graph executable " << exec << " with graph " << graph;

#if CUDA_VERSION >= 12000
  CUgraphExecUpdateResultInfo cu_result;
  memset(&cu_result, 0, sizeof(cu_result));
  CUresult err_code = cuGraphExecUpdate(exec, graph, &cu_result);
  auto cu_result_enum = cu_result.result;
  if (cu_result.errorFromNode) {
    result->error_from_node = cu_result.errorFromNode;
  }
  if (cu_result.errorNode) {
    result->error_node = cu_result.errorNode;
  }
#else
  CUgraphExecUpdateResult cu_result;
  CUresult err_code = cuGraphExecUpdate(exec, graph, nullptr, &cu_result);
  auto cu_result_enum = cu_result;
#endif  // CUDA_VERSION >= 12000

  switch (cu_result_enum) {
    case CU_GRAPH_EXEC_UPDATE_SUCCESS:
      result->result = GraphExecUpdateResult::kSuccess;
      break;
    case CU_GRAPH_EXEC_UPDATE_ERROR:
      result->result = GraphExecUpdateResult::kError;
      break;
    case CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED:
      result->result = GraphExecUpdateResult::kTopologyChanged;
      break;
    case CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED:
      result->result = GraphExecUpdateResult::kNodeTypeChanged;
      break;
    case CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED:
      result->result = GraphExecUpdateResult::kFunctionChanged;
      break;
    case CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED:
      result->result = GraphExecUpdateResult::kParametersChanged;
      break;
    case CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED:
      result->result = GraphExecUpdateResult::kNotSupported;
      break;
#if CUDA_VERSION >= 12000
    case CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE:
      result->result = GraphExecUpdateResult::kUnsupportedFunctionChange;
      break;
    case CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED:
      result->result = GraphExecUpdateResult::kAttributesChanged;
      break;
#endif  // CUDA_VERSION >= 12000
    default:
      return absl::InternalError("Unknown graph update result");
  }
  return cuda::ToStatus(err_code, "Failed to update CUDA graph");
}

absl::StatusOr<GpuDriver::GraphNodeType> GpuDriver::GraphNodeGetType(
    CUgraphNode node) {
  CUgraphNodeType cu_node_type;
  memset(&cu_node_type, 0, sizeof(cu_node_type));
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuGraphNodeGetType(node, &cu_node_type),
                                    "Failed to get CUDA graph node type"));

  switch (cu_node_type) {
    case CU_GRAPH_NODE_TYPE_KERNEL:
      return GraphNodeType::kKernel;
    case CU_GRAPH_NODE_TYPE_MEMCPY:
      return GraphNodeType::kMemcpy;
    case CU_GRAPH_NODE_TYPE_MEMSET:
      return GraphNodeType::kMemset;
    case CU_GRAPH_NODE_TYPE_HOST:
      return GraphNodeType::kHost;
    case CU_GRAPH_NODE_TYPE_GRAPH:
      return GraphNodeType::kGraph;
    case CU_GRAPH_NODE_TYPE_EMPTY:
      return GraphNodeType::kEmpty;
#if CUDA_VERSION >= 12000
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT:
      return GraphNodeType::kWaitEvent;
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
      return GraphNodeType::kEventRecord;
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL:
      return GraphNodeType::kExtSemasSignal;
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT:
      return GraphNodeType::kExtSemasWait;
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC:
      return GraphNodeType::kMemAlloc;
    case CU_GRAPH_NODE_TYPE_MEM_FREE:
      return GraphNodeType::kMemFree;
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP:
      return GraphNodeType::kBatchMemOp;
#endif  // CUDA_VERSION >= 12000
    default:
      return absl::InternalError("Unknown graph node type");
  }

  return absl::InternalError("Invalid CUDA graph node type");
}

absl::StatusOr<std::vector<GpuGraphNodeHandle>>
GpuDriver::GraphNodeGetDependencies(GpuGraphNodeHandle node) {
  VLOG(2) << "Get CUDA graph node " << node << " dependencies";

  std::vector<CUgraphNode> dependencies;

  size_t num_dependencies = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphNodeGetDependencies(node, nullptr, &num_dependencies),
      "Failed to get CUDA graph node depedencies size"));

  dependencies.resize(num_dependencies, nullptr);
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphNodeGetDependencies(node, dependencies.data(), &num_dependencies),
      "Failed to get CUDA graph node depedencies"));

  return dependencies;
}

absl::Status GpuDriver::DestroyGraphExec(CUgraphExec exec) {
  VLOG(2) << "Destroying CUDA executable graph " << exec;
  return cuda::ToStatus(cuGraphExecDestroy(exec),
                        "Failed to destroy CUDA executable graph");
}

absl::StatusOr<std::string> GpuDriver::GraphDebugDotPrint(
    CUgraph graph, const char* path, bool return_printed_graph) {
#if CUDA_VERSION >= 12000
  VLOG(2) << "Print CUDA graph " << graph << " debug dot file to " << path;

  int flags = CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuGraphDebugDotPrint(graph, path, flags),
                                    "Failed to print gpu graph debug file"));

  if (return_printed_graph) {
    std::string data;
    if (tsl::ReadFileToString(tsl::Env::Default(), path, &data).ok()) {
      return data;
    } else {
      LOG(WARNING) << "failed to read gpu graph debug file " << path;
    }
  }
#endif  // CUDA_VERSION >= 12000

  return std::string(path);
}

absl::Status GpuDriver::DeviceGraphMemTrim(CUdevice device) {
  VLOG(2) << "Trim CUDA device graph memory " << device;
  return cuda::ToStatus(cuDeviceGraphMemTrim(device),
                        "Failed to trim device graph memory");
}

absl::StatusOr<bool> GpuDriver::StreamIsCapturing(CUstream stream) {
  VLOG(2) << "Checking if stream " << stream << " is capturing";

  CUstreamCaptureStatus status;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuStreamIsCapturing(stream, &status),
                                    "Failed to check stream capturing status"));

  return status == CU_STREAM_CAPTURE_STATUS_ACTIVE;
}

absl::Status GpuDriver::GraphConditionalHandleCreate(
    GpuGraphConditionalHandle* handle, CUgraph graph, GpuContext* context,
    unsigned int default_launch_value, unsigned int flags) {
  VLOG(2) << "Create conditional handle for a graph " << graph
          << "; context: " << context
          << "; default_launch_value: " << default_launch_value
          << "; flags: " << flags;

#if CUDA_VERSION >= 12030
  return cuda::ToStatus(
      cuGraphConditionalHandleCreate(handle, graph, context->context(),
                                     default_launch_value, flags),
      "Failed to create conditional handle for a CUDA graph");
#else
  return absl::UnimplementedError(
      "CUDA graph conditional nodes are not implemented");
#endif  // CUDA_VERSION >= 12030
}

static std::string ConditionalTypeToString(
    GpuDriver::GpuGraphConditionalNodeParams::Type type) {
  switch (type) {
    case GpuDriver::GpuGraphConditionalNodeParams::Type::kIf:
      return "IF";
    case GpuDriver::GpuGraphConditionalNodeParams::Type::kWhile:
      return "WHILE";
  }
}

absl::StatusOr<GpuDriver::GpuGraphNodeResult> GpuDriver::GraphAddNode(
    CUgraphNode* node, CUgraph graph, absl::Span<const CUgraphNode> deps,
    const GpuGraphNodeParams& params) {
#if CUDA_VERSION >= 12030
  // Add conditional node to a graph.
  if (auto* conditional = std::get_if<GpuGraphConditionalNodeParams>(&params)) {
    VLOG(2) << "Add conditional node to a graph " << graph
            << "; type: " << ConditionalTypeToString(conditional->type)
            << "; deps: " << deps.size();

    CUgraphNodeParams cu_params;
    memset(&cu_params, 0, sizeof(cu_params));

    cu_params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
    cu_params.conditional.handle = conditional->handle;
    cu_params.conditional.ctx = conditional->context->context();
    cu_params.conditional.size = 1;

    switch (conditional->type) {
      case GpuDriver::GpuGraphConditionalNodeParams::Type::kIf:
        cu_params.conditional.type = CU_GRAPH_COND_TYPE_IF;
        break;
      case GpuDriver::GpuGraphConditionalNodeParams::Type::kWhile:
        cu_params.conditional.type = CU_GRAPH_COND_TYPE_WHILE;
        break;
    }

    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuGraphAddNode(node, graph, deps.data(), deps.size(), &cu_params),
        "Failed to add conditional node to a CUDA graph"));

    GpuGraphConditionalNodeParams::Result result;
    result.graph = cu_params.conditional.phGraph_out[0];

    VLOG(2) << "Created conditional CUDA graph " << result.graph;
    return result;
  }
#endif  // CUDA_VERSION >= 12030

  return absl::UnimplementedError("unsupported node type");
}

absl::Status GpuDriver::GraphAddEmptyNode(CUgraphNode* node, CUgraph graph,
                                          absl::Span<const CUgraphNode> deps) {
  VLOG(2) << "Add empty node to a graph " << graph << "; deps: " << deps.size();

  return cuda::ToStatus(
      cuGraphAddEmptyNode(node, graph, deps.data(), deps.size()),
      "Failed to add empty node to a CUDA graph");
}

absl::Status GpuDriver::GraphAddKernelNode(
    CUgraphNode* node, CUgraph graph, absl::Span<const CUgraphNode> deps,
    absl::string_view kernel_name, CUfunction function, unsigned int grid_dim_x,
    unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, void** kernel_params, void** extra) {
  VLOG(2) << "Add kernel node to a graph " << graph
          << "; kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << "; shmem: " << shared_mem_bytes
          << "; deps: " << deps.size();

  CUDA_KERNEL_NODE_PARAMS params;
  memset(&params, 0, sizeof(params));

  params.func = function;
  params.gridDimX = grid_dim_x;
  params.gridDimY = grid_dim_y;
  params.gridDimZ = grid_dim_z;
  params.blockDimX = block_dim_x;
  params.blockDimY = block_dim_y;
  params.blockDimZ = block_dim_z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = kernel_params;
  params.extra = extra;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return cuda::ToStatus(
      cuGraphAddKernelNode(node, graph, deps.data(), deps.size(), &params),
      "Failed to add kernel node to a CUDA graph");
}

/*static*/ absl::Status GpuDriver::GraphExecKernelNodeSetParams(
    CUgraphExec exec, CUgraphNode node, absl::string_view kernel_name,
    CUfunction function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes,
    void** kernel_params, void** extra) {
  VLOG(2) << "Set kernel node params " << node << " in graph executable "
          << exec << "; kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << "; shmem: " << shared_mem_bytes;

  CUDA_KERNEL_NODE_PARAMS params;
  memset(&params, 0, sizeof(params));

  params.func = function;
  params.gridDimX = grid_dim_x;
  params.gridDimY = grid_dim_y;
  params.gridDimZ = grid_dim_z;
  params.blockDimX = block_dim_x;
  params.blockDimY = block_dim_y;
  params.blockDimZ = block_dim_z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = kernel_params;
  params.extra = extra;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return cuda::ToStatus(cuGraphExecKernelNodeSetParams(exec, node, &params),
                        "Failed to set CUDA graph kernel node params");
}

static CUmemAccess_flags ToCudaMemAccessFlags(
    GpuDriver::MemAccessFlags access_flags) {
  switch (access_flags) {
    case GpuDriver::MemAccessFlags::kNone:
      return CU_MEM_ACCESS_FLAGS_PROT_NONE;
    case GpuDriver::MemAccessFlags::kRead:
      return CU_MEM_ACCESS_FLAGS_PROT_READ;
    case GpuDriver::MemAccessFlags::kReadWrite:
      return CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }
}

static CUmemLocationType ToCudaLocationType(
    GpuDriver::MemLocationType location_type) {
  switch (location_type) {
    case GpuDriver::MemLocationType::kInvalid:
      return CU_MEM_LOCATION_TYPE_INVALID;
    case GpuDriver::MemLocationType::kDevice:
      return CU_MEM_LOCATION_TYPE_DEVICE;
#if CUDA_VERSION >= 12030
    case GpuDriver::MemLocationType::kHost:
      return CU_MEM_LOCATION_TYPE_HOST;
    case GpuDriver::MemLocationType::kHostNuma:
      return CU_MEM_LOCATION_TYPE_HOST_NUMA;
    case GpuDriver::MemLocationType::kHostNumaCurrent:
      return CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT;
#else
    case GpuDriver::MemLocationType::kHost:
    case GpuDriver::MemLocationType::kHostNuma:
    case GpuDriver::MemLocationType::kHostNumaCurrent:
      return CU_MEM_LOCATION_TYPE_INVALID;
#endif  // CUDA_VERSION >= 12030
  }
}

static CUmemAllocationType ToCudaAllocationType(
    GpuDriver::MemAllocationType alocation_type) {
  switch (alocation_type) {
    case GpuDriver::MemAllocationType::kInvalid:
      return CU_MEM_ALLOCATION_TYPE_INVALID;
    case GpuDriver::MemAllocationType::kPinned:
      return CU_MEM_ALLOCATION_TYPE_PINNED;
  }
}

/*static*/ absl::Status GpuDriver::GraphAddMemAllocNode(
    CUgraphNode* node, CUgraph graph, absl::Span<const CUgraphNode> deps,
    GpuDriver::MemAccessFlags access_flags,
    GpuDriver::MemLocationType location_type, int device_id,
    GpuDriver::MemAllocationType allocation_type, uint64_t size,
    CUdeviceptr* d_ptr, uint64_t max_pool_size) {
  CUDA_MEM_ALLOC_NODE_PARAMS params;
  memset(&params, 0, sizeof(params));

  CUmemLocation mem_location;
  mem_location.id = device_id;
  mem_location.type = ToCudaLocationType(location_type);

  CUmemAccessDesc mem_desc;
  mem_desc.flags = ToCudaMemAccessFlags(access_flags);
  mem_desc.location = mem_location;

  CUmemPoolProps mem_pool_props;
  mem_pool_props.allocType = ToCudaAllocationType(allocation_type);
  mem_pool_props.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  mem_pool_props.location = mem_location;
#if CUDA_VERSION >= 12030
  mem_pool_props.maxSize = max_pool_size;
#endif  // CUDA_VERSION >= 12030
  // cuda graph requires reserved space initialized to 0
  memset(mem_pool_props.reserved, 0, sizeof(mem_pool_props.reserved));

  params.accessDescCount = 1;
  params.bytesize = size;
  params.accessDescs = &mem_desc;
  params.poolProps = mem_pool_props;

  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddMemAllocNode(node, graph, deps.data(), deps.size(), &params),
      "Failed to add memory allocation node to a CUDA graph"));

  VLOG(2) << "Add MemAllocNode to a graph " << graph << " size " << size
          << " address " << reinterpret_cast<void*>(params.dptr);

  *d_ptr = params.dptr;
  return absl::OkStatus();
}

/*static*/ absl::StatusOr<std::pair<CUdeviceptr, uint64_t>>
GpuDriver::GraphGetMemAllocNodeParams(CUgraphNode node) {
  CUDA_MEM_ALLOC_NODE_PARAMS params;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphMemAllocNodeGetParams(node, &params),
                     "Failed to get memory allocation node parameter"));
  return std::pair<CUdeviceptr, uint64_t>{params.dptr, params.bytesize};
}

/*static*/ absl::Status GpuDriver::GraphAddMemFreeNode(
    CUgraphNode* node, CUgraph graph, absl::Span<const CUgraphNode> deps,
    CUdeviceptr gpu_dst) {
  return cuda::ToStatus(
      cuGraphAddMemFreeNode(node, graph, deps.data(), deps.size(), gpu_dst),
      "Failed to add memory free node to a CUDA graph");
}

absl::Status GpuDriver::GraphAddMemcpyD2DNode(
    GpuContext* context, CUgraphNode* node, CUgraph graph,
    absl::Span<const CUgraphNode> deps, CUdeviceptr gpu_dst,
    CUdeviceptr gpu_src, uint64_t size) {
  VLOG(2) << "Add memcpy d2d node to a graph " << graph
          << "; dst: " << reinterpret_cast<void*>(gpu_dst)
          << "; src: " << reinterpret_cast<void*>(gpu_src) << "; size: " << size
          << "; context: " << context->context() << "; deps: " << deps.size();

  CUDA_MEMCPY3D params;
  memset(&params, 0, sizeof(params));

  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = gpu_src;
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = gpu_dst;
  params.WidthInBytes = size;
  params.Height = 1;
  params.Depth = 1;

  return cuda::ToStatus(
      cuGraphAddMemcpyNode(node, graph, deps.data(), deps.size(), &params,
                           context->context()),
      "Failed to add memcpy d2d node to a CUDA graph");
}

absl::Status GpuDriver::GraphExecMemcpyD2DNodeSetParams(
    GpuContext* context, GpuGraphExecHandle exec, GpuGraphNodeHandle node,
    GpuDevicePtr gpu_dst, GpuDevicePtr gpu_src, uint64_t size) {
  VLOG(2) << "Set memcpy d2d node params " << node << " in graph executable "
          << exec << "; dst: " << reinterpret_cast<void*>(gpu_dst)
          << "; src: " << reinterpret_cast<void*>(gpu_src) << "; size: " << size
          << "; context: " << context->context();

  CUDA_MEMCPY3D params;
  memset(&params, 0, sizeof(params));

  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = gpu_src;
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = gpu_dst;
  params.WidthInBytes = size;
  params.Height = 1;
  params.Depth = 1;

  return cuda::ToStatus(
      cuGraphExecMemcpyNodeSetParams(exec, node, &params, context->context()),
      "Failed to set memcpy d2d node params");
}

namespace {

struct BitPatternToString {
  std::string operator()(uint8_t pattern) {
    return absl::StrCat("u8:", pattern);
  }
  std::string operator()(uint16_t pattern) {
    return absl::StrCat("u16:", pattern);
  }
  std::string operator()(uint32_t pattern) {
    return absl::StrCat("u32:", pattern);
  }
};

// Broadcasts a pattern value of 1/2/4 bytes to a 4 byte value.
struct BitPatternToValue {
  std::pair<unsigned, unsigned> operator()(uint8_t pattern) {
    unsigned value = pattern;
    return {(value << 24) | (value << 16) | (value << 8) | value,
            /*element_size=*/1};
  }
  std::pair<unsigned, unsigned> operator()(uint16_t pattern) {
    unsigned value = pattern;
    return {(value << 16) | value, /*element_size=*/2};
  }
  std::pair<unsigned, unsigned> operator()(uint32_t pattern) {
    return {pattern, /*element_size=*/4};
  }
};

}  // namespace

absl::Status GpuDriver::GraphAddMemsetNode(
    GpuContext* context, CUgraphNode* node, GpuGraphHandle graph,
    absl::Span<const CUgraphNode> deps, CUdeviceptr dst,
    std::variant<uint8_t, uint16_t, uint32_t> bit_pattern,
    uint64_t num_elements) {
  VLOG(2) << "Add memset node to a graph " << graph
          << "; dst: " << reinterpret_cast<void*>(dst)
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements
          << "; context: " << context->context() << "; deps: " << deps.size();

  CUDA_MEMSET_NODE_PARAMS params;
  memset(&params, 0, sizeof(params));

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  params.dst = dst;
  params.elementSize = element_size;
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = value;
  params.width = num_elements;

  return cuda::ToStatus(
      cuGraphAddMemsetNode(node, graph, deps.data(), deps.size(), &params,
                           context->context()),
      "Failed to add memset node to a CUDA graph");
}

absl::Status GpuDriver::GraphExecMemsetNodeSetParams(
    GpuContext* context, CUgraphExec exec, CUgraphNode node, CUdeviceptr dst,
    std::variant<uint8_t, uint16_t, uint32_t> bit_pattern,
    uint64_t num_elements) {
  VLOG(2) << "Set memset node params " << node << " in graph executable "
          << exec << "; dst: " << reinterpret_cast<void*>(dst)
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements
          << "; context: " << context->context();

  CUDA_MEMSET_NODE_PARAMS params;
  memset(&params, 0, sizeof(params));

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  params.dst = dst;
  params.elementSize = element_size;
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = value;
  params.width = num_elements;

  return cuda::ToStatus(
      cuGraphExecMemsetNodeSetParams(exec, node, &params, context->context()),
      "Failed to set memset node params");
}

absl::Status GpuDriver::GraphAddChildNode(CUgraphNode* node, CUgraph graph,
                                          absl::Span<const CUgraphNode> deps,
                                          CUgraph child) {
  VLOG(2) << "Create a new node by cloning the child graph " << child
          << " and add it to " << graph << "; deps: " << deps.size();

  return cuda::ToStatus(
      cuGraphAddChildGraphNode(node, graph, deps.data(), deps.size(), child),
      "Failed to create a child graph node and add it to a CUDA graph");
}

/*static*/ absl::Status GpuDriver::GraphExecChildNodeSetParams(CUgraphExec exec,
                                                               CUgraphNode node,
                                                               CUgraph child) {
  VLOG(2) << "Set child node params " << node << " in graph executable " << exec
          << "to params contained in " << child;

  return cuda::ToStatus(cuGraphExecChildGraphNodeSetParams(exec, node, child),
                        "Failed to set CUDA graph child node params");
}

absl::Status GpuDriver::LaunchKernel(
    GpuContext* context, absl::string_view kernel_name, CUfunction function,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes, CUstream stream,
    void** kernel_params, void** extra) {
  ScopedActivateContext activation(context);
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z
          << "; shared_mem_bytes: " << shared_mem_bytes;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return cuda::ToStatus(
      cuLaunchKernel(function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                     block_dim_y, block_dim_z, shared_mem_bytes, stream,
                     kernel_params, extra),
      absl::StrCat("Failed to launch CUDA kernel: ", kernel_name,
                   "; block dims: ", block_dim_x, "x", block_dim_y, "x",
                   block_dim_z, "; grid dims: ", grid_dim_x, "x", grid_dim_y,
                   "x", grid_dim_z,
                   "; shared memory size: ", shared_mem_bytes));
}

absl::Status GpuDriver::LaunchKernel(
    GpuContext* context, absl::string_view kernel_name,
    GpuFunctionHandle function, unsigned int cluster_dim_x,
    unsigned int cluster_dim_y, unsigned int cluster_dim_z,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes,
    GpuStreamHandle stream, void** kernel_params, void** extra) {
  ScopedActivateContext activation(context);
  VLOG(2) << "launching kernel: " << kernel_name << "; cdx: " << cluster_dim_x
          << " cdy: " << cluster_dim_y << " cdz: " << cluster_dim_z
          << " gdx: " << grid_dim_x << " gdy: " << grid_dim_y
          << " gdz: " << grid_dim_z << " bdx: " << block_dim_x
          << " bdy: " << block_dim_y << " bdz: " << block_dim_z
          << "; shared_mem_bytes: " << shared_mem_bytes;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  CUlaunchConfig launch_config;
  memset(&launch_config, 0, sizeof(launch_config));
  launch_config.blockDimX = block_dim_x;
  launch_config.blockDimY = block_dim_y;
  launch_config.blockDimZ = block_dim_z;
  launch_config.gridDimX = grid_dim_x;
  launch_config.gridDimY = grid_dim_y;
  launch_config.gridDimZ = grid_dim_z;
  launch_config.hStream = stream;
  launch_config.sharedMemBytes = shared_mem_bytes;

  CUlaunchAttribute cluster_dims;
  memset(&cluster_dims, 0, sizeof(cluster_dims));
  cluster_dims.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  cluster_dims.value.clusterDim.x = cluster_dim_x;
  cluster_dims.value.clusterDim.y = cluster_dim_y;
  cluster_dims.value.clusterDim.z = cluster_dim_z;

  launch_config.attrs = &cluster_dims;
  launch_config.numAttrs = 1;

  return cuda::ToStatus(
      cuLaunchKernelEx(&launch_config, function, kernel_params, extra),
      absl::StrCat("Failed to launch CUDA kernel: ", kernel_name,
                   "; cluster dims: ", cluster_dim_x, "x", cluster_dim_y, "x",
                   cluster_dim_z, "; block dims: ", block_dim_x, "x",
                   block_dim_y, "x", block_dim_z, "; grid dims: ", grid_dim_x,
                   "x", grid_dim_y, "x", grid_dim_z,
                   "; shared memory size: ", shared_mem_bytes));
}

absl::Status GpuDriver::LoadCubin(GpuContext* context, const char* cubin_bytes,
                                  CUmodule* module) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(
      cuModuleLoadFatBinary(module, cubin_bytes),
      "Failed to load in-memory CUBIN (compiled for a different GPU?).");
}

absl::Status GpuDriver::LoadPtx(GpuContext* context, const char* ptx_contents,
                                CUmodule* module) {
  absl::Notification notification;
  absl::Status ret = absl::OkStatus();
  GetDriverExecutor()->Schedule(
      [context, ptx_contents, module, &ret, &notification]() {
        ScopedActivateContext activation(context);
        void* ptx_data = const_cast<char*>(ptx_contents);
        static const unsigned int kLogBufferBytesLimit = 1024;
        unsigned int error_log_buffer_bytes = kLogBufferBytesLimit;
        unsigned int info_log_buffer_bytes = kLogBufferBytesLimit;
        absl::InlinedVector<char, 4> error_log_buffer(error_log_buffer_bytes);
        absl::InlinedVector<char, 4> info_log_buffer(info_log_buffer_bytes);
        bool log_verbose = true;
        CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                  CU_JIT_ERROR_LOG_BUFFER,
                                  CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                                  CU_JIT_INFO_LOG_BUFFER, CU_JIT_LOG_VERBOSE};
        // Note that the driver API wants the contents of this values to be
        // stored in an array of void*s, so we coerce them accordingly.
        void* option_values[] = {
            absl::bit_cast<void*>(uintptr_t(error_log_buffer_bytes)),
            absl::bit_cast<void*>(error_log_buffer.data()),
            absl::bit_cast<void*>(uintptr_t(info_log_buffer_bytes)),
            absl::bit_cast<void*>(info_log_buffer.data()),
            absl::bit_cast<void*>(uintptr_t(log_verbose))};
        CHECK(TF_ARRAYSIZE(options) == TF_ARRAYSIZE(option_values));

        absl::Status status;
        {
          // TODO(leary) Need to see if NVIDIA can expunge the leakiness in
          // their module loading: see http://b/13248943
          absl::LeakCheckDisabler disabler;
          status = cuda::ToStatus(cuModuleLoadDataEx(
              module, ptx_data, TF_ARRAYSIZE(options), options, option_values));
        }

        // The PTX JIT mutates the values in the option values array to reflect
        // the size of the logs it output; now that we've made the call, read
        // the values back out.
        error_log_buffer_bytes = reinterpret_cast<uintptr_t>(option_values[0]);
        info_log_buffer_bytes = reinterpret_cast<uintptr_t>(option_values[2]);
        CHECK_LE(error_log_buffer_bytes, kLogBufferBytesLimit);
        CHECK_LE(info_log_buffer_bytes, kLogBufferBytesLimit);

        if (!status.ok()) {
          LOG(ERROR) << "failed to load PTX text as a module: " << status;
          // As a precaution for null termination of the API-provided value,
          // ensure that at least the last byte is null.
          error_log_buffer[error_log_buffer_bytes ? error_log_buffer_bytes - 1
                                                  : 0] = '\0';
          LOG(ERROR) << "error log buffer (" << error_log_buffer_bytes
                     << " bytes): " << error_log_buffer.data();
          if (absl::StrContains(error_log_buffer.data(),
                                "Register allocation failed")) {
            ret = absl::ResourceExhaustedError(
                absl::StrFormat("Failed to load PTX text as a module (register "
                                "allocation failed): %s",
                                status.ToString()));
          } else {
            ret = status;
          }
          notification.Notify();
          return;
        }

        VLOG(3) << "PTX compilation info log (" << info_log_buffer_bytes
                << " bytes): " << info_log_buffer.data();
        VLOG(3) << "PTX compilation error log (" << error_log_buffer_bytes
                << " bytes): " << error_log_buffer.data();
        CHECK(module != nullptr);
        notification.Notify();
      });
  notification.WaitForNotification();

  return ret;
}

absl::Status GpuDriver::LoadHsaco(GpuContext* context,
                                  const char* hsaco_contents,
                                  CUmodule* module) {
  return absl::InternalError(
      "Feature not supported on CUDA platform (LoadHsaco)");
}

absl::Status GpuDriver::SynchronousMemsetUint8(GpuContext* context,
                                               CUdeviceptr location,
                                               uint8_t value, size_t size) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuMemsetD8(location, value, size),
                        "Failed to memset memory");
}

absl::Status GpuDriver::SynchronousMemsetUint32(GpuContext* context,
                                                CUdeviceptr location,
                                                uint32_t value,
                                                size_t uint32_count) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuMemsetD32(location, value, uint32_count),
                        "Failed to memset memory");
}

absl::Status GpuDriver::AsynchronousMemsetUint8(GpuContext* context,
                                                CUdeviceptr location,
                                                uint8_t value,
                                                size_t uint32_count,
                                                CUstream stream) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuMemsetD8Async(location, value, uint32_count, stream),
                        "Failed to enqueue async memset operation");
}

absl::Status GpuDriver::AsynchronousMemsetUint32(GpuContext* context,
                                                 CUdeviceptr location,
                                                 uint32_t value,
                                                 size_t uint32_count,
                                                 CUstream stream) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuMemsetD32Async(location, value, uint32_count, stream),
                        "Failed to enqueue async memset operation");
}

bool GpuDriver::AddStreamCallback(GpuContext* context, CUstream stream,
                                  StreamCallback callback, void* data) {
  // Note: flags param is required to be zero according to CUDA 6.0.
  auto status = cuda::ToStatus(cuLaunchHostFunc(stream, callback, data));
  if (!status.ok()) {
    LOG(ERROR) << "unable to add host callback: " << status;
    return false;
  }
  return true;
}

absl::Status GpuDriver::GetModuleFunction(GpuContext* context, CUmodule module,
                                          const char* kernel_name,
                                          CUfunction* function) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  cudaError_t cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return absl::InternalError(
        absl::StrCat("There was an error before calling cuModuleGetFunction (",
                     cuda_error, "): ", cudaGetErrorName(cuda_error), " : ",
                     cudaGetErrorString(cuda_error)));
  }
  return cuda::ToStatus(cuModuleGetFunction(function, module, kernel_name),
                        "Failed to get module function");
}

absl::Status GpuDriver::GetModuleSymbol(GpuContext* context, CUmodule module,
                                        const char* symbol_name,
                                        CUdeviceptr* dptr, size_t* bytes) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  return cuda::ToStatus(
      cuModuleGetGlobal(dptr, bytes, module, symbol_name),
      absl::StrCat("Failed to get symbol '", symbol_name, "'"));
}

void GpuDriver::UnloadModule(GpuContext* context, CUmodule module) {
  ScopedActivateContext activated{context};
  auto status = cuda::ToStatus(cuModuleUnload(module));
  if (!status.ok()) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << status;
  }
}

absl::StatusOr<CUdevice> GpuDriver::DeviceFromContext(GpuContext* context) {
  ScopedActivateContext activated{context};
  CUdevice device = -1;
  auto status = cuda::ToStatus(cuCtxGetDevice(&device));
  if (status.ok()) {
    return device;
  }

  return status;
}

bool GpuDriver::CreateStream(GpuContext* context, CUstream* stream,
                             int priority) {
  ScopedActivateContext activated{context};
  absl::Status status;
  // If the priority is 0, then use the previous api to create the stream with
  // the default priority for backward compatibility. Probably there is no
  // difference in using the new api call but leaving it as is for now.
  if (priority == 0) {
    status = cuda::ToStatus(cuStreamCreate(stream, CU_STREAM_NON_BLOCKING));
  } else {
    status = cuda::ToStatus(
        cuStreamCreateWithPriority(stream, CU_STREAM_NON_BLOCKING, priority));
  }
  if (!status.ok()) {
    LOG(ERROR) << "could not allocate CUDA stream for context "
               << context->context() << ": " << status;
    return false;
  }

  VLOG(2) << "successfully created stream " << *stream << " for context "
          << context->context() << " on thread";
  return true;
}

void GpuDriver::DestroyStream(GpuContext* context, CUstream* stream) {
  if (*stream == nullptr) {
    return;
  }

  ScopedActivateContext activated{context};
  auto status = cuda::ToStatus(cuStreamDestroy(*stream));
  if (!status.ok()) {
    LOG(ERROR) << "failed to destroy CUDA stream for context "
               << context->context() << ": " << status;
  } else {
    VLOG(2) << "successfully destroyed stream " << *stream << " for context "
            << context->context();
    *stream = nullptr;
  }
}

void* GpuDriver::DeviceAllocate(GpuContext* context, uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  ScopedActivateContext activated{context};
  CUdeviceptr result = 0;
  auto status = cuda::ToStatus(cuMemAlloc(&result, bytes));
  if (!status.ok()) {
    // LOG(INFO) because this isn't always important to users (e.g. BFCAllocator
    // implements a retry if the first allocation fails).
    LOG(INFO) << "failed to allocate "
              << tsl::strings::HumanReadableNumBytes(bytes) << " (" << bytes
              << " bytes) from device: " << status;
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  VLOG(2) << "allocated " << ptr << " for context " << context->context()
          << " of " << bytes << " bytes";
  return ptr;
}

void GpuDriver::DeviceDeallocate(GpuContext* context, void* location) {
  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  auto status = cuda::ToStatus(cuMemFree(pointer));
  if (!status.ok()) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << status;
  } else {
    VLOG(2) << "deallocated " << location << " for context "
            << context->context();
  }
}

void* GpuDriver::UnifiedMemoryAllocate(GpuContext* context, uint64_t bytes) {
  ScopedActivateContext activation(context);
  CUdeviceptr result = 0;
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  auto status =
      cuda::ToStatus(cuMemAllocManaged(&result, bytes, CU_MEM_ATTACH_GLOBAL));
  if (!status.ok()) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes unified memory; result: " << status;
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  VLOG(2) << "allocated " << ptr << " for context " << context->context()
          << " of " << bytes << " bytes in unified memory";
  return ptr;
}

void GpuDriver::UnifiedMemoryDeallocate(GpuContext* context, void* location) {
  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  auto status = cuda::ToStatus(cuMemFree(pointer));
  if (!status.ok()) {
    LOG(ERROR) << "failed to free unified memory at " << location
               << "; result: " << status;
  } else {
    VLOG(2) << "deallocated unified memory at " << location << " for context "
            << context->context();
  }
}

void* GpuDriver::HostAllocate(GpuContext* context, uint64_t bytes) {
  ScopedActivateContext activation(context);
  void* host_mem = nullptr;
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  auto status = cuda::ToStatus(
      cuMemHostAlloc(&host_mem, bytes, CU_MEMHOSTALLOC_PORTABLE));
  if (!status.ok()) {
    LOG(ERROR) << "failed to alloc " << bytes << " bytes on host: " << status;
  }
  return host_mem;
}

void GpuDriver::HostDeallocate(GpuContext* context, void* location) {
  ScopedActivateContext activation(context);
  auto status = cuda::ToStatus(cuMemFreeHost(location));
  if (!status.ok()) {
    LOG(ERROR) << "error deallocating host memory at " << location << ": "
               << status;
  }
}

bool GpuDriver::HostRegister(GpuContext* context, void* location,
                             uint64_t bytes) {
  ScopedActivateContext activation(context);
  // "Portable" memory is visible to all CUDA contexts. Safe for our use model.
  auto status = cuda::ToStatus(
      cuMemHostRegister(location, bytes, CU_MEMHOSTREGISTER_PORTABLE));
  if (!status.ok()) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << status;
    return false;
  }
  return true;
}

bool GpuDriver::HostUnregister(GpuContext* context, void* location) {
  ScopedActivateContext activation(context);
  auto status = cuda::ToStatus(cuMemHostUnregister(location));
  if (!status.ok()) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << status;
    return false;
  }
  return true;
}

int GpuDriver::GetGpuStreamPriority(
    GpuContext* context, stream_executor::StreamPriority stream_priority) {
  ScopedActivateContext activation(context);
  if (stream_priority == stream_executor::StreamPriority::Default) {
    return 0;
  }
  int lowest, highest;
  auto status = cuda::ToStatus(cuCtxGetStreamPriorityRange(&lowest, &highest));
  if (!status.ok()) {
    LOG(ERROR)
        << "Could not query stream priority range. Returning default priority.";
    return 0;
  }
  return stream_priority == stream_executor::StreamPriority::Highest ? highest
                                                                     : lowest;
}

absl::StatusOr<GpuDriver::VmemSpan> GpuDriver::ReserveVirtualMemory(
    GpuContext* context, uint64_t bytes) {
  ScopedActivateContext activation(context);
  CUdeviceptr base;
  return cuda::ToStatus(
      cuMemAddressReserve(&base, bytes, /*alignment=*/0,
                          /*addr=*/0, /*flags=*/0),
      absl::StrFormat("error reserving %d bytes of virtual GPU memory", bytes));
}

void GpuDriver::FreeVirtualMemory(GpuContext* context,
                                  GpuDriver::VmemSpan reservation) {
  ScopedActivateContext activation(context);
  auto status = cuda::ToStatus(
      cuMemAddressFree(reservation.base, reservation.size_bytes));
  if (!status.ok()) {
    LOG(ERROR) << "error freeing vmem reservation of size "
               << reservation.size_bytes << " at address " << reservation.base;
  }
}

absl::StatusOr<uint64_t> GpuDriver::GetMinAllocationGranularity(
    GpuDeviceHandle device) {
  CUmemAllocationProp props = {};
  props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = device;

  size_t granularity;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemGetAllocationGranularity(&granularity, &props,
                                    CU_MEM_ALLOC_GRANULARITY_MINIMUM),
      "failed to get min allocation granularity"));
  return granularity;
}

absl::StatusOr<GpuDriver::GenericMemoryHandle> GpuDriver::CreateMemoryHandle(
    GpuContext* context, uint64_t bytes) {
  ScopedActivateContext activation(context);
  auto device = DeviceFromContext(context);
  if (!device.ok()) {
    LOG(ERROR) << "Failed to get device from context" << device.status();
    return device.status();
  }

  CUmemAllocationProp props = {};
  props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = device.value();

  CUmemGenericAllocationHandle mem_handle;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemCreate(&mem_handle, bytes, &props, 0),
      absl::StrFormat("failed to create memory allocation of size %d", bytes)));
  return GpuDriver::GenericMemoryHandle{mem_handle, bytes};
}

void GpuDriver::ReleaseMemoryHandle(GpuContext* context,
                                    GpuDriver::GenericMemoryHandle handle) {
  ScopedActivateContext activation(context);

  auto status = cuda::ToStatus(cuMemRelease(handle.handle));
  if (!status.ok()) {
    LOG(ERROR) << "Failed to release memory handle " << handle.handle
               << " of size " << handle.bytes << ": " << status;
  }
}

absl::Status GpuDriver::MapMemory(
    GpuContext* context, CUdeviceptr va,
    const GpuDriver::GenericMemoryHandle& handle,
    const std::vector<GpuDeviceHandle>& device_handles) {
  ScopedActivateContext activation(context);

  auto device = DeviceFromContext(context);
  if (!device.ok()) {
    return device.status();
  }

  // NB: Zero is the only valid value for both flags and offset.
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemMap(va, handle.bytes, /*offset=*/0, handle.handle, /*flags=*/0)));

  std::vector<CUmemAccessDesc> access_descriptors(device_handles.size());
  for (int i = 0; i < access_descriptors.size(); ++i) {
    access_descriptors[i].location.id = device_handles[i];
    access_descriptors[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_descriptors[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }

  auto status = cuda::ToStatus(cuMemSetAccess(
      va, handle.bytes, access_descriptors.data(), access_descriptors.size()));
  if (!status.ok()) {
    // Unmap the memory that we failed to set access for.
    if (!cuda::ToStatus(cuMemUnmap(va, handle.bytes)).ok()) {
      LOG(ERROR)
          << "Failed to unmap memory in GpuDriver::MapMemory error path.";
    }
    return status;
  }
  return absl::OkStatus();
}

void GpuDriver::UnmapMemory(GpuContext* context, CUdeviceptr va,
                            uint64_t bytes) {
  ScopedActivateContext activation(context);

  auto status = cuda::ToStatus(cuMemUnmap(va, bytes));
  if (!status.ok()) {
    LOG(ERROR) << "Failed to unmap memory at " << va << " of size " << bytes
               << ": " << status;
  }
}

absl::Status GpuDriver::DestroyEvent(GpuContext* context, CUevent* event) {
  if (*event == nullptr) {
    return absl::InvalidArgumentError("input event cannot be null");
  }

  ScopedActivateContext activated{context};
  return cuda::ToStatus(cuEventDestroy(*event), "Error destroying CUDA event");
}

absl::Status GpuDriver::RecordEvent(GpuContext* context, CUevent event,
                                    CUstream stream) {
  ScopedActivateContext activated{context};
  return cuda::ToStatus(cuEventRecord(event, stream),
                        "Error recording CUDA event");
}

bool GpuDriver::GetEventElapsedTime(GpuContext* context,
                                    float* elapsed_milliseconds, CUevent start,
                                    CUevent stop) {
  ScopedActivateContext activated{context};
  // The stop event must have completed in order for cuEventElapsedTime to
  // work.
  auto status = cuda::ToStatus(cuEventSynchronize(stop));
  if (!status.ok()) {
    LOG(ERROR) << "failed to synchronize the stop event: " << status;
    return false;
  }
  status =
      cuda::ToStatus(cuEventElapsedTime(elapsed_milliseconds, start, stop));
  if (!status.ok()) {
    LOG(ERROR) << "failed to get elapsed time between events: " << status;
    return false;
  }

  return true;
}

bool GpuDriver::WaitStreamOnEvent(GpuContext* context, CUstream stream,
                                  CUevent event) {
  ScopedActivateContext activation(context);
  auto status =
      cuda::ToStatus(cuStreamWaitEvent(stream, event, 0 /* = flags */));
  if (!status.ok()) {
    LOG(ERROR) << "could not wait stream on event: " << status;
    return false;
  }

  return true;
}

bool GpuDriver::SynchronizeContext(GpuContext* context) {
  ScopedActivateContext activation(context);
  auto status = cuda::ToStatus(cuCtxSynchronize());
  if (!status.ok()) {
    LOG(ERROR) << "could not synchronize on CUDA context: " << status
               << " :: " << tsl::CurrentStackTrace();
    return false;
  }

  return true;
}

absl::Status GpuDriver::SynchronizeStream(GpuContext* context,
                                          CUstream stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  return cuda::ToStatus(cuStreamSynchronize(stream),
                        "Could not synchronize CUDA stream");
}

bool GpuDriver::IsStreamIdle(GpuContext* context, CUstream stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  CUresult res = cuStreamQuery(stream);
  if (res == CUDA_SUCCESS) {
    return true;
  }

  if (res != CUDA_ERROR_NOT_READY) {
    LOG(ERROR) << "stream in bad state on status query: "
               << cuda::ToStatus(res);
  }
  return false;
}

absl::Status GpuDriver::SynchronousMemcpyD2H(GpuContext* context,
                                             void* host_dst,
                                             CUdeviceptr gpu_src,
                                             uint64_t size) {
  ScopedActivateContext activation(context);
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemcpyDtoH(host_dst, gpu_src, size),
      absl::StrFormat("failed to synchronous memcpy from device to host "
                      "host dst: %p; GPU src: %p; size: %u=0x%x",
                      host_dst, absl::bit_cast<void*>(gpu_src), size, size)));
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return absl::OkStatus();
}

absl::Status GpuDriver::SynchronousMemcpyH2D(GpuContext* context,
                                             CUdeviceptr gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  ScopedActivateContext activation(context);
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemcpyHtoD(gpu_dst, host_src, size),
      absl::StrFormat(
          "failed to synchronous memcpy from host to device: GPU dst: %p;"
          " host src: %p; size: %u=0x%x",
          absl::bit_cast<void*>(gpu_dst), host_src, size, size)));
  VLOG(2) << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return absl::OkStatus();
}

absl::Status GpuDriver::SynchronousMemcpyD2D(GpuContext* context,
                                             CUdeviceptr gpu_dst,
                                             CUdeviceptr gpu_src,
                                             uint64_t size) {
  ScopedActivateContext activation(context);

  CUresult result;
  // CreatedContexts::GetAnyContext() doesn't works when ptr == 0.
  // This happens when the size is 0.
  if (gpu_dst == 0 || gpu_src == 0) {
    result = cuMemcpyDtoD(gpu_dst, gpu_src, size);
  } else {
    // Any context work here.
    CUcontext dst_context =
        CreatedContexts::GetAnyContext(absl::bit_cast<void*>(gpu_dst));
    CUcontext src_context =
        CreatedContexts::GetAnyContext(absl::bit_cast<void*>(gpu_src));

    if (static_cast<void*>(dst_context) == nullptr) {
      absl::StatusOr<GpuContext*> tmp_context = GetPointerContext(gpu_dst);
      if (tmp_context.ok()) {
        dst_context = tmp_context.value()->context();
      }
    }

    if (static_cast<void*>(src_context) == nullptr) {
      absl::StatusOr<GpuContext*> tmp_context = GetPointerContext(gpu_src);
      if (tmp_context.ok()) {
        src_context = tmp_context.value()->context();
      }
    }

    result = cuMemcpyPeer(gpu_dst, dst_context, gpu_src, src_context, size);
  }

  TF_RETURN_IF_ERROR(cuda::ToStatus(
      result,
      absl::StrFormat(
          "failed to synchronous memcpy from host to device: GPU dst: %p; "
          "GPU src: %p; size: %u=0x%x",
          absl::bit_cast<void*>(gpu_dst), absl::bit_cast<void*>(gpu_src), size,
          size)));
  VLOG(2) << "successfully sync memcpy'd d2d of " << size << " bytes";
  return absl::OkStatus();
}

bool GpuDriver::AsynchronousMemcpyD2H(GpuContext* context, void* host_dst,
                                      CUdeviceptr gpu_src, uint64_t size,
                                      CUstream stream) {
  ScopedActivateContext activation(context);
  auto status =
      cuda::ToStatus(cuMemcpyDtoHAsync(host_dst, gpu_src, size, stream));
  if (!status.ok()) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from device to host: %s; host dst: %p; "
        "GPU src: %p; size: %u=0x%x",
        status.ToString(), host_dst, absl::bit_cast<void*>(gpu_src), size,
        size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream;
  return true;
}

bool GpuDriver::AsynchronousMemcpyH2D(GpuContext* context, CUdeviceptr gpu_dst,
                                      const void* host_src, uint64_t size,
                                      CUstream stream) {
  ScopedActivateContext activation(context);
  auto status =
      cuda::ToStatus(cuMemcpyHtoDAsync(gpu_dst, host_src, size, stream));
  if (!status.ok()) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from host to device: %s; GPU dst: %p; "
        "host src: %p; size: %u=0x%x",
        status.ToString(), absl::bit_cast<void*>(gpu_dst), host_src, size,
        size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " from " << host_src << " to " << absl::bit_cast<void*>(gpu_dst)
          << " on stream " << stream;
  return true;
}

bool GpuDriver::AsynchronousMemcpyD2D(GpuContext* context, CUdeviceptr gpu_dst,
                                      CUdeviceptr gpu_src, uint64_t size,
                                      CUstream stream) {
  ScopedActivateContext activation(context);
  CUresult result;

  // In graph capture mode we never have operations that access peer memory, so
  // we can always make a call to cuMemcpyDtoDAsync.
  absl::StatusOr<bool> is_capturing = StreamIsCapturing(stream);
  if (!is_capturing.ok()) {
    LOG(ERROR) << is_capturing.status().message();
    return false;
  }

  if ((gpu_dst == 0 || gpu_src == 0) || (*is_capturing)) {
    // CreatedContexts::GetAnyContext() doesn't works when ptr == 0.
    // This happens when the size is 0.
    result = cuMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream);
  } else {
    // Any context work here.
    CUcontext dst_context =
        CreatedContexts::GetAnyContext(absl::bit_cast<void*>(gpu_dst));
    CUcontext src_context =
        CreatedContexts::GetAnyContext(absl::bit_cast<void*>(gpu_src));

    if (static_cast<void*>(dst_context) == nullptr) {
      absl::StatusOr<GpuContext*> tmp_context = GetPointerContext(gpu_dst);
      if (tmp_context.ok()) {
        dst_context = tmp_context.value()->context();
      }
    }

    if (static_cast<void*>(src_context) == nullptr) {
      absl::StatusOr<GpuContext*> tmp_context = GetPointerContext(gpu_src);
      if (tmp_context.ok()) {
        src_context = tmp_context.value()->context();
      }
    }

    if (dst_context == src_context) {
      // Since the CUDA context is the same, the src and dst are within the same
      // GPU. So we can use cuMemcpyDtoD.
      result = cuMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream);
    } else {
      result = cuMemcpyPeerAsync(gpu_dst, dst_context, gpu_src, src_context,
                                 size, stream);
    }
  }
  if (result != CUDA_SUCCESS) {
    LOG(ERROR) << absl::StrFormat(
        "failed to enqueue async memcpy from device to device: %s"
        "; GPU dst: %p on %s %s"
        "; GPU src: %p on %s %s"
        "; can access? %s; size: %u=0x%x",
        cuda::ToStatus(result).ToString(), absl::bit_cast<void*>(gpu_dst),
        CUDAPointerToMemorySpaceString(gpu_dst),
        CUDAPointerToDeviceString(gpu_dst), absl::bit_cast<void*>(gpu_src),
        CUDAPointerToMemorySpaceString(gpu_src),
        CUDAPointerToDeviceString(gpu_src),
        CUDAPointersToCanAccessString(gpu_src, gpu_dst), size, size);

    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes"
          << " from " << absl::bit_cast<void*>(gpu_src) << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream;
  return true;
}

absl::Status GpuDriver::InitEvent(GpuContext* context, CUevent* result,
                                  EventFlags flags) {
  int cuflags;
  switch (flags) {
    case EventFlags::kDefault:
      cuflags = CU_EVENT_DEFAULT;
      break;
    case EventFlags::kDisableTiming:
      cuflags = CU_EVENT_DISABLE_TIMING;
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(flags);
  }

  ScopedActivateContext activated{context};
  return cuda::ToStatus(cuEventCreate(result, cuflags));
}

int GpuDriver::GetDeviceCount() {
  int device_count = 0;
  auto status = cuda::ToStatus(cuDeviceGetCount(&device_count));
  if (!status.ok()) {
    LOG(ERROR) << "could not retrieve CUDA device count: " << status;
    return 0;
  }

  return device_count;
}

absl::StatusOr<GpuContext*> GpuDriver::GetPointerContext(CUdeviceptr pointer) {
  GpuContext* context = nullptr;
  auto status = cuda::ToStatus(
      cuPointerGetAttribute(&context, CU_POINTER_ATTRIBUTE_CONTEXT, pointer));
  if (status.ok()) {
    // For cudaMallocAsync, the context returned is null.  For now
    // return not-available. But how to manage that correctly
    // everywhere in TF?  Currently this is only used during error
    // handling.  So all is working fine, but TF have a different
    // error then the original one.
    if (context == nullptr) {
      return absl::UnavailableError(
          "Empty context returned while querying context for device pointer");
    }
    return context;
  }

  return status;
}

absl::StatusOr<MemoryType> GpuDriver::GetPointerMemorySpace(
    CUdeviceptr pointer) {
  unsigned int value;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuPointerGetAttribute(
      &value, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer)));
  switch (value) {
    case CU_MEMORYTYPE_DEVICE:
      return MemoryType::kDevice;
    case CU_MEMORYTYPE_HOST:
      return MemoryType::kHost;
    default:
      return absl::InternalError(
          absl::StrCat("unknown memory space provided by CUDA API: ", value));
  }
}

absl::Status GpuDriver::GetPointerAddressRange(CUdeviceptr dptr,
                                               CUdeviceptr* base,
                                               size_t* size) {
  return cuda::ToStatus(cuMemGetAddressRange(base, size, dptr));
}

absl::StatusOr<CUdevice> GpuDriver::GetPointerDevice(CUdeviceptr pointer) {
  auto result = GetPointerContext(pointer);
  if (!result.ok()) {
    return result.status();
  }

  return DeviceFromContext(result.value());
}

absl::Status GpuDriver::GetComputeCapability(int* cc_major, int* cc_minor,
                                             CUdevice device) {
  *cc_major = 0;
  *cc_minor = 0;

  TF_RETURN_IF_ERROR(cuda::ToStatus(cuDeviceGetAttribute(
      cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)));

  return cuda::ToStatus(cuDeviceGetAttribute(
      cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
}

absl::Status GpuDriver::GetGpuISAVersion(int* version, CUdevice device) {
  return absl::Status{
      absl::StatusCode::kInternal,
      "Feature not supported on CUDA platform (GetGpuISAVersion)"};
}

absl::Status GpuDriver::GetGpuGCNArchName(CUdevice, std::string*) {
  return absl::Status{
      absl::StatusCode::kInternal,
      "Feature not supported on CUDA platform (GetGpuGCNArchName)"};
}

// Helper function that turns the integer output of cuDeviceGetAttribute to type
// T and wraps it in a absl::StatusOr.
template <typename T>
static absl::StatusOr<T> GetSimpleAttribute(CUdevice device,
                                            CUdevice_attribute attribute) {
  int value = -1;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetAttribute(&value, attribute, device),
      absl::StrCat("Could not retrieve CUDA device attribute (", attribute)));
  T converted = value;
  return converted;
}

absl::StatusOr<int> GpuDriver::GetMultiprocessorCount(CUdevice device) {
  return GetSimpleAttribute<int>(device,
                                 CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
}

absl::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerCore(CUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
}

absl::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerBlock(CUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
}

absl::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerBlockOptin(
    CUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
}

absl::StatusOr<int64_t> GpuDriver::GetMaxThreadsPerMultiprocessor(
    CUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
}

absl::StatusOr<int64_t> GpuDriver::GetMaxThreadsPerBlock(CUdevice device) {
  return GetSimpleAttribute<int64_t>(device,
                                     CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

absl::StatusOr<int64_t> GpuDriver::GetMaxRegistersPerBlock(CUdevice device) {
  return GetSimpleAttribute<int64_t>(
      device, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
}

absl::StatusOr<int64_t> GpuDriver::GetThreadsPerWarp(CUdevice device) {
  return GetSimpleAttribute<int64_t>(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
}

absl::Status GpuDriver::GetGridLimits(int* x, int* y, int* z, CUdevice device) {
  int value;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device),
      "Could not get device attribute"));
  *x = value;

  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device),
      "Could not get device attribute"));
  *y = value;

  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device),
      "Could not get device attribute"));
  *z = value;
  return absl::OkStatus();
}

absl::StatusOr<int32_t> GpuDriver::GetDriverVersion() {
  int32_t version;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuDriverGetVersion(&version),
                                    "Could not get driver version"));
  return version;
}

bool GpuDriver::GetDeviceProperties(CUdevprop* device_properties,
                                    int device_ordinal) {
  auto status =
      cuda::ToStatus(cuDeviceGetProperties(device_properties, device_ordinal));
  return status.ok();
}

absl::StatusOr<int> GpuDriver::GetDeviceAttribute(CUdevice_attribute attribute,
                                                  CUdevice device) {
  int val;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGetAttribute(&val, attribute, device)));
  return val;
}

bool GpuDriver::IsEccEnabled(CUdevice device, bool* result) {
  int value = -1;
  auto status = cuda::ToStatus(
      cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device));
  if (!status.ok()) {
    LOG(ERROR) << "failed to query ECC status: " << status;
    return false;
  }

  *result = value;
  return true;
}

bool GpuDriver::GetDeviceMemoryInfo(GpuContext* context, int64_t* free_out,
                                    int64_t* total_out) {
  ScopedActivateContext activation(context);
  size_t free = 0;
  size_t total = 0;
  auto status = cuda::ToStatus(cuMemGetInfo(&free, &total));
  if (!status.ok()) {
    LOG(ERROR) << "failed to query device memory info: " << status;
    return false;
  }

  *free_out = free;
  *total_out = total;
  return true;
}

bool GpuDriver::GetDeviceTotalMemory(CUdevice device, uint64_t* result) {
  size_t value{};
  auto status = cuda::ToStatus(cuDeviceTotalMem(&value, device));
  if (!status.ok()) {
    LOG(ERROR) << "failed to query total available memory: " << status;
    return false;
  }

  *result = value;
  return true;
}

std::string GpuDriver::GetPCIBusID(CUdevice device) {
  std::string pci_bus_id;
  static const int kBufferSize = 64;
  absl::InlinedVector<char, 4> chars(kBufferSize);
  chars[kBufferSize - 1] = '\0';
  auto status = cuda::ToStatus(
      cuDeviceGetPCIBusId(chars.begin(), kBufferSize - 1, device));
  if (!status.ok()) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << status;
    return pci_bus_id;
  }
  pci_bus_id = chars.begin();
  return pci_bus_id;
}

bool GpuDriver::CanEnablePeerAccess(GpuContext* from, GpuContext* to) {
  if (from == to) {
    return true;  // A context can always access its own memory.
  }

  auto from_device = DeviceFromContext(from);
  if (!from_device.ok()) {
    LOG(ERROR) << "failed to resolve 'from' peer access context to a device: "
               << from_device.status();
    return false;
  }
  auto to_device = DeviceFromContext(to);
  if (!to_device.ok()) {
    LOG(ERROR) << "failed to resolve 'to' peer access context to a device: "
               << to_device.status();
    return false;
  }
  return CanEnablePeerAccess(from_device.value(), to_device.value());
}

bool GpuDriver::CanEnablePeerAccess(GpuDeviceHandle from, GpuDeviceHandle to) {
  int can_access_peer = -1;
  auto status =
      cuda::ToStatus(cuDeviceCanAccessPeer(&can_access_peer, from, to));
  if (!status.ok()) {
    LOG(ERROR) << "failed to detect peer access capability: " << status;
    return false;
  }
  return can_access_peer;
}

absl::Status GpuDriver::EnablePeerAccess(GpuContext* from, GpuContext* to) {
  if (from == to) {
    return absl::OkStatus();  // A context can always access its own
                              // memory.
  }

  ScopedActivateContext activated{from};
  CUresult result = cuCtxEnablePeerAccess(to->context(), 0 /* = flags */);
  if (result != CUDA_SUCCESS &&
      result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    return absl::InternalError(
        absl::StrFormat("failed to enable peer access from %p to %p: %s", from,
                        to, cuda::ToStatus(result).ToString()));
  }

  return absl::OkStatus();
}

absl::StatusOr<int> GpuDriver::GetMaxOccupiedBlocksPerCore(
    GpuContext* context, CUfunction kernel, int threads_per_block,
    size_t dynamic_shared_memory_bytes) {
  ScopedActivateContext activation(context);

  int max_blocks;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
          &max_blocks, kernel, threads_per_block, dynamic_shared_memory_bytes,
          CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE),
      absl::StrFormat("Failed to calculate occupancy of kernel %p", kernel)));
  return max_blocks;
}

}  // namespace gpu

namespace cuda {

CUcontext CurrentContextOrDie() {
  CUcontext current = nullptr;
  TF_CHECK_OK(cuda::ToStatus(cuCtxGetCurrent(&current),
                             "Failed to query current context"));
  return current;
}

}  // namespace cuda
}  // namespace stream_executor
