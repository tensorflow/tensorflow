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
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/context_map.h"
#include "xla/stream_executor/gpu/gpu_diagnostics.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

namespace {

// Returns the device associated with the given context.
absl::StatusOr<CUdevice> DeviceFromContext(Context* context) {
  ScopedActivateContext activated{context};
  CUdevice device = -1;
  auto status = cuda::ToStatus(cuCtxGetDevice(&device));
  if (status.ok()) {
    return device;
  }

  return status;
}

}  // namespace

namespace {

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

}  // namespace

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
    GpuGraphConditionalHandle* handle, CUgraph graph, Context* context,
    unsigned int default_launch_value, unsigned int flags) {
  VLOG(2) << "Create conditional handle for a graph " << graph
          << "; context: " << context
          << "; default_launch_value: " << default_launch_value
          << "; flags: " << flags;

#if CUDA_VERSION >= 12030
  return cuda::ToStatus(
      cuGraphConditionalHandleCreate(
          handle, graph,
          tensorflow::down_cast<CudaContext*>(context)->context(),
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
    CudaContext* gpu_context =
        tensorflow::down_cast<CudaContext*>(conditional->context);

    cu_params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
    cu_params.conditional.handle = conditional->handle;
    cu_params.conditional.ctx = gpu_context->context();
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

absl::Status GpuDriver::GraphAddMemcpyD2DNode(
    Context* context, CUgraphNode* node, CUgraph graph,
    absl::Span<const CUgraphNode> deps, CUdeviceptr gpu_dst,
    CUdeviceptr gpu_src, uint64_t size) {
  CudaContext* gpu_context = tensorflow::down_cast<CudaContext*>(context);
  VLOG(2) << "Add memcpy d2d node to a graph " << graph
          << "; dst: " << reinterpret_cast<void*>(gpu_dst)
          << "; src: " << reinterpret_cast<void*>(gpu_src) << "; size: " << size
          << "; context: " << gpu_context->context()
          << "; deps: " << deps.size();

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
                           gpu_context->context()),
      "Failed to add memcpy d2d node to a CUDA graph");
}

absl::Status GpuDriver::GraphExecMemcpyD2DNodeSetParams(
    Context* context, GpuGraphExecHandle exec, GpuGraphNodeHandle node,
    GpuDevicePtr gpu_dst, GpuDevicePtr gpu_src, uint64_t size) {
  CudaContext* gpu_context = tensorflow::down_cast<CudaContext*>(context);
  VLOG(2) << "Set memcpy d2d node params " << node << " in graph executable "
          << exec << "; dst: " << reinterpret_cast<void*>(gpu_dst)
          << "; src: " << reinterpret_cast<void*>(gpu_src) << "; size: " << size
          << "; context: " << gpu_context->context();

  CUDA_MEMCPY3D params;
  memset(&params, 0, sizeof(params));

  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = gpu_src;
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = gpu_dst;
  params.WidthInBytes = size;
  params.Height = 1;
  params.Depth = 1;

  return cuda::ToStatus(cuGraphExecMemcpyNodeSetParams(exec, node, &params,
                                                       gpu_context->context()),
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
    Context* context, CUgraphNode* node, GpuGraphHandle graph,
    absl::Span<const CUgraphNode> deps, CUdeviceptr dst,
    std::variant<uint8_t, uint16_t, uint32_t> bit_pattern,
    uint64_t num_elements) {
  CudaContext* gpu_context = tensorflow::down_cast<CudaContext*>(context);
  VLOG(2) << "Add memset node to a graph " << graph
          << "; dst: " << reinterpret_cast<void*>(dst)
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements
          << "; context: " << gpu_context->context()
          << "; deps: " << deps.size();

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
                           gpu_context->context()),
      "Failed to add memset node to a CUDA graph");
}

absl::Status GpuDriver::GraphExecMemsetNodeSetParams(
    Context* context, CUgraphExec exec, CUgraphNode node, CUdeviceptr dst,
    std::variant<uint8_t, uint16_t, uint32_t> bit_pattern,
    uint64_t num_elements) {
  CudaContext* gpu_context = tensorflow::down_cast<CudaContext*>(context);
  VLOG(2) << "Set memset node params " << node << " in graph executable "
          << exec << "; dst: " << reinterpret_cast<void*>(dst)
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements
          << "; context: " << gpu_context->context();

  CUDA_MEMSET_NODE_PARAMS params;
  memset(&params, 0, sizeof(params));

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  params.dst = dst;
  params.elementSize = element_size;
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = value;
  params.width = num_elements;

  return cuda::ToStatus(cuGraphExecMemsetNodeSetParams(exec, node, &params,
                                                       gpu_context->context()),
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
    Context* context, absl::string_view kernel_name, CUfunction function,
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
    Context* context, absl::string_view kernel_name, GpuFunctionHandle function,
    unsigned int cluster_dim_x, unsigned int cluster_dim_y,
    unsigned int cluster_dim_z, unsigned int grid_dim_x,
    unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, GpuStreamHandle stream, void** kernel_params,
    void** extra) {
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

absl::Status GpuDriver::SynchronousMemsetUint8(Context* context,
                                               CUdeviceptr location,
                                               uint8_t value, size_t size) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuMemsetD8(location, value, size),
                        "Failed to memset memory");
}

absl::Status GpuDriver::SynchronousMemsetUint32(Context* context,
                                                CUdeviceptr location,
                                                uint32_t value,
                                                size_t uint32_count) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuMemsetD32(location, value, uint32_count),
                        "Failed to memset memory");
}

absl::Status GpuDriver::AsynchronousMemsetUint8(Context* context,
                                                CUdeviceptr location,
                                                uint8_t value,
                                                size_t uint8_count,
                                                CUstream stream) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuMemsetD8Async(location, value, uint8_count, stream),
                        "Failed to enqueue async memset operation");
}

absl::Status GpuDriver::AsynchronousMemsetUint32(Context* context,
                                                 CUdeviceptr location,
                                                 uint32_t value,
                                                 size_t uint32_count,
                                                 CUstream stream) {
  ScopedActivateContext activation(context);
  return cuda::ToStatus(cuMemsetD32Async(location, value, uint32_count, stream),
                        "Failed to enqueue async memset operation");
}

absl::Status GpuDriver::AddStreamCallback(Context* context, CUstream stream,
                                          StreamCallback callback, void* data) {
  // Note: flags param is required to be zero according to CUDA 6.0.
  return cuda::ToStatus(cuLaunchHostFunc(stream, callback, data));
}


void GpuDriver::DestroyStream(Context* context, GpuStreamHandle stream) {
  if (stream == nullptr) {
    return;
  }

  ScopedActivateContext activated{context};
  CUresult res = cuStreamQuery(stream);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "stream not idle on destroy: " << cuda::ToStatus(res);
  }

  auto status = cuda::ToStatus(cuStreamDestroy(stream));
  if (!status.ok()) {
    LOG(ERROR) << "failed to destroy CUDA stream for context " << context
               << ": " << status;
  } else {
    VLOG(2) << "successfully destroyed stream " << stream << " for context "
            << context;
  }
}

void* GpuDriver::DeviceAllocate(Context* context, uint64_t bytes) {
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
  VLOG(2) << "allocated " << ptr << " for context " << context << " of "
          << bytes << " bytes";
  return ptr;
}

void GpuDriver::DeviceDeallocate(Context* context, void* location) {
  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  auto status = cuda::ToStatus(cuMemFree(pointer));
  if (!status.ok()) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << status;
  } else {
    VLOG(2) << "deallocated " << location << " for context " << context;
  }
}

void* GpuDriver::UnifiedMemoryAllocate(Context* context, uint64_t bytes) {
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
  VLOG(2) << "allocated " << ptr << " for context " << context << " of "
          << bytes << " bytes in unified memory";
  return ptr;
}

void GpuDriver::UnifiedMemoryDeallocate(Context* context, void* location) {
  ScopedActivateContext activation(context);
  CUdeviceptr pointer = absl::bit_cast<CUdeviceptr>(location);
  auto status = cuda::ToStatus(cuMemFree(pointer));
  if (!status.ok()) {
    LOG(ERROR) << "failed to free unified memory at " << location
               << "; result: " << status;
  } else {
    VLOG(2) << "deallocated unified memory at " << location << " for context "
            << context;
  }
}

void* GpuDriver::HostAllocate(Context* context, uint64_t bytes) {
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

void GpuDriver::HostDeallocate(Context* context, void* location) {
  ScopedActivateContext activation(context);
  auto status = cuda::ToStatus(cuMemFreeHost(location));
  if (!status.ok()) {
    LOG(ERROR) << "error deallocating host memory at " << location << ": "
               << status;
  }
}

bool GpuDriver::HostRegister(Context* context, void* location, uint64_t bytes) {
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

bool GpuDriver::HostUnregister(Context* context, void* location) {
  ScopedActivateContext activation(context);
  auto status = cuda::ToStatus(cuMemHostUnregister(location));
  if (!status.ok()) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << status;
    return false;
  }
  return true;
}

absl::Status GpuDriver::DestroyEvent(Context* context, CUevent* event) {
  if (*event == nullptr) {
    return absl::InvalidArgumentError("input event cannot be null");
  }

  ScopedActivateContext activated{context};
  return cuda::ToStatus(cuEventDestroy(*event), "Error destroying CUDA event");
}

absl::Status GpuDriver::SynchronizeStream(Context* context, CUstream stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  return cuda::ToStatus(cuStreamSynchronize(stream),
                        "Could not synchronize CUDA stream");
}

absl::Status GpuDriver::SynchronousMemcpyD2H(Context* context, void* host_dst,
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

absl::Status GpuDriver::SynchronousMemcpyH2D(Context* context,
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

absl::Status GpuDriver::AsynchronousMemcpyD2H(Context* context, void* host_dst,
                                              CUdeviceptr gpu_src,
                                              uint64_t size, CUstream stream) {
  ScopedActivateContext activation(context);

  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuMemcpyDtoHAsync(host_dst, gpu_src, size, stream)));

  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status GpuDriver::AsynchronousMemcpyH2D(Context* context,
                                              CUdeviceptr gpu_dst,
                                              const void* host_src,
                                              uint64_t size, CUstream stream) {
  ScopedActivateContext activation(context);
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuMemcpyHtoDAsync(gpu_dst, host_src, size, stream)));

  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " from " << host_src << " to " << absl::bit_cast<void*>(gpu_dst)
          << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status GpuDriver::AsynchronousMemcpyD2D(Context* context,
                                              CUdeviceptr gpu_dst,
                                              CUdeviceptr gpu_src,
                                              uint64_t size, CUstream stream) {
  ScopedActivateContext activation(context);

  // In graph capture mode we never have operations that access peer memory, so
  // we can always make a call to cuMemcpyDtoDAsync.
  TF_ASSIGN_OR_RETURN(bool is_capturing, StreamIsCapturing(stream));

  if ((gpu_dst == 0 || gpu_src == 0) || is_capturing) {
    // GetContextMap()->GetAnyContext() doesn't work when ptr == 0.
    // This happens when the size is 0.
    TF_RETURN_IF_ERROR(
        cuda::ToStatus(cuMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream)));
  } else {
    // Any context work here.
    CUcontext dst_context = CudaContext::GetContextMap()->GetAnyContext(
        absl::bit_cast<void*>(gpu_dst));
    CUcontext src_context = CudaContext::GetContextMap()->GetAnyContext(
        absl::bit_cast<void*>(gpu_src));

    if (dst_context == src_context) {
      // Since the CUDA context is the same, the src and dst are within the same
      // GPU. So we can use cuMemcpyDtoD.
      TF_RETURN_IF_ERROR(
          cuda::ToStatus(cuMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream)));
    } else {
      TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemcpyPeerAsync(
          gpu_dst, dst_context, gpu_src, src_context, size, stream)));
    }
  }

  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes"
          << " from " << absl::bit_cast<void*>(gpu_src) << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream;
  return absl::OkStatus();
}

absl::Status GpuDriver::InitEvent(Context* context, CUevent* result,
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

bool GpuDriver::CanEnablePeerAccess(Context* from, Context* to) {
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

absl::Status GpuDriver::EnablePeerAccess(Context* from, Context* to) {
  if (from == to) {
    return absl::OkStatus();  // A context can always access its own
                              // memory.
  }

  ScopedActivateContext activated{from};
  CUresult result = cuCtxEnablePeerAccess(
      tensorflow::down_cast<CudaContext*>(to)->context(), 0 /* = flags */);
  if (result != CUDA_SUCCESS &&
      result != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    return absl::InternalError(
        absl::StrFormat("failed to enable peer access from %p to %p: %s", from,
                        to, cuda::ToStatus(result).ToString()));
  }

  return absl::OkStatus();
}

absl::StatusOr<int> GpuDriver::GetMaxOccupiedBlocksPerCore(
    Context* context, CUfunction kernel, int threads_per_block,
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

absl::StatusOr<size_t> GpuDriver::GraphGetNodeCount(GpuGraphHandle graph) {
  size_t num_nodes;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphGetNodes(graph, /*nodes=*/nullptr, &num_nodes)));
  return num_nodes;
}

}  // namespace gpu
}  // namespace stream_executor
