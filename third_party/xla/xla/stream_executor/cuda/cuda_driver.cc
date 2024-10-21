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

int GpuDriver::GetDeviceCount() {
  int device_count = 0;
  auto status = cuda::ToStatus(cuDeviceGetCount(&device_count));
  if (!status.ok()) {
    LOG(ERROR) << "could not retrieve CUDA device count: " << status;
    return 0;
  }

  return device_count;
}

absl::StatusOr<int32_t> GpuDriver::GetDriverVersion() {
  int32_t version;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuDriverGetVersion(&version),
                                    "Could not get driver version"));
  return version;
}

absl::StatusOr<size_t> GpuDriver::GraphGetNodeCount(GpuGraphHandle graph) {
  size_t num_nodes;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphGetNodes(graph, /*nodes=*/nullptr, &num_nodes)));
  return num_nodes;
}

}  // namespace gpu
}  // namespace stream_executor
