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

#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/context_map.h"
#include "xla/stream_executor/gpu/gpu_diagnostics.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/rocm/rocm_context.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/numbers.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/platform/status.h"

namespace stream_executor::gpu {

absl::Status GpuDriver::CreateGraph(hipGraph_t* graph) {
  VLOG(2) << "Create new HIP graph";
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipGraphCreate(graph, /*flags=*/0),
                              "Failed to create HIP graph"));
  VLOG(2) << "Created HIP graph " << *graph;
  return absl::OkStatus();
}

absl::Status GpuDriver::DestroyGraph(hipGraph_t graph) {
  VLOG(2) << "Destroy HIP graph " << graph;
  return ToStatus(wrap::hipGraphDestroy(graph), "Failed to destroy HIP graph");
}

absl::Status GpuDriver::GraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                         const GraphInstantiateFlags& flags) {
  VLOG(2) << "Instantiate HIP executable graph from graph " << graph << " ("
          << "auto_free_on_launch=" << flags.auto_free_on_launch << ", "
          << "device_launch=" << flags.device_launch << ", "
          << "use_node_priority=" << flags.use_node_prirotiy << ", "
          << "upload=" << flags.upload << ")";
  return ToStatus(wrap::hipGraphInstantiate(exec, graph, nullptr, nullptr, 0),
                  "Failed to instantiate HIP graph");
}

absl::Status GpuDriver::GraphExecUpdate(hipGraphExec_t exec, hipGraph_t graph,
                                        GraphExecUpdateResultInfo* result) {
  VLOG(2) << "Update HIP graph executable " << exec << " with graph " << graph;

  hipGraphExecUpdateResult hip_result = hipGraphExecUpdateError;
  hipGraphNode_t error_node = nullptr;
  auto hip_error =
      wrap::hipGraphExecUpdate(exec, graph, &error_node, &hip_result);

  if (error_node) {
    result->error_node = error_node;
  }

  switch (hip_result) {
    case hipGraphExecUpdateSuccess:
      result->result = GraphExecUpdateResult::kSuccess;
      break;
    case hipGraphExecUpdateError:
      result->result = GraphExecUpdateResult::kError;
      break;
    case hipGraphExecUpdateErrorTopologyChanged:
      result->result = GraphExecUpdateResult::kTopologyChanged;
      break;
    case hipGraphExecUpdateErrorNodeTypeChanged:
      result->result = GraphExecUpdateResult::kNodeTypeChanged;
      break;
    case hipGraphExecUpdateErrorFunctionChanged:
      result->result = GraphExecUpdateResult::kFunctionChanged;
      break;
    case hipGraphExecUpdateErrorParametersChanged:
      result->result = GraphExecUpdateResult::kParametersChanged;
      break;
    case hipGraphExecUpdateErrorNotSupported:
      result->result = GraphExecUpdateResult::kNotSupported;
      break;
    case hipGraphExecUpdateErrorUnsupportedFunctionChange:
      result->result = GraphExecUpdateResult::kUnsupportedFunctionChange;
      break;
      // TODO: HIP hasn't GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED yet
  }

  return ToStatus(hip_error, "Failed to update HIP graph");
}

absl::StatusOr<std::vector<GpuGraphNodeHandle>>
GpuDriver::GraphNodeGetDependencies(GpuGraphNodeHandle node) {
  VLOG(2) << "Get HIP graph node " << node << " dependencies";

  std::vector<hipGraphNode_t> dependencies;

  size_t num_dependencies = 0;
  TF_RETURN_IF_ERROR(
      ToStatus(hipGraphNodeGetDependencies(node, nullptr, &num_dependencies),
               "Failed to get HIP graph node depedencies size"));

  dependencies.resize(num_dependencies, nullptr);
  TF_RETURN_IF_ERROR(ToStatus(
      hipGraphNodeGetDependencies(node, dependencies.data(), &num_dependencies),
      "Failed to get HIP graph node depedencies"));

  return dependencies;
}

absl::Status GpuDriver::DestroyGraphExec(hipGraphExec_t exec) {
  VLOG(2) << "Destroying HIP executable graph" << exec;
  return ToStatus(wrap::hipGraphExecDestroy(exec),
                  "Failed to destroy HIP graph");
}

absl::StatusOr<std::string> GpuDriver::GraphDebugDotPrint(
    hipGraph_t graph, const char* path, bool return_printed_graph) {
  VLOG(2) << "Print HIP graph " << graph << " debug dot file to " << path;

  int flags = hipGraphDebugDotFlagsVerbose;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipGraphDebugDotPrint(graph, path, flags),
                              "Failed to print gpu graph debug file"));

  if (return_printed_graph) {
    std::string data;
    if (tsl::ReadFileToString(tsl::Env::Default(), path, &data).ok()) {
      return data;
    } else {
      LOG(WARNING) << "failed to read gpu graph debug file " << path;
    }
  }

  return std::string(path);
}

absl::Status GpuDriver::GraphConditionalHandleCreate(
    GpuGraphConditionalHandle* handle, hipGraph_t graph, Context* context,
    unsigned int default_launch_value, unsigned int flags) {
  VLOG(2) << "Create conditional handle for a graph " << graph
          << "; context: " << context
          << "; default_launch_value: " << default_launch_value
          << "; flags: " << flags;

  return absl::UnimplementedError(
      "HIP graph conditional nodes are not implemented yet");
}

absl::StatusOr<GpuDriver::GpuGraphNodeResult> GpuDriver::GraphAddNode(
    hipGraphNode_t* node, hipGraph_t graph,
    absl::Span<const hipGraphNode_t> deps, const GpuGraphNodeParams& params) {
  return absl::UnimplementedError("unsupported node type");
}

absl::Status GpuDriver::GraphAddKernelNode(
    hipGraphNode_t* node, hipGraph_t graph,
    absl::Span<const hipGraphNode_t> deps, absl::string_view kernel_name,
    hipFunction_t function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes,
    void** kernel_params, void** extra) {
  VLOG(2) << "Add kernel node to a graph " << graph
          << "; kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << "; shmem: " << shared_mem_bytes;

  hipKernelNodeParams params;
  memset(&params, 0, sizeof(params));

  params.func = function;
  params.gridDim.x = grid_dim_x;
  params.gridDim.y = grid_dim_y;
  params.gridDim.z = grid_dim_z;
  params.blockDim.x = block_dim_x;
  params.blockDim.y = block_dim_y;
  params.blockDim.z = block_dim_z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = kernel_params;
  params.extra = extra;

  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipFuncSetAttribute(function,
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return ToStatus(wrap::hipGraphAddKernelNode(node, graph, deps.data(),
                                              deps.size(), &params),
                  "Failed to add kernel node to a HIP graph");
}

absl::StatusOr<size_t> GpuDriver::GraphGetNodeCount(hipGraph_t graph) {
  VLOG(2) << "Get node count in graph " << graph;
  size_t numNodes;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipGraphGetNodes(graph, nullptr, &numNodes),
                              "Failed to get HIP graph node count"));

  return numNodes;
}

int GpuDriver::GetDeviceCount() {
  int device_count = 0;
  hipError_t res = wrap::hipGetDeviceCount(&device_count);
  if (res != hipSuccess) {
    LOG(ERROR) << "could not retrieve ROCM device count: " << ToString(res);
    return 0;
  }

  return device_count;
}

absl::StatusOr<int32_t> GpuDriver::GetDriverVersion() {
  int32_t version;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipDriverGetVersion(&version),
                              "Could not get driver version"));
  return version;
}

}  // namespace stream_executor::gpu
