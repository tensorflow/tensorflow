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

namespace {

// Returns the device associated with the given context.
absl::StatusOr<hipDevice_t> DeviceFromContext(Context* context) {
  ScopedActivateContext activated{context};
  hipDevice_t device = -1;
  hipError_t result = wrap::hipCtxGetDevice(&device);
  if (result == hipSuccess) return device;

  return absl::InternalError(
      absl::StrCat("failed to get device for context: ", ToString(result)));
}

}  // namespace

namespace {

// Call hipDeviceSynchronize and crash if it doesn't succeed.
void SynchronizeOrDie() {
  auto res = wrap::hipDeviceSynchronize();
  if (res != hipSuccess) {
    LOG(FATAL) << "Synchronize found " << ToString(res)
               << " :: " << tsl::CurrentStackTrace();
  }
}

}  // namespace

namespace {

// Actually performs the work of ROCM initialization. Wrapped up in one-time
// execution guard.
static absl::Status InternalInit() {
  hipError_t res = wrap::hipInit(0 /* = flags */);

  if (res == hipSuccess) {
    return absl::OkStatus();
  }

  LOG(ERROR) << "failed call to hipInit: " << ToString(res);
  Diagnostician::LogDiagnosticInformation();
  return absl::AbortedError(
      absl::StrCat("failed call to hipInit: ", ToString(res)));
}

}  // namespace

absl::Status GpuDriver::Init() {
  // Cached return value from calling InternalInit(), as hipInit need only be
  // called once, but GpuDriver::Init may be called many times.
  static absl::Status* init_retval = [] {
    return new absl::Status(InternalInit());
  }();
  return *init_retval;
}

absl::Status GpuDriver::GetDevice(int device_ordinal, hipDevice_t* device) {
  hipError_t res = wrap::hipDeviceGet(device, device_ordinal);
  if (res == hipSuccess) {
    return absl::OkStatus();
  }

  return absl::InternalError(
      absl::StrCat("failed call to hipDeviceGet: ", ToString(res)));
}

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

absl::Status GpuDriver::StreamBeginCapture(GpuStreamHandle stream,
                                           StreamCaptureMode mode) {
  hipStreamCaptureMode hip_mode;
  switch (mode) {
    case StreamCaptureMode::kGlobal:
      hip_mode = hipStreamCaptureModeGlobal;
      break;
    case StreamCaptureMode::kThreadLocal:
      hip_mode = hipStreamCaptureModeThreadLocal;
      break;
    case StreamCaptureMode::kRelaxed:
      hip_mode = hipStreamCaptureModeRelaxed;
      break;
  }

  VLOG(2) << "Beging stream " << stream << " capture in "
          << StreamCaptureModeToString(mode) << " mode";
  return ToStatus(wrap::hipStreamBeginCapture(stream, hip_mode),
                  "Failed to begin stream capture");
}

absl::Status GpuDriver::StreamBeginCaptureToGraph(GpuStreamHandle stream,
                                                  GpuGraphHandle graph,
                                                  StreamCaptureMode mode) {
  return absl::UnimplementedError(
      "StreamBeginCaptureToGraph is not implemented");
}

absl::Status GpuDriver::StreamEndCapture(GpuStreamHandle stream,
                                         hipGraph_t* graph) {
  VLOG(2) << "End stream " << stream << " capture";

  return ToStatus(wrap::hipStreamEndCapture(stream, graph),
                  "Failed to end stream capture");
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

absl::Status GpuDriver::GraphLaunch(hipGraphExec_t exec,
                                    GpuStreamHandle stream) {
  VLOG(2) << "Launching HIP executable graph " << exec << " on a stream "
          << stream;
  return ToStatus(wrap::hipGraphLaunch(exec, stream),
                  "Failed to launch HIP graph");
}

absl::Status GpuDriver::GraphNodeSetEnabled(hipGraphExec_t exec,
                                            hipGraphNode_t node, bool enabled) {
  // Node is enabled if value != 0, otherwise the node is disabled.
  unsigned value = enabled ? 1 : 0;
  VLOG(2) << "Set HIP executable graph " << exec << " node " << node
          << " enabled flag to " << value;
  return ToStatus(wrap::hipGraphNodeSetEnabled(exec, node, value),
                  "Failed to set HIP graph node enabled flag");
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

absl::Status GpuDriver::DeviceGraphMemTrim(GpuDeviceHandle device) {
  VLOG(2) << "Trim ROCM device graph memory " << device;
  return ToStatus(wrap::hipDeviceGraphMemTrim(device),
                  "Failed to trim device graph memory");
}

absl::StatusOr<bool> GpuDriver::StreamIsCapturing(GpuStreamHandle stream) {
  VLOG(2) << "Checking if stream " << stream << " is capturing";

  hipStreamCaptureStatus status;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipStreamIsCapturing(stream, &status),
                              "Failed to check stream capturing status"));

  return status == hipStreamCaptureStatusActive;
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

absl::Status GpuDriver::GraphAddEmptyNode(
    hipGraphNode_t* node, hipGraph_t graph,
    absl::Span<const hipGraphNode_t> deps) {
  VLOG(2) << "Add empty node to a graph " << graph << "; deps: " << deps.size();

  return ToStatus(
      wrap::hipGraphAddEmptyNode(node, graph, deps.data(), deps.size()),
      "Failed to add empty node to a HIP graph");
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

/*static*/ absl::Status GpuDriver::GraphExecKernelNodeSetParams(
    GpuGraphExecHandle exec, GpuGraphNodeHandle node,
    absl::string_view kernel_name, GpuFunctionHandle function,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes,
    void** kernel_params, void** extra) {
  VLOG(2) << "Set kernel node params " << node << " in graph executabe " << exec
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

  return ToStatus(wrap::hipGraphExecKernelNodeSetParams(exec, node, &params),
                  "Failed to set HIP graph kernel node params");
}

absl::Status GpuDriver::GraphAddChildNode(hipGraphNode_t* node,
                                          hipGraph_t graph,
                                          absl::Span<const hipGraphNode_t> deps,
                                          hipGraph_t child) {
  VLOG(2) << "Create a new node by cloning the child graph " << child
          << " and add it to " << graph << "; deps: " << deps.size();

  return ToStatus(
      wrap::hipGraphAddChildGraphNode(node, graph, deps.data(), deps.size(),
                                      child),
      "Failed to create a child graph node and add it to a HIP graph");
}

/*static*/ absl::Status GpuDriver::GraphExecChildNodeSetParams(
    GpuGraphExecHandle exec, GpuGraphNodeHandle node, GpuGraphHandle child) {
  VLOG(2) << "Set child node params " << node << " in graph executable " << exec
          << "to params contained in " << child;

  return ToStatus(wrap::hipGraphExecChildGraphNodeSetParams(exec, node, child),
                  "Failed to set HIP graph child node params");
}

absl::Status GpuDriver::GraphAddMemcpyD2DNode(
    Context* context, GpuGraphNodeHandle* node, GpuGraphHandle graph,
    absl::Span<const GpuGraphNodeHandle> deps, GpuDevicePtr gpu_dst,
    GpuDevicePtr gpu_src, uint64_t size) {
  VLOG(2) << "Add memcpy d2d node to a graph " << graph
          << "; dst: " << reinterpret_cast<void*>(gpu_dst)
          << "; src: " << reinterpret_cast<void*>(gpu_src) << "; size: " << size
          << "; context: " << context << "; deps: " << deps.size();

  return ToStatus(wrap::hipGraphAddMemcpyNode1D(node, graph, deps.data(),
                                                deps.size(), gpu_dst, gpu_src,
                                                size, hipMemcpyDeviceToDevice),
                  "Failed to add memcpy d2d node to a HIP graph");
}

absl::Status GpuDriver::GraphExecMemcpyD2DNodeSetParams(
    Context* context, GpuGraphExecHandle exec, GpuGraphNodeHandle node,
    GpuDevicePtr gpu_dst, GpuDevicePtr gpu_src, uint64_t size) {
  VLOG(2) << "Set memcpy d2d node params " << node << " in graph executable "
          << exec << "; dst: " << reinterpret_cast<void*>(gpu_dst)
          << "; src: " << reinterpret_cast<void*>(gpu_src) << "; size: " << size
          << "; context: " << context;

  return ToStatus(
      wrap::hipGraphExecMemcpyNodeSetParams1D(exec, node, gpu_dst, gpu_src,
                                              size, hipMemcpyDeviceToDevice),
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
    Context* context, GpuGraphNodeHandle* node, GpuGraphHandle graph,
    absl::Span<const GpuGraphNodeHandle> deps, GpuDevicePtr dst,
    std::variant<uint8_t, uint16_t, uint32_t> bit_pattern,
    uint64_t num_elements) {
  VLOG(2) << "Add memset node to a graph " << graph
          << "; dst: " << reinterpret_cast<void*>(dst)
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements << "; context: " << context
          << "; deps: " << deps.size();

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  hipMemsetParams params{
      .dst = dst,
      .elementSize = element_size,
      .height = 1,
      .pitch = 0,  // unused if height is 1
      .value = value,
      .width = num_elements,
  };

  return ToStatus(wrap::hipGraphAddMemsetNode(node, graph, deps.data(),
                                              deps.size(), &params),
                  "Failed to add memset node to a HIP graph");
}

absl::Status GpuDriver::GraphExecMemsetNodeSetParams(
    Context* context, GpuGraphExecHandle exec, GpuGraphNodeHandle node,
    GpuDevicePtr dst, std::variant<uint8_t, uint16_t, uint32_t> bit_pattern,
    uint64_t num_elements) {
  VLOG(2) << "Set memset node params " << node << " in graph executable "
          << exec << "; dst: " << reinterpret_cast<void*>(dst)
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements << "; context: " << context;

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  hipMemsetParams params{
      .dst = dst,
      .elementSize = element_size,
      .height = 1,
      .pitch = 0,  // unused if height is 1
      .value = value,
      .width = num_elements,
  };

  return ToStatus(wrap::hipGraphExecMemsetNodeSetParams(exec, node, &params),
                  "Failed to set memset node params");
}

absl::Status GpuDriver::LaunchKernel(
    Context* context, absl::string_view kernel_name, hipFunction_t function,
    unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
    unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_mem_bytes,
    GpuStreamHandle stream, void** kernel_params, void** extra) {
  ScopedActivateContext activation{context};
  VLOG(2) << "launching kernel: " << kernel_name << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << " smem: " << shared_mem_bytes
          << " func: " << (const void*)function;

  auto res = hipSuccess;
#if TF_ROCM_VERSION < 60200
  // for in-process kernel this function returns mangled kernel function name,
  // and null otherwise
  auto name = wrap::hipKernelNameRefByPtr((const void*)function, stream);
  if (name != nullptr) {
    res = wrap::hipLaunchKernel((const void*)function,
                                dim3(grid_dim_x, grid_dim_y, grid_dim_z),
                                dim3(block_dim_x, block_dim_y, block_dim_z),
                                kernel_params, shared_mem_bytes, stream);
  } else  // NOLINT(readability/braces)
#endif    // TF_ROCM_VERSION < 60200
  {
    res = wrap::hipModuleLaunchKernel(
        function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
        block_dim_z, shared_mem_bytes, stream, kernel_params, extra);
  }
  TF_RETURN_IF_ERROR(
      ToStatus(res, absl::StrCat("Failed to launch ROCm kernel: ", kernel_name,
                                 " with block dimensions: ", block_dim_x, "x",
                                 block_dim_y, "x", block_dim_z)));

  VLOG(2) << "successfully launched kernel";
  return absl::OkStatus();
}

absl::Status GpuDriver::LaunchKernel(
    Context* context, absl::string_view kernel_name, GpuFunctionHandle function,
    unsigned int cluster_dim_x, unsigned int cluster_dim_y,
    unsigned int cluster_dim_z, unsigned int grid_dim_x,
    unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, GpuStreamHandle stream, void** kernel_params,
    void** extra) {
  if (cluster_dim_x != 1 || cluster_dim_y != 1 || cluster_dim_z != 1)
    return absl::UnimplementedError("Not implemented for ROCm");
  return LaunchKernel(context, kernel_name, function, grid_dim_x, grid_dim_y,
                      grid_dim_z, block_dim_x, block_dim_y, block_dim_z,
                      shared_mem_bytes, stream, kernel_params, extra);
}

absl::Status GpuDriver::SynchronousMemsetUint8(Context* context,
                                               hipDeviceptr_t location,
                                               uint8_t value, size_t size) {
  ScopedActivateContext activation{context};
  return ToStatus(wrap::hipMemsetD8(location, value, size),
                  "Failed to memset memory");
}

absl::Status GpuDriver::SynchronousMemsetUint32(Context* context,
                                                hipDeviceptr_t location,
                                                uint32_t value,
                                                size_t uint32_count) {
  ScopedActivateContext activation{context};
  void* pointer = absl::bit_cast<void*>(location);
  return ToStatus(wrap::hipMemsetD32(pointer, value, uint32_count),
                  "Failed to memset memory");
}

absl::Status GpuDriver::AsynchronousMemsetUint8(Context* context,
                                                hipDeviceptr_t location,
                                                uint8_t value,
                                                size_t uint8_count,
                                                GpuStreamHandle stream) {
  ScopedActivateContext activation{context};
  return ToStatus(wrap::hipMemsetAsync(location, value, uint8_count, stream),
                  "Failed to enqueue async memset operation");
}

absl::Status GpuDriver::AsynchronousMemsetUint32(Context* context,
                                                 hipDeviceptr_t location,
                                                 uint32_t value,
                                                 size_t uint32_count,
                                                 GpuStreamHandle stream) {
  ScopedActivateContext activation{context};
  void* pointer = absl::bit_cast<void*>(location);
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipMemsetD32Async(pointer, value, uint32_count, stream),
               "Failed to enqueue async memset operation"));
  VLOG(2) << "successfully enqueued async memset operation";
  return absl::OkStatus();
}

absl::Status GpuDriver::AddStreamCallback(Context* context,
                                          GpuStreamHandle stream,
                                          StreamCallback callback, void* data) {
  return ToStatus(wrap::hipLaunchHostFunc(stream, (hipHostFn_t)callback, data),
                  "unable to add host callback");
}

void GpuDriver::DestroyStream(Context* context, GpuStreamHandle stream) {
  if (stream == nullptr) {
    return;
  }
  hipError_t res = wrap::hipStreamQuery(stream);
  if (res != hipSuccess) {
    LOG(ERROR) << "stream not idle on destroy: " << ToString(res);
  }

  ScopedActivateContext activated(context);
  res = wrap::hipStreamDestroy(stream);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to destroy ROCM stream for device "
               << context->device_ordinal() << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << stream << " for device "
            << context->device_ordinal();
  }
}

void* GpuDriver::DeviceAllocate(Context* context, uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }

  ScopedActivateContext activated{context};
  hipDeviceptr_t result = 0;
  hipError_t res = wrap::hipMalloc(&result, bytes);
  if (res != hipSuccess) {
    // LOG(INFO) because this isn't always important to users (e.g. BFCAllocator
    // implements a retry if the first allocation fails).
    LOG(INFO) << "failed to allocate "
              << tsl::strings::HumanReadableNumBytes(bytes) << " (" << bytes
              << " bytes) from device: " << ToString(res);
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  VLOG(2) << "allocated " << ptr << " for device " << context->device_ordinal()
          << " of " << bytes << " bytes";
  return ptr;
}

void GpuDriver::DeviceDeallocate(Context* context, void* location) {
  ScopedActivateContext activation{context};
  hipDeviceptr_t pointer = absl::bit_cast<hipDeviceptr_t>(location);
  hipError_t res = wrap::hipFree(pointer);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated " << location << " for device "
            << context->device_ordinal();
  }
}

void* GpuDriver::UnifiedMemoryAllocate(Context* context, uint64_t bytes) {
  ScopedActivateContext activated{context};
  hipDeviceptr_t result = 0;
  // "managed" memory is visible to both CPU and GPU.
  hipError_t res = wrap::hipMallocManaged(&result, bytes, hipMemAttachGlobal);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes unified memory; result: " << ToString(res);
    return nullptr;
  }
  void* ptr = reinterpret_cast<void*>(result);
  VLOG(2) << "allocated " << ptr << " for context " << context << " of "
          << bytes << " bytes in unified memory";
  return ptr;
}

void GpuDriver::UnifiedMemoryDeallocate(Context* context, void* location) {
  ScopedActivateContext activation(context);
  hipDeviceptr_t pointer = absl::bit_cast<hipDeviceptr_t>(location);
  hipError_t res = wrap::hipFree(pointer);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to free unified memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated unified memory at " << location << " for context "
            << context;
  }
}

void* GpuDriver::HostAllocate(Context* context, uint64_t bytes) {
  ScopedActivateContext activation{context};
  void* host_mem = nullptr;
  // "Portable" memory is visible to all ROCM contexts. Safe for our use model.
  hipError_t res = wrap::hipHostMalloc(&host_mem, bytes, hipHostMallocPortable);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes on host: " << ToString(res);
  }
  return host_mem;
}

void GpuDriver::HostDeallocate(Context* context, void* location) {
  ScopedActivateContext activation{context};
  hipError_t res = wrap::hipHostFree(location);
  if (res != hipSuccess) {
    LOG(ERROR) << "error deallocating host memory at " << location << ": "
               << ToString(res);
  }
}

absl::Status GpuDriver::DestroyEvent(Context* context, GpuEventHandle* event) {
  if (*event == nullptr) {
    return absl::InvalidArgumentError("input event cannot be null");
  }

  ScopedActivateContext activated{context};
  hipError_t res = wrap::hipEventDestroy(*event);
  *event = nullptr;

  switch (res) {
    case hipSuccess:
      return absl::OkStatus();
    case hipErrorDeinitialized:
    case hipErrorNotInitialized:
      return absl::FailedPreconditionError(
          absl::StrFormat("error destroying ROCM event in device %d: %s",
                          context->device_ordinal(), ToString(res).c_str()));
    default:
      return absl::InternalError(
          absl::StrFormat("error destroying ROCM event in device %d: %s",
                          context->device_ordinal(), ToString(res).c_str()));
  }
}

absl::Status GpuDriver::SynchronizeStream(Context* context,
                                          GpuStreamHandle stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipStreamSynchronize(stream),
                              "Could not synchronize on ROCM stream"));
  VLOG(2) << "successfully synchronized stream " << stream << " on device "
          << context->device_ordinal();
  return absl::OkStatus();
}

absl::Status GpuDriver::SynchronousMemcpyD2H(Context* context, void* host_dst,
                                             hipDeviceptr_t gpu_src,
                                             uint64_t size) {
  ScopedActivateContext activation{context};
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyDtoH(host_dst, gpu_src, size),
      absl::StrFormat("failed to synchronous memcpy from device to host: "
                      "host dst: %p; Gpu src: %p; size: %llu=0x%llx",
                      host_dst, absl::bit_cast<void*>(gpu_src), size, size)));
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return absl::OkStatus();
}

absl::Status GpuDriver::SynchronousMemcpyH2D(Context* context,
                                             hipDeviceptr_t gpu_dst,
                                             const void* host_src,
                                             uint64_t size) {
  ScopedActivateContext activation{context};
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyHtoD(gpu_dst, const_cast<void*>(host_src), size),
      absl::StrFormat(
          "failed to synchronous memcpy from host to device: Gpu dst: %p;"
          " host src: %p; size: %llu=0x%llx",
          absl::bit_cast<void*>(gpu_dst), host_src, size, size)));
  VLOG(2) << "successfully sync memcpy'd h2d of " << size << " bytes";
  return absl::OkStatus();
}

absl::Status GpuDriver::AsynchronousMemcpyD2H(Context* context, void* host_dst,
                                              hipDeviceptr_t gpu_src,
                                              uint64_t size,
                                              GpuStreamHandle stream) {
  ScopedActivateContext activation{context};
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyDtoHAsync(host_dst, gpu_src, size, stream),
      absl::StrFormat(
          "failed to enqueue async memcpy from device to host: host dst: %p; "
          "Gpu src: %p; size: %llu=0x%llx",
          host_dst, absl::bit_cast<void*>(gpu_src), size, size)));

  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << host_dst << " on stream " << stream
          << " device: " << context->device_ordinal();
  return absl::OkStatus();
}

absl::Status GpuDriver::AsynchronousMemcpyH2D(Context* context,
                                              hipDeviceptr_t gpu_dst,
                                              const void* host_src,
                                              uint64_t size,
                                              GpuStreamHandle stream) {
  ScopedActivateContext activation{context};
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyHtoDAsync(gpu_dst, const_cast<void*>(host_src), size,
                               stream),
      absl::StrFormat(
          "failed to enqueue async memcpy from host to device: Gpu dst: %p; "
          "host src: %p; size: %llu=0x%llx",
          absl::bit_cast<void*>(gpu_dst), host_src, size, size)));

  VLOG(2) << "successfully enqueued async memcpy h2d of " << size
          << " bytes from " << host_src << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream
          << " device: " << context->device_ordinal();
  return absl::OkStatus();
}

absl::Status GpuDriver::AsynchronousMemcpyD2D(Context* context,
                                              hipDeviceptr_t gpu_dst,
                                              hipDeviceptr_t gpu_src,
                                              uint64_t size,
                                              GpuStreamHandle stream) {
  ScopedActivateContext activation{context};
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream),
      absl::StrFormat("failed to enqueue async memcpy from device to device: "
                      "Gpu dst: %p ; Gpu src: %p ; size: %llu=0x%llx",
                      absl::bit_cast<void*>(gpu_dst),
                      absl::bit_cast<void*>(gpu_src), size, size)));

  VLOG(2) << "successfully enqueued async memcpy d2d of " << size
          << " bytes from " << absl::bit_cast<void*>(gpu_src) << " to "
          << absl::bit_cast<void*>(gpu_dst) << " on stream " << stream
          << " device: " << context->device_ordinal();
  return absl::OkStatus();
}

absl::Status GpuDriver::InitEvent(Context* context, GpuEventHandle* event,
                                  EventFlags flags) {
  int hipflags;
  switch (flags) {
    case EventFlags::kDefault:
      hipflags = hipEventDefault;
      break;
    case EventFlags::kDisableTiming:
      hipflags = hipEventDisableTiming | hipEventReleaseToSystem;
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(hipflags);
  }

  ScopedActivateContext activated{context};
  hipError_t res = wrap::hipEventCreateWithFlags(event, hipflags);

  if (res == hipSuccess) {
    return absl::OkStatus();
  } else if (res == hipErrorMemoryAllocation) {
    return absl::ResourceExhaustedError(
        "could not create ROCM event: out of device memory");
  } else {
    return absl::FailedPreconditionError(
        absl::StrCat("could not create ROCM event: ", ToString(res)));
  }
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

absl::Status GpuDriver::GetPointerAddressRange(hipDeviceptr_t dptr,
                                               hipDeviceptr_t* base,
                                               size_t* size) {
  hipError_t result = wrap::hipMemGetAddressRange(base, size, dptr);
  if (result == hipSuccess) {
    return absl::OkStatus();
  } else if (result == hipErrorNotFound) {
    // We differentiate between "this pointer is unknown" (return here) and
    // "there was an internal error while performing this operation" (return
    // below).
    return absl::NotFoundError(absl::StrFormat("not a device pointer %p; %s",
                                               reinterpret_cast<void*>(dptr),
                                               ToString(result).c_str()));
  }

  return absl::InternalError(
      absl::StrFormat("failed to get pointer into for device pointer %p; %s",
                      reinterpret_cast<void*>(dptr), ToString(result).c_str()));
}

absl::StatusOr<MemoryType> GpuDriver::GetPointerMemorySpace(
    hipDeviceptr_t pointer) {
  unsigned int value;
  hipError_t result = wrap::hipPointerGetAttribute(
      &value, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer);
  if (result == hipSuccess) {
    switch (value) {
      case hipMemoryTypeDevice:
        return MemoryType::kDevice;
      case hipMemoryTypeHost:
        return MemoryType::kHost;
      default:
        return absl::InternalError(
            absl::StrCat("unknown memory space provided by ROCM API: ", value));
    }
  }

  return absl::InternalError(absl::StrCat(
      "failed to query device pointer for memory space: ", ToString(result)));
}

// Helper function that turns the integer output of hipDeviceGetAttribute to
// type T and wraps it in a absl::StatusOr.
template <typename T>
static absl::StatusOr<T> GetSimpleAttribute(hipDevice_t device,
                                            hipDeviceAttribute_t attribute) {
  int value = -1;
  hipError_t result = wrap::hipDeviceGetAttribute(&value, attribute, device);
  if (result != hipSuccess) {
    return absl::NotFoundError(
        absl::StrCat("could not retrieve ROCM device attribute (", attribute,
                     "): ", ToString(result)));
  }
  T converted = value;
  return converted;
}

absl::StatusOr<int> GpuDriver::GetMultiprocessorCount(hipDevice_t device) {
  return GetSimpleAttribute<int>(device, hipDeviceAttributeMultiprocessorCount);
}

absl::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerCore(
    hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(
      device, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor);
}

absl::StatusOr<int64_t> GpuDriver::GetMaxSharedMemoryPerBlock(
    hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(device,
                                     hipDeviceAttributeMaxSharedMemoryPerBlock);
}

absl::StatusOr<int64_t> GpuDriver::GetMaxThreadsPerMultiprocessor(
    hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(
      device, hipDeviceAttributeMaxThreadsPerMultiProcessor);
}

absl::StatusOr<int64_t> GpuDriver::GetMaxRegistersPerBlock(hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(device,
                                     hipDeviceAttributeMaxRegistersPerBlock);
}

absl::StatusOr<int64_t> GpuDriver::GetThreadsPerWarp(hipDevice_t device) {
  return GetSimpleAttribute<int64_t>(device, hipDeviceAttributeWarpSize);
}

absl::Status GpuDriver::GetGridLimits(int* x, int* y, int* z,
                                      hipDevice_t device) {
  int value;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipDeviceGetAttribute(
                   &value, hipDeviceAttributeMaxGridDimX, device),
               "failed to query max grid dim x"));
  *x = value;

  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipDeviceGetAttribute(
                   &value, hipDeviceAttributeMaxGridDimY, device),
               "failed to query max grid dim y"));
  *y = value;

  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipDeviceGetAttribute(
                   &value, hipDeviceAttributeMaxGridDimZ, device),
               "failed to query max grid dim z"));
  *z = value;
  return absl::OkStatus();
}

absl::StatusOr<int32_t> GpuDriver::GetDriverVersion() {
  int32_t version;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipDriverGetVersion(&version),
                              "Could not get driver version"));
  return version;
}

bool GpuDriver::GetDeviceProperties(hipDeviceProp_t* device_properties,
                                    int device_ordinal) {
  hipError_t res =
      wrap::hipGetDeviceProperties(device_properties, device_ordinal);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  return true;
}

bool GpuDriver::IsEccEnabled(hipDevice_t device, bool* result) {
  int value = -1;
  hipError_t res = hipSuccess;
  // TODO(ROCm) implement this feature in HIP
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query ECC status: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

std::string GpuDriver::GetPCIBusID(hipDevice_t device) {
  std::string pci_bus_id;
  static const int kBufferSize = 64;
  absl::InlinedVector<char, 4> chars(kBufferSize);
  chars[kBufferSize - 1] = '\0';
  hipError_t res =
      wrap::hipDeviceGetPCIBusId(chars.begin(), kBufferSize - 1, device);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << ToString(res);
    return pci_bus_id;
  }
  pci_bus_id = chars.begin();
  return pci_bus_id;
}

bool GpuDriver::CanEnablePeerAccess(Context* from, Context* to) {
  // A context can always access its own memory.
  if (from == to) return true;

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
  hipError_t result = wrap::hipDeviceCanAccessPeer(&can_access_peer, from, to);
  if (result != hipSuccess) {
    LOG(ERROR) << "failed to detect peer access capability: "
               << ToString(result);
    return false;
  }
  return can_access_peer;
}

absl::Status GpuDriver::EnablePeerAccess(Context* from, Context* to) {
  if (from == to) {
    return absl::OkStatus();  // A device can always access its own memory.
  }

  ScopedActivateContext activated{from};
  hipError_t result = wrap::hipCtxEnablePeerAccess(
      tensorflow::down_cast<RocmContext*>(to)->context(), 0 /* = flags */);
  if (result != hipSuccess && result != hipErrorPeerAccessAlreadyEnabled) {
    return absl::InternalError(
        absl::StrFormat("failed to enable peer access from %d to %d: %s",
                        from->device_ordinal(), to->device_ordinal(),
                        ToString(result).c_str()));
  }

  return absl::OkStatus();
}

absl::StatusOr<int> GpuDriver::GetMaxOccupiedBlocksPerCore(
    Context* context, hipFunction_t kernel, int threads_per_block,
    size_t dynamic_shared_memory_bytes) {
  ScopedActivateContext activation{context};

  int max_blocks = 0;
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
          &max_blocks, kernel, threads_per_block, dynamic_shared_memory_bytes),
      "Failed to calculate maximal active blocks per SM"));
  return max_blocks;
}

}  // namespace stream_executor::gpu
