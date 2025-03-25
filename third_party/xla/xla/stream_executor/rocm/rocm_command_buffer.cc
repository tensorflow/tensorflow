/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_command_buffer.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/scoped_update_mode.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_kernel.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<hipGraph_t> CreateGraph() {
  VLOG(2) << "Create new HIP graph";
  hipGraph_t graph;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipGraphCreate(&graph, /*flags=*/0),
                              "Failed to create HIP graph"));
  VLOG(2) << "Created HIP graph " << graph;
  return graph;
}

hipDeviceptr_t AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<hipDeviceptr_t>(mem.opaque());
}

using NodeHandle = CommandBuffer::NodeHandle;

// Converts a platform independent NodeHandle into a HIP specific
// hipGraphNode_t.
hipGraphNode_t ToHipGraphHandle(CommandBuffer::NodeHandle handle) {
  return absl::bit_cast<hipGraphNode_t>(handle);
}

// Converts a HIP specific hipGraphNode_t into a platform independent
// NodeHandle. This function will be removed once all Node factory
// functions have been migrated into the subclasses.
NodeHandle FromHipGraphHandle(hipGraphNode_t handle) {
  return absl::bit_cast<CommandBuffer::NodeHandle>(handle);
}
}  // namespace

// Converts a list of platform independent std::vector<NodeHandle>
// into a list of Rocm specific hipGraphNode_t.
absl::StatusOr<std::vector<hipGraphNode_t>> ToHipGraphHandles(
    absl::Span<const NodeHandle> opaque_handles) {
  std::vector<hipGraphNode_t> handles;
  for (const NodeHandle opaque_handle : opaque_handles) {
    handles.push_back(ToHipGraphHandle(opaque_handle));
  }
  return handles;
}

absl::StatusOr<std::unique_ptr<RocmCommandBuffer>> RocmCommandBuffer::Create(
    Mode mode, StreamExecutor* parent) {
  TF_ASSIGN_OR_RETURN(hipGraph_t graph, CreateGraph());
  return std::unique_ptr<RocmCommandBuffer>(
      new RocmCommandBuffer(mode, parent, graph,
                            /*is_owned_graph=*/true));
}

absl::StatusOr<NodeHandle> RocmCommandBuffer::CreateMemsetNode(
    Dependencies dependencies, DeviceMemoryBase destination,
    BitPattern bit_pattern, size_t num_elements) {
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements;

  hipMemsetParams params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  TF_ASSIGN_OR_RETURN(std::vector<hipGraphNode_t> deps,
                      ToHipGraphHandles(dependencies));

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphAddMemsetNode(&node_handle, graph_, deps.data(),
                                           deps.size(), &params),
               "Failed to add memset node to a HIP graph"));
  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateMemsetNode(NodeHandle node_handle,
                                                 DeviceMemoryBase destination,
                                                 BitPattern bit_pattern,
                                                 size_t num_elements) {
  VLOG(2) << "Set memset node params " << node_handle << " in graph executable "
          << exec_ << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements;

  hipMemsetParams params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  return ToStatus(wrap::hipGraphExecMemsetNodeSetParams(
                      exec_, ToHipGraphHandle(node_handle), &params),
                  "Failed to set memset node params");
}

absl::StatusOr<NodeHandle> RocmCommandBuffer::CreateMemcpyD2DNode(
    Dependencies dependencies, DeviceMemoryBase destination,
    DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Add memcpy d2d node to a graph " << graph_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size;

  TF_ASSIGN_OR_RETURN(std::vector<hipGraphNode_t> deps,
                      ToHipGraphHandles(dependencies));

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipGraphAddMemcpyNode1D(&node_handle, graph_, deps.data(),
                                    deps.size(), AsDevicePtr(destination),
                                    AsDevicePtr(source), size,
                                    hipMemcpyDeviceToDevice),
      "Failed to add memcpy d2d node to a HIP graph"));
  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateMemcpyD2DNode(
    NodeHandle node_handle, DeviceMemoryBase destination,
    DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Set memcpy d2d node params " << node_handle
          << " in graph executable " << exec_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size;

  return ToStatus(
      wrap::hipGraphExecMemcpyNodeSetParams1D(
          exec_, ToHipGraphHandle(node_handle), AsDevicePtr(destination),
          AsDevicePtr(source), size, hipMemcpyDeviceToDevice),
      "Failed to set memcpy d2d node params");
}

absl::StatusOr<NodeHandle> RocmCommandBuffer::CreateChildNode(
    Dependencies dependencies, const CommandBuffer& nested) {
  hipGraph_t child_graph =
      tensorflow::down_cast<const RocmCommandBuffer&>(nested).graph_;
  VLOG(2) << "Create a new node by cloning the child graph " << child_graph
          << " and add it to " << graph_;

  TF_ASSIGN_OR_RETURN(std::vector<hipGraphNode_t> deps,
                      ToHipGraphHandles(dependencies));

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipGraphAddChildGraphNode(&node_handle, graph_, deps.data(),
                                      deps.size(), child_graph),
      "Failed to create a child graph node and add it to a HIP graph"));
  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateChildNode(NodeHandle node_handle,
                                                const CommandBuffer& nested) {
  hipGraph_t child_graph =
      tensorflow::down_cast<const RocmCommandBuffer&>(nested).graph_;

  VLOG(2) << "Set child node params " << node_handle << " in graph executable "
          << exec_ << "to params contained in " << child_graph;

  return ToStatus(wrap::hipGraphExecChildGraphNodeSetParams(
                      exec_, ToHipGraphHandle(node_handle), child_graph),
                  "Failed to set HIP graph child node params");
}

absl::StatusOr<NodeHandle> RocmCommandBuffer::CreateLaunchNode(
    Dependencies dependencies, const ThreadDim& threads, const BlockDim& blocks,
    const Kernel& kernel, const KernelArgs& args) {
  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return CreateKernelNode(dependencies, threads, blocks, kernel, *packed);
  }

  // For device memory array we rely on a custom kernel arguments packing.
  if (auto* device_mem = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
    auto& pack = kernel.args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array");
    }

    TF_ASSIGN_OR_RETURN(auto packed, pack(kernel, *device_mem));
    return CreateKernelNode(dependencies, threads, blocks, kernel, *packed);
  }

  return absl::InternalError("Unsupported kernel arguments type");
}

absl::Status RocmCommandBuffer::UpdateLaunchNode(NodeHandle node,
                                                 const ThreadDim& threads,
                                                 const BlockDim& blocks,
                                                 const Kernel& kernel,
                                                 const KernelArgs& args) {
  // If arguments are already packed we can just launch the kernel.
  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return UpdateKernelNode(node, threads, blocks, kernel, *packed);
  }

  // For device memory array we rely on a custom kernel arguments packing.
  if (auto* device_mem = DynCast<KernelArgsDeviceMemoryArray>(&args)) {
    auto& pack = kernel.args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array");
    }

    TF_ASSIGN_OR_RETURN(auto packed, pack(kernel, *device_mem));
    return UpdateKernelNode(node, threads, blocks, kernel, *packed);
  }
  return absl::InternalError("Unsupported kernel arguments type");
}

absl::StatusOr<NodeHandle> RocmCommandBuffer::CreateKernelNode(
    Dependencies dependencies, const ThreadDim& threads, const BlockDim& blocks,
    const Kernel& kernel, const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Add kernel node to a graph " << graph_
          << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes;

  hipKernelNodeParams params{};

  hipFunction_t function =
      static_cast<const RocmKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDim.x = blocks.x;
  params.gridDim.y = blocks.y;
  params.gridDim.z = blocks.z;
  params.blockDim.x = threads.x;
  params.blockDim.y = threads.y;
  params.blockDim.z = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipFuncSetAttribute(function,
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  TF_ASSIGN_OR_RETURN(std::vector<hipGraphNode_t> deps,
                      ToHipGraphHandles(dependencies));

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphAddKernelNode(&node_handle, graph_, deps.data(),
                                           deps.size(), &params),
               "Failed to add kernel node to a HIP graph"));

  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateKernelNode(
    NodeHandle node_handle, const ThreadDim& threads, const BlockDim& blocks,
    const Kernel& kernel, const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Set kernel node params " << node_handle << " in graph executable "
          << exec_ << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes;

  hipKernelNodeParams params{};

  hipFunction_t function =
      static_cast<const RocmKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDim.x = blocks.x;
  params.gridDim.y = blocks.y;
  params.gridDim.z = blocks.z;
  params.blockDim.x = threads.x;
  params.blockDim.y = threads.y;
  params.blockDim.z = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipFuncSetAttribute(function,
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return ToStatus(wrap::hipGraphExecKernelNodeSetParams(
                      exec_, ToHipGraphHandle(node_handle), &params),
                  "Failed to set HIP graph kernel node params");
}

absl::StatusOr<NodeHandle> RocmCommandBuffer::CreateEmptyNode(
    Dependencies dependencies) {
  VLOG(2) << "Add empty node to a graph " << graph_;

  hipGraphNode_t barrier_handle = nullptr;
  TF_ASSIGN_OR_RETURN(std::vector<hipGraphNode_t> deps,
                      ToHipGraphHandles(dependencies));

  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphAddEmptyNode(&barrier_handle, graph_, deps.data(),
                                          deps.size()),
               "Failed to add empty node to a HIP graph"));

  return FromHipGraphHandle(barrier_handle);
}

absl::Status RocmCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status()> function) {
  TF_ASSIGN_OR_RETURN(size_t count, GetNodeCount());
  if (count != 0 || !is_owned_graph_)
    return absl::InternalError(
        "Stream can't be traced on non empty command buffer");

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream;

  hipStream_t stream_handle =
      static_cast<hipStream_t>(stream->platform_specific_handle().stream);

  // Switch stream into the capture mode.
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipStreamBeginCapture(stream_handle,
                                           hipStreamCaptureModeThreadLocal),
               "Failed to begin stream capture"));
  auto traced = function();

  // Always stop capturing the stream before checking `traced` result.
  VLOG(5) << "End stream " << stream << " capture";
  hipGraph_t captured_graph;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipStreamEndCapture(stream_handle, &captured_graph),
               "Failed to end stream capture"));
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphDestroy(std::exchange(graph_, captured_graph)),
               "Failed to destroy HIP graph"));
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!traced.ok())
    return absl::InternalError(
        absl::StrCat("Failed to capture gpu graph: ", traced.message()));

  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  return absl::OkStatus();
}

absl::Status RocmCommandBuffer::SetNodeExecutionEnabled(NodeHandle node_handle,
                                                        bool enabled) {
  // Node is enabled if value != 0, otherwise the node is disabled.
  unsigned value = enabled ? 1 : 0;
  VLOG(2) << "Set HIP executable graph " << exec_ << " node " << node_handle
          << " enabled flag to " << value;
  return ToStatus(
      wrap::hipGraphNodeSetEnabled(exec_, ToHipGraphHandle(node_handle), value),
      "Failed to set HIP graph node enabled flag");
}

absl::Status RocmCommandBuffer::LaunchGraph(Stream* stream) {
  VLOG(3) << "Launch command buffer executable graph " << exec_
          << " on a stream: " << stream;
  return ToStatus(wrap::hipGraphLaunch(
                      exec_, static_cast<hipStream_t>(
                                 stream->platform_specific_handle().stream)),
                  "Failed to launch HIP graph");
}
absl::StatusOr<size_t> RocmCommandBuffer::GetNodeCount() const {
  size_t numNodes;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphGetNodes(graph_, /*nodes=*/nullptr, &numNodes),
               "Failed to get HIP graph node count"));

  return numNodes;
}

absl::Status RocmCommandBuffer::PrepareFinalization() {
  return absl::OkStatus();
}

absl::StatusOr<CommandBuffer::ConditionalHandle>
RocmCommandBuffer::CreateConditionalHandle() {
  return absl::UnimplementedError(
      "Graph conditionals are not yet supported on HIP graphs.");
}

absl::Status RocmCommandBuffer::WriteGraphToDotFile(absl::string_view path) {
  VLOG(2) << "Print HIP graph " << graph_ << " debug dot file to " << path;

  int flags = hipGraphDebugDotFlagsVerbose;
  return ToStatus(
      wrap::hipGraphDebugDotPrint(graph_, std::string{path}.c_str(), flags),
      "Failed to print gpu graph debug file");
}

absl::Status RocmCommandBuffer::InstantiateGraph() {
  VLOG(2) << "Instantiate HIP executable graph from graph " << graph_;
  return ToStatus(
      wrap::hipGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0),
      "Failed to instantiate HIP graph");
}

absl::Status RocmCommandBuffer::Finalize() {
  TF_RETURN_IF_ERROR(PrepareFinalization());

  size_t num_nodes = 0, num_cond_cmd_buffers = 0;

  if (mode() == Mode::kPrimary) {
    uint64_t start_nanos = tsl::Env::Default()->NowNanos();

    if (exec_ == nullptr) {
      // If this is the first time we finalize command buffer after
      // construction, we need to instantiate it to an executable graph.
      auto instantiated = InstantiateGraph();

      if (instantiated.code() == absl::StatusCode::kResourceExhausted) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "Underlying backend ran out of memory trying to instantiate "
            "graph "
            "with %d nodes and %d conditionals. Give more memory to the "
            "driver by reducing XLA_CLIENT_MEM_FRACTION (b) Disable "
            "command buffers with "
            "'XLA_FLAGS=--xla_gpu_enable_command_buffer=' "
            "(empty set). Original error: %s",
            num_nodes, num_cond_cmd_buffers, instantiated.message()));
      }
      TF_RETURN_IF_ERROR(instantiated);

      uint64_t end_nanos = tsl::Env::Default()->NowNanos();
      TF_ASSIGN_OR_RETURN(auto node_count, GetNodeCount());
      VLOG(5) << "Instantiated executable graph in "
              << (end_nanos - start_nanos) / 1000 << " μs"
              << "; nodes: " << node_count
              << "; conditionals: " << num_cond_cmd_buffers;
    } else {
      VLOG(5) << "Update existing executor " << exec_
              << " by updating with graph " << graph_;
      hipGraphExecUpdateResult resultInfo;
      hipGraphNode_t errNode;
      TF_RETURN_IF_ERROR(
          ToStatus(hipGraphExecUpdate(exec_, graph_, &errNode, &resultInfo),
                   "Failed to update HIP graph executable"));
      VLOG(5) << "Updated HIP graph executable " << exec_;
    }
  } else {
    // Nested command buffers do not have executable graphs.
    VLOG(5) << "Finalize nested command buffer without instantiating "
               "executable graph";
  }
  return absl::OkStatus();
}

absl::Status RocmCommandBuffer::Submit(Stream* stream) {
  if (mode() != Mode::kPrimary) {
    return absl::InvalidArgumentError(
        "Can't submit non-primary command buffer for execution");
  }
  return LaunchGraph(stream);
}

RocmCommandBuffer::~RocmCommandBuffer() {
  if (exec_ != nullptr && is_owned_graph_exec_) {
    if (auto status = ToStatus(hipGraphExecDestroy(exec_),
                               "Failed to destroy HIP executable graph");
        !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
  if (graph_ != nullptr && is_owned_graph_) {
    if (auto status =
            ToStatus(hipGraphDestroy(graph_), "Failed to destroy HIP graph");
        !status.ok()) {
      LOG(ERROR) << status.message();
    }
  }
}
absl::Status RocmCommandBuffer::CheckCanBeUpdated() {
  if (exec_ == nullptr) {
    return absl::InternalError(
        "Command buffer has to have a graph executable to be updated.");
  }
  return absl::OkStatus();
}

}  // namespace stream_executor::gpu
