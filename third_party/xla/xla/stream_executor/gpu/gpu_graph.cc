/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/stream_executor/gpu/gpu_graph.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "absl/strings/str_cat.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

//===----------------------------------------------------------------------===//
// RAII helpers for gpu graph types.
//===----------------------------------------------------------------------===//

std::atomic<size_t> GpuGraphSupport::allocated_gpu_graph_execs_;
std::atomic<size_t> GpuGraphSupport::alive_gpu_graph_execs_;

/*static*/ size_t GpuGraphSupport::NotifyGraphExecCreated() {
  alive_gpu_graph_execs_.fetch_add(1, std::memory_order_relaxed);
  return allocated_gpu_graph_execs_.fetch_add(1, std::memory_order_relaxed);
}

/*static*/ size_t GpuGraphSupport::NotifyGraphExecDestroyed() {
  return alive_gpu_graph_execs_.fetch_sub(1, std::memory_order_relaxed) - 1;
}

/*static*/ size_t GpuGraphSupport::allocated_gpu_graph_execs() {
  return allocated_gpu_graph_execs_.load(std::memory_order_relaxed);
}

/*static*/ size_t GpuGraphSupport::alive_gpu_graph_execs() {
  return alive_gpu_graph_execs_.load(std::memory_order_relaxed);
}

void GpuGraphSupport::DestroyGraph::operator()(GpuGraphHandle graph) {
  auto st = GpuDriver::DestroyGraph(graph);
  CHECK(st.ok()) << "Failed to destroy gpu graph: " << st.message();
}

void GpuGraphSupport::DestroyGraphExec::operator()(GpuGraphExecHandle exec) {
  auto st = GpuDriver::DestroyGraphExec(exec);
  CHECK(st.ok()) << "Failed to destroy executable gpu graph: " << st.message();
}

tsl::StatusOr<std::string> GraphExecUpdateResultToString(
    GpuDriver::GraphExecUpdateResult result) {
  switch (result) {
    case GpuDriver::GraphExecUpdateResult::kSuccess:
      return "kSuccess";
    case GpuDriver::GraphExecUpdateResult::kError:
      return "kFailure";
    case GpuDriver::GraphExecUpdateResult::kTopologyChanged:
      return "kTopologyChanged";
    case GpuDriver::GraphExecUpdateResult::kAttributesChanged:
      return "kAttributesChanged";
    case GpuDriver::GraphExecUpdateResult::kFunctionChanged:
      return "kFunctionChanged";
    case GpuDriver::GraphExecUpdateResult::kParametersChanged:
      return "kParametersChanged";
    case GpuDriver::GraphExecUpdateResult::kUnsupportedFunctionChange:
      return "kUnsupportedFunctionChange";
    case GpuDriver::GraphExecUpdateResult::kNodeTypeChanged:
      return "kNodeTypeChanged";
    case GpuDriver::GraphExecUpdateResult::kNotSupported:
      return "kNotSupported";
  }
  return tsl::errors::Internal("Unexpected value for GraphExecUpdateResult");
}

tsl::StatusOr<std::string> GraphNodeTypeToString(
    GpuDriver::GraphNodeType node_type) {
  switch (node_type) {
    case GpuDriver::GraphNodeType::kKernel:
      return "kKernel";
    case GpuDriver::GraphNodeType::kMemcpy:
      return "kMemcpy";
    case GpuDriver::GraphNodeType::kMemset:
      return "kMemset";
    case GpuDriver::GraphNodeType::kHost:
      return "kHost";
    case GpuDriver::GraphNodeType::kGraph:
      return "kGraph";
    case GpuDriver::GraphNodeType::kEmpty:
      return "kEmpty";
    case GpuDriver::GraphNodeType::kWaitEvent:
      return "kWaitEvent";
    case GpuDriver::GraphNodeType::kEventRecord:
      return "kEventRecord";
    case GpuDriver::GraphNodeType::kExtSemasSignal:
      return "kExtSemasSignal";
    case GpuDriver::GraphNodeType::kExtSemasWait:
      return "kExtSemasWait";
    case GpuDriver::GraphNodeType::kMemAlloc:
      return "kMemAlloc";
    case GpuDriver::GraphNodeType::kMemFree:
      return "kMemFree";
    case GpuDriver::GraphNodeType::kBatchMemOp:
      return "kBatchMemOp";
  }
  return tsl::errors::Internal("Unexpected value for GraphNodeType");
}

tsl::Status OwnedGpuGraphExec::Update(OwnedGpuGraph graph) {
  VLOG(3) << "Update gpu graph exec with a new graph after " << num_launches_
          << " launches since last update"
          << " #" << num_updates_++;

  num_launches_ = 0;

  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  GpuDriver::GraphExecUpdateResultInfo result;
  memset(&result, 0, sizeof(result));
  auto st = GpuDriver::GraphExecUpdate(get(), graph.get(), &result);
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!st.ok()) {
    TF_ASSIGN_OR_RETURN(std::string result_str,
                        GraphExecUpdateResultToString(result.result));
    std::string error_message = absl::StrCat(
        "Failed to update gpu graph: Graph update result=", result_str);

    if (result.error_node) {
      TF_ASSIGN_OR_RETURN(GpuDriver::GraphNodeType node_type,
                          GpuDriver::GraphNodeGetType(result.error_node));
      TF_ASSIGN_OR_RETURN(std::string node_type_str,
                          GraphNodeTypeToString(node_type));
      absl::StrAppend(&error_message, ", Error node name=", node_type_str);
    }

    if (result.error_from_node) {
      TF_ASSIGN_OR_RETURN(GpuDriver::GraphNodeType node_type,
                          GpuDriver::GraphNodeGetType(result.error_from_node));
      TF_ASSIGN_OR_RETURN(std::string node_type_str,
                          GraphNodeTypeToString(node_type));
      absl::StrAppend(&error_message, ", Error from node name=", node_type_str);
    }

    absl::StrAppend(&error_message, ": ", st.message());
    return tsl::errors::Internal(error_message);
  }

  VLOG(5) << "Updated gpu graph exec #" << id_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " us)";

  return tsl::OkStatus();
}

tsl::Status OwnedGpuGraphExec::Launch(stream_executor::Stream* stream) {
  VLOG(3) << "Launch gpu graph " << get()
          << " on a stream: " << stream->DebugStreamPointers() << " #"
          << ++num_launches_;

  return GpuDriver::GraphLaunch(get(), AsGpuStreamValue(stream));
}

OwnedGpuGraphExec::~OwnedGpuGraphExec() {
  if (*this)  // do not log for moved-from instances
    VLOG(5) << "Destroy GPU graph exec #" << id_
            << " (remaining alive instances: "
            << GpuGraphSupport::NotifyGraphExecDestroyed() << ")";
}

//===----------------------------------------------------------------------===//
// GPU Graph Helpers.
//===----------------------------------------------------------------------===//

tsl::StatusOr<OwnedGpuGraph> CreateGpuGraph() {
  GpuGraphHandle graph;
  TF_RETURN_IF_ERROR(GpuDriver::CreateGraph(&graph));
  return OwnedGpuGraph(graph);
}

tsl::StatusOr<GpuGraphNodeHandle> AddKernelNode(
    GpuGraphHandle graph, absl::Span<GpuGraphNodeHandle> deps,
    ThreadDim threads, BlockDim blocks, const KernelBase& kernel,
    const KernelArgsArrayBase& args) {
  const GpuKernel* gpu_kernel = AsGpuKernel(&kernel);
  GpuFunctionHandle gpu_func = gpu_kernel->AsGpuFunctionHandle();

  void** kernel_params = const_cast<void**>(args.argument_addresses().data());

  GpuGraphNodeHandle node;
  TF_RETURN_IF_ERROR(GpuDriver::GraphAddKernelNode(
      &node, graph, deps, kernel.name(), gpu_func, blocks.x, blocks.y, blocks.z,
      threads.x, threads.y, threads.z, args.number_of_shared_bytes(),
      kernel_params, /*extra=*/nullptr));

  return node;
}

static GpuDevicePtr AsDevicePtr(const DeviceMemoryBase& mem) {
  return reinterpret_cast<GpuDevicePtr>(const_cast<void*>(mem.opaque()));
}

tsl::StatusOr<GpuGraphNodeHandle> AddMemcpyD2DNode(
    GpuContext* context, GpuGraphHandle graph,
    absl::Span<GpuGraphNodeHandle> deps, const DeviceMemoryBase& dst,
    const DeviceMemoryBase& src) {
  GpuGraphNodeHandle node;
  TF_RETURN_IF_ERROR(GpuDriver::GraphAddMemcpyD2DNode(
      context, &node, graph, deps, AsDevicePtr(dst), AsDevicePtr(src),
      dst.size()));
  return node;
}

tsl::StatusOr<OwnedGpuGraph> CaptureGpuGraph(
    stream_executor::Stream* stream,
    absl::AnyInvocable<tsl::Status()> capture) {
  VLOG(3) << "Capture gpu graph on a stream: " << stream->DebugStreamPointers();
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();

  GpuGraphHandle graph;

  // Get the underlying stream for passing to GPU runtime APIs.
  auto gpu_stream = AsGpuStreamValue(stream);

  // Capture graph constructed by the exported graph capture function.
  TF_RETURN_IF_ERROR(GpuDriver::StreamBeginCapture(
      gpu_stream, GpuDriver::StreamCaptureMode::kThreadLocal));

  // Call into graph capture function.
  auto captured = capture();

  // Always stop capturing the stream before checking `captured` result.
  TF_RETURN_IF_ERROR(GpuDriver::StreamEndCapture(gpu_stream, &graph));

  if (!captured.ok())
    return tsl::errors::Internal("failed to capture gpu graph: ",
                                 captured.message());

  uint64_t end_nanos = tsl::Env::Default()->NowNanos();
  VLOG(5) << "Captured XLA:GPU operations into the graph " << graph << " (took "
          << (end_nanos - start_nanos) / 1000 << " us)";

  if (const char* path = getenv("XLA_GPU_GRAPH_DEBUG_DIRECTORY"); path) {
    std::string file = tsl::io::JoinPath(std::string(path), "/gpu-graph-");

    if (tsl::Env::Default()->CreateUniqueFileName(&file, ".dot")) {
      VLOG(100) << "Print gpu graph " << graph
                << " debug dot file to: " << file;
      auto printed = GpuDriver::GraphDebugDotPrint(graph, file.c_str());
      printed.IgnoreError();  // warning will be printed by GpuDriver
    } else {
      LOG(WARNING) << "Cannot create unique filename, won't enable gpu "
                      "graph debugging";
    }
  }

  return OwnedGpuGraph(graph);
}

tsl::StatusOr<OwnedGpuGraphExec> InstantiateGpuGraph(OwnedGpuGraph graph) {
  GpuGraphExecHandle exec;

  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  GpuDriver::GraphInstantiateFlags flags;
  TF_RETURN_IF_ERROR(GpuDriver::GraphInstantiate(&exec, graph.get(), flags));
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  size_t id = GpuGraphSupport::NotifyGraphExecCreated();
  VLOG(5) << "Instantiated gpu graph exec instance #" << id << " in "
          << (end_nanos - start_nanos) / 1000 << " us (alive instances: "
          << GpuGraphSupport::alive_gpu_graph_execs() << ")";
  return OwnedGpuGraphExec(id, exec);
}

tsl::StatusOr<bool> IsStreamCapturing(stream_executor::Stream* stream) {
  return GpuDriver::StreamIsCapturing(AsGpuStreamValue(stream));
}

}  // namespace gpu
}  // namespace stream_executor
