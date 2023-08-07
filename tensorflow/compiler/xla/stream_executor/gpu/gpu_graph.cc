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

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_graph.h"

#include <atomic>
#include <string>

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/path.h"

#if TENSORFLOW_USE_ROCM
using namespace stream_executor::wrap;  // NOLINT[build/namespaces]
#define GPU_PREFIX hip
#else
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#define GPU_PREFIX cuda
#endif

#define GPU_CAT_NX(A, B) A##B
#define GPU_CAT(A, B) GPU_CAT_NX(A, B)
#define GPU(A) GPU_CAT(GPU_PREFIX, A)

#define GpuGetErrorString GPU(GetErrorString)
#define GpuGraphDebugDotFlagsVerbose GPU(GraphDebugDotFlagsVerbose)
#define GpuGraphDebugDotPrint GPU(GraphDebugDotPrint)
#define GpuGraphDestroy GPU(GraphDestroy)
#define GpuErrorMemoryAllocation GPU(ErrorMemoryAllocation)
#define GpuGraphExecDestroy GPU(GraphExecDestroy)
#define GpuGraphExecUpdate GPU(GraphExecUpdate)
#define GpuGraphExecUpdateResult GPU(GraphExecUpdateResult)
#define GpuGraphExecUpdateSuccess GPU(GraphExecUpdateSuccess)
#define GpuGraphInstantiate GPU(GraphInstantiate)
#define GpuGraphLaunch GPU(GraphLaunch)
#define GpuGraphNode GPU(GraphNode_t)
#define GpuStreamBeginCapture GPU(StreamBeginCapture)
#define GpuStreamCaptureModeThreadLocal GPU(StreamCaptureModeThreadLocal)
#define GpuStreamCaptureStatus GPU(StreamCaptureStatus)
#define GpuStreamCaptureStatusActive GPU(StreamCaptureStatusActive)
#define GpuStreamEndCapture GPU(StreamEndCapture)
#define GpuStreamIsCapturing GPU(StreamIsCapturing)
#define GpuSuccess GPU(Success)

#define RETURN_IF_GPU_GRAPH_ERROR(expr, ...)                 \
  do {                                                       \
    auto _res = (expr);                                      \
    if (TF_PREDICT_FALSE(_res != GpuSuccess)) {              \
      return tsl::errors::Internal(__VA_ARGS__, ": ",        \
                                   GpuGetErrorString(_res)); \
    }                                                        \
  } while (0)

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
  auto err = GpuGraphDestroy(graph);
  CHECK(err == GpuSuccess) << "Failed to destroy gpu graph: "
                           << GpuGetErrorString(err);
}

void GpuGraphSupport::DestroyGraphExec::operator()(
    GpuGraphExecHandle instance) {
  auto err = GpuGraphExecDestroy(instance);
  CHECK(err == GpuSuccess) << "Failed to destroy gpu graph instance: "
                           << GpuGetErrorString(err);
}

tsl::Status OwnedGpuGraphExec::Update(OwnedGpuGraph graph) {
  VLOG(3) << "Update gpu graph exec with a new graph after " << num_launches_
          << " launches since last update"
          << " #" << num_updates_++;

  num_launches_ = 0;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  cudaGraphExecUpdateResultInfo updated;

  auto err = cudaGraphExecUpdate(get(), graph.get(), &updated);
  if (err != cudaSuccess || updated.result != cudaGraphExecUpdateSuccess)
    return tsl::errors::Internal("Failed to update gpu graph: ",
                                 GpuGetErrorString(err));

#else
  GpuGraphExecUpdateResult updated;
  GpuGraphNode error_node;

  auto err = GpuGraphExecUpdate(get(), graph.get(), &error_node, &updated);
  if (err != GpuSuccess || updated != GpuGraphExecUpdateSuccess)
    return tsl::errors::Internal("Failed to update gpu graph: ",
                                 GpuGetErrorString(err));
#endif

  return tsl::OkStatus();
}

tsl::Status OwnedGpuGraphExec::Launch(stream_executor::Stream* stream) {
  VLOG(3) << "Launch gpu graph " << get()
          << " on a stream: " << stream->DebugStreamPointers() << " #"
          << ++num_launches_;

  RETURN_IF_GPU_GRAPH_ERROR(GpuGraphLaunch(get(), AsGpuStreamValue(stream)),
                            "failed to run gpu graph");

  return tsl::OkStatus();
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

tsl::StatusOr<OwnedGpuGraph> CaptureGpuGraph(
    stream_executor::Stream* stream,
    absl::AnyInvocable<tsl::Status()> capture) {
  VLOG(3) << "Capture gpu graph on a stream: " << stream->DebugStreamPointers();

  GpuGraphHandle graph;

  // Get the underlying stream for passing to GPU runtime APIs.
  auto gpu_stream = AsGpuStreamValue(stream);

  // Capture graph constructed by the exported graph capture function.
  RETURN_IF_GPU_GRAPH_ERROR(
      GpuStreamBeginCapture(gpu_stream, GpuStreamCaptureModeThreadLocal),
      "stream begin capture failed");

  // Call into graph capture function.
  auto captured = capture();

  // Always stop capturing the stream before checking `captured` result.
  RETURN_IF_GPU_GRAPH_ERROR(GpuStreamEndCapture(gpu_stream, &graph),
                            "stream end capture failed");

  if (!captured.ok())
    return tsl::errors::Internal("failed to capture gpu graph: ",
                                 captured.message());

  VLOG(5) << "Captured gpu graph " << graph;

#if TENSORFLOW_USE_ROCM || CUDA_VERSION >= 12000
  // If verbose logging is enabled print captured gpu graph debug information.
  if (VLOG_IS_ON(100)) {
    if (const char* path = getenv("XLA_GPU_GRAPH_DEBUG_DIRECTORY"); path) {
      std::string file = tsl::io::JoinPath(std::string(path), "/gpu_graph-");

      if (tsl::Env::Default()->CreateUniqueFileName(&file, ".dot")) {
        VLOG(100) << "Print gpu graph " << graph
                  << " debug dot file to: " << file;

        int flags = GpuGraphDebugDotFlagsVerbose;
        if (auto err = GpuGraphDebugDotPrint(graph, file.c_str(), flags);
            err != GpuSuccess) {
          LOG(WARNING) << "failed to print gpu graph debug file: "
                       << GpuGetErrorString(err);

        } else if (VLOG_IS_ON(200)) {
          std::string data;
          if (tsl::ReadFileToString(tsl::Env::Default(), file, &data).ok()) {
            VLOG(200) << "gpu graph " << graph << " debug file:\n" << data;
          } else {
            LOG(WARNING) << "failed to read gpu graph debug file";
          }
        }

      } else {
        LOG(WARNING) << "cannot create unique filename, won't enable gpu "
                        "graph debugging";
      }
    }
  }
#endif  // TENSORFLOW_USE_ROCM || CUDA_VERSION >= 12000

  return OwnedGpuGraph(graph);
}

tsl::StatusOr<OwnedGpuGraphExec> InstantiateGpuGraph(OwnedGpuGraph graph) {
  GpuGraphExecHandle exec;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
  if (auto err = cudaGraphInstantiate(&exec, &*graph);
#else
  if (auto err = GpuGraphInstantiate(&exec, &*graph, nullptr, nullptr, 0);
#endif
      err != GpuSuccess) {
    if (err == GpuErrorMemoryAllocation) {
      // OOM is a recoverable error, we evict all instantiated cuda graphs to
      // free up some space (see graph launch.cc). Clear error status.
      return absl::ResourceExhaustedError(absl::StrFormat(
          "graph instantiation failed: %s", GpuGetErrorString(err)));
    } else {
      return absl::InternalError(absl::StrFormat(
          "graph instantiation failed: %s", GpuGetErrorString(err)));
    }
  }

  size_t id = GpuGraphSupport::NotifyGraphExecCreated();
  VLOG(5) << "Instantiated gpu graph exec instance #" << id
          << " (alive instances: " << GpuGraphSupport::alive_gpu_graph_execs()
          << ")";
  return OwnedGpuGraphExec(id, exec);
}

tsl::StatusOr<bool> IsStreamCapturing(stream_executor::Stream* stream) {
  GpuStreamCaptureStatus capture_status;
  RETURN_IF_GPU_GRAPH_ERROR(
      GpuStreamIsCapturing(stream_executor::gpu::AsGpuStreamValue(stream),
                           &capture_status),
      "Failed to get stream's capture status");

  return capture_status == GpuStreamCaptureStatusActive;
}

}  // namespace gpu
}  // namespace stream_executor
