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

#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_graph.h"

#include "absl/strings/str_format.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"

namespace stream_executor {
namespace gpu {

template <typename... Args>
static tsl::Status InternalError(const absl::FormatSpec<Args...>& format,
                                 const Args&... args) {
  return tsl::errors::Internal(absl::StrFormat(format, args...));
}

//===----------------------------------------------------------------------===//
// RAII helpers for CUDA graph types.
//===----------------------------------------------------------------------===//

void CudaGraphSupport::DestroyGraph::operator()(cudaGraph_t graph) {
  cudaError_t err = cudaGraphDestroy(graph);
  CHECK(err == cudaSuccess)
      << "Failed to destroy CUDA graph: " << cudaGetErrorString(err);
}

void CudaGraphSupport::DestroyGraphExec::operator()(cudaGraphExec_t instance) {
  cudaError_t err = cudaGraphExecDestroy(instance);
  CHECK(err == cudaSuccess)
      << "Failed to destroy CUDA graph instance: " << cudaGetErrorString(err);
}

tsl::Status OwnedCudaGraphExec::Update(OwnedCudaGraph graph) {
  VLOG(3) << "Update CUDA graph exec with a new graph";

#if CUDA_VERSION >= 12000
  cudaGraphExecUpdateResultInfo updated;

  auto err = cudaGraphExecUpdate(get(), graph.get(), &updated);
  if (err != cudaSuccess || updated.result != cudaGraphExecUpdateSuccess)
    return InternalError("failed to update cuda graph: %s",
                         cudaGetErrorString(err));

#else
  cudaGraphExecUpdateResult updated;
  cudaGraphNode_t error_node;

  auto err = cudaGraphExecUpdate(get(), graph.get(), &error_node, &updated);
  if (err != cudaSuccess || updated != cudaGraphExecUpdateSuccess)
    return InternalError("Failed to update cuda graph %s",
                         cudaGetErrorString(err));
#endif

  return tsl::OkStatus();
}

tsl::Status OwnedCudaGraphExec::Launch(stream_executor::Stream* stream) {
  VLOG(3) << "Launch CUDA graph on a stream: " << stream->DebugStreamPointers();

  if (auto err = cudaGraphLaunch(get(), AsGpuStreamValue(stream));
      err != cudaSuccess)
    return InternalError("failed to run cuda graph: %s",
                         cudaGetErrorString(err));

  return tsl::OkStatus();
}

//===----------------------------------------------------------------------===//
// CUDA Graph Helpers.
//===----------------------------------------------------------------------===//

tsl::StatusOr<OwnedCudaGraph> CaptureCudaGraph(
    stream_executor::Stream* stream, absl::AnyInvocable<tsl::Status()> capture,
    cudaStreamCaptureMode mode) {
  VLOG(3) << "Capture CUDA graph on a stream: "
          << stream->DebugStreamPointers();

  cudaGraph_t graph;

  // Get the underlying CUDA stream for passing to CUDA APIs.
  auto gpu_stream = AsGpuStreamValue(stream);

  // Capture graph constructed by the exported graph capture function.
  if (auto err = cudaStreamBeginCapture(gpu_stream, mode); err != cudaSuccess)
    return InternalError("stream begin capture failed: %s",
                         cudaGetErrorString(err));

  // Call into graph capture function.
  auto captured = capture();

  // Always stop capturing the stream before checking `captured` result.
  if (auto err = cudaStreamEndCapture(gpu_stream, &graph); err != cudaSuccess)
    return InternalError("stream end capture failed: %s",
                         cudaGetErrorString(err));

  if (!captured.ok())
    return InternalError("failed to capture CUDA graph: %s",
                         captured.error_message());

  return OwnedCudaGraph(graph);
}

tsl::StatusOr<OwnedCudaGraphExec> InstantiateCudaGraph(OwnedCudaGraph graph) {
  cudaGraphExec_t exec;

#if CUDA_VERSION >= 12000
  if (auto err = cudaGraphInstantiate(&exec, &*graph);
#else
  if (auto err = cudaGraphInstantiate(&exec, &*graph, nullptr, nullptr, 0);
#endif
      err != cudaSuccess) {
    return InternalError("graph instantiation failed: %s",
                         cudaGetErrorString(err));
  }

  return OwnedCudaGraphExec(exec);
}

}  // namespace gpu
}  // namespace stream_executor
