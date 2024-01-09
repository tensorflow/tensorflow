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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_GRAPH_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_GRAPH_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

// Forward declare.
class GpuContext;

class GpuGraphSupport {
 public:
  // Deleters for gpu graph and graph exec instance that check the returned
  // status and terminate on error.
  struct DestroyGraph {
    void operator()(GpuGraphHandle);
  };
  struct DestroyGraphExec {
    void operator()(GpuGraphExecHandle);
  };

  static size_t NotifyGraphExecCreated();
  static size_t NotifyGraphExecDestroyed();

  static size_t allocated_gpu_graph_execs();
  static size_t alive_gpu_graph_execs();

  static void TrimDeviceMemory(StreamExecutor* executor);

 private:
  // Global counters for the total number of allocated and alive gpu graph
  // execs to track the resource usage at run time.
  static std::atomic<size_t> allocated_gpu_graph_execs_;
  static std::atomic<size_t> alive_gpu_graph_execs_;
};

//===----------------------------------------------------------------------===//
// RAII helpers for gpu graph types.
//===----------------------------------------------------------------------===//

class OwnedGpuGraph
    : public std::unique_ptr<std::remove_pointer_t<GpuGraphHandle>,
                             GpuGraphSupport::DestroyGraph> {
  // Bring std::unique_ptr constructors in scope.
  using std::unique_ptr<std::remove_pointer_t<GpuGraphHandle>,
                        GpuGraphSupport::DestroyGraph>::unique_ptr;
};

class OwnedGpuGraphExec
    : public std::unique_ptr<std::remove_pointer_t<GpuGraphExecHandle>,
                             GpuGraphSupport::DestroyGraphExec> {
  using Base = std::unique_ptr<std::remove_pointer_t<GpuGraphExecHandle>,
                               GpuGraphSupport::DestroyGraphExec>;

 public:
  OwnedGpuGraphExec(uint64_t id, GpuGraphExecHandle exec)
      : Base(exec), id_(id) {}
  ~OwnedGpuGraphExec();

  OwnedGpuGraphExec(OwnedGpuGraphExec&&) = default;
  OwnedGpuGraphExec& operator=(OwnedGpuGraphExec&&) = default;

  // Updates executable graph instance with a newly captured graph. Returns an
  // error if the new graph is not compatible (see `cudaGraphExecUpdate`).
  tsl::Status Update(OwnedGpuGraph graph);

  // Launches captured graph on a given stream.
  tsl::Status Launch(stream_executor::Stream* stream);

  uint64_t id() const { return id_; }

 private:
  uint64_t id_;
  uint64_t num_updates_ = 0;
  uint64_t num_launches_ = 0;
};

//===----------------------------------------------------------------------===//
// Gpu Graph Helpers.
//===----------------------------------------------------------------------===//

// Creates new empty Gpu graph.
tsl::StatusOr<OwnedGpuGraph> CreateGpuGraph();

// Adds a kernel node to the graph.
tsl::StatusOr<GpuGraphNodeHandle> AddKernelNode(
    GpuGraphHandle graph, absl::Span<GpuGraphNodeHandle> deps,
    ThreadDim threads, BlockDim blocks, const Kernel& kernel,
    const KernelArgs& args);

// Adds a memory copy node to the graph.
tsl::StatusOr<GpuGraphNodeHandle> AddMemcpyD2DNode(
    GpuContext* context, GpuGraphHandle graph,
    absl::Span<GpuGraphNodeHandle> deps, const DeviceMemoryBase& dst,
    const DeviceMemoryBase& src);

// Captures all operations added to a `stream` by the `capture` function into
// the gpu graph instance.
tsl::StatusOr<OwnedGpuGraph> CaptureGpuGraph(
    stream_executor::Stream* stream, absl::AnyInvocable<tsl::Status()> capture);

// Instantiates a captured gpu graph instance into a gpu graph executable.
tsl::StatusOr<OwnedGpuGraphExec> InstantiateGpuGraph(OwnedGpuGraph graph);

// Returns true if the stream is in graph capture mode
tsl::StatusOr<bool> IsStreamCapturing(stream_executor ::Stream* stream);

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_GRAPH_H_
