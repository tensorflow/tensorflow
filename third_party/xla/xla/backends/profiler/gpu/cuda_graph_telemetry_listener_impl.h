/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUDA_GRAPH_TELEMETRY_LISTENER_IMPL_H_
#define XLA_BACKENDS_PROFILER_GPU_CUDA_GRAPH_TELEMETRY_LISTENER_IMPL_H_

#include <cstddef>

#include "absl/strings/string_view.h"
#include "xla/stream_executor/gpu/gpu_command_buffer_listener.h"

namespace xla {
namespace profiler {

// Implementation of the StreamExecutor's GpuCommandBufferListener interface.
// Intercepts low-level CUDA Graph creation events, retrieves the active
// thread-local HLO Module annotation, and populates the global
// CudaGraphAnnotationRegistry.
class CudaGraphTelemetryListenerImpl
    : public stream_executor::gpu::GpuCommandBufferListener {
 public:
  CudaGraphTelemetryListenerImpl() = default;
  ~CudaGraphTelemetryListenerImpl() override = default;

  // Called when an HLO annotation is associated with a graph node.
  void OnRegisterNodeAnnotation(
      void* graph, stream_executor::gpu::GpuCommandBuffer::GraphNodeHandle node,
      absl::string_view annotation) override;

  // Called when a graph's total size (number of nodes) is recorded.
  // Automatically retrieves the active HLO Module annotation from the
  // thread-local ScopedCommandBufferAnnotation context.
  void OnRegisterGraphSize(void* graph, size_t size) override;

  // Called when a child graph is associated with a node in a parent graph.
  void OnRegisterChildGraph(
      void* parent_graph, void* child_graph,
      stream_executor::gpu::GpuCommandBuffer::GraphNodeHandle child_node,
      bool is_conditional) override;

  // Called when an executable graph is instantiated from a graph definition.
  void OnRegisterGraphExec(void* graph_exec, void* graph) override;

  // Called when an executable graph is destroyed.
  void OnUnregisterGraphExec(void* graph_exec) override;

  // Called when a graph definition is destroyed.
  void OnUnregisterGraphAnnotations(void* graph) override;

 private:
  bool IsEnabled() const override;

  CudaGraphTelemetryListenerImpl(const CudaGraphTelemetryListenerImpl&) =
      delete;
  CudaGraphTelemetryListenerImpl& operator=(
      const CudaGraphTelemetryListenerImpl&) = delete;
};

// Registers the CudaGraphTelemetryListenerImpl instance globally.
void RegisterCudaGraphTelemetryListener();

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUDA_GRAPH_TELEMETRY_LISTENER_IMPL_H_
