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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_LISTENER_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_LISTENER_H_

#include <cstddef>

#include "absl/strings/string_view.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"

namespace stream_executor::gpu {

// Listener interface for monitoring CUDA Graph events without direct
// dependencies on the Profiler.
class GpuCommandBufferListener {
 public:
  virtual ~GpuCommandBufferListener() = default;

  // Called when an HLO annotation is associated with a graph node.
  virtual void OnRegisterNodeAnnotation(void* graph,
                                        GpuCommandBuffer::GraphNodeHandle node,
                                        absl::string_view annotation) = 0;

  // Called when a graph's total size (number of nodes) is recorded.
  virtual void OnRegisterGraphSize(void* graph, size_t size) = 0;

  // Called when a child graph (e.g., conditional or loop body) is associated
  // with a node in a parent graph.
  virtual void OnRegisterChildGraph(
      void* parent_graph, void* child_graph,
      GpuCommandBuffer::GraphNodeHandle child_node, bool is_conditional) = 0;

  // Called when an executable graph is instantiated from a graph definition.
  virtual void OnRegisterGraphExec(void* graph_exec, void* graph) = 0;

  // Called when an executable graph is destroyed.
  virtual void OnUnregisterGraphExec(void* graph_exec) = 0;

  // Called when a graph definition is destroyed, to clean up associated
  // annotations.
  virtual void OnUnregisterGraphAnnotations(void* graph) = 0;
};

// Registers the global GpuCommandBufferListener. Returns true if successful,
// or false if a listener is already registered. Thread-safe.
bool RegisterGpuCommandBufferListener(GpuCommandBufferListener* listener);

// Unregisters the given GpuCommandBufferListener. Returns true if successful,
// or false if the given listener was not the currently registered one.
// Thread-safe.
bool UnregisterGpuCommandBufferListener(GpuCommandBufferListener* listener);

// Retrieves the registered global GpuCommandBufferListener, or nullptr if none.
// Thread-safe.
GpuCommandBufferListener* GetGpuCommandBufferListener();

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_LISTENER_H_
