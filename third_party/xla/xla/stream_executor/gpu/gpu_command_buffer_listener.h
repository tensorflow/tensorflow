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

// Opaque wrapper for a GPU graph handle (e.g., `CUgraph` on CUDA).
// This wrapper is used to keep platform-specific graph handle types out of the
// public stream_executor API, while enabling type-safe tracking of graph
// resources by telemetry tools.
class GpuGraphHandle {
 public:
  // Constructs a GpuGraphHandle from a raw opaque pointer.
  explicit GpuGraphHandle(void* opaque = nullptr) : opaque_(opaque) {}

  // Returns the underlying raw opaque pointer.
  void* opaque() const { return opaque_; }

  // Returns true if the handle is null.
  bool is_null() const { return opaque_ == nullptr; }

  // Comparisons are performed on the underlying raw pointer values.
  bool operator==(const GpuGraphHandle& other) const {
    return opaque_ == other.opaque_;
  }
  bool operator!=(const GpuGraphHandle& other) const {
    return opaque_ != other.opaque_;
  }
  bool operator<(const GpuGraphHandle& other) const {
    return opaque_ < other.opaque_;
  }

  // Support for absl hashing to allow using this handle as a key in hash maps.
  template <typename H>
  friend H AbslHashValue(H h, const GpuGraphHandle& handle) {
    return H::combine(std::move(h), handle.opaque_);
  }

 private:
  void* opaque_;
};

// Opaque wrapper for an executable GPU graph handle (e.g., `CUgraphExec` on
// CUDA). This wrapper is used to keep platform-specific executable graph handle
// types out of the public stream_executor API, while enabling type-safe
// tracking of executable graph resources by telemetry tools.
class GpuGraphExecHandle {
 public:
  // Constructs a GpuGraphExecHandle from a raw opaque pointer.
  explicit GpuGraphExecHandle(void* opaque = nullptr) : opaque_(opaque) {}

  // Returns the underlying raw opaque pointer.
  void* opaque() const { return opaque_; }

  // Returns true if the handle is null.
  bool is_null() const { return opaque_ == nullptr; }

  // Comparisons are performed on the underlying raw pointer values.
  bool operator==(const GpuGraphExecHandle& other) const {
    return opaque_ == other.opaque_;
  }
  bool operator!=(const GpuGraphExecHandle& other) const {
    return opaque_ != other.opaque_;
  }
  bool operator<(const GpuGraphExecHandle& other) const {
    return opaque_ < other.opaque_;
  }

  // Support for absl hashing to allow using this handle as a key in hash maps.
  template <typename H>
  friend H AbslHashValue(H h, const GpuGraphExecHandle& handle) {
    return H::combine(std::move(h), handle.opaque_);
  }

 private:
  void* opaque_;
};

class ScopedGpuCommandBufferListenerOverrideForTesting;

// Listener interface for monitoring CUDA Graph events without direct
// dependencies on the Profiler.
class GpuCommandBufferListener {
 public:
  virtual ~GpuCommandBufferListener() = default;

  // Returns true if the listener is enabled and should receive events.
  virtual bool IsEnabled() const { return true; }

  // Called when an HLO annotation is associated with a graph node.
  virtual void OnRegisterNodeAnnotation(GpuGraphHandle graph,
                                        GpuCommandBuffer::GraphNodeHandle node,
                                        absl::string_view annotation) = 0;

  // Called when a graph's total size (number of nodes) is recorded.
  virtual void OnRegisterGraphSize(GpuGraphHandle graph, size_t size) = 0;

  // Called when a child graph (e.g., conditional or loop body) is associated
  // with a node in a parent graph.
  virtual void OnRegisterChildGraph(
      GpuGraphHandle parent_graph, GpuGraphHandle child_graph,
      GpuCommandBuffer::GraphNodeHandle child_node, bool is_conditional) = 0;

  // Called when an executable graph is instantiated from a graph definition.
  virtual void OnRegisterGraphExec(GpuGraphExecHandle graph_exec,
                                   GpuGraphHandle graph) = 0;

  // Called when an executable graph is destroyed.
  virtual void OnUnregisterGraphExec(GpuGraphExecHandle graph_exec) = 0;

  // Called when a graph definition is destroyed, to clean up associated
  // annotations.
  virtual void OnUnregisterGraphAnnotations(GpuGraphHandle graph) = 0;

 private:
  friend class ScopedGpuCommandBufferListenerOverrideForTesting;
  static GpuCommandBufferListener* ExchangeForTesting(
      GpuCommandBufferListener* listener);
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
