/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_GRAPH_LAUNCH_H_
#define XLA_SERVICE_GPU_RUNTIME_GRAPH_LAUNCH_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/node_hash_map.h"
#include "absl/types/span.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/executable.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/stream_executor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_graph.h"
#endif  // #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

// Registers XLA Gpu runtime graph launch custom calls.
void RegisterGraphLaunchCustomCalls(
    runtime::DirectCustomCallRegistry& registry);

struct GraphInstance;                // Forward declare
class StreamExecutorGraphInstances;  // Forward declare

// A state vector that keeps track of the number of times a capture function
// gets executed. Graph capture function ordinal is the key in this container.
class CapturedFunctionExecutionCount
    : public runtime::StateVector<std::unique_ptr<std::atomic<uint64_t>>> {};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// A state vector that owns all instantiated GPU graphs. Graph capture function
// ordinal is the key in this container.
class StreamExecutorGraphInstances
    : public runtime::StateVector<GraphInstance> {};

// Instantiated GPU graph instance guarded with a mutex for exclusive access.
struct GraphInstance {
  GraphInstance(size_t ptr_hash, se::gpu::OwnedGpuGraphExec exec)
      : ptr_hash(ptr_hash), exec(std::move(exec)), mutex(new absl::Mutex) {}

  // Graph instance is fully identified by the hash of its pointer arguments
  // because currently it's guaranteed that all shapes and launch dimensions
  // will be constant from run to run.
  size_t ptr_hash ABSL_GUARDED_BY(*mutex);
  se::gpu::OwnedGpuGraphExec exec ABSL_GUARDED_BY(*mutex);

  // Access to a graph instance must be synchronized, because we potentially can
  // run concurrent graph instance updates.
  std::unique_ptr<absl::Mutex> mutex;
};

#else  // #if !GOOGLE_CUDA && !TENSORFLOW_USE_ROCM

// Define empty struct and empty state when GPU is not enabled.
struct GraphInstance {};
class StreamExecutorGraphInstances
    : public runtime::StateVector<GraphInstance> {};

#endif  // #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Xla executable keeps a mapping from stream executors to graph instances.
//
// Graph instances allocate on-device memory, so we periodically destroy
// them to free up some space on device. JAX for example keeps all XLA
// executables alive, and destroys them when the process shuts down, so we can
// end up with thousands of unused (or rarely used) graphs in device memory.
class GraphInstances {
 public:
  struct Impl;

  GraphInstances(std::string module_name, int64_t num_graphs);
  ~GraphInstances();

  std::shared_ptr<StreamExecutorGraphInstances> operator()(
      se::StreamExecutor* executor);

  // Instantiates all Gpu graphs defined by the given executable using user
  // provided run options. This guarantees that once we start execution, all Gpu
  // graphs are ready, and will only require cheap update operation and will not
  // require allocating new resources (we avoid non deterministic OOM errors).
  //
  // If timeout is not nullopt it will evict all previously instantiated graphs
  // that were used more than `eviction_timeout_seconds` seconds ago.
  Status InstantiateAllGraphs(
      const ServiceExecutableRunOptions* run_options,
      const runtime::Executable& executable,
      const runtime::CustomCall::UserData& user_data,
      const BufferAllocations& buffer_allocations,
      absl::Span<const int64_t> buffer_sizes,
      absl::Span<const std::vector<int64_t>> allocation_indices,
      std::optional<uint64_t> eviction_timeout_seconds = std::nullopt);

  // Returns true if all Gpu graphs were already instantiated.
  bool InstantiatedAllGraphs(const ServiceExecutableRunOptions* run_options,
                             const runtime::Executable& executable);

 private:
  std::shared_ptr<Impl> impl_;
};

// Xla executable keeps a mapping from stream executors to execution counts.
class CapturedFunctionExecutionCounts {
 public:
  CapturedFunctionExecutionCount* operator()(se::StreamExecutor* executor);

 private:
  mutable absl::Mutex mutex_;
  absl::node_hash_map<se::StreamExecutor*, CapturedFunctionExecutionCount>
      counts_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_GRAPH_LAUNCH_H_
