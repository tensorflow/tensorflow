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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_GRAPH_LAUNCH_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_GRAPH_LAUNCH_H_

#include <memory>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>

#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif  // #if GOOGLE_CUDA

namespace xla {
namespace gpu {

#if GOOGLE_CUDA
struct GraphInstance;  // Forward declare

// A state vector that owns all instantiated CUDA graphs. Graph capture function
// ordinal is the key in this container.
class GraphInstances : public runtime::StateVector<GraphInstance> {
  // Deleters for CUDA graph and graph exec instance that check the returned
  // status and terminate if it's not `cudaSuccess`.
  struct DestroyGraph {
    void operator()(cudaGraph_t);
  };
  struct DestroyGraphExec {
    void operator()(cudaGraphExec_t);
  };

 public:
  using OwnedGraph =
      std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, DestroyGraph>;
  using OwnedGraphExec =
      std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, DestroyGraphExec>;
};

// Instantiated CUDA graph instance guarded with a mutex for exclusive access.
struct GraphInstance {
  GraphInstance(size_t ptr_hash, cudaGraphExec_t exec)
      : ptr_hash(ptr_hash), exec(exec), mutex(new absl::Mutex) {}

  // Graph instance is fully identified by the hash of its pointer arguments
  // because currently it's guaranteed that all shapes and launch dimensions
  // will be constant from run to run.
  size_t ptr_hash ABSL_GUARDED_BY(*mutex);
  GraphInstances::OwnedGraphExec exec ABSL_GUARDED_BY(*mutex);

  // Access to a graph instance must be synchronized, because we potentially can
  // run concurrent graph instance updates.
  std::unique_ptr<absl::Mutex> mutex;
};

#else  // #if !GOOGLE_CUDA

// Define empty struct and empty state when CUDA is not enabled.
struct GraphInstance {};
class GraphInstances : public runtime::StateVector<GraphInstance> {};

#endif  // #if GOOGLE_CUDA

// Registers XLA Gpu runtime graph launch custom calls.
void RegisterGraphLaunchCustomCalls(
    runtime::DirectCustomCallRegistry& registry);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_GRAPH_LAUNCH_H_
